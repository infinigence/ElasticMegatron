import gc
import os
import pathlib
from collections.abc import Callable

import torch
from megatron.core.distributed import DistributedDataParallel as DDP
from megatron.core.distributed import finalize_model_grads
from megatron.core.optimizer import (
    ChainedOptimizer,
    MegatronOptimizer,
)
from megatron.core.optimizer_param_scheduler import OptimizerParamScheduler
from megatron.core.utils import get_model_config
from megatron.training.checkpointing import save_checkpoint
from megatron.training.global_vars import get_args, get_timers
from megatron.training.training import preprocess_common_state_dict

from ..host_memory import trim_host_memory
from ..resharding.optimizer_adapter import OptimizerAdapter
from ..resharding.resharding_metadata import (
    OptimizerTensorInfo,
    generate_resharding_metadata,
)
from .parallel_strategy import ParallelStrategy


def clear_memory(
    sync_enable: bool = False, empty_cache_enable: bool = False, gc_enable: bool = False
):
    if sync_enable:
        torch.cuda.synchronize()
    if gc_enable:
        gc.collect()
    if empty_cache_enable:
        torch.cuda.empty_cache()


class TrainingState:
    """Megatron Training State."""

    setup_model_and_optimizer_func: Callable = None

    @staticmethod
    def param_migrate(
        src_training_state: "TrainingState", dst_training_state: "TrainingState"
    ):
        """Assert src/dst represent the same model expecet dp(without_ddp or group_zero)."""
        src_optimizer_tensor_info_list = src_training_state.optimizer_tensor_info_list
        dst_optimizer_tensor_info_list = dst_training_state.optimizer_tensor_info_list
        assert len(src_optimizer_tensor_info_list) == len(
            dst_optimizer_tensor_info_list
        )

        for src_optimizer_tensor_info, dst_optimizer_tensor_info in zip(
            src_optimizer_tensor_info_list, dst_optimizer_tensor_info_list
        ):
            dst_optimizer_tensor_info.rebuild()
            for src_tensor, dst_tensor in zip(
                src_optimizer_tensor_info.optimizer_tensors,
                dst_optimizer_tensor_info.optimizer_tensors,
            ):
                dst_tensor.data.copy_(src_tensor.data)
            src_optimizer_tensor_info.release()

    @classmethod
    def set_setup_model_and_optimizer_func(
        cls, setup_model_and_optimizer_func: Callable
    ):
        """In Megatron, bind the model_provider and model_type parameters in advance to the setup_model_and_optimizer_func."""
        assert cls.setup_model_and_optimizer_func is None, (
            "setup_model_and_optimize_func is already initialized"
        )
        cls.setup_model_and_optimizer_func = setup_model_and_optimizer_func
        assert callable(cls.setup_model_and_optimizer_func), (
            "setup_model_and_optimize_func should be a callable function"
        )

    @classmethod
    def setup_model_and_optimizer(cls, *args, **kwargs):
        is_meta_device = kwargs.pop("is_meta_device", False)
        from contextlib import nullcontext

        from .meta_device_context import meta_device_context

        build_model_context = nullcontext()
        if is_meta_device:
            build_model_context = meta_device_context()
        with build_model_context:
            model, optimizer, opt_param_scheduler = cls.setup_model_and_optimizer_func(
                *args, **kwargs
            )
        return cls(model, optimizer, opt_param_scheduler)

    def __init__(
        self,
        model: list[DDP],
        optimizer: MegatronOptimizer,
        opt_param_scheduler: OptimizerParamScheduler,
    ):
        self.model = model
        self.optimizer = optimizer
        self.opt_param_scheduler = opt_param_scheduler

        self.model_offloaded = False
        self.params_to_resharding_metadata = {}
        self.optimizer_tensor_info_list: list[OptimizerTensorInfo] = []
        self.optimizer_tensor_list: list[torch.Tensor] = []

        self.is_meta_device = (
            model[0].named_parameters().__next__()[1].device.type == "meta"
        )

    @staticmethod
    def refresh_config(model, optimizer):
        args = get_args()
        timers = get_timers()
        config = get_model_config(model[0])
        config.grad_scale_func = optimizer.scale_loss
        config.timers = timers
        if isinstance(model[0], DDP) and args.overlap_grad_reduce:
            assert config.no_sync_func is None, (
                "When overlap_grad_reduce is True, config.no_sync_func must be None; "
                "a custom no_sync_func is not supported when overlapping grad-reduce"
            )
            config.no_sync_func = [model_chunk.no_sync for model_chunk in model]
            if len(model) == 1:
                config.no_sync_func = config.no_sync_func[0]
            if args.align_grad_reduce:
                config.grad_sync_func = [
                    model_chunk.start_grad_sync for model_chunk in model
                ]
                if len(model) == 1:
                    config.grad_sync_func = config.grad_sync_func[0]
        if args.overlap_param_gather and args.align_param_gather:
            config.param_sync_func = [
                model_chunk.start_param_sync for model_chunk in model
            ]
            if len(model) == 1:
                config.param_sync_func = config.param_sync_func[0]
        config.finalize_model_grads_func = finalize_model_grads
        return config

    @property
    def optimizers(self) -> list[MegatronOptimizer]:
        if isinstance(self.optimizer, ChainedOptimizer):
            return self.optimizer.chained_optimizers
        return [self.optimizer]

    def get_metadata(self):
        """Get the metadata of the model and optimizer.

        Note. If not model_only, the model will be released.
        """
        assert self.params_to_resharding_metadata is not None
        return self.params_to_resharding_metadata

    def init_metadata(
        self, parallel_strategy: ParallelStrategy, offload_opt_tensors: bool
    ):
        """Init the metadata of the model and optimizer.

        Each TrainingState can only init metadata once.
        For not the first TrainingState, optimizer should be released.
        """
        assert not self.params_to_resharding_metadata
        assert not self.model_offloaded
        self.params_to_resharding_metadata = generate_resharding_metadata(
            self.model,
            self.optimizer,
            parallel_strategy,
            offload_opt_tensors,
        )

        for resharding_metadata in self.params_to_resharding_metadata.values():
            optimizer_tensor_info = resharding_metadata.optimizer_tensor_info
            if optimizer_tensor_info:
                self.optimizer_tensor_info_list.append(optimizer_tensor_info)
                self.optimizer_tensor_list += optimizer_tensor_info.optimizer_tensors

    def update_model_weight(self):
        if self.is_meta_device:
            return
        if self.model_offloaded:
            self.rebuild_model()

        # Copy main params to model params. The adapter does the plain
        # _copy_main_params_to_model_params, and — under --use-precision-aware-optimizer
        # (required by HybridDeviceOptimizer), where that call is a no-op (the
        # master->model copy normally happens inside optimizer.step(), which a reshard
        # never runs) — also the explicit refill of model param_data from the transferred
        # fp32 master. See docs/hybrid_adam/megatron_hybrid_optimizer.md §6.
        optimizers = (
            self.optimizer.chained_optimizers
            if isinstance(self.optimizer, ChainedOptimizer)
            else [self.optimizer]
        )
        for optimizer in optimizers:
            OptimizerAdapter.create(optimizer).copy_main_to_model()

        args = get_args()
        if not args.use_distributed_optimizer:
            return

        # Clear any stale async param-gather handle before the forced sync. A
        # re-entered cached model chunk (e.g. reshard back to a previously-used
        # strategy) can still hold a pending param_gather_handle from before its
        # release; start_param_sync(force_sync=True) would `.wait()` on it and
        # deadlock, because the matching all-gather is never re-issued after the
        # reshard rebuilt the buffers/groups. Resetting to None makes every rank
        # fall through and issue a FRESH, matched all-gather. (No-op when the handle
        # is already None, i.e. the non-precision-aware path.) See
        # docs/hybrid_adam/changelog.md (TP2->TP1 start_param_sync hang).
        for model_chunk in self.optimizer.model_chunks:
            for bucket_group in (
                getattr(model_chunk, "bucket_groups", [])
                + getattr(model_chunk, "expert_parallel_bucket_groups", [])
            ):
                bucket_group.param_gather_handle = None
                # The shard views in cached_param_buffer_shard_list reference the OLD
                # param_data storage captured during this chunk's previous active period;
                # release_model/rebuild_model changed the storage, so they are stale on a
                # re-entered chunk. Force a rebuild against the current param_data.
                if getattr(bucket_group, "cached_param_buffer_shard_list", None):
                    bucket_group.cached_param_buffer_shard_list = [
                        None
                    ] * len(bucket_group.cached_param_buffer_shard_list)

        # Sync model params in DP-Group
        self.optimizer.update_successful = True
        for model_chunk in self.optimizer.model_chunks:
            model_chunk.start_param_sync(force_sync=True)
        self.optimizer.update_successful = False

    def release_model(
        self, release_grad=True, release_weight=True, offload_weight=False
    ):
        """Release Megatron Model Buffers.

        Args:
            release_grad: Whether to release gradient buffers.
            release_weight: Whether to release weight buffers without saving weight to cpu.
            offload_weight: Whether to offload weight buffers to CPU.

        Note. release_weight and release_grad are mutually exclusive.
        """
        if self.is_meta_device:
            return
        assert not self.model_offloaded, "model is already offloaded"
        assert not (release_weight and offload_weight), (
            "release_weight and offload_weight are mutually exclusive."
        )
        release_or_offload_weight = release_weight or offload_weight

        for model_idx, model_chunk in enumerate(self.model):
            if isinstance(model_chunk, DDP):
                for buffer_idx, buffer in enumerate(
                    model_chunk.buffers + model_chunk.expert_parallel_buffers
                ):
                    if (
                        release_or_offload_weight
                        and buffer.param_data.untyped_storage().size() > 0
                    ):
                        buffer.param_data_size = (
                            buffer.param_data.untyped_storage().size()
                        )

                        if offload_weight:
                            buffer.param_data.cpu_data = (
                                buffer.param_data.data.cpu().pin_memory()
                            )
                            assert (
                                buffer.param_data_size
                                == buffer.param_data.cpu_data.untyped_storage().size()
                            )

                        buffer.param_data.untyped_storage().resize_(0)

                    if release_grad and buffer.grad_data.untyped_storage().size() > 0:
                        grad_size = buffer.grad_data.untyped_storage().size()
                        buffer.grad_data_size = grad_size
                        buffer.grad_data.untyped_storage().resize_(0)

            else:
                # for non-DDP model, using offload instead of release
                for param_name, param in model_chunk.named_parameters():
                    if release_or_offload_weight and param.data is not None:
                        param.data = param.data.to("cpu", non_blocking=True)

                    if release_grad and param.grad is not None:
                        param.grad = param.grad.to("cpu", non_blocking=True)

        # clear memory
        clear_memory()

        self.model_offloaded = True

    def rebuild_model(
        self, rebuild_grad=True, rebuild_weight=True, onload_weight=False
    ):
        """Rebuild Megatron Model Buffers.

        Args:
            rebuild_grad: Whether to rebuild gradient buffers.
            rebuild_weight: Whether to rebuild weight buffers without loading weight from cpu.
            onload_weight: Whether to onload weight buffers to GPU.

        Note. rebuild_weight and rebuild_grad are mutually exclusive.
        """
        if self.is_meta_device:
            return
        assert self.model_offloaded, "model is not offloaded"

        assert not (rebuild_weight and onload_weight), (
            "rebuild_weight and rebuild_grad are mutually exclusive."
        )
        assert rebuild_weight or onload_weight, (
            "rebuild_weight or onload_weight must be True"
        )

        for model_chunk in self.model:
            if isinstance(model_chunk, DDP):
                for buffer in model_chunk.buffers + model_chunk.expert_parallel_buffers:
                    # sometimes, we don't want to load grad for pure inference
                    if rebuild_grad and hasattr(buffer, "grad_data_size"):
                        buffer.grad_data.untyped_storage().resize_(
                            buffer.grad_data_size
                        )
                        buffer.grad_data.zero_()

                    if buffer.param_data.untyped_storage().size() == 0:
                        # Normally release_model recorded param_data_size. Under
                        # --use-precision-aware-optimizer the freshly-built dst model's
                        # param_data is already storage-0 at release time (the weights
                        # are held in the optimizer's master buffer), so the size was
                        # never recorded — fall back to the buffer's own element count
                        # (bytes). _copy_main_params_to_model_params (update_model_weight
                        # step 5.2) then refills it from the transferred master.
                        param_data_size = getattr(
                            buffer,
                            "param_data_size",
                            buffer.param_data.numel() * buffer.param_data.element_size(),
                        )
                        buffer.param_data.untyped_storage().resize_(param_data_size)
                        if onload_weight:
                            buffer.param_data.copy_(
                                buffer.param_data.cpu_data, non_blocking=True
                            )
            else:
                device_id = torch.cuda.current_device()
                for _, param in model_chunk.named_parameters():
                    param.data = param.data.to(device_id, non_blocking=True)
                    if rebuild_grad and param.grad is not None:
                        param.grad = param.grad.to(device_id, non_blocking=True)
        clear_memory()

        self.model_offloaded = False

    def release_optimizer(self):
        for optimizer_tensor_info in self.optimizer_tensor_info_list:
            optimizer_tensor_info.release()
        # The per-param release above resize_(0)'s the param-shaped master + moments
        # (POOL A). The CPU-offload optimizer (HybridDeviceOptimizer) also holds
        # pinned host grad buffers (POOL B) that nothing above reaches; free them on
        # this becoming-dormant gear so they don't accumulate N x across cached
        # strategies. No-op for non-offload optimizers; lazily re-created on the
        # gear's next step (see HybridDeviceOptimizerAdapter.release_offload_host_buffers).
        # Default ON; ELASTIC_RELEASE_HDO_HOST_BUFFERS=0 reverts to the old (leaking)
        # behaviour for A/B measurement and as a safety hatch.
        if os.environ.get("ELASTIC_RELEASE_HDO_HOST_BUFFERS", "1") == "1":
            for optimizer in self.optimizers:
                OptimizerAdapter.create(optimizer).release_offload_host_buffers()
        trim_host_memory()

    def rebuild_optimizer(self):
        for optimizer_tensor_info in self.optimizer_tensor_info_list:
            optimizer_tensor_info.rebuild()

    @property
    def all_optimizer_tensors(self):
        return self.optimizer_tensor_list

    def save_checkpoint(
        self,
        iteration: int | None = None,
        num_floating_point_operations_so_far: float | None = None,
        preprocess_common_state_dict_fn=preprocess_common_state_dict,
        is_before_reshard: bool = True,
        save_path: str | None = None,
    ):
        # On a scale-down (world-shrink) reshard, redirect the DCP save collective
        # to the current elastic world group; otherwise the inactive ranks feed a
        # None SavePlan into DCP's dedup_save_plans and the after_reshard save
        # crashes. No-op for full-world (symmetric / scale-up) saves.
        from ..distributed.dist_ckpt_patch import ensure_dist_ckpt_save_patched

        ensure_dist_ckpt_save_patched()
        args = get_args()
        if iteration is None:
            iteration = getattr(args, "curr_iteration", 0) + 1
        if num_floating_point_operations_so_far is None:
            num_floating_point_operations_so_far = getattr(
                args, "num_floating_point_operations_so_far", 0.0
            )
        prev_save = getattr(args, "save", None)
        # 0.16 下 swiglu 用 closure-based sharded factory,legacy(torch pickle)ckpt
        # 序列化失败。保持 args.use_dist_ckpt 默认 True,让 save_checkpoint 走 torch_dist。
        # 这里不再改/复原 args.use_dist_ckpt;调用方若要 legacy,自己负责 save/restore。

        if save_path is None:
            # 基于本文件位置定位 ElasticMegatron repo 根,与 cwd 解耦,
            # 避免用户在不同子目录 import 调用导致路径错位。
            repo_root = pathlib.Path(__file__).resolve().parents[2]
            save_dir = (
                repo_root
                / "tools"
                / "ckpt"
                / ("before_reshard" if is_before_reshard else "after_reshard")
            )
        else:
            save_dir = pathlib.Path(save_path)
        os.makedirs(save_dir, exist_ok=True)

        args.save = str(save_dir)
        try:
            save_checkpoint(
                iteration,
                self.model,
                self.optimizer,
                self.opt_param_scheduler,
                num_floating_point_operations_so_far,
                preprocess_common_state_dict_fn=preprocess_common_state_dict_fn,
            )
        finally:
            # 恢复:否则 pretrain() 收尾段会用 args.save / args.save_interval
            # 触发不必要的 save 路径(save_interval 可能是 None,直接报 TypeError)。
            args.save = prev_save


def load_state_dict_with_no_step(lr: OptimizerParamScheduler, state_dict: dict):
    origin_step = OptimizerParamScheduler.step

    def foo(*args, **kwargs):
        pass

    OptimizerParamScheduler.step = foo
    lr.load_state_dict(state_dict)
    lr.num_steps = state_dict["num_steps"]
    OptimizerParamScheduler.step = origin_step


def update_state_dict(
    src_state_dict: dict, dst_state_dict: dict, exclude_keys: list[str] | None = None
):
    if exclude_keys is None:
        exclude_keys = ["params"]
    for key, value in src_state_dict.items():
        if key not in exclude_keys:
            dst_state_dict[key] = value


def update_optimizer_by_state_dict(
    optimizer: MegatronOptimizer, state_param_groups: list[dict]
):
    assert isinstance(optimizer, MegatronOptimizer) and not isinstance(
        optimizer, ChainedOptimizer
    )
    torch_optimizer = optimizer.optimizer
    for opt_param_group, state_param_group in zip(
        torch_optimizer.param_groups, state_param_groups
    ):
        update_state_dict(state_param_group, opt_param_group)


def update_optimizer_and_opt_param_scheduler(
    src_optimizers: list[MegatronOptimizer],
    src_opt_param_scheduler: OptimizerParamScheduler,
    dst_optimizers: list[MegatronOptimizer],
    dst_opt_param_scheduler: OptimizerParamScheduler,
):
    # zip copy is positional (lr/betas etc.); step must be broadcast explicitly to
    # ALL dst chaineds, including newly-created ones beyond the zip range.
    for src_optimizer, dst_optimizer in zip(src_optimizers, dst_optimizers):
        src_state_dict = src_optimizer.state_dict()
        dst_state_dict = dst_optimizer.state_dict()

        # Update optimizer.state_dict()
        for id in range(len(src_state_dict["optimizer"]["param_groups"])):
            update_state_dict(
                src_state_dict["optimizer"]["param_groups"][id],
                dst_state_dict["optimizer"]["param_groups"][id],
            )

        # Update optimizer.param_groups
        update_optimizer_by_state_dict(
            dst_optimizer, dst_state_dict["optimizer"]["param_groups"]
        )

    # Broadcast step to all dst chaineds (mcore _synchronize_steps requires all
    # chaineds to carry the same step value).
    src_step = None
    for src_optimizer in src_optimizers:
        sd = src_optimizer.state_dict()
        for pg in sd["optimizer"]["param_groups"]:
            if "step" in pg:
                src_step = pg["step"]
                break
        if src_step is not None:
            break

    if src_step is not None:
        for dst_optimizer in dst_optimizers:
            for pg in dst_optimizer.optimizer.param_groups:
                # 只设有非空 params 的 group(空 group 在 TE FusedAdam 下没有 step,
                # mcore _synchronize_steps 也跳过它们,保持一致)
                if len(pg.get("params", [])) > 0:
                    pg["step"] = src_step

    # Update opt_param_scheduler
    load_state_dict_with_no_step(
        dst_opt_param_scheduler, src_opt_param_scheduler.state_dict()
    )

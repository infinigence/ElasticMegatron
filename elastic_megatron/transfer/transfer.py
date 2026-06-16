import os
from collections import defaultdict
from collections.abc import Callable
from contextlib import contextmanager
from typing import Dict, List, Tuple

import torch
from megatron.core import parallel_state
from megatron.core.optimizer import MegatronOptimizer
from megatron.core.optimizer_param_scheduler import OptimizerParamScheduler

from ..megatron_manager.parallel_strategy import ParallelStrategy
from ..megatron_manager.training_state import (
    TrainingState,
    load_state_dict_with_no_step,
    update_optimizer_by_state_dict,
)
from ..resharding.resharding import ReshardPlan
from ..resharding.resharding_metadata import OptimizerTensorInfo
from ..resharding.resharding_pp import LayerType, ParamPositionAttr
from ..resharding.util import ParamRange, Range, Timer
from ..resharding.virtual_param import VirtualParam, VirtualParamSpace
from .chunk_schedule import STAGING_CAP_DEFAULT, derive_staging_cap
from .communicator import BatchedTransfer, Communicator


class TransferManager:
    def __init__(self, send_fn: Callable, recv_fn: Callable, broadcast_fn: Callable):
        self._rank = torch.distributed.get_rank()
        self.communicator = Communicator(self._rank, send_fn, recv_fn, broadcast_fn)
        self.batched_transfer = BatchedTransfer(self.communicator)
        self.send = self.communicator.send
        self.recv = self.communicator.recv
        self.broadcast = self.communicator.broadcast
        self.fake_transfer = False
        # When enabled, each phase of the batched transfer is timed (CUDA-
        # synchronized) and the per-phase durations are printed on rank 0.
        # Disabled => the timing context managers are no-ops (zero overhead).
        self._log_transfer_timing = os.getenv("ELASTIC_TRANSFER_LOG_LEVEL", "1") != "0"
        # Pack each peer's slices into one buffer before NCCL (fast path) vs one
        # p2p op per slice. The packed buffer lives on GPU, so packing doubles as
        # CPU<->GPU staging for offloaded (HybridDeviceOptimizer) state.
        self._pack = os.getenv("ELASTIC_USE_ASYNCBUFFER_P2P", "1") == "1"
        # Byte cap per staging chunk in the packed path; staging residency is
        # bounded by ~2x this value. Resolved per reshard in _resolve_staging_cap
        # right before the exchange. ELASTIC_STAGING_CAP_MODE selects the source:
        #   free (default): (smallest current free GPU memory across the union
        #     ranks − a 2 GiB reserve) / 2, clamped to [512 MiB, 8 GiB] — the
        #     reserve keeps 2x cap from filling the card; the only mode that
        #     reflects real device memory, including co-tenant processes.
        #   fixed: a 2 GiB constant.
        # ELASTIC_MAX_INFLIGHT_BYTES, when set, overrides the mode: a positive
        # value is the exact cap; <=0 disables chunking (one chunk per peer,
        # legacy residency). The cap is identical on every rank — both ends of a
        # pair derive chunk counts from it.
        self._cap_mode = os.getenv("ELASTIC_STAGING_CAP_MODE", "free")
        raw_inflight = os.getenv("ELASTIC_MAX_INFLIGHT_BYTES")
        self._explicit_inflight: int | None = (
            int(raw_inflight) if raw_inflight is not None else None
        )
        self._max_inflight_bytes: int | None = None

    def _resolve_staging_cap(self) -> int | None:
        """Per-chunk staging byte cap for this reshard, identical on every rank.

        ELASTIC_MAX_INFLIGHT_BYTES, when set, wins (positive = exact cap; <=0 =
        one chunk per peer). Otherwise ELASTIC_STAGING_CAP_MODE picks the source:
        ``fixed`` => a 2 GiB constant; ``free`` (default) => (the smallest current
        free GPU memory across the union ranks (one MIN all-reduce; the no-group
        collective lands on the union via the dist patch) − a 2 GiB reserve) / 2,
        clamped to [512 MiB, 8 GiB]. Called right before the exchange so the free reading
        reflects the post-release_model state of the card.
        """
        if self._explicit_inflight is not None:
            return self._explicit_inflight
        if self._cap_mode == "fixed":
            return STAGING_CAP_DEFAULT
        free_bytes = torch.cuda.mem_get_info()[0]
        min_free = torch.tensor(
            [free_bytes], device=torch.cuda.current_device(), dtype=torch.int64
        )
        torch.distributed.all_reduce(min_free, op=torch.distributed.ReduceOp.MIN)
        min_free_bytes = int(min_free.item())
        cap = derive_staging_cap(min_free_bytes)
        if self._rank == 0 and 2 * cap > min_free_bytes:
            print(
                f"[ElasticMegatron-Transfer] staging cap {cap / (1 << 20):.0f} MiB x2 "
                f"exceeds min free {min_free_bytes / (1 << 20):.0f} MiB across union "
                f"ranks — staging may over-commit on a tight/co-tenanted card.",
                flush=True,
            )
        return cap

    @contextmanager
    def _timed(self, timings: dict[str, float] | None, name: str):
        """Time the wrapped block into ``timings[name]`` (milliseconds).

        A no-op when ``timings`` is None, so timing adds no overhead (and no
        ``cuda.synchronize``) when disabled.
        """
        if timings is None:
            yield
            return
        with Timer() as timer:
            yield
        timings[name] = timer.elapsed

    def set_fake_transfer(self, fake_transfer: bool):
        self.fake_transfer = fake_transfer
        self.communicator.set_fake_transfer(fake_transfer)

    def _send_optimizer_tensors(
        self,
        send_transfer_range_dict: dict[int, ParamRange] | dict[int, Range],
        src_optimizer_tensor_info: OptimizerTensorInfo,
    ):
        """Send (a dp-gather/scatter slice of) one param's optimizer state, then
        release its storage. Transport is delegated to the communicator."""
        if send_transfer_range_dict is None or src_optimizer_tensor_info is None:
            return
        send_tasks: Dict[int, List[torch.Tensor]] = defaultdict(list)
        self._collect_send(
            send_transfer_range_dict, src_optimizer_tensor_info, send_tasks
        )
        self.batched_transfer.transfer(
            send_tasks, {}, pack=self._pack, max_inflight_bytes=self._max_inflight_bytes
        )
        src_optimizer_tensor_info.release()

    def _recv_optimizer_tensors(
        self,
        recv_transfer_range_dict: dict[int, ParamRange],
        dst_optimizer_tensor_info: OptimizerTensorInfo,
    ):
        """Rebuild the destination state then receive (a dp-gather/scatter slice
        of) one param's optimizer state. Transport is delegated to the
        communicator; non-contiguous targets are copied back here."""
        if recv_transfer_range_dict is None or dst_optimizer_tensor_info is None:
            return
        dst_optimizer_tensor_info.rebuild()
        recv_tasks: Dict[int, List[torch.Tensor]] = defaultdict(list)
        recv_copy_back: List[Tuple[torch.Tensor, torch.Tensor]] = []
        self._collect_recv(
            recv_transfer_range_dict,
            dst_optimizer_tensor_info,
            recv_tasks,
            recv_copy_back,
        )
        self.batched_transfer.transfer(
            {}, recv_tasks, pack=self._pack, max_inflight_bytes=self._max_inflight_bytes
        )
        for recv_slice, recv_buffer in recv_copy_back:
            recv_slice.data.copy_(recv_buffer)

    def transfer_word_embedding_and_output_layer(
        self, word_embedding: VirtualParam, output_layer: VirtualParam
    ):
        """If shared_embedding and new PP > 1, need transfer word embedding to output layer by broadcast."""
        if not output_layer.shared_embedding:
            return

        dst_parallel_strategy: ParallelStrategy = (
            word_embedding.reshard_plan.dst_parallel_strategy
        )
        if dst_parallel_strategy.pipeline_model_parallel_size == 1:
            return

        if (
            not parallel_state.is_rank_in_embedding_group(ignore_virtual=True)
            or torch.distributed.get_world_size(parallel_state.get_embedding_group())
            <= 1
        ):
            return

        if parallel_state.is_pipeline_first_stage(ignore_virtual=True):
            dst_optimizer_tensor_info: OptimizerTensorInfo = (
                word_embedding.dst_optimizer_tensor_info
            )
            assert dst_optimizer_tensor_info.optimizer_tensors[0].storage().size() > 0
        else:
            dst_optimizer_tensor_info: OptimizerTensorInfo = (
                output_layer.dst_optimizer_tensor_info
            )
            dst_optimizer_tensor_info.rebuild()

        for optimizer_tensor in dst_optimizer_tensor_info.optimizer_tensors:
            self.broadcast(
                optimizer_tensor,
                src=torch.distributed.get_global_rank(
                    parallel_state.get_embedding_group(), 0
                ),
                group=parallel_state.get_embedding_group(),
            )

    def _pre_process(self, virtual_param: VirtualParam):
        """Pre Process. For sender, gather param in dp ranks(ZeRO-1) and execute swiglu shuffle."""
        src_optimizer_tensor_info: OptimizerTensorInfo = (
            virtual_param.src_optimizer_tensor_info
        )
        if src_optimizer_tensor_info is None:
            return

        # Step-1 : Get DP-distribution and aligned global rank
        reshard_plan: ReshardPlan = virtual_param.reshard_plan
        dp_distribution: dict[int, Range] = (
            reshard_plan.data_parallel_resharding_info.src_dp_distribution_with_global_rank
        )
        aligned_global_rank = (
            reshard_plan.data_parallel_resharding_info.src_aligned_global_rank
        )
        is_aligned_rank = aligned_global_rank == self._rank
        assert dp_distribution.get(self._rank, None) is not None, (
            f"DP-distribution is not found for rank {self._rank}"
        )

        # Step-2 : Gather param in dp ranks(ZeRO-1)
        if len(dp_distribution) > 1:
            self._send_optimizer_tensors(
                {aligned_global_rank: dp_distribution.get(self._rank).normalize()},
                src_optimizer_tensor_info,
            )
            if aligned_global_rank == self._rank:
                src_optimizer_tensor_info.create_padded_optimizer_tensor()
                self._recv_optimizer_tensors(dp_distribution, src_optimizer_tensor_info)

        # Step-3 : Shuffle swiglu
        if virtual_param.is_swiglu_fc and is_aligned_rank:
            scale_up_ratio = len(reshard_plan.global_send_info.get(self._rank))
            assert scale_up_ratio >= 1, (
                f"{scale_up_ratio=} less than 1 menas no send/recv for this param. Global transfer plan or dp-align might be incorrect."
            )
            src_optimizer_tensor_info.shuffle_swiglu(scale_up_ratio)

    def _should_skip_virtual_param(self, virtual_param: VirtualParam) -> bool:
        if (
            not virtual_param.src_optimizer_tensor_info
            and not virtual_param.dst_optimizer_tensor_info
        ):
            return True
        if (
            virtual_param.shared_embedding
            and virtual_param.param_position_attr.layer_type == LayerType.OUTPUT_LAYER
        ):
            return True
        return False

    def _resolve_transfer_layout(
        self,
        transfer_range_dict: Dict[int, ParamRange] | Dict[int, Range],
        optimizer_tensor_info: OptimizerTensorInfo,
    ) -> Tuple[ParamRange | Range, torch.Size | int]:
        model_param_range: ParamRange | Range = optimizer_tensor_info.model_param_range
        model_param_shape: torch.Size | int = model_param_range.to_torch_size()

        # Dp-align : the param range is a Range object
        if isinstance(next(iter(transfer_range_dict.values())), Range):
            model_param_range = Range(
                0, optimizer_tensor_info.optimizer_tensors[0].nelement()
            )
            model_param_shape = -1
        return model_param_range, model_param_shape

    def _collect_send(
        self,
        send_transfer_range_dict: Dict[int, ParamRange] | Dict[int, Range],
        optimizer_tensor_info: OptimizerTensorInfo,
        send_tasks: Dict[int, List[torch.Tensor]],
    ) -> None:
        """Append each cross-rank send slice to ``send_tasks[dst]``; self-rank
        slices are copied immediately via :meth:`send`. Raw views are appended
        (the communicator makes them contiguous / stages them as needed)."""
        if self.fake_transfer:
            for dst_rank in send_transfer_range_dict:
                self.send(None, dst=dst_rank)
            return

        model_param_range, model_param_shape = self._resolve_transfer_layout(
            send_transfer_range_dict, optimizer_tensor_info
        )
        for dst_rank, send_param_range in send_transfer_range_dict.items():
            transfer_param_slices = model_param_range.get_sub_range_slices(
                send_param_range
            )
            for optimizer_tensor in optimizer_tensor_info.optimizer_tensors:
                assert optimizer_tensor.nelement() == model_param_range.size
                send_slice = optimizer_tensor.view(model_param_shape)[
                    transfer_param_slices
                ]
                if dst_rank == self._rank:
                    self.send(send_slice, dst=dst_rank)
                else:
                    send_tasks[dst_rank].append(send_slice)

    def _collect_recv(
        self,
        recv_transfer_range_dict: Dict[int, ParamRange] | Dict[int, Range],
        optimizer_tensor_info: OptimizerTensorInfo,
        recv_tasks: Dict[int, List[torch.Tensor]],
        recv_copy_back: List[Tuple[torch.Tensor, torch.Tensor]],
    ) -> None:
        """Append each cross-rank recv landing buffer to ``recv_tasks[src]``;
        self-rank slices are filled immediately via :meth:`recv`. A
        non-contiguous target gets a contiguous buffer plus a ``recv_copy_back``
        entry the caller scatters back after the transfer."""
        if self.fake_transfer:
            for src_rank in recv_transfer_range_dict:
                self.recv(None, src=src_rank)
            return

        model_param_range, model_param_shape = self._resolve_transfer_layout(
            recv_transfer_range_dict, optimizer_tensor_info
        )
        for src_rank, recv_param_range in recv_transfer_range_dict.items():
            transfer_param_slices = model_param_range.get_sub_range_slices(
                recv_param_range
            )
            for optimizer_tensor in optimizer_tensor_info.optimizer_tensors:
                assert optimizer_tensor.nelement() == model_param_range.size
                recv_slice = optimizer_tensor.view(model_param_shape)[
                    transfer_param_slices
                ]
                if src_rank == self._rank:
                    self.recv(recv_slice, src=src_rank)
                    continue

                if recv_slice.is_contiguous():
                    recv_buffer = recv_slice
                else:
                    recv_buffer = torch.empty_like(
                        recv_slice, memory_format=torch.contiguous_format
                    )
                    recv_copy_back.append((recv_slice, recv_buffer))
                recv_tasks[src_rank].append(recv_buffer)

    def _main_process(self, virtual_params: List[VirtualParam]):
        """Execute the cross-rank transfer plan for every param.

        Collect every send/recv slice into per-peer task dicts, hand them to the
        communicator in one batched transfer, then scatter non-contiguous
        receives and release sent storage. Self-rank slices (survival node that
        is both sender and receiver) are copied during collection, in order, via
        :meth:`send` / :meth:`recv`; cross-rank slices move through the
        communicator.
        """
        send_tasks: Dict[int, List[torch.Tensor]] = defaultdict(list)
        recv_tasks: Dict[int, List[torch.Tensor]] = defaultdict(list)
        recv_copy_back: List[Tuple[torch.Tensor, torch.Tensor]] = []
        src_to_release: List[OptimizerTensorInfo] = []
        src_to_release_padded: List[OptimizerTensorInfo] = []
        timings: dict[str, float] | None = {} if self._log_transfer_timing else None

        with self._timed(timings, "Collect tasks"):
            for virtual_param in virtual_params:
                if self._should_skip_virtual_param(virtual_param):
                    continue

                reshard_plan: ReshardPlan = virtual_param.reshard_plan
                src_info = virtual_param.src_optimizer_tensor_info
                dst_info = virtual_param.dst_optimizer_tensor_info

                # When this rank holds both sides (self/survival transfer) src and
                # dst must carry the same ordered state set -- slices are zipped
                # positionally. Always holds for Adam (both = 3 states).
                if src_info is not None and dst_info is not None:
                    assert src_info.state_names == dst_info.state_names, (
                        f"src/dst optimizer state set mismatch: {src_info.state_names} "
                        f"vs {dst_info.state_names}; heterogeneous state sets are "
                        "unsupported."
                    )

                # Sender side: collect cross-rank sends, schedule releases.
                if src_info is not None:
                    src_to_release_padded.append(src_info)
                    send_range = reshard_plan.global_send_info.get(self._rank)
                    if send_range is not None:
                        self._collect_send(send_range, src_info, send_tasks)
                        src_to_release.append(src_info)

                # Receiver side (aligned rank only): allocate the padded buffer
                # then collect cross-rank receives into it.
                is_aligned_receiver = (
                    reshard_plan.data_parallel_resharding_info.dst_aligned_global_rank
                    == self._rank
                )
                if is_aligned_receiver:
                    assert dst_info is not None, (
                        f"rank {self._rank} is the aligned recv rank of param "
                        f"(name={virtual_param.name}) but dst_optimizer_tensor_info "
                        "is None."
                    )
                    recv_range = reshard_plan.global_recv_info.get(self._rank)
                    assert recv_range is not None, (
                        f"rank {self._rank} is the aligned recv rank of param "
                        f"(name={virtual_param.name}) but recv info is None."
                    )
                    dst_info.create_padded_optimizer_tensor()
                    dst_info.rebuild()
                    self._collect_recv(recv_range, dst_info, recv_tasks, recv_copy_back)

        with self._timed(timings, "Transfer"):
            self.batched_transfer.transfer(
                send_tasks,
                recv_tasks,
                pack=self._pack,
                max_inflight_bytes=self._max_inflight_bytes,
                collect_phase_ms=timings is not None,
            )
        if timings is not None and self.batched_transfer.last_phase_ms:
            for phase, ms in self.batched_transfer.last_phase_ms.items():
                timings[f"Transfer/{phase}"] = ms

        with self._timed(timings, "Copy recv tensors"):
            for recv_slice, recv_buffer in recv_copy_back:
                recv_slice.data.copy_(recv_buffer)

        with self._timed(timings, "Release optimizer tensors"):
            for optimizer_tensor_info in src_to_release:
                optimizer_tensor_info.release()
            for optimizer_tensor_info in src_to_release_padded:
                optimizer_tensor_info.release_padded_optimizer_tensor()

        if timings is not None and self._rank == 0:
            summary = "  ".join(f"{name}: {ms:.2f}ms" for name, ms in timings.items())
            print(f"[ElasticMegatron-Transfer] {summary}", flush=True)

    def _post_process(self, virtual_param: VirtualParam):
        """Post Process. For receiver, unshuffle swiglu and scatter param to dp ranks(ZeRO-1), and release all padded optimizer tensors."""
        dst_optimizer_tensor_info: OptimizerTensorInfo = (
            virtual_param.dst_optimizer_tensor_info
        )
        if dst_optimizer_tensor_info is None:
            return

        # Step-1 : Find the aligned_dp_rank
        reshard_plan = virtual_param.reshard_plan
        aligned_global_rank = (
            reshard_plan.data_parallel_resharding_info.dst_aligned_global_rank
        )
        is_aligned_rank = aligned_global_rank == self._rank

        # Step-2 : Unshuffle swiglu
        if virtual_param.is_swiglu_fc and is_aligned_rank:
            scale_down_ratio = len(reshard_plan.global_recv_info.get(self._rank))
            assert scale_down_ratio >= 1, (
                f"{scale_down_ratio=} less than 1 menas no send/recv for this param. Global transfer plan or dp-align might be incorrect."
            )
            dst_optimizer_tensor_info.unshuffle_swiglu(scale_down_ratio)

        # Step-3 : Scatter params to all dp ranks and release the padded optimizer tensor

        dp_distribution: dict[int, Range] = (
            reshard_plan.data_parallel_resharding_info.dst_dp_distribution_with_global_rank
        )
        if (
            len(dp_distribution) > 1
        ):  # Add this condition would speed up the scatter process when the param is not split
            if is_aligned_rank:
                self._send_optimizer_tensors(dp_distribution, dst_optimizer_tensor_info)
                dst_optimizer_tensor_info.release_padded_optimizer_tensor()
            self._recv_optimizer_tensors(
                {aligned_global_rank: dp_distribution.get(self._rank).normalize()},
                dst_optimizer_tensor_info,
            )
        elif is_aligned_rank:  # When len(dp_distribution) == 1, we still need to release the padded optimizer tensor for aligned_rank
            dst_optimizer_tensor_info.release_padded_optimizer_tensor()

    def _block_and_print(self, virtual_param: VirtualParam):
        param_position_attr: ParamPositionAttr = virtual_param.param_position_attr
        msg = f"\nTransfer Info : {param_position_attr}\n"

        reshard_plan: ReshardPlan = virtual_param.reshard_plan
        msg += "\nglobal_send_info:\n"
        for src_rank, send_transfer_range_dict in reshard_plan.global_send_info.items():
            for dst_rank, param_range in send_transfer_range_dict.items():
                msg += f"src_rank={src_rank} => dst_rank={dst_rank}, param_range={param_range}\n"
            msg += "\n"

        msg += "\nglobal_recv_info:\n"
        for dst_rank, recv_transfer_range_dict in reshard_plan.global_recv_info.items():
            for src_rank, param_range in recv_transfer_range_dict.items():
                msg += f"dst_rank={dst_rank} <= src_rank={src_rank}, param_range = {param_range}\n"
            msg += "\n"
        if self._rank == 0:
            print(msg, flush=True)
        torch.distributed.barrier()

    def transfer_optimizer_tensors(
        self,
        virtual_param_space: VirtualParamSpace,
        use_block_and_print: bool = False,
    ):
        """Transfer all optimizer tensors.

        There are three steps to transfer all optimizer tensors:

        1. Pre-process  : For sender, gather param in dp ranks(ZeRO-1) and execute swiglu shuffle

        2. Main-process : Execute transfer plan
            2.1 : For sender, send tensors and release memory
            2.2 : For receiver, allocate memory and recv tensors

        3. Post-process : For receiver, unshuffle swiglu and scatter param to dp ranks(ZeRO-1). And release all padded optimizer tensors.

        Note.
        The above three steps are strictly synchronized, so if the transmission is carried out according to the way of each parameter traversing the above three steps, it will cause blockage.
        """

        # Resolve the staging-chunk cap for THIS reshard, right before the
        # exchange (after release_model + cache reclaim, inside the union world
        # group). Identical on every rank.
        self._max_inflight_bytes = self._resolve_staging_cap()
        if self._pack and self._rank == 0:
            cap = self._max_inflight_bytes
            shown = "no-split" if not cap or cap <= 0 else f"{cap / (1 << 20):.0f} MiB"
            print(
                f"[ElasticMegatron-Transfer] staging cap (mode={self._cap_mode}): {shown}",
                flush=True,
            )

        def process_virtual_params(process_fn: Callable):
            assert callable(process_fn), "process_fn must be a callable"
            for virtual_param in virtual_param_space.all_virtual_params:
                # For debug purpose, and this would couse low performance.
                if use_block_and_print:
                    self._block_and_print(virtual_param)

                # Skip if no optimizer tensor in this virtual param
                if (
                    not virtual_param.src_optimizer_tensor_info
                    and not virtual_param.dst_optimizer_tensor_info
                ):
                    continue

                # Skip if shared_embedding is True and the virtual_param is output_layer param
                if (
                    virtual_param.shared_embedding
                    and virtual_param.param_position_attr.layer_type
                    == LayerType.OUTPUT_LAYER
                ):
                    continue

                process_fn(virtual_param)

        process_virtual_params(self._pre_process)
        self._main_process(virtual_param_space.all_virtual_params)
        process_virtual_params(self._post_process)
        self.transfer_word_embedding_and_output_layer(
            virtual_param_space.all_virtual_params[0],
            virtual_param_space.all_virtual_params[-1],
        )

    def redundant_backup(
        self,
        is_scale_up: bool,
        src_training_state: TrainingState,
        dst_training_state: TrainingState,
        group_src: int = 0,
    ):
        # Delete redundant backup
        if not is_scale_up:
            assert src_training_state is not None
            if dst_training_state is not None:
                TrainingState.param_migrate(src_training_state, dst_training_state)
            else:
                src_training_state.release_optimizer()
            return

        # Create redundant backup
        assert dst_training_state is not None
        inter_partial_data_parallel_group = (
            parallel_state.get_inter_distributed_optimizer_instance_group()
        )
        inter_partial_data_ranks = torch.distributed.get_process_group_ranks(
            inter_partial_data_parallel_group
        )
        master_inter_partial_data_rank = inter_partial_data_ranks[group_src]

        # Master rank migrate param from src to dst
        if self._rank == master_inter_partial_data_rank:
            TrainingState.param_migrate(src_training_state, dst_training_state)
        else:  # Other rank manually release & rebduild optimizer
            if src_training_state is not None:
                src_training_state.release_optimizer()
            dst_training_state.rebuild_optimizer()

        # Master broadcast param to inter group
        all_optimizer_tensors: list[torch.Tensor] = (
            dst_training_state.all_optimizer_tensors
        )

        # Use torch.distributed.broadcast_object_list may lead to OOM
        for optimizer_tensor in all_optimizer_tensors:
            self.broadcast(
                optimizer_tensor,
                group=inter_partial_data_parallel_group,
                src=master_inter_partial_data_rank,
            )

    # ------------------------------------------------------------------------------------------------
    # |                             Transfer optimizer state dict                                     |
    # ------------------------------------------------------------------------------------------------
    def transfer_opt_param_scheduler(
        self,
        optimizers: list[MegatronOptimizer],
        opt_param_scheduler: OptimizerParamScheduler,
    ):
        for optimizer in optimizers:
            self._transfer_optimizer_state_dict(optimizer)
        self._transfer_param_scheduler(opt_param_scheduler)

    def _transfer_optimizer_state_dict(
        self,
        optimizer: MegatronOptimizer,
    ):
        broadcast_object = None
        if self._rank == 0:
            broadcast_object = []
            state_dict = optimizer.state_dict()
            for param_group in state_dict["optimizer"]["param_groups"]:
                broadcast_object.append(
                    {
                        key: value
                        for key, value in param_group.items()
                        if key != "params"
                    }
                )
        broadcast_object_list = [broadcast_object]
        torch.distributed.broadcast_object_list(broadcast_object_list, src=0)
        if self._rank != 0:
            update_optimizer_by_state_dict(optimizer, broadcast_object_list[0])

    def _transfer_param_scheduler(
        self,
        opt_param_scheduler: OptimizerParamScheduler,
    ):
        broadcast_object = None
        if self._rank == 0:
            broadcast_object = opt_param_scheduler.state_dict()
        broadcast_object_list = [broadcast_object]
        torch.distributed.broadcast_object_list(broadcast_object_list, src=0)
        if self._rank != 0:
            load_state_dict_with_no_step(opt_param_scheduler, broadcast_object_list[0])

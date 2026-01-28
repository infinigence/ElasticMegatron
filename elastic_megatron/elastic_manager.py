from typing import Dict, Callable, List, Optional
import torch
import torch.distributed
from megatron.core import parallel_state
from megatron.core.optimizer import MegatronOptimizer
from megatron.core.distributed import DistributedDataParallel as DDP
from megatron.core.optimizer_param_scheduler import OptimizerParamScheduler
from .megatron_manager.megatron_state import (
    MegatronStateManager,
    with_world_group,
)
from .transfer.transfer import TransferManager
from .megatron_manager.training_state import (
    TrainingState,
    update_optimizer_and_opt_param_scheduler,
)
from .resharding.virtual_param import build_virtual_model, VirtualParamSpace
from .resharding.resharding_metadata import OptimizerTensorInfo
from .distributed import global_barrier_by_gloo
from .megatron_manager.parallel_strategy import ParallelStrategy

from contextlib import contextmanager
from .resharding.util import Timer
from logging import Logger
from .resharding.util import get_megatron_version_minor
from .megatron_manager.dataloader_state import DataloaderState

from contextlib import nullcontext


class ElasticMegatronManager:
    @classmethod
    def register(
        cls,
        setup_model_and_optimizer_func: Callable,
        train_valid_test_datasets_provider: Callable,
    ):
        TrainingState.set_setup_model_and_optimizer_func(setup_model_and_optimizer_func)
        DataloaderState.register(train_valid_test_datasets_provider)

    def __init__(
        self,
        parallel_strategy_list: List[Dict[str, int]],
        model: List[DDP],
        optimizer: MegatronOptimizer,
        opt_param_scheduler: OptimizerParamScheduler,
        send_fn: Callable = torch.distributed.send,
        recv_fn: Callable = torch.distributed.recv,
        broadcast_fn: Callable = None,
        use_p2p_to_collective: bool = True,
    ):
        self._rank = torch.distributed.get_rank()

        self.p2p_manager = None
        if use_p2p_to_collective:
            from .transfer.p2p_to_collective import P2PToCollective

            self.p2p_manager = P2PToCollective()

        self._init_transfer_manager(send_fn, recv_fn, broadcast_fn)
        self._init_state_manager(
            parallel_strategy_list, model, optimizer, opt_param_scheduler
        )

        if get_megatron_version_minor() == 11:
            parallel_state.get_inter_distributed_optimizer_instance_group = (
                parallel_state.get_inter_partial_data_parallel_group
            )

    def _init_transfer_manager(
        self, send_fn: Callable, recv_fn: Callable, broadcast_fn: Callable
    ):
        if self.p2p_manager is not None:
            send_fn = self.p2p_manager.send
            recv_fn = self.p2p_manager.recv
        self.transfer_manager = TransferManager(
            send_fn=send_fn,
            recv_fn=recv_fn,
            broadcast_fn=torch.distributed.broadcast
            if broadcast_fn is None
            else broadcast_fn,
        )

    def _init_state_manager(
        self,
        parallel_strategy_list: List[Dict[str, int]],
        model: List[DDP],
        optimizer: MegatronOptimizer,
        opt_param_scheduler: OptimizerParamScheduler,
    ):
        self.state_manager = MegatronStateManager()
        self.state_manager.init_parallel_strategy_list(parallel_strategy_list)
        self.state_manager.init_training_state(
            model=model,
            optimizer=optimizer,
            opt_param_scheduler=opt_param_scheduler,
            offload_opt_tensors=False,
        )
        self.virtual_param_space: VirtualParamSpace = build_virtual_model(
            self.state_manager.training_state.params_to_resharding_metadata
        )

    def build_iterators(self):
        return DataloaderState.build_iterators()

    def transfer_params(
        self, src_megatron_state, dst_megatron_state, union_world_group, is_meta_device
    ) -> float:
        # Step-1 : Register resharding info to global Virtual Space.
        self.virtual_param_space.register_reshard(
            src_megatron_state, dst_megatron_state, with_ddp=True
        )

        # Step-2 : Transfer params
        transfer_context = nullcontext()
        if is_meta_device:
            transfer_context = build_nccl_connection_context(self.transfer_manager)

        with with_world_group(union_world_group), transfer_context, Timer() as t:
            self.transfer_manager.transfer_optimizer_tensors(self.virtual_param_space)

        return t.elapsed

    def redundant_backup(
        self, src_megatron_state, dst_megatron_state, is_sacle_up, is_meta_device
    ) -> float:
        if is_meta_device and is_sacle_up:
            with Timer() as t:
                torch.distributed.broadcast(
                    torch.empty(1, device=torch.cuda.current_device()),
                    group=parallel_state.get_inter_distributed_optimizer_instance_group(),
                )
            return t.elapsed

        with Timer() as t:
            self.transfer_manager.redundant_backup(
                is_sacle_up,
                src_megatron_state.training_state,
                dst_megatron_state.training_state,
            )
        return t.elapsed

    def transfer_learning_rate(self, src_megatron_state, dst_megatron_state):
        """Update the learning rate of the optimizer."""
        if dst_megatron_state.training_state is not None:
            if self._rank == 0:
                update_optimizer_and_opt_param_scheduler(
                    src_megatron_state.training_state.optimizers,
                    src_megatron_state.training_state.opt_param_scheduler,
                    dst_megatron_state.training_state.optimizers,
                    dst_megatron_state.training_state.opt_param_scheduler,
                )
            self.transfer_manager.transfer_opt_param_scheduler(
                dst_megatron_state.training_state.optimizers,
                dst_megatron_state.training_state.opt_param_scheduler,
            )

    def log_communication_info(
        self, transfer_time: float, union_world_group, logger: Logger | None = None
    ):
        info = torch.tensor(
            [
                transfer_time,
                self.transfer_manager.communicator.get_communication_bytes(),
            ],
            device=torch.cuda.current_device(),
            dtype=torch.float32,
        )

        with with_world_group(union_world_group):
            torch.distributed.all_reduce(info, op=torch.distributed.ReduceOp.MAX)

        transfer_time, communication_bytes = info.tolist()
        msg = f"[ElasticMegatron-Perf] : transfer time: {transfer_time:.2f} ms, communication bytes: {communication_bytes:.2f} GB, bandwidth: {1000 * communication_bytes / (transfer_time):.2f} GB/s"
        if logger is not None:
            logger.info(msg)
        elif self._rank == 0:
            print(msg)

    def clear_meta_state(self, parallel_strategy: dict[str, int]):
        parallel_strategy = ParallelStrategy(**parallel_strategy)
        megatron_state = self.state_manager._parallel_strategy_to_megatron_state[
            str(parallel_strategy)
        ]
        if megatron_state.training_state is not None:
            assert megatron_state.training_state.is_meta_device
        megatron_state.training_state = None

    def reshard(
        self,
        new_parallel_strategy: Dict[str, int],
        logger: Logger | None = None,
        is_meta_device: bool = False,
        save_ckpt: bool = False,
    ) -> TrainingState | None:
        assert not (is_meta_device and save_ckpt), (
            "Can't save checkpoint in meta device"
        )

        # torch.cuda.empty_cache()
        if save_ckpt and self.state_manager.current_megatron_state.training_state:
            self.state_manager.current_megatron_state.training_state.save_checkpoint(
                is_before_reshard=True
            )

        # Step-1 : Generate src/dst megatron states.
        src_megatron_state, dst_megatron_state, union_world_group = (
            self.state_manager.reshard(new_parallel_strategy, is_meta_device)
        )

        # For node not in union world group, return None
        if src_megatron_state is None:
            return None
        elif (
            dst_megatron_state is None
        ):  # For non-resharding case, return src training state
            return src_megatron_state.training_state

        # Step-2 : Resharding params and transfer.
        is_redundant_sacle_up: Optional[bool] = ParallelStrategy.is_redundant_backup(
            src_megatron_state.parallel_strategy, dst_megatron_state.parallel_strategy
        )
        if is_redundant_sacle_up is not None:
            transfer_time = self.redundant_backup(
                src_megatron_state,
                dst_megatron_state,
                is_redundant_sacle_up,
                is_meta_device,
            )
        else:
            transfer_time = self.transfer_params(
                src_megatron_state,
                dst_megatron_state,
                union_world_group,
                is_meta_device,
            )

        # Step-3 : Transfer learning rate.
        self.transfer_learning_rate(src_megatron_state, dst_megatron_state)

        # Step-4 : Fully release src training state
        if src_megatron_state.training_state is not None:
            src_megatron_state.training_state.release_optimizer()

        # Step-5 : Update model weight.
        if dst_megatron_state.training_state is not None:
            dst_megatron_state.training_state.update_model_weight()

        # Step-6 : Collect communication info
        self.log_communication_info(transfer_time, union_world_group, logger)

        # This might take 50~100+ ms
        torch.cuda.empty_cache()

        if save_ckpt and dst_megatron_state.training_state is not None:
            dst_megatron_state.training_state.save_checkpoint(is_before_reshard=False)

        return dst_megatron_state.training_state

    def get_data_parallel_group(self, parallel_strategy: Dict[str, int]):
        return self.state_manager.get_data_parallel_group(parallel_strategy)

    @classmethod
    def global_barrier_by_gloo(cls):
        global_barrier_by_gloo()


@contextmanager
def build_nccl_connection_context(transfer_manager: TransferManager):
    transfer_manager.set_fake_transfer(True)

    def foo(*args, **kwargs):
        pass

    origin_rebuild = OptimizerTensorInfo.rebuild
    origin_release = OptimizerTensorInfo.release
    origin_create_padded_optimizer_tensor = (
        OptimizerTensorInfo.create_padded_optimizer_tensor
    )
    origin_release_padded_optimizer_tensor = (
        OptimizerTensorInfo.release_padded_optimizer_tensor
    )
    origin_shuffle_swiglu = OptimizerTensorInfo.shuffle_swiglu
    origin_unshuffle_swiglu = OptimizerTensorInfo.unshuffle_swiglu
    OptimizerTensorInfo.rebuild = foo
    OptimizerTensorInfo.release = foo
    OptimizerTensorInfo.create_padded_optimizer_tensor = foo
    OptimizerTensorInfo.release_padded_optimizer_tensor = foo
    OptimizerTensorInfo.shuffle_swiglu = foo
    OptimizerTensorInfo.unshuffle_swiglu = foo
    try:
        yield
    finally:
        OptimizerTensorInfo.rebuild = origin_rebuild
        OptimizerTensorInfo.release = origin_release
        OptimizerTensorInfo.create_padded_optimizer_tensor = (
            origin_create_padded_optimizer_tensor
        )
        OptimizerTensorInfo.release_padded_optimizer_tensor = (
            origin_release_padded_optimizer_tensor
        )
        OptimizerTensorInfo.shuffle_swiglu = origin_shuffle_swiglu
        OptimizerTensorInfo.unshuffle_swiglu = origin_unshuffle_swiglu
        transfer_manager.set_fake_transfer(False)

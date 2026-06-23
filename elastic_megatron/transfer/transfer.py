from collections.abc import Callable

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
from ..resharding.optimizer_adapter import OptimizerAdapter
from ..resharding.resharding import ReshardPlan
from ..resharding.resharding_metadata import OptimizerTensorInfo
from ..resharding.resharding_pp import LayerType, ParamPositionAttr
from ..resharding.util import ParamRange, Range
from ..resharding.virtual_param import VirtualParam, VirtualParamSpace
from .communicator import Communicator


class TransferManager:
    def __init__(self, send_fn: Callable, recv_fn: Callable, broadcast_fn: Callable):
        self._rank = torch.distributed.get_rank()
        self.communicator = Communicator(self._rank, send_fn, recv_fn, broadcast_fn)
        self.send = self.communicator.send
        self.recv = self.communicator.recv
        self.broadcast = self.communicator.broadcast
        self.fake_transfer = False

    def set_fake_transfer(self, fake_transfer: bool):
        self.fake_transfer = fake_transfer
        self.communicator.set_fake_transfer(fake_transfer)

    def _send_optimizer_tensors(
        self,
        send_transfer_range_dict: dict[int, ParamRange] | dict[int, Range],
        src_optimizer_tensor_info: OptimizerTensorInfo,
    ):
        if send_transfer_range_dict is None or src_optimizer_tensor_info is None:
            return

        src_model_param_range: ParamRange = src_optimizer_tensor_info.model_param_range
        src_model_param_shape: torch.Size = src_model_param_range.to_torch_size()

        # Dp-align : the param range is a Range object
        if isinstance(list(send_transfer_range_dict.values())[0], Range):
            src_model_param_range = Range(
                0, src_optimizer_tensor_info.optimizer_tensors[0].nelement()
            )
            src_model_param_shape = -1

        for dst_rank, send_param_range in send_transfer_range_dict.items():
            if self.fake_transfer:
                self.send(None, dst=dst_rank)
                continue

            transfer_param_slices: tuple[slice] = (
                src_model_param_range.get_sub_range_slices(send_param_range)
            )
            for src_optimizer_tensor in src_optimizer_tensor_info.optimizer_tensors:
                assert src_optimizer_tensor.nelement() == src_model_param_range.size
                send_optimizer_tensor = src_optimizer_tensor.view(
                    src_model_param_shape
                )[transfer_param_slices]
                self.send(send_optimizer_tensor, dst=dst_rank)
        src_optimizer_tensor_info.release()

    def _recv_optimizer_tensors(
        self,
        recv_transfer_range_dict: dict[int, ParamRange],
        dst_optimizer_tensor_info: OptimizerTensorInfo,
    ):
        if recv_transfer_range_dict is None or dst_optimizer_tensor_info is None:
            return
        dst_optimizer_tensor_info.rebuild()
        dst_model_param_range: ParamRange = dst_optimizer_tensor_info.model_param_range
        dst_model_param_shape: torch.Size = dst_model_param_range.to_torch_size()

        # Dp-align : the param range is a Range object
        if isinstance(list(recv_transfer_range_dict.values())[0], Range):
            dst_model_param_range = Range(
                0, dst_optimizer_tensor_info.optimizer_tensors[0].nelement()
            )
            dst_model_param_shape = -1

        for src_rank, recv_param_range in recv_transfer_range_dict.items():
            if self.fake_transfer:
                self.recv(None, src=src_rank)
                continue
            transfer_param_slices: tuple[slice] = (
                dst_model_param_range.get_sub_range_slices(recv_param_range)
            )

            for dst_optimizer_tensor in dst_optimizer_tensor_info.optimizer_tensors:
                assert dst_optimizer_tensor.nelement() == dst_model_param_range.size
                recv_optimizer_tensor = dst_optimizer_tensor.view(
                    dst_model_param_shape
                )[transfer_param_slices]
                self.recv(recv_optimizer_tensor, src=src_rank)

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

    def _main_process(self, virtual_param: VirtualParam):
        """Main Process. Execute transfer plan.

        For sender:
        1. Send tensors and release memory by the aligned_dp_rank
        2. If this param has been gathered by dp-ranks, release padded optimizer tensor

        For receiver:
        1. Allocate memory for padded optimizer tensor in aligned_dp_rank
        2. Receive tensors

        .. note::
            For survival-node, it might be the sender and receiver in the same time, and self-send and self-recv is allowed and which is implemented by tensor-copy.

        .. note::
            The call order of send_fn and recv_fn cannot be changed, otherwise it will cause a copy error on the surviving node.

        .. note::
            This process will only be executed on the aligned_dp_rank.
        """
        reshard_plan: ReshardPlan = virtual_param.reshard_plan

        # Best-effort guard: when this rank holds both sides (self/survival
        # transfer), the src and dst must carry the same ordered state set —
        # _send/_recv zip optimizer_tensors positionally. No-op when only one
        # side is present on this rank. Always holds for Adam (both = 3 states).
        src_info = virtual_param.src_optimizer_tensor_info
        dst_info = virtual_param.dst_optimizer_tensor_info
        if src_info is not None and dst_info is not None:
            assert src_info.state_names == dst_info.state_names, (
                f"src/dst optimizer state set mismatch: {src_info.state_names} vs "
                f"{dst_info.state_names}; heterogeneous state sets are unsupported."
            )

        # Step-1 : Send tensors
        self._send_optimizer_tensors(
            send_transfer_range_dict=reshard_plan.global_send_info.get(self._rank),
            src_optimizer_tensor_info=virtual_param.src_optimizer_tensor_info,
        )
        if virtual_param.src_optimizer_tensor_info:
            virtual_param.src_optimizer_tensor_info.release_padded_optimizer_tensor()

        # Step-2 : Receive tensors
        # Step-2.1 : Find the aligned_dp_rank for receiver
        is_aligned_rank_for_receiver = (
            reshard_plan.data_parallel_resharding_info.dst_aligned_global_rank
            == self._rank
        )
        if not is_aligned_rank_for_receiver:
            return

        # Step-2.2 : Allocate memory for padded optimizer tensor in aligned_dp_rank
        virtual_param.dst_optimizer_tensor_info.create_padded_optimizer_tensor()

        # Step-2.3 : Receive tensors
        self._recv_optimizer_tensors(
            recv_transfer_range_dict=reshard_plan.global_recv_info.get(self._rank),
            dst_optimizer_tensor_info=virtual_param.dst_optimizer_tensor_info,
        )

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
        self, virtual_param_space: VirtualParamSpace, use_block_and_print: bool = False
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
        process_virtual_params(self._main_process)
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
            # The reshard cannot carry the per-param Adam step (dropped as
            # non-param-shaped, I-15); restore it from the param_groups step just
            # broadcast above, for optimizers that keep step per-param (CPU-offload
            # HDO). No-op for every other optimizer. Runs on each dst-active rank.
            OptimizerAdapter.create(optimizer).repair_per_param_step()
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

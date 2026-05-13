from collections import defaultdict
import os
from typing import Dict, List, Tuple, Callable
import torch
from .communicator import Communicator


from ..resharding.virtual_param import VirtualParamSpace, VirtualParam
from megatron.core import parallel_state
from ..resharding.resharding_metadata import OptimizerTensorInfo
from ..resharding.resharding_pp import ParamPositionAttr, LayerType
from ..resharding.util import ParamRange, Timer
from ..resharding.util import Range
from ..megatron_manager.parallel_strategy import ParallelStrategy
from ..megatron_manager.training_state import (
    TrainingState,
    update_optimizer_by_state_dict,
    load_state_dict_with_no_step,
)
from megatron.core.optimizer_param_scheduler import OptimizerParamScheduler
from megatron.core.optimizer import MegatronOptimizer
from ..resharding.resharding import ReshardPlan

from megatron.core.timers import Timers as MegatronTimers


class TransferManager:
    def __init__(self, send_fn: Callable, recv_fn: Callable, broadcast_fn: Callable):
        self._rank = torch.distributed.get_rank()
        self.communicator = Communicator(self._rank, send_fn, recv_fn, broadcast_fn)
        self.send = self.communicator.send
        self.recv = self.communicator.recv
        self.broadcast = self.communicator.broadcast
        self.fake_transfer = False
        self._transfer_timers = MegatronTimers(
            log_level=int(os.getenv("ELASTIC_TRANSFER_LOG_LEVEL", "1")),
            log_option="minmax",
        )
        self.use_asyncbuffer_p2p = os.getenv("ELASTIC_USE_ASYNCBUFFER_P2P", "0") == "1"

    def set_fake_transfer(self, fake_transfer: bool):
        self.fake_transfer = fake_transfer
        self.communicator.set_fake_transfer(fake_transfer)

    def _send_optimizer_tensors(
        self,
        send_transfer_range_dict: Dict[int, ParamRange] | Dict[int, Range],
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

        if self.fake_transfer:
            for dst_rank in send_transfer_range_dict:
                self.send(None, dst=dst_rank)
            src_optimizer_tensor_info.release()
            return

        batch = self.communicator.batch_p2p()
        for dst_rank, send_param_range in send_transfer_range_dict.items():
            transfer_param_slices: Tuple[slice] = (
                src_model_param_range.get_sub_range_slices(send_param_range)
            )
            for src_optimizer_tensor in src_optimizer_tensor_info.optimizer_tensors:
                assert src_optimizer_tensor.nelement() == src_model_param_range.size
                send_optimizer_tensor = src_optimizer_tensor.view(
                    src_model_param_shape
                )[transfer_param_slices]
                batch.isend(send_optimizer_tensor, dst=dst_rank)
        batch.wait()
        src_optimizer_tensor_info.release()

    def _recv_optimizer_tensors(
        self,
        recv_transfer_range_dict: Dict[int, ParamRange],
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

        if self.fake_transfer:
            for src_rank in recv_transfer_range_dict:
                self.recv(None, src=src_rank)
            return

        batch = self.communicator.batch_p2p()
        for src_rank, recv_param_range in recv_transfer_range_dict.items():
            transfer_param_slices: Tuple[slice] = (
                dst_model_param_range.get_sub_range_slices(recv_param_range)
            )

            for dst_optimizer_tensor in dst_optimizer_tensor_info.optimizer_tensors:
                assert dst_optimizer_tensor.nelement() == dst_model_param_range.size
                recv_optimizer_tensor = dst_optimizer_tensor.view(
                    dst_model_param_shape
                )[transfer_param_slices]
                batch.irecv(recv_optimizer_tensor, src=src_rank)
        batch.wait()

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
        dp_distribution: Dict[int, Range] = (
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

    def _collect_send_tasks_for_virtual_param(
        self,
        virtual_param: VirtualParam,
        send_tasks: Dict[int, List[torch.Tensor]],
        send_buffers: List[torch.Tensor],
        src_to_release: List[OptimizerTensorInfo],
        src_to_release_padded: List[OptimizerTensorInfo],
    ) -> None:
        src_optimizer_tensor_info = virtual_param.src_optimizer_tensor_info
        if src_optimizer_tensor_info is None:
            return
        src_to_release_padded.append(src_optimizer_tensor_info)

        send_transfer_range_dict = virtual_param.reshard_plan.global_send_info.get(
            self._rank
        )
        if send_transfer_range_dict is None:
            return

        if self.fake_transfer:
            for dst_rank in send_transfer_range_dict:
                self.send(None, dst=dst_rank)
            src_to_release.append(src_optimizer_tensor_info)
            return

        src_model_param_range, src_model_param_shape = self._resolve_transfer_layout(
            send_transfer_range_dict,
            src_optimizer_tensor_info,
        )
        for dst_rank, send_param_range in send_transfer_range_dict.items():
            transfer_param_slices: Tuple[slice] = (
                src_model_param_range.get_sub_range_slices(send_param_range)
            )
            for src_optimizer_tensor in src_optimizer_tensor_info.optimizer_tensors:
                assert src_optimizer_tensor.nelement() == src_model_param_range.size
                send_optimizer_tensor = src_optimizer_tensor.view(
                    src_model_param_shape
                )[transfer_param_slices]
                if dst_rank == self._rank:
                    self.send(send_optimizer_tensor, dst=dst_rank)
                    continue

                send_buffer = send_optimizer_tensor.contiguous()
                send_buffers.append(send_buffer)
                send_tasks[dst_rank].append(send_buffer)
        src_to_release.append(src_optimizer_tensor_info)

    def _collect_recv_tasks_for_virtual_param(
        self,
        virtual_param: VirtualParam,
        recv_tasks: Dict[int, List[torch.Tensor]],
        recv_copy_back: List[Tuple[torch.Tensor, torch.Tensor]],
    ) -> None:
        reshard_plan: ReshardPlan = virtual_param.reshard_plan
        is_aligned_rank_for_receiver = (
            reshard_plan.data_parallel_resharding_info.dst_aligned_global_rank
            == self._rank
        )
        if not is_aligned_rank_for_receiver:
            return

        dst_optimizer_tensor_info = virtual_param.dst_optimizer_tensor_info
        assert dst_optimizer_tensor_info is not None, (
            f"rank {self._rank} is the aligned rank of param (name={virtual_param.name}), "
            "but dst_optimizer_tensor_info is None."
        )

        recv_transfer_range_dict = reshard_plan.global_recv_info.get(self._rank)
        assert recv_transfer_range_dict is not None, (
            f"rank {self._rank} is the aligned rank of param (name={virtual_param.name}), "
            "but recv_transfer_range_dict is None."
        )

        # Allocate memory for padded optimizer tensor in aligned_dp_rank.
        dst_optimizer_tensor_info.create_padded_optimizer_tensor()
        dst_optimizer_tensor_info.rebuild()
        dst_model_param_range, dst_model_param_shape = self._resolve_transfer_layout(
            recv_transfer_range_dict,
            dst_optimizer_tensor_info,
        )

        if self.fake_transfer:
            for src_rank in recv_transfer_range_dict:
                self.recv(None, src=src_rank)
            return

        for src_rank, recv_param_range in recv_transfer_range_dict.items():
            transfer_param_slices: Tuple[slice] = (
                dst_model_param_range.get_sub_range_slices(recv_param_range)
            )
            for dst_optimizer_tensor in dst_optimizer_tensor_info.optimizer_tensors:
                assert dst_optimizer_tensor.nelement() == dst_model_param_range.size
                recv_optimizer_tensor = dst_optimizer_tensor.view(
                    dst_model_param_shape
                )[transfer_param_slices]
                if src_rank == self._rank:
                    self.recv(recv_optimizer_tensor, src=src_rank)
                    continue

                if recv_optimizer_tensor.is_contiguous():
                    recv_buffer = recv_optimizer_tensor
                else:
                    recv_buffer = torch.empty_like(
                        recv_optimizer_tensor,
                        memory_format=torch.contiguous_format,
                    )
                    recv_copy_back.append((recv_optimizer_tensor, recv_buffer))

                recv_tasks[src_rank].append(recv_buffer)

    def _main_process_batch(self, virtual_params: List[VirtualParam]):
        send_tasks: Dict[int, List[torch.Tensor]] = defaultdict(list)
        recv_tasks: Dict[int, List[torch.Tensor]] = defaultdict(list)
        send_buffers: List[torch.Tensor] = []
        recv_copy_back: List[Tuple[torch.Tensor, torch.Tensor]] = []
        recv_unpack_tasks: List[Tuple[torch.Tensor, List[torch.Tensor]]] = []
        src_to_release: List[OptimizerTensorInfo] = []
        src_to_release_padded: List[OptimizerTensorInfo] = []

        self._transfer_timers("Main process batch", log_level=0).start()
        for virtual_param in virtual_params:
            if self._should_skip_virtual_param(virtual_param):
                continue

            self._collect_send_tasks_for_virtual_param(
                virtual_param=virtual_param,
                send_tasks=send_tasks,
                send_buffers=send_buffers,
                src_to_release=src_to_release,
                src_to_release_padded=src_to_release_padded,
            )
            self._collect_recv_tasks_for_virtual_param(
                virtual_param=virtual_param,
                recv_tasks=recv_tasks,
                recv_copy_back=recv_copy_back,
            )
        self._transfer_timers("Main process batch").stop()

        batch = self.communicator.batch_p2p()
        world_size = torch.distributed.get_world_size()

        def _pack_tensors(
            peer_tensors: List[torch.Tensor], copy_to_buffer: bool = True
        ) -> torch.Tensor:
            first = peer_tensors[0]
            for tensor in peer_tensors[1:]:
                assert tensor.device == first.device, (
                    "Peer tensors must be on the same device when packing. "
                    f"Found {first.device} and {tensor.device}."
                )

            total_nbytes = sum(tensor.nbytes for tensor in peer_tensors)
            packed = torch.empty(total_nbytes, dtype=torch.uint8, device=first.device)

            if copy_to_buffer:
                offset = 0
                for tensor in peer_tensors:
                    tensor_bytes = tensor.view(torch.uint8).reshape(-1)
                    nbytes = tensor_bytes.numel()
                    packed[offset : offset + nbytes].copy_(tensor_bytes)
                    offset += nbytes
            return packed

        def _unpack_tensors(
            packed: torch.Tensor, peer_tensors: List[torch.Tensor]
        ) -> None:
            """Inverse of :func:`_pack_tensors`: scatter ``packed`` back into
            the original tensors in the same order they were packed."""
            offset = 0
            for tensor in peer_tensors:
                tensor_bytes = tensor.view(torch.uint8).reshape(-1)
                nbytes = tensor_bytes.numel()
                tensor_bytes.copy_(packed[offset : offset + nbytes])
                offset += nbytes

        self._transfer_timers("Pack peer tensors", log_level=1).start()
        num_steps = 1 << ((world_size - 1).bit_length())
        for step in range(1, num_steps):
            peer = self._rank ^ step
            if peer >= world_size:
                continue

            peer_sends = send_tasks.get(peer, [])
            peer_recvs = recv_tasks.get(peer, [])
            if not peer_sends and not peer_recvs:
                continue

            packed_send_tensor = None
            packed_recv_tensor = None
            if peer_sends:
                packed_send_tensor = _pack_tensors(peer_sends)
            if peer_recvs:
                packed_recv_tensor = _pack_tensors(peer_recvs, False)
                recv_unpack_tasks.append((packed_recv_tensor, peer_recvs))

            # Order send/recv to avoid potential deadlocks: the lower-ranked
            # peer enqueues send first, the higher-ranked peer enqueues recv
            # first. ``BatchP2P`` preserves enqueue order in the op list.
            if self._rank < peer:
                if packed_send_tensor is not None:
                    batch.isend(packed_send_tensor, dst=peer)
                if packed_recv_tensor is not None:
                    batch.irecv(packed_recv_tensor, src=peer)
            else:
                if packed_recv_tensor is not None:
                    batch.irecv(packed_recv_tensor, src=peer)
                if packed_send_tensor is not None:
                    batch.isend(packed_send_tensor, dst=peer)
        self._transfer_timers("Pack peer tensors").stop()

        if batch.has_ops():
            self._transfer_timers("Real transfer", log_level=1).start()
            batch.wait()
            self._transfer_timers("Real transfer").stop()

            self._transfer_timers("Unpack recv tensors", log_level=1).start()
            for packed_recv_tensor, original_recv_tensors in recv_unpack_tasks:
                _unpack_tensors(packed_recv_tensor, original_recv_tensors)
            self._transfer_timers("Unpack recv tensors").stop()
            self._transfer_timers("Copy recv tensors", log_level=1).start()
            for recv_optimizer_tensor, recv_buffer in recv_copy_back:
                recv_optimizer_tensor.data.copy_(recv_buffer)
            self._transfer_timers("Copy recv tensors").stop()
        else:
            # for timer synchronization, otherwise will get stuck
            self._transfer_timers("Real transfer", log_level=1).start()
            self._transfer_timers("Real transfer").stop()
            self._transfer_timers("Unpack recv tensors", log_level=1).start()
            self._transfer_timers("Unpack recv tensors").stop()
            self._transfer_timers("Copy recv tensors", log_level=1).start()
            self._transfer_timers("Copy recv tensors").stop()

        self._transfer_timers("Release optimizer tensors", log_level=1).start()
        for src_optimizer_tensor_info in src_to_release:
            src_optimizer_tensor_info.release()
        for src_optimizer_tensor_info in src_to_release_padded:
            src_optimizer_tensor_info.release_padded_optimizer_tensor()
        self._transfer_timers("Release optimizer tensors").stop()
        self._transfer_timers.log(names=None, rank=0)

    def _post_process(self, virtual_param: VirtualParam):
        """Post Process. For receiver, unshuffle swiglu and scatter param to dp ranks(ZeRO-1), and release all padded optimizer tensors."""
        dst_optimizer_tensor_info: OptimizerTensorInfo = (
            virtual_param.dst_optimizer_tensor_info
        )
        if dst_optimizer_tensor_info is None:
            return None

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

        dp_distribution: Dict[int, Range] = (
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
        if self.use_asyncbuffer_p2p:
            self._main_process_batch(virtual_param_space.all_virtual_params)
        else:
            process_virtual_params(self._main_process)
        process_virtual_params(self._post_process)
        self.transfer_word_embedding_and_output_layer(
            virtual_param_space.all_virtual_params[0],
            virtual_param_space.all_virtual_params[-1],
        )

    def redundant_backup(
        self,
        is_sacle_up: bool,
        src_training_state: TrainingState,
        dst_training_state: TrainingState,
        group_src: int = 0,
    ):
        # Delete redundant backup
        if not is_sacle_up:
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
        optimizers: List[MegatronOptimizer],
        opt_param_scheduler: OptimizerParamScheduler,
    ):
        for optimizer in optimizers:
            self._transfer_optimizer_state_dict(optimizer)
        self._transfer_param_scheduler(opt_param_scheduler)

    def _transfer_optimizer_state_dict(
        self,
        optimizer: MegatronOptimizer,
    ):
        broacast_object = None
        if self._rank == 0:
            broacast_object = []
            state_dict = optimizer.state_dict()
            for param_group in state_dict["optimizer"]["param_groups"]:
                broacast_object.append(
                    {
                        key: value
                        for key, value in param_group.items()
                        if key != "params"
                    }
                )
        broacast_object_list = [broacast_object]
        torch.distributed.broadcast_object_list(broacast_object_list, src=0)
        if self._rank != 0:
            update_optimizer_by_state_dict(optimizer, broacast_object_list[0])

    def _transfer_param_scheduler(
        self,
        opt_param_scheduler: OptimizerParamScheduler,
    ):
        broacast_object = None
        if self._rank == 0:
            broacast_object = opt_param_scheduler.state_dict()
        broacast_object_list = [broacast_object]
        torch.distributed.broadcast_object_list(broacast_object_list, src=0)
        if self._rank != 0:
            load_state_dict_with_no_step(opt_param_scheduler, broacast_object_list[0])

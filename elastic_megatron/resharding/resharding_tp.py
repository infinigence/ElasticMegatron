from dataclasses import dataclass

import torch

from .util import (
    ParamRange,
)


@dataclass
class TensorParallelReshardingInfo:
    # Key : src_tp_rank
    # Value : {dst_tp_rank: param_range}
    send_info: dict[int, dict[int, ParamRange]]

    # Key : dst_tp_rank
    # Value : {src_tp_rank: param_range}
    recv_info: dict[int, dict[int, ParamRange]]


class TensorParallelAttr:
    """TensorParallelAttr is used to store the tensor parallel attributes of a model parameter.

    Attributes:
        model_param: param in megatron model
        parallel_group_size: the size of the parallel group
        parallel_group_str: the string of the parallel group
        tensor_model_parallel: whether the model parameter is tensor model parallel
        partition_dim: the dimension of the partition
        partition_stride: the stride of the partition
        model_param_shape: the shape of the model parameter
        model_param_range_in_group: the range of the model parameter in the parallel group
    """

    def __init__(
        self,
        model_param: torch.nn.Parameter,
        parallel_group_size: int = 1,
        parallel_group_str: str = "TP",
        force_unsharded: bool = False,
    ):
        self.parallel_group_size = parallel_group_size
        self.parallel_group_str = parallel_group_str

        if force_unsharded:
            self.tensor_model_parallel = False
            self.partition_dim = -1
            self.partition_stride = 1
        else:
            self.tensor_model_parallel = getattr(
                model_param, "tensor_model_parallel", False
            )
            self.partition_dim = getattr(model_param, "partition_dim", -1)
            self.partition_stride = getattr(model_param, "partition_stride", 1)
        assert self.partition_stride == 1, "Only support partition_stride==1"

        self._init_model_param_ranges(model_param.shape)

        # Cache
        self._resharding_info_cache: dict[
            tuple[int, int], TensorParallelReshardingInfo
        ] = {}

    def _init_model_param_ranges(self, model_param_shape: torch.Size):
        """Init the model param ranges in the parallel group."""
        model_param_shape = list(model_param_shape)
        if self.tensor_model_parallel and self.parallel_group_size != 1:
            assert self.partition_dim != -1
            model_param_shape[self.partition_dim] *= self.parallel_group_size

        # This ParamRange represents the model param range when group size is 1.
        self.model_param_range_in_group: ParamRange = ParamRange(
            param_shape=torch.Size(model_param_shape)
        )
        # Key : group Size
        # Value : ParamRange of the model weight of group rank-0
        self._group_size_to_model_param_ranges: dict[int, ParamRange] = {
            1: self.model_param_range_in_group,
        }

    def get_model_param_range(self, parallel_group_size: int) -> ParamRange:
        """Get the model param range in the parallel group by given parallel group size."""
        if parallel_group_size == 1 or not self.tensor_model_parallel:
            return self.model_param_range_in_group

        if self._group_size_to_model_param_ranges.get(parallel_group_size) is not None:
            return self._group_size_to_model_param_ranges[parallel_group_size]

        model_param_range_chunks: list[ParamRange] = (
            self.model_param_range_in_group.chunk(
                split=parallel_group_size, dim=self.partition_dim
            )
        )
        self._group_size_to_model_param_ranges[parallel_group_size] = (
            model_param_range_chunks[0]
        )
        return model_param_range_chunks[0]

    def reshard(
        self, src_parallel_group_size: int, dst_parallel_group_size: int
    ) -> TensorParallelReshardingInfo:
        """Reshard the model param range in the parallel group from src_parallel_group_size to dst_parallel_group_size."""
        assert (src_parallel_group_size & (src_parallel_group_size - 1)) == 0
        assert (dst_parallel_group_size & (dst_parallel_group_size - 1)) == 0
        key = (src_parallel_group_size, dst_parallel_group_size)

        if key in self._resharding_info_cache:
            return self._resharding_info_cache[key]

        if not self.tensor_model_parallel:
            resharding_info: TensorParallelReshardingInfo = (
                self._resharding_without_split(
                    src_parallel_group_size, dst_parallel_group_size
                )
            )
        else:
            resharding_info: TensorParallelReshardingInfo = self._resharding_with_split(
                src_parallel_group_size, dst_parallel_group_size
            )

        self._resharding_info_cache[key] = resharding_info
        return resharding_info

    def _resharding_without_split(
        self, src_parallel_group_size: int, dst_parallel_group_size: int
    ) -> TensorParallelReshardingInfo:
        """Reshard param without tensor split. eg. layernorm param.

        For simplicity, use group rank 0 broadcast to other ranks in the group.
        """
        src_param_range: ParamRange = self.get_model_param_range(
            src_parallel_group_size
        )
        dst_param_range: ParamRange = self.get_model_param_range(
            dst_parallel_group_size
        )
        assert src_param_range.size == dst_param_range.size

        send_info: dict[int, dict[int, ParamRange]] = {0: {}}
        recv_info: dict[int, dict[int, ParamRange]] = {}
        for dst_rank in range(dst_parallel_group_size):
            send_info[0][dst_rank] = src_param_range
            recv_info[dst_rank] = {0: src_param_range}
        return TensorParallelReshardingInfo(send_info=send_info, recv_info=recv_info)

    def _resharding_with_split(
        self, src_parallel_group_size: int, dst_parallel_group_size: int
    ) -> TensorParallelReshardingInfo:
        """Reshard the model param range in the parallel group from src_parallel_group_size to dst_parallel_group_size with split."""
        src_param_range: ParamRange = self.get_model_param_range(
            src_parallel_group_size
        )
        dst_param_range: ParamRange = self.get_model_param_range(
            dst_parallel_group_size
        )
        if src_parallel_group_size == dst_parallel_group_size:
            assert src_param_range.size == dst_param_range.size

        send_info: dict[int, dict[int, ParamRange]] = {}
        recv_info: dict[int, dict[int, ParamRange]] = {}

        # Scale-up : split src_tensor and send split results to dst_ranks
        if src_parallel_group_size < dst_parallel_group_size:
            # Split src tensor
            scale_up_ratio = dst_parallel_group_size // src_parallel_group_size
            src_param_range_chunks: list[ParamRange] = src_param_range.chunk(
                split=scale_up_ratio, dim=self.partition_dim
            )

            for src_rank in range(src_parallel_group_size):
                dst_rank_offset = src_rank * scale_up_ratio
                for chunk_id, src_param_range_chunk in enumerate(
                    src_param_range_chunks
                ):
                    dst_rank = dst_rank_offset + chunk_id
                    send_info.setdefault(src_rank, {})[dst_rank] = src_param_range_chunk
                    recv_info[dst_rank] = {src_rank: src_param_range_chunks[0]}
            return TensorParallelReshardingInfo(
                send_info=send_info, recv_info=recv_info
            )

        # Scale-down : split dst tensor and recv src tensors from src_ranks
        scale_down_ratio = src_parallel_group_size // dst_parallel_group_size
        # Split dst tensor
        dst_param_range_chunks: list[ParamRange] = dst_param_range.chunk(
            split=scale_down_ratio, dim=self.partition_dim
        )

        for dst_rank in range(dst_parallel_group_size):
            src_rank_offset = dst_rank * scale_down_ratio
            for chunk_id, dst_param_range_chunk in enumerate(dst_param_range_chunks):
                src_rank = src_rank_offset + chunk_id
                # When scale-down, one dst tensor should be split into multiple src tensors
                recv_info.setdefault(dst_rank, {})[src_rank] = dst_param_range_chunk
                send_info[src_rank] = {dst_rank: dst_param_range_chunks[0]}
        return TensorParallelReshardingInfo(send_info=send_info, recv_info=recv_info)

    def __str__(self):
        return f"TensorParallelAttr(tensor_model_parallel={self.tensor_model_parallel}, partition_dim={self.partition_dim}, partition_stride={self.partition_stride}), model_param_range_in_group={self.model_param_range_in_group}"

    def __hash__(self):
        return hash(
            (
                self.tensor_model_parallel,
                self.partition_dim,
                self.partition_stride,
                self.model_param_range_in_group,
            )
        )

    def __eq__(self, other):
        assert isinstance(other, TensorParallelAttr)
        return hash(self) == hash(other)

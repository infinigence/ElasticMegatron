from .resharding_pp import PipelineParallelReshardingInfo
from .resharding_tp import TensorParallelReshardingInfo
from .resharding_dp import DataParallelReshardingInfo, ExpertParallelReshardingInfo
from .util import (
    ParamRange,
)
from ..megatron_manager.parallel_strategy import ParallelStrategy
from typing import Dict, Optional

from dataclasses import dataclass


def get_global_rank(
    pipeline_model_parallel_rank: int,
    data_parallel_rank: int,
    expert_model_parallel_rank: Optional[int],
    tensor_model_parallel_rank: int,
    parallel_strategy: ParallelStrategy,
) -> int:
    """Assume order is TP->EP->DP->PP"""
    is_expert = expert_model_parallel_rank is not None
    if is_expert:
        data_parallel_size = parallel_strategy.expert_data_parallel_size
        expert_model_parallel_size = parallel_strategy.expert_model_parallel_size
    else:
        data_parallel_size = parallel_strategy.get_data_parallel_world_size(
            with_context_parallel=True, partial_data_parallel=True
        )
        expert_model_parallel_size = 1
        expert_model_parallel_rank = 0

    tensor_model_parallel_size = parallel_strategy.get_tensor_model_parallel_size(
        is_expert
    )

    global_rank = (
        pipeline_model_parallel_rank
        * data_parallel_size
        * expert_model_parallel_size
        * tensor_model_parallel_size
        + data_parallel_rank * expert_model_parallel_size * tensor_model_parallel_size
        + expert_model_parallel_rank * tensor_model_parallel_size
        + tensor_model_parallel_rank
    )
    return global_rank


@dataclass
class ReshardPlan:
    tensor_parallel_resharding_info: TensorParallelReshardingInfo
    pipeline_parallel_resharding_info: PipelineParallelReshardingInfo
    data_parallel_resharding_info: DataParallelReshardingInfo
    expert_parallel_reshardign_info: ExpertParallelReshardingInfo

    src_parallel_strategy: ParallelStrategy
    dst_parallel_strategy: ParallelStrategy

    def __post_init__(self):
        if self.expert_parallel_reshardign_info is not None:
            assert (
                self.src_parallel_strategy.get_tensor_model_parallel_size(
                    is_expert=True
                )
                == 1
            )
            assert (
                self.dst_parallel_strategy.get_tensor_model_parallel_size(
                    is_expert=True
                )
                == 1
            )
            assert len(self.tensor_parallel_resharding_info.send_info) == 1
            assert len(self.tensor_parallel_resharding_info.recv_info) == 1

            self.src_ep_rank = (
                self.expert_parallel_reshardign_info.src_expert_parallel_rank
            )
            self.dst_ep_rank = (
                self.expert_parallel_reshardign_info.dst_expert_parallel_rank
            )
        else:
            self.src_ep_rank = None
            self.dst_ep_rank = None

        self.src_pp_rank = self.pipeline_parallel_resharding_info.src_pp_rank
        self.dst_pp_rank = self.pipeline_parallel_resharding_info.dst_pp_rank
        if self.src_pp_rank == -1 or self.dst_pp_rank == -1:
            self.global_send_info = None
            self.global_recv_info = None
            return

        self.src_dp_rank = self.data_parallel_resharding_info.src_aligned_dp_rank
        self.dst_dp_rank = self.data_parallel_resharding_info.dst_aligned_dp_rank

        self.global_send_info: Dict[int, Dict[int, ParamRange]] = (
            self._generate_global_send_info()
        )
        self.global_recv_info: Dict[int, Dict[int, ParamRange]] = (
            self._generate_global_recv_info()
        )

    def _generate_global_send_info(self) -> Dict[int, Dict[int, ParamRange]]:
        """Base on the send_info of TensorParallelReshardingInfo, update it's rank from TP Rank to Global Rank."""
        global_send_info: Dict[int, Dict[int, ParamRange]] = {}
        tp_send_info: Dict[int, Dict[int, ParamRange]] = (
            self.tensor_parallel_resharding_info.send_info
        )
        for src_tp_rank, model_param_ranges_dict in tp_send_info.items():
            src_global_rank = get_global_rank(
                pipeline_model_parallel_rank=self.src_pp_rank,
                data_parallel_rank=self.src_dp_rank,
                expert_model_parallel_rank=self.src_ep_rank,
                tensor_model_parallel_rank=src_tp_rank,
                parallel_strategy=self.src_parallel_strategy,
            )

            global_send_info[src_global_rank] = {}
            for dst_tp_rank, model_param_range in model_param_ranges_dict.items():
                global_dst_rank = get_global_rank(
                    pipeline_model_parallel_rank=self.dst_pp_rank,
                    data_parallel_rank=self.dst_dp_rank,
                    expert_model_parallel_rank=self.dst_ep_rank,
                    tensor_model_parallel_rank=dst_tp_rank,
                    parallel_strategy=self.dst_parallel_strategy,
                )
                global_send_info[src_global_rank][global_dst_rank] = model_param_range

        return global_send_info

    def _generate_global_recv_info(self) -> Dict[int, Dict[int, ParamRange]]:
        """Base on the recv_info of TensorParallelReshardingInfo, update it's rank from TP Rank to Global Rank."""
        global_recv_info: Dict[int, Dict[int, ParamRange]] = {}
        tp_recv_info: Dict[int, Dict[int, ParamRange]] = (
            self.tensor_parallel_resharding_info.recv_info
        )

        for dst_tp_rank, model_param_ranges_dict in tp_recv_info.items():
            global_dst_rank = get_global_rank(
                pipeline_model_parallel_rank=self.dst_pp_rank,
                data_parallel_rank=self.dst_dp_rank,
                expert_model_parallel_rank=self.dst_ep_rank,
                tensor_model_parallel_rank=dst_tp_rank,
                parallel_strategy=self.dst_parallel_strategy,
            )

            global_recv_info[global_dst_rank] = {}
            for src_tp_rank, model_param_range in model_param_ranges_dict.items():
                global_src_rank = get_global_rank(
                    pipeline_model_parallel_rank=self.src_pp_rank,
                    data_parallel_rank=self.src_dp_rank,
                    expert_model_parallel_rank=self.src_ep_rank,
                    tensor_model_parallel_rank=src_tp_rank,
                    parallel_strategy=self.src_parallel_strategy,
                )
                global_recv_info[global_dst_rank][global_src_rank] = model_param_range
        return global_recv_info

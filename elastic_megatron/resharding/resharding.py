from dataclasses import dataclass

from ..megatron_manager.parallel_strategy import ParallelStrategy
from .resharding_dp import DataParallelReshardingInfo, ExpertParallelReshardingInfo
from .resharding_pp import PipelineParallelReshardingInfo
from .resharding_tp import TensorParallelReshardingInfo
from .util import (
    ParamRange,
)


def get_global_rank(
    pipeline_model_parallel_rank: int,
    data_parallel_rank: int,
    expert_model_parallel_rank: int | None,
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
    expert_parallel_resharding_info: ExpertParallelReshardingInfo

    src_parallel_strategy: ParallelStrategy
    dst_parallel_strategy: ParallelStrategy

    def __post_init__(self):
        if self.expert_parallel_resharding_info is not None:
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
            # When EP=1 on either side, experts are treated as dense and replicated
            # across dense TP ranks, so send/recv plans expand beyond one entry.
            # Only enforce the single-entry invariant when both sides have EP>1.
            src_ep1 = self.src_parallel_strategy.expert_is_dense_bucketed()
            dst_ep1 = self.dst_parallel_strategy.expert_is_dense_bucketed()
            if not (src_ep1 or dst_ep1):
                assert len(self.tensor_parallel_resharding_info.send_info) == 1
                assert len(self.tensor_parallel_resharding_info.recv_info) == 1

            # 注意:self.src_ep_rank / self.dst_ep_rank 是 *物理* expert_parallel_rank,
            # 仅对外暴露 ep 语义。要传给 get_global_rank() 的应当用下面计算的
            # self._src_ep_rank_for_gr / self._dst_ep_rank_for_gr —— 它们在该侧 EP=1
            # 时变成 None,让 get_global_rank 走 dense 分支(dense tp_size 作 TP 系数)。
            self.src_ep_rank = (
                self.expert_parallel_resharding_info.src_expert_parallel_rank
            )
            self.dst_ep_rank = (
                self.expert_parallel_resharding_info.dst_expert_parallel_rank
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

        # ep_rank passed to get_global_rank: None when that side treats experts as
        # dense (EP=1), so get_global_rank uses the dense tp_size as the TP factor.
        self._src_ep_rank_for_gr = self._ep_rank_for_global_rank(is_src=True)
        self._dst_ep_rank_for_gr = self._ep_rank_for_global_rank(is_src=False)

        self.global_send_info: dict[int, dict[int, ParamRange]] = (
            self._generate_global_send_info()
        )
        self.global_recv_info: dict[int, dict[int, ParamRange]] = (
            self._generate_global_recv_info()
        )

    def _ep_rank_for_global_rank(self, is_src: bool) -> int | None:
        """Return the ep_rank to pass into `get_global_rank` for this plan's side.

        返回 None 表示"该侧 expert 被 Megatron 当成 dense 处理",get_global_rank
        会走 dense 分支(不考虑 ep 维度)。
        """
        if self.expert_parallel_resharding_info is None:
            return None
        ps = self.src_parallel_strategy if is_src else self.dst_parallel_strategy
        if ps.expert_is_dense_bucketed():
            return None
        return self.src_ep_rank if is_src else self.dst_ep_rank

    def _generate_global_send_info(self) -> dict[int, dict[int, ParamRange]]:
        """Base on the send_info of TensorParallelReshardingInfo, update it's rank from TP Rank to Global Rank."""
        global_send_info: dict[int, dict[int, ParamRange]] = {}
        tp_send_info: dict[int, dict[int, ParamRange]] = (
            self.tensor_parallel_resharding_info.send_info
        )
        for src_tp_rank, model_param_ranges_dict in tp_send_info.items():
            src_global_rank = get_global_rank(
                pipeline_model_parallel_rank=self.src_pp_rank,
                data_parallel_rank=self.src_dp_rank,
                expert_model_parallel_rank=self._src_ep_rank_for_gr,
                tensor_model_parallel_rank=src_tp_rank,
                parallel_strategy=self.src_parallel_strategy,
            )

            global_send_info[src_global_rank] = {}
            for dst_tp_rank, model_param_range in model_param_ranges_dict.items():
                global_dst_rank = get_global_rank(
                    pipeline_model_parallel_rank=self.dst_pp_rank,
                    data_parallel_rank=self.dst_dp_rank,
                    expert_model_parallel_rank=self._dst_ep_rank_for_gr,
                    tensor_model_parallel_rank=dst_tp_rank,
                    parallel_strategy=self.dst_parallel_strategy,
                )
                global_send_info[src_global_rank][global_dst_rank] = model_param_range

        return global_send_info

    def _generate_global_recv_info(self) -> dict[int, dict[int, ParamRange]]:
        """Base on the recv_info of TensorParallelReshardingInfo, update it's rank from TP Rank to Global Rank."""
        global_recv_info: dict[int, dict[int, ParamRange]] = {}
        tp_recv_info: dict[int, dict[int, ParamRange]] = (
            self.tensor_parallel_resharding_info.recv_info
        )

        for dst_tp_rank, model_param_ranges_dict in tp_recv_info.items():
            global_dst_rank = get_global_rank(
                pipeline_model_parallel_rank=self.dst_pp_rank,
                data_parallel_rank=self.dst_dp_rank,
                expert_model_parallel_rank=self._dst_ep_rank_for_gr,
                tensor_model_parallel_rank=dst_tp_rank,
                parallel_strategy=self.dst_parallel_strategy,
            )

            global_recv_info[global_dst_rank] = {}
            for src_tp_rank, model_param_range in model_param_ranges_dict.items():
                global_src_rank = get_global_rank(
                    pipeline_model_parallel_rank=self.src_pp_rank,
                    data_parallel_rank=self.src_dp_rank,
                    expert_model_parallel_rank=self._src_ep_rank_for_gr,
                    tensor_model_parallel_rank=src_tp_rank,
                    parallel_strategy=self.src_parallel_strategy,
                )
                global_recv_info[global_dst_rank][global_src_rank] = model_param_range
        return global_recv_info

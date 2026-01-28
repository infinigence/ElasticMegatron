from dataclasses import dataclass
import torch
from megatron.core import parallel_state
from megatron.training.global_vars import get_args
from typing import List, Optional

from .rank_generator import ElasticRankGenerator


@dataclass
class ParallelStrategy:
    world_size: int
    tensor_model_parallel_size: int
    pipeline_model_parallel_size: int
    context_parallel_size: int = 1
    num_distributed_optimizer_instances: int = 1
    expert_model_parallel_size: int = 1
    expert_tensor_parallel_size: int = None
    sequence_parallel: bool = None

    order: str = "tp-cp-ep-dp-pp"
    ranks: List[int] = None

    def __post_init__(self):
        self._rank = torch.distributed.get_rank()
        self.is_running = self._rank < self.world_size
        assert self.world_size > 0
        assert self.world_size % self.tensor_model_parallel_size == 0
        assert self.world_size % self.pipeline_model_parallel_size == 0

        self.model_parallel_size = (
            self.tensor_model_parallel_size * self.pipeline_model_parallel_size
        )
        self.total_model_size = self.model_parallel_size * self.context_parallel_size
        assert self.world_size % self.total_model_size == 0
        self.data_parallel_size = self.world_size // self.total_model_size

        if self.tensor_model_parallel_size == 1 and self.sequence_parallel:
            raise ValueError(
                "Sequence parallel is not supported with tensor model parallel size 1"
            )
        if self.sequence_parallel is None:
            self.sequence_parallel = self.tensor_model_parallel_size > 1

        self._init_group_zero()
        self._init_moe()
        self._init_transformer_layer()

        self.rank_generator = ElasticRankGenerator(self)

    def _init_transformer_layer(self):
        args = get_args()
        self.num_layers = args.num_layers
        pipeline_model_parallel_size = self.pipeline_model_parallel_size

        assert (
            self.num_layers >= 1 and args.num_layers % pipeline_model_parallel_size == 0
        )
        self.transformer_layer_num_per_stage = (
            self.num_layers // pipeline_model_parallel_size
        )

        # Moe
        self.num_experts = args.num_experts
        if self.num_experts == 0:
            self.num_experts = None
        if not self.num_experts:
            self.expert_num_per_ep_rank = None
            return

        assert self.num_experts % self.expert_model_parallel_size == 0
        self.expert_num_per_ep_rank = (
            self.num_experts // self.expert_model_parallel_size
        )
        assert self.expert_tensor_parallel_size == 1, (
            "Only support TPE==1 for MoE resharding"
        )

    @property
    def transformer_layer_id_offset(self):
        return self.transformer_layer_num_per_stage * self.pipeline_model_parallel_rank

    @property
    def expert_id_offset(self):
        ep_ranks = self.get_ranks("ep", is_expert=True)
        assert ep_ranks is not None
        return self.expert_num_per_ep_rank * ep_ranks.index(self._rank)

    def get_tensor_model_parallel_size(self, is_expert: bool) -> int:
        if is_expert:
            return self.expert_tensor_parallel_size
        return self.tensor_model_parallel_size

    def get_tensor_model_parallel_rank(self, is_expert: bool) -> int:
        if is_expert:
            return self.expert_tensor_parallel_rank
        return self.tensor_model_parallel_rank

    def _init_group_zero(self):
        if self.num_distributed_optimizer_instances > 1:
            assert self.context_parallel_size > 1, (
                "Partial DP for Optimizer needs to include CP"
            )
            assert (
                self.data_parallel_size % self.num_distributed_optimizer_instances == 0
            )

    def _init_moe(self):
        if self.expert_tensor_parallel_size is None:
            self.expert_tensor_parallel_size = self.tensor_model_parallel_size
        self.expert_tensor_model_pipeline_parallel_size = (
            self.expert_tensor_parallel_size
            * self.expert_model_parallel_size
            * self.pipeline_model_parallel_size
        )
        self.expert_data_parallel_size = (
            self.world_size // self.expert_tensor_model_pipeline_parallel_size
        )
        if self.world_size % self.expert_tensor_model_pipeline_parallel_size != 0:
            raise RuntimeError(
                f"decoder world_size ({self.world_size}) is not divisible by expert_tensor_model_pipeline_parallel size ({self.expert_tensor_model_pipeline_parallel_size})"
            )

    def get_ranks(self, group_type, is_expert=False, **kwargs) -> Optional[list[int]]:
        for ranks in self.rank_generator.generator_wrapper(
            group_type, is_expert, **kwargs
        ):
            if self._rank in ranks:
                return ranks

    def get_data_parallel_world_size(
        self, with_context_parallel=False, partial_data_parallel=False
    ):
        """Get the data-parallel group the caller rank belongs to."""
        if with_context_parallel:
            if partial_data_parallel:
                return (
                    self.data_parallel_size
                    * self.context_parallel_size
                    // self.num_distributed_optimizer_instances
                )
            return self.data_parallel_size * self.context_parallel_size
        else:
            assert partial_data_parallel is False, (
                "Partial DP for Optimizer needs to include CP"
            )
            return self.data_parallel_size

    def register_mpu_state(self):
        if not self.is_running:
            return
        assert self.world_size == torch.distributed.get_world_size()
        self.tensor_model_parallel_rank = (
            parallel_state.get_tensor_model_parallel_rank()
        )
        self.pipeline_model_parallel_rank = (
            parallel_state.get_pipeline_model_parallel_rank()
        )
        self.data_parallel_rank = parallel_state.get_data_parallel_rank(
            with_context_parallel=False, partial_data_parallel=False
        )

        if self.num_experts:
            self.expert_model_parallel_rank = (
                parallel_state.get_expert_model_parallel_rank()
            )
            self.expert_tensor_parallel_rank = (
                parallel_state.get_expert_tensor_parallel_rank()
            )
            self.expert_data_parallel_rank = (
                parallel_state.get_expert_data_parallel_rank()
            )
        else:
            self.expert_model_parallel_rank = None
            self.expert_tensor_parallel_rank = None
            self.expert_data_parallel_rank = None

        # check
        assert (
            parallel_state.get_data_parallel_world_size()
            == self.get_data_parallel_world_size()
        )
        assert parallel_state.get_data_parallel_world_size(
            True, False
        ) == self.get_data_parallel_world_size(True, False)
        assert parallel_state.get_data_parallel_world_size(
            True, True
        ) == self.get_data_parallel_world_size(True, True)

    def check_consistency(self, args):
        keys = [
            "world_size",
            "tensor_model_parallel_size",
            "pipeline_model_parallel_size",
            "context_parallel_size",
            "sequence_parallel",
            "num_distributed_optimizer_instances",
        ]
        moe_keys = ["expert_model_parallel_size", "expert_tensor_parallel_size"]
        if args.num_experts is None or args.num_experts == 0:
            moe_keys = []
        for key in keys + moe_keys:
            assert getattr(args, key) == getattr(self, key), (
                f"args.{key} = {getattr(args, key)}, self.{key} = {getattr(self, key)}"
            )

    def update_global_args(self):
        args = get_args()
        for key in [
            "world_size",
            "tensor_model_parallel_size",
            "pipeline_model_parallel_size",
            "context_parallel_size",
            "num_distributed_optimizer_instances",
            "expert_model_parallel_size",
            "expert_tensor_parallel_size",
        ]:
            setattr(args, key, getattr(self, key))
        args.data_parallel_size = self.get_data_parallel_world_size(
            with_context_parallel=False, partial_data_parallel=False
        )

    def __str__(self):
        return f"{self.world_size=}, {self.tensor_model_parallel_size=}, {self.pipeline_model_parallel_size=}, {self.context_parallel_size=}, {self.num_distributed_optimizer_instances=}, {self.expert_model_parallel_size=}, {self.expert_tensor_parallel_size=}"

    def __hash__(self):
        return hash(str(self))

    def __eq__(self, other):
        if not isinstance(other, ParallelStrategy):
            return False
        return self.__hash__() == other.__hash__()

    @staticmethod
    def is_redundant_backup(
        src_parallel_strategy: "ParallelStrategy",
        dst_parallel_strategy: "ParallelStrategy",
    ) -> Optional[bool]:
        src_group_size = src_parallel_strategy.num_distributed_optimizer_instances
        dst_group_size = dst_parallel_strategy.num_distributed_optimizer_instances
        if src_group_size == dst_group_size:
            if src_group_size != 1:
                raise ValueError(
                    f"Resharding within a group is not supported, current group size is {src_group_size}"
                )
            return None

        assert (
            src_parallel_strategy.model_parallel_size
            == dst_parallel_strategy.model_parallel_size
        )
        assert (
            src_parallel_strategy.context_parallel_size
            == dst_parallel_strategy.context_parallel_size
        )
        assert src_parallel_strategy.world_size != dst_parallel_strategy.world_size
        assert (
            src_parallel_strategy.world_size * dst_group_size
            == dst_parallel_strategy.world_size * src_group_size
        )

        is_scale_up = src_group_size < dst_group_size
        return is_scale_up

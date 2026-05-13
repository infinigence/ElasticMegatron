from typing import Dict, List
from contextlib import contextmanager
import torch
from .mpu_state import MPUState, get_union_world_group
from .training_state import TrainingState
from .parallel_strategy import ParallelStrategy

from megatron.training.global_vars import get_args
from megatron.core.optimizer import MegatronOptimizer
from megatron.core.distributed import DistributedDataParallel as DDP
from megatron.core.optimizer_param_scheduler import OptimizerParamScheduler
from ..distributed import ElasticProcessGroup

WORLD_GROUP = None


def set_world_group(group: ElasticProcessGroup | None):
    global WORLD_GROUP
    WORLD_GROUP = group
    return WORLD_GROUP


def get_world_group() -> ElasticProcessGroup | None:
    global WORLD_GROUP
    return WORLD_GROUP


class MegatronState:
    """MegatronState is a class that contains the parallel strategy, mpu state, and training state.

    Attributes:
        parallel_strategy (ParallelStrategy) : The parallel strategy of the megatron state.
        mpu_state (MPUState) : The mpu state of the megatron state.
        training_state (TrainingState) : The training state of the megatron state.
    """

    def __init__(self, parallel_strategy: ParallelStrategy, skip_initialize_mpu: bool):
        """Init mpu state and register mpu state."""
        self.parallel_strategy = parallel_strategy
        self.parallel_strategy.update_global_args()
        self.mpu_state: MPUState = MPUState(
            world_size=parallel_strategy.world_size, ranks=parallel_strategy.ranks
        )
        args = get_args()
        set_world_group(self.mpu_state.world_group)
        self.mpu_state.initialize_model_parallel(
            skip_initialize_mpu,
            args.tensor_model_parallel_size,
            args.pipeline_model_parallel_size,
            args.virtual_pipeline_model_parallel_size,
            args.pipeline_model_parallel_split_rank,
            context_parallel_size=args.context_parallel_size,
            hierarchical_context_parallel_sizes=getattr(
                args, "hierarchical_context_parallel_sizes", None
            ),
            expert_model_parallel_size=args.expert_model_parallel_size,
            num_distributed_optimizer_instances=getattr(
                args, "num_distributed_optimizer_instances", 1
            ),
            expert_tensor_parallel_size=getattr(
                args, "expert_tensor_parallel_size", None
            ),
            distributed_timeout_minutes=args.distributed_timeout_minutes,
            nccl_communicator_config_path=args.nccl_communicator_config_path,
            order="tp-cp-ep-dp-pp"
            if not args.use_tp_pp_dp_mapping
            else "tp-cp-ep-pp-dp",
        )

        self.parallel_strategy.register_mpu_state()
        self.training_state = None

    def init_training_state(
        self,
        model: List[DDP],
        optimizer: MegatronOptimizer,
        opt_param_scheduler: OptimizerParamScheduler,
        offload_opt_tensors: bool,
    ):
        assert self.training_state is None
        self.training_state = TrainingState(model, optimizer, opt_param_scheduler)
        self.training_state.init_metadata(self.parallel_strategy, offload_opt_tensors)

    def apply(self):
        set_world_group(self.mpu_state.world_group)
        self.mpu_state.apply()
        self.parallel_strategy.update_global_args()

    @property
    def world_size(self) -> int:
        return self.parallel_strategy.world_size


class MegatronStateManager:
    def __init__(self):
        self.current_megatron_state: MegatronState = None
        self._parallel_strategy_to_megatron_state: Dict[str, MegatronState] = {}
        self._state_id_to_parallel_strategy: Dict[int, str] = {}
        self._state_id: int = -1

        self.specified_world_group: ElasticProcessGroup = None

    def init_parallel_strategy(
        self, parallel_strategy: ParallelStrategy, skip_initialize_mpu: bool
    ):
        parallel_strategy_str = str(parallel_strategy)
        assert parallel_strategy_str not in self._parallel_strategy_to_megatron_state

        self._parallel_strategy_to_megatron_state[parallel_strategy_str] = (
            MegatronState(parallel_strategy, skip_initialize_mpu)
        )
        self._state_id += 1
        self._state_id_to_parallel_strategy[self._state_id] = parallel_strategy_str
        self.current_megatron_state = self._parallel_strategy_to_megatron_state[
            parallel_strategy_str
        ]

    def init_parallel_strategy_list(self, parallel_strategy_list: List[Dict[str, int]]):
        for i, parallel_strategy_dict in enumerate(parallel_strategy_list):
            parallel_strategy = ParallelStrategy(**parallel_strategy_dict)
            if i == 0:
                parallel_strategy.check_consistency(get_args())
            self.init_parallel_strategy(parallel_strategy, skip_initialize_mpu=(i == 0))
            torch.distributed.barrier(group=torch.distributed.GroupMember.WORLD)
        self.apply(state_id=0)

    def init_training_state(
        self,
        model: List[DDP],
        optimizer: MegatronOptimizer,
        opt_param_scheduler: OptimizerParamScheduler,
        offload_opt_tensors: bool,
    ):
        self.current_megatron_state.init_training_state(
            model, optimizer, opt_param_scheduler, offload_opt_tensors
        )

    def apply(self, state_id: int = None, parallel_strategy: ParallelStrategy = None):
        if state_id is not None:
            assert state_id in self._state_id_to_parallel_strategy
            parallel_strategy_str = self._state_id_to_parallel_strategy[state_id]
        else:
            parallel_strategy_str = str(parallel_strategy)
            assert parallel_strategy_str in self._parallel_strategy_to_megatron_state

        self.current_megatron_state = self._parallel_strategy_to_megatron_state[
            parallel_strategy_str
        ]
        self.current_megatron_state.apply()

    def need_resharding(self, new_parallel_strategy: ParallelStrategy):
        assert self.current_megatron_state is not None
        assert (
            str(new_parallel_strategy) in self._parallel_strategy_to_megatron_state
        ), "Add new parallel strategy after resharding_init() is not supported"
        return new_parallel_strategy != self.current_megatron_state.parallel_strategy

    def reshard(
        self,
        new_parallel_strategy: ParallelStrategy | Dict[str, int],
        is_meta_device: bool,
    ) -> tuple[MegatronState, MegatronState, ElasticProcessGroup]:
        if not isinstance(new_parallel_strategy, ParallelStrategy):
            new_parallel_strategy = ParallelStrategy(**new_parallel_strategy)

        src_megatron_state: MegatronState = self.current_megatron_state

        if not self.need_resharding(new_parallel_strategy):
            print(
                "Warning: new_parallel_strategy is the same as the current parallel strategy\n",
                flush=True,
            )
            return src_megatron_state, None, None

        dst_megatron_state: MegatronState = self._parallel_strategy_to_megatron_state[
            str(new_parallel_strategy)
        ]
        self.apply(parallel_strategy=new_parallel_strategy)

        # Get union world group and ranks
        union_world_group, union_world_ranks = get_union_world_group(
            src_megatron_state.mpu_state, dst_megatron_state.mpu_state
        )
        rank = torch.distributed.get_rank()
        if rank not in union_world_ranks:
            assert src_megatron_state.training_state is None, (
                "Training state should be None for node not in union world group"
            )
            return None, None, None

        # Release src training state
        if src_megatron_state.training_state is not None:
            assert is_meta_device == src_megatron_state.training_state.is_meta_device, (
                f"{is_meta_device=} but {src_megatron_state.training_state.is_meta_device=}"
            )
            src_megatron_state.training_state.release_model()

        # Setup dst training state
        if (
            rank < dst_megatron_state.world_size
            and dst_megatron_state.training_state is None
        ):
            torch.cuda.empty_cache()
            dst_megatron_state.training_state = TrainingState.setup_model_and_optimizer(
                is_meta_device=is_meta_device
            )
            dst_megatron_state.training_state.init_metadata(
                dst_megatron_state.parallel_strategy, offload_opt_tensors=True
            )
            dst_megatron_state.training_state.release_model()

        return src_megatron_state, dst_megatron_state, union_world_group

    def get_data_parallel_group(
        self, parallel_strategy: ParallelStrategy | Dict[str, int]
    ) -> ElasticProcessGroup:
        if not isinstance(parallel_strategy, ParallelStrategy):
            parallel_strategy = ParallelStrategy(**parallel_strategy)
        assert isinstance(parallel_strategy, ParallelStrategy)
        parallel_strategy_str = str(parallel_strategy)
        assert parallel_strategy_str in self._parallel_strategy_to_megatron_state
        data_parallel_group = self._parallel_strategy_to_megatron_state[
            parallel_strategy_str
        ].mpu_state.data_parallel_group
        return data_parallel_group

    @property
    def megatron_state(self) -> MegatronState | None:
        return self.current_megatron_state

    @property
    def parallel_strategy(self) -> ParallelStrategy | None:
        return getattr(self.megatron_state, "parallel_strategy", None)

    @property
    def training_state(self) -> TrainingState | None:
        return getattr(self.megatron_state, "training_state", None)

    @property
    def mpu_state(self) -> MPUState | None:
        return getattr(self.megatron_state, "mpu_state", None)

    def set_specified_world_group(self, group: ElasticProcessGroup):
        self.specified_world_group = group


@contextmanager
def with_world_group(group: ElasticProcessGroup):
    assert group is not None
    origin_world_group = get_world_group()
    set_world_group(group)
    try:
        yield
    finally:
        set_world_group(origin_world_group)

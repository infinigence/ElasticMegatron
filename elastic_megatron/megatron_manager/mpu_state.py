from typing import List, Tuple
import torch
from megatron.core import parallel_state
from ..distributed import ElasticProcessGroup


class MPUState:
    _mpu_default_attrs = None

    @classmethod
    def _get_mpu_default_attrs(cls):
        if not cls._mpu_default_attrs:
            cls._mpu_default_attrs = {}

            attr_name_list = filter(
                lambda attr_name: attr_name.startswith("_")
                and not attr_name.startswith("__"),
                dir(parallel_state),
            )

            for attr_name in attr_name_list:
                value = getattr(parallel_state, attr_name)
                if callable(value):
                    continue
                cls._mpu_default_attrs[attr_name] = value

        return cls._mpu_default_attrs

    def __init__(self, world_size: int, ranks: List[int] = None):
        self._WORLD_GROUP: ElasticProcessGroup = None
        self._WORLD_RANKS: List[int] = ranks
        if self._WORLD_RANKS is None:
            self._WORLD_RANKS = list(range(world_size))
        else:
            assert len(self._WORLD_RANKS) == world_size
            assert all(rank >= 0 and rank < world_size for rank in self._WORLD_RANKS)

        # Note. Declaring _WORLD_GROUP as None before create_group is necessary, otherwise, when the torch version is higher than 2.5, it will enter a deadlock loop during create_group.
        self._WORLD_GROUP = parallel_state.create_group(ranks=self._WORLD_RANKS)
        self.is_running = torch.distributed.get_rank() in self._WORLD_RANKS

    def _clear_mpu_state(self):
        """Set default value for attr in parallel_state."""
        for attr_name, default_value in self._get_mpu_default_attrs().items():
            setattr(parallel_state, attr_name, default_value)

    def _save_mpu_state(self):
        """Save attr value in parallel_state to current MPUState."""
        self._mpu_global_vars = {}
        for attr_name in self._get_mpu_default_attrs().keys():
            self._mpu_global_vars[attr_name] = getattr(parallel_state, attr_name)

    def _apply_mpu_state(self):
        """Replace attr value in parallel_state by current MPUState."""
        for attr_name, attr_value in self._mpu_global_vars.items():
            setattr(parallel_state, attr_name, attr_value)

    def is_initialized(self):
        return self._mpu_global_vars.get("_DATA_PARALLEL_GROUP", None) is not None

    def apply(self):
        if self.is_running:
            assert self.is_initialized(), "MPUState is not initialized"
            self._apply_mpu_state()

    def initialize_model_parallel(self, skip_initialize_mpu: bool, *args, **kwargs):
        """Initialize megatron. If mpu is already initialized, skip the initialization."""
        if skip_initialize_mpu:
            assert self.is_running, (
                f"rank={torch.distributed.get_rank()}, world_ranks={self._WORLD_RANKS}, skip_initialize_mpu should only be True when all ranks are in the world group. "
            )
            self._save_mpu_state()
            return

        self._clear_mpu_state()
        parallel_state.initialize_model_parallel(*args, **kwargs)
        self._save_mpu_state()

    @property
    def world_group(self) -> ElasticProcessGroup | None:
        return self._WORLD_GROUP

    @property
    def world_size(self) -> int:
        return len(self._WORLD_RANKS)

    @property
    def world_ranks(self) -> List[int]:
        return self._WORLD_RANKS

    @property
    def data_parallel_group(self) -> ElasticProcessGroup:
        return self._mpu_global_vars["_DATA_PARALLEL_GROUP"]


def get_union_world_group(
    src_mpu_state: MPUState, dst_mpu_state: MPUState
) -> Tuple[ElasticProcessGroup, List[int]]:
    assert src_mpu_state.world_group is not None
    assert dst_mpu_state.world_group is not None

    union_world_ranks = src_mpu_state.world_ranks + dst_mpu_state.world_ranks
    union_world_ranks = list(set(union_world_ranks))
    union_world_ranks.sort()
    union_world_group = parallel_state.create_group(ranks=union_world_ranks)
    return union_world_group, union_world_ranks


def init_mpu_default_attrs():
    """Init default attr value in parallel_state.
    Note. This should be called before calling parallel_state.initialize_model_parallel().
    """
    MPUState._get_mpu_default_attrs()

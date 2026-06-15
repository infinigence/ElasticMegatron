import inspect
from dataclasses import asdict, dataclass
from functools import wraps

import torch
from torch.distributed import ProcessGroup

_ORIGIN_NEW_GROUP_FN = None


def set_origin_new_group_fn(fn):
    assert fn is not None and callable(fn)
    global _ORIGIN_NEW_GROUP_FN
    _ORIGIN_NEW_GROUP_FN = fn


def get_origin_new_group_fn():
    global _ORIGIN_NEW_GROUP_FN
    assert _ORIGIN_NEW_GROUP_FN is not None
    return _ORIGIN_NEW_GROUP_FN


@dataclass
class ElasticProcessGroupParams:
    ranks_set: frozenset = None
    backend: str = None
    init_args: tuple = None
    init_kwargs: dict = None

    def __post_init__(self):
        if isinstance(self.ranks_set, list):
            self.ranks_set = frozenset(self.ranks_set)
        if self.backend is None:
            self.backend = "nccl"
        assert self.backend in ["nccl", "gloo"], "Only support nccl and gloo backend"

    def __hash__(self) -> int:
        return hash((self.ranks_set, self.backend))

    def __eq__(self, other) -> bool:
        if not isinstance(other, ElasticProcessGroupParams):
            return False
        return hash(self) == hash(other)


class ElasticProcessGroup:
    """A wrapper of ProcessGroup that can be offloaded.

    1. Only support NCCL backend
    2. GroupMember.WORLD shouldn't be wrapped by this class
    """

    def __init__(self, params: ElasticProcessGroupParams, group: ProcessGroup = None):
        self.params = params
        self.ranks_set = params.ranks_set
        self.backend = params.backend
        self.init_args = params.init_args
        self.init_kwargs = params.init_kwargs

        self.initialized = False
        self.group = group

        if self.group is not None:
            self.initialized = True

    def create(self):
        assert not self.initialized
        self.group = get_origin_new_group_fn()(*self.init_args, **self.init_kwargs)
        self.initialized = True

    def destroy(self):
        assert self.group is not None and self.initialized
        assert self.backend is None or self.backend == "nccl"
        torch.distributed.destroy_process_group(self.group)
        self.group = None
        self.initialized = False

    def __getattr__(self, name):
        return getattr(self.group, name)


class _ElasticProcessGroupManager:
    def __init__(self):
        self._init_fn_patch()

        # Key : (ranks_set, backend)
        self._pg_cache: dict[tuple[frozenset, str], ElasticProcessGroup] = {}
        self._lazy_create_pg_cache: dict[
            tuple[frozenset, str], ElasticProcessGroup
        ] = {}

        self.world_group_initialized = False
        self.lazy_create = False

    def _init_fn_patch(self):
        set_origin_new_group_fn(torch.distributed.new_group)
        parameters = list(
            inspect.signature(torch.distributed.new_group).parameters.values()
        )
        self.ranks_index = None
        self.backend_index = None
        for index, param in enumerate(parameters):
            if param.name == "ranks":
                self.ranks_index = index
            elif param.name == "backend":
                self.backend_index = index

    def _parse_fn_args(self, *args, **kwargs) -> ElasticProcessGroupParams:
        if len(args) > self.ranks_index:
            ranks = args[self.ranks_index]
        else:
            ranks = kwargs.get("ranks", None)
        ranks = self.WORLD_RANKS if ranks is None else ranks

        if len(args) > self.backend_index:
            backend = args[self.backend_index]
        else:
            backend = kwargs.get("backend", None)
        backend = "nccl" if backend is None else backend

        return ElasticProcessGroupParams(
            ranks_set=frozenset(ranks),
            backend=backend,
            init_args=args,
            init_kwargs=kwargs,
        )

    def _init_world_group(self):
        self.world_group_initialized = True
        self.WORLD_RANKS = torch.distributed.get_process_group_ranks(
            group=torch.distributed.GroupMember.WORLD
        )

        self.WORLD = ElasticProcessGroup(
            ElasticProcessGroupParams(ranks_set=self.WORLD_RANKS, backend="nccl"),
            group=torch.distributed.GroupMember.WORLD,
        )

        WORLD_GLOO = get_origin_new_group_fn()(ranks=self.WORLD_RANKS, backend="gloo")
        self.WORLD_GLOO = ElasticProcessGroup(
            ElasticProcessGroupParams(ranks_set=self.WORLD_RANKS, backend="gloo"),
            group=WORLD_GLOO,
        )

    def set_lazy_create(self, lazy_create: bool):
        self.lazy_create = lazy_create

    @property
    def lazy_create_params(self) -> list[dict]:
        return [asdict(group.params) for group in self._lazy_create_pg_cache.values()]

    @property
    def world_size(self):
        return len(self.WORLD_RANKS)

    def process_lazy_create_tasks(
        self, lazy_create_params: list[ElasticProcessGroupParams]
    ):
        if len(lazy_create_params) == 0:
            return

        for pg_params in lazy_create_params:
            cache_key = (pg_params.ranks_set, pg_params.backend)
            group = self._lazy_create_pg_cache.get(cache_key)
            if group is None:
                group = ElasticProcessGroup(params=pg_params)
            group.create()
            self._pg_cache[cache_key] = group
        self._lazy_create_pg_cache.clear()

    def new_group(self, *args, **kwargs) -> ElasticProcessGroup:
        # Init WORLD group
        if not self.world_group_initialized:
            self._init_world_group()

        pg_params: ElasticProcessGroupParams = self._parse_fn_args(*args, **kwargs)

        if pg_params.ranks_set == frozenset(self.WORLD_RANKS):
            return self.WORLD if pg_params.backend == "nccl" else self.WORLD_GLOO

        # Find in cache
        cache_key = (pg_params.ranks_set, pg_params.backend)
        if cache_key in self._pg_cache:
            return self._pg_cache[cache_key]

        if cache_key in self._lazy_create_pg_cache and self.lazy_create:
            return self._lazy_create_pg_cache[cache_key]

        # Create new group
        elastic_group = ElasticProcessGroup(params=pg_params)
        if self.lazy_create:
            self._lazy_create_pg_cache[cache_key] = elastic_group
        else:
            self._pg_cache[cache_key] = elastic_group
            elastic_group.create()
        return elastic_group

    def nccl_group_recreate(self):
        for (ranks_set, backend), group in sorted(
            self._pg_cache.items(), key=lambda kv: (sorted(kv[0][0]), kv[0][1])
        ):
            if backend != "nccl" or len(ranks_set) <= 1:
                continue
            group.destroy()
            group.create()

    def global_barrier_by_gloo(self):
        """Perform a global barrier synchronization by gloo backend."""
        assert getattr(self, "WORLD_GLOO", None)
        torch.distributed.all_reduce(torch.empty(1), group=self.WORLD_GLOO)


ElasticProcessGroupManager = _ElasticProcessGroupManager()


def sync_group_create():
    lazy_create_params: list[dict] = ElasticProcessGroupManager.lazy_create_params
    all_lists = [None] * ElasticProcessGroupManager.world_size
    torch.distributed.all_gather_object(
        all_lists, lazy_create_params, ElasticProcessGroupManager.WORLD
    )

    total_lazy_create_params: list[ElasticProcessGroupParams] = []
    for lazy_create_params_list in all_lists:
        if not lazy_create_params_list:
            continue
        for lazy_create_param_dict in lazy_create_params_list:
            lazy_create_param = ElasticProcessGroupParams(**lazy_create_param_dict)
            if lazy_create_param not in total_lazy_create_params:
                total_lazy_create_params.append(lazy_create_param)

    ElasticProcessGroupManager.process_lazy_create_tasks(total_lazy_create_params)


def create_p2p_collective_groups():
    world_size = ElasticProcessGroupManager.world_size
    rank = torch.distributed.get_rank()

    ElasticProcessGroupManager.set_lazy_create(True)
    if rank == 0:
        for i in range(world_size):
            for j in range(i + 1, world_size):
                ElasticProcessGroupManager.new_group(ranks=[i, j], backend="nccl")
    ElasticProcessGroupManager.set_lazy_create(False)
    sync_group_create()


def get_p2p_collective_group(
    ranks: list[int], backend: str = "nccl"
) -> ElasticProcessGroup:
    assert backend == "nccl", "p2p to collective group only supports nccl backend"
    assert len(ranks) == 2, "p2p to collective group only supports 2 ranks"
    group = ElasticProcessGroupManager.new_group(ranks=ranks, backend=backend)
    assert group is not None, f"Group {ranks} {backend} is not created"
    assert group.initialized, f"Group {ranks} {backend} is not initialized"
    return group


def mpu_create_group_wrapper(fn):
    origin_new_group = torch.distributed.new_group

    @wraps(fn)
    def wrapper(*args, **kwargs):
        torch.distributed.new_group = ElasticProcessGroupManager.new_group
        res = fn(*args, **kwargs)
        torch.distributed.new_group = origin_new_group
        return res

    return wrapper


def mpu_initialize_model_parallel_wrapper(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        ElasticProcessGroupManager.set_lazy_create(True)
        is_running = torch.distributed.get_rank() < torch.distributed.get_world_size()
        if is_running:
            fn(*args, **kwargs)
        ElasticProcessGroupManager.set_lazy_create(False)
        sync_group_create()

    return wrapper


def nccl_group_recreate():
    """Release GPU memory occupied by NCCL groups"""
    ElasticProcessGroupManager.nccl_group_recreate()


def global_barrier_by_gloo():
    ElasticProcessGroupManager.global_barrier_by_gloo()


def mpu_patch():
    from megatron.core import parallel_state

    origin_mpu_create_group = parallel_state.create_group
    origin_mpu_initialize_model_parallel = parallel_state.initialize_model_parallel

    parallel_state.create_group = mpu_create_group_wrapper(origin_mpu_create_group)
    parallel_state.initialize_model_parallel = mpu_initialize_model_parallel_wrapper(
        origin_mpu_initialize_model_parallel
    )

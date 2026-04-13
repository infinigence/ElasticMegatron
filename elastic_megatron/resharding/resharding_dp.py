from typing import Any, TYPE_CHECKING
from contextlib import contextmanager
import torch
from megatron.training.global_vars import get_args
from megatron.core.distributed.param_and_grad_buffer import (
    _ParamAndGradBuffer,
    _ParamAndGradBucket,
)
from megatron.core import parallel_state

from ..megatron_manager.parallel_strategy import ParallelStrategy

from .util import Range, get_megatron_version_minor

from .resharding_pp import PipelineParallelReshardingInfo

if TYPE_CHECKING:
    from .virtual_param import VirtualParam


class ExpertParallelReshardingInfo:
    def __init__(
        self,
        expert_id: int,
        src_parallel_strategy: ParallelStrategy,
        dst_parallel_strategy: ParallelStrategy,
    ):
        self.expert_id = expert_id
        self.src_expert_parallel_rank = (
            expert_id // src_parallel_strategy.expert_num_per_ep_rank
        )
        self.dst_expert_parallel_rank = (
            expert_id // dst_parallel_strategy.expert_num_per_ep_rank
        )


class DataParallelReshardingInfo:
    """Data parallel resharding info for given virtual param.

    Convert dp_rank in dp_distribution to global_rank.

    .. note::
        Single virtual param may represent multiple parameters in the model when TP Size > 1.
        After _init_dp_distribution_with_global_rank() is called, global dp distribution is the same within the same TP Rank.
    """

    def __init__(
        self,
        src_dp_distribution: dict[int, Range],
        dst_dp_distribution: dict[int, Range],
        src_parallel_strategy: ParallelStrategy,
        dst_parallel_strategy: ParallelStrategy,
        pipeline_parallel_resharding_info: PipelineParallelReshardingInfo,
        expert_parallel_reshardign_info: ExpertParallelReshardingInfo,
    ):
        self.src_dp_distribution = src_dp_distribution
        self.dst_dp_distribution = dst_dp_distribution

        self.src_pp_rank = pipeline_parallel_resharding_info.src_pp_rank
        self.dst_pp_rank = pipeline_parallel_resharding_info.dst_pp_rank
        self.is_expert = expert_parallel_reshardign_info is not None
        self.src_ep_rank = (
            expert_parallel_reshardign_info.src_expert_parallel_rank
            if self.is_expert
            else None
        )
        self.dst_ep_rank = (
            expert_parallel_reshardign_info.dst_expert_parallel_rank
            if self.is_expert
            else None
        )

        self._init_dp_distribution_with_global_rank(
            src_parallel_strategy, dst_parallel_strategy
        )
        self._init_aligned_rank()

    def _init_dp_distribution_with_global_rank(
        self,
        src_parallel_strategy: ParallelStrategy,
        dst_parallel_strategy: ParallelStrategy,
    ) -> dict[int, Range]:
        from .resharding import get_global_rank

        def _build_dp_distribution_with_global_rank(is_src: bool) -> dict[int, Range]:
            dp_distribution_with_global_rank = {}

            parallel_strategy = (
                src_parallel_strategy if is_src else dst_parallel_strategy
            )
            dp_distribution = (
                self.src_dp_distribution if is_src else self.dst_dp_distribution
            )
            pp_rank = self.src_pp_rank if is_src else self.dst_pp_rank
            ep_rank = self.src_ep_rank if is_src else self.dst_ep_rank

            if not parallel_strategy.is_running:
                return dp_distribution_with_global_rank

            tp_rank = parallel_strategy.get_tensor_model_parallel_rank(self.is_expert)

            for dp_rank, param_range in dp_distribution.items():
                global_rank = get_global_rank(
                    pipeline_model_parallel_rank=pp_rank,
                    data_parallel_rank=dp_rank,
                    expert_model_parallel_rank=ep_rank,
                    tensor_model_parallel_rank=tp_rank,
                    parallel_strategy=parallel_strategy,
                )
                dp_distribution_with_global_rank[global_rank] = param_range
            return dp_distribution_with_global_rank

        self.src_dp_distribution_with_global_rank = (
            _build_dp_distribution_with_global_rank(True)
        )
        self.dst_dp_distribution_with_global_rank = (
            _build_dp_distribution_with_global_rank(False)
        )

    def _init_aligned_rank(self):
        self.src_aligned_dp_rank = list(self.src_dp_distribution.keys())[0]
        self.dst_aligned_dp_rank = list(self.dst_dp_distribution.keys())[0]

        self.src_aligned_global_rank = None
        self.dst_aligned_global_rank = None
        if self.src_dp_distribution_with_global_rank:
            self.src_aligned_global_rank = list(
                self.src_dp_distribution_with_global_rank.keys()
            )[0]
        if self.dst_dp_distribution_with_global_rank:
            self.dst_aligned_global_rank = list(
                self.dst_dp_distribution_with_global_rank.keys()
            )[0]


def intersect_range(main_range: Range, sub_range: Range) -> Range:
    intersect_start = max(main_range.start, sub_range.start)
    intersect_end = min(main_range.end, sub_range.end)
    if intersect_start < intersect_end:
        return Range(intersect_start, intersect_end)
    return None


def intersect_sub_ranges(
    main_range: Range, sub_ranges_dict: dict[Any, Range]
) -> dict[Any, Range]:
    sub_ranges_in_main_range = {}
    for key, sub_range in sub_ranges_dict.items():
        intersect_sub_range = intersect_range(main_range, sub_range)
        if intersect_sub_range is not None:
            sub_ranges_in_main_range[key] = intersect_sub_range
    return sub_ranges_in_main_range


@contextmanager
def mock_ddp_buffer_init(data_parallel_size: int):
    origin_get_world_size = torch.distributed.get_world_size
    torch.distributed.get_world_size = lambda *args, **kwargs: data_parallel_size

    origin_torch_zeros = torch.zeros

    class _FakeTensor:
        def __init__(
            self,
            shape,
            dtype=None,
            device=None,
            requires_grad: bool = False,
        ):
            self.shape = shape
            if hasattr(shape, "numel"):
                self._numel = int(shape.numel())
            elif hasattr(shape, "size"):
                self._numel = int(getattr(shape, "size"))
            else:
                self._numel = int(shape)
            self.dtype = dtype
            self.device = device
            self.requires_grad = requires_grad
            self._base = None

        def nelement(self) -> int:
            return self._numel

        def numel(self) -> int:
            return self._numel

        def detach(self):
            return self

        def copy_(self, src):
            return self

    def mock_torch_zeros(size, *args, **kwargs):
        return _FakeTensor(
            size,
            dtype=kwargs.get("dtype"),
            device=kwargs.get("device"),
            requires_grad=kwargs.get("requires_grad", False),
        )

    torch.zeros = mock_torch_zeros

    origin_ParamAndGradBuffer_get = _ParamAndGradBuffer._get
    # _ParamAndGradBuffer._get = lambda *args, **kwargs: None

    def _mock_ParamAndGradBuffer_get(self, shape, start_index, buffer_type):
        return _FakeTensor(shape, dtype=None, device=None, requires_grad=False)

    _ParamAndGradBuffer._get = _mock_ParamAndGradBuffer_get

    origin_get_data_parallel_rank = parallel_state.get_data_parallel_rank
    parallel_state.get_data_parallel_rank = lambda *args, **kwargs: -1

    try:
        yield
    finally:
        torch.distributed.get_world_size = origin_get_world_size
        torch.zeros = origin_torch_zeros
        _ParamAndGradBuffer._get = origin_ParamAndGradBuffer_get
        parallel_state.get_data_parallel_rank = origin_get_data_parallel_rank


def get_ddp_buffer_distribution(
    params: list["VirtualParam"], data_parallel_size: int, ddp_config, bucket_size
) -> dict["VirtualParam", dict[int, Range]]:
    if not params:
        return {}

    original_param_data = {param: getattr(param, "data", None) for param in params}

    class _MockProcessGroup:
        def __init__(self, size: int):
            self._size = size

        def size(self) -> int:
            return self._size

    kwargs = {
        "ddp_config": ddp_config,
        "param_dtype": "param_dtype",
        "grad_dtype": "grad_dtype",
        "params": params,
        "data_parallel_group": _MockProcessGroup(data_parallel_size),
        "bucket_size": bucket_size,
        "param_to_name": {param: param.name for param in params},
        "gradient_scaling_factor": 1.0,
        "param_indices": [0],
    }

    if get_megatron_version_minor() >= 13:
        kwargs["nccl_ub"] = False

    with mock_ddp_buffer_init(data_parallel_size):
        ddp_buffer = _ParamAndGradBuffer(**kwargs)
    for param, data in original_param_data.items():
        param.data = data

    param_index_map = ddp_buffer.param_index_map
    bucket_indices = ddp_buffer.bucket_indices

    def get_bucket_params_distribution(
        bucket: _ParamAndGradBucket,
    ) -> list[dict["VirtualParam", Range]]:
        """
        返回每个dp上, 若某个param的opt tensor落在该dp上时, opt tensor在bucket中的range
        """
        bucket_start_index, bucket_end_index = bucket_indices[bucket.bucket_id]
        bucket_numel = bucket_end_index - bucket_start_index
        bucket_params: list["VirtualParam"] = bucket.params_list

        shard_size = bucket_numel // data_parallel_size

        # Conver param_index(in buffer) into param_index(in bucket)
        param_index_in_bucket_map: dict["VirtualParam", Range] = {}
        for param in bucket_params:
            param_start_in_buffer, param_end_in_buffer, _ = param_index_map[param]

            param_index_in_bucket_map[param] = Range(
                param_start_in_buffer, param_end_in_buffer
            ).normalize(bucket_start_index)

        # Split bucket by dp size
        dp_to_params_distribution: list[dict["VirtualParam", Range]] = [
            None for i in range(data_parallel_size)
        ]
        for dp_rank in range(data_parallel_size):
            dp_bucket_range = Range(dp_rank * shard_size, (dp_rank + 1) * shard_size)

            dp_to_params_distribution[dp_rank] = intersect_sub_ranges(
                dp_bucket_range, param_index_in_bucket_map
            )

            # Set param_start(in bucket) as global_offset
            for param, param_range in dp_to_params_distribution[dp_rank].items():
                param_global_offset_in_buffer = param_index_in_bucket_map[param].start
                setattr(param_range, "global_offset", param_global_offset_in_buffer)
        return dp_to_params_distribution

    def get_buckets_params_distribution() -> list[dict["VirtualParam", Range]]:
        """
        返回每个dp上, 
        buffer中所有bucket的opt tensor的range
        range.global_offset = param_global_offset_in_buffer
        """
        dp_to_params_distribution: list[dict["VirtualParam", Range]] = [
            {} for i in range(data_parallel_size)
        ]
        for bucket in ddp_buffer.buckets:
            dp_to_params_distribution_in_bucket: list[dict["VirtualParam", Range]] = (
                get_bucket_params_distribution(bucket)
            )
            assert len(dp_to_params_distribution_in_bucket) == data_parallel_size

            for dp_rank in range(data_parallel_size):
                dp_to_params_distribution[dp_rank].update(
                    dp_to_params_distribution_in_bucket[dp_rank]
                )
        return dp_to_params_distribution

    dp_to_params_distribution = get_buckets_params_distribution()

    return transpose_and_sort_dp_distribution(dp_to_params_distribution, params)


def get_params_dp_distribution(
    params: list["VirtualParam"],
    parallel_strategy: ParallelStrategy,
    ddp_config,
    disable_bucketing: bool = False,
) -> dict["VirtualParam", dict[int, Range]]:
    args = get_args()

    data_parallel_size = parallel_strategy.get_data_parallel_world_size(
        with_context_parallel=True, partial_data_parallel=True
    )
    expert_data_parallel_size = parallel_strategy.expert_data_parallel_size
    expert_num_per_ep_rank = parallel_strategy.expert_num_per_ep_rank

    bucket_size = args.ddp_bucket_size
    if bucket_size is None:
        bucket_size = max(40000000, 1000000 * data_parallel_size)
    if not args.overlap_grad_reduce or disable_bucketing:
        bucket_size = None

    dense_params: list[VirtualParam] = []
    moe_params_dict: dict[int, list[VirtualParam]] = {}  # key = ep_rank
    for param in params:
        if param.is_expert:
            expert_id = param.param_position_attr.expert_id
            expert_parallel_rank = expert_id // expert_num_per_ep_rank
            moe_params_dict.setdefault(expert_parallel_rank, []).append(param)
        else:
            dense_params.append(param)

    dense_params_dp_distribution: dict["VirtualParam", dict[int, Range]] = (
        get_ddp_buffer_distribution(
            dense_params, data_parallel_size, ddp_config, bucket_size
        )
    )

    if moe_params_dict:
        # Get param dp distribution in EP-Rank0
        moe_params_in_ep_rank0 = moe_params_dict[0]
        moe_params_dp_distribution: dict["VirtualParam", dict[int, Range]] = (
            get_ddp_buffer_distribution(
                moe_params_in_ep_rank0,
                expert_data_parallel_size,
                ddp_config,
                bucket_size,
            )
        )
        assert len(moe_params_in_ep_rank0) == len(moe_params_dp_distribution)
        # Expand dp distribution for all EP-Rank
        for ep_rank, moe_params in moe_params_dict.items():
            if ep_rank == 0:
                continue
            for moe_param, moe_param_in_ep_rank0 in zip(
                moe_params, moe_params_in_ep_rank0
            ):
                moe_params_dp_distribution[moe_param] = moe_params_dp_distribution[
                    moe_param_in_ep_rank0
                ]
    else:
        moe_params_dp_distribution = {}

    assert len(dense_params_dp_distribution) + len(moe_params_dp_distribution) == len(
        params
    ), (
        f"{len(dense_params_dp_distribution)=} + {len(moe_params_dp_distribution)=} != {len(params)=}"
    )

    dense_params_dp_distribution.update(moe_params_dp_distribution)
    return dense_params_dp_distribution


def transpose_and_sort_dp_distribution(
    dp_to_params_distribution: list[dict["VirtualParam", Range]],
    params: list["VirtualParam"],
) -> dict["VirtualParam", dict[int, Range]]:
    params_dp_distribution: dict["VirtualParam", dict[int, Range]] = {
        param: {} for param in params
    }
    for dp_rank, params_distribution in enumerate(dp_to_params_distribution):
        assert len(params_distribution) > 0
        for param, param_range in params_distribution.items():
            param_range = param_range.normalize(param_range.global_offset)
            params_dp_distribution[param][dp_rank] = param_range

    for param in params:
        dp_distribution: dict[int, Range] = params_dp_distribution[param]
        params_dp_distribution[param] = {
            key: value
            for key, value in sorted(dp_distribution.items(), key=lambda x: x[0])
        }

    return params_dp_distribution

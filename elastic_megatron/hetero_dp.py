from collections.abc import Callable
from contextlib import contextmanager
from copy import deepcopy
from functools import wraps

import torch
from megatron.core import num_microbatches_calculator, parallel_state
from megatron.core.num_microbatches_calculator import ConstantNumMicroBatchesCalculator
from megatron.core.pipeline_parallel import schedules

from .resharding.util import get_megatron_version_minor


@contextmanager
def disable_data_parallel_sync():
    """Disable data parallel sync.

    1. This context should be used with `model.config.no_sync_func` in forward-backward pass.
    2. Only support `forward_backward_no_pipelining` schedule now.
    """
    origin_get_data_parallel_group = parallel_state.get_data_parallel_group
    global_rank = torch.distributed.get_rank()
    self_group = parallel_state.create_group(ranks=[global_rank])

    def _get_data_parallel_group(
        with_context_parallel=False, partial_data_parallel=False
    ):
        if with_context_parallel is False and partial_data_parallel is False:
            return self_group
        return origin_get_data_parallel_group(
            with_context_parallel, partial_data_parallel
        )

    parallel_state.get_data_parallel_group = _get_data_parallel_group
    try:
        yield
    finally:
        parallel_state.get_data_parallel_group = origin_get_data_parallel_group


def wrap_forward_step(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        with disable_data_parallel_sync():
            return fn(*args, **kwargs)

    return wrapper


def wrap_forward_backward_func(
    forward_backward_fn: Callable, _process_fwd_bwd_outputs: Callable
):
    @wraps(forward_backward_fn)
    def wrapper(*args, **kwargs):
        outputs = forward_backward_fn(*args, **kwargs)
        if parallel_state.is_pipeline_last_stage():
            return _process_fwd_bwd_outputs(outputs)
        return outputs

    return wrapper


def process_fwd_bwd_outputs_for_pretrain_gpt(forward_data_store: list):
    assert isinstance(forward_data_store, list), (
        f"forward_backward_func should return a list, but got {type(forward_data_store)}"
    )

    forward_data_store_len = len(forward_data_store)
    assert forward_data_store_len > 0, (
        f"forward_backward_func should return a non-empty list, but got {forward_data_store_len}"
    )

    max_forward_data_store_len = torch.tensor(
        forward_data_store_len, dtype=torch.int
    ).cuda()

    torch.distributed.all_reduce(
        max_forward_data_store_len,
        group=parallel_state.get_data_parallel_group(),
        op=torch.distributed.ReduceOp.MAX,
    )

    default_forward_data = {
        "lm loss": (
            torch.tensor(0, dtype=torch.float32).cuda(),
            torch.tensor(0, dtype=torch.int).cuda(),
        )
    }

    padded_len = max_forward_data_store_len - forward_data_store_len
    assert padded_len >= 0, (
        f"max_forward_data_store_len={max_forward_data_store_len} < len(forward_data_store)={forward_data_store_len}"
    )
    forward_data_store += [deepcopy(default_forward_data) for _ in range(padded_len)]

    megatron_version_minor = get_megatron_version_minor()
    for i in range(len(forward_data_store)):
        loss_reduced = forward_data_store[i]["lm loss"]

        if megatron_version_minor <= 11:
            loss_reduced_tensor = torch.tensor(
                loss_reduced, dtype=loss_reduced[0].dtype
            ).cuda()
        elif megatron_version_minor == 13:
            loss_reduced_tensor = loss_reduced.view(-1).cuda()
        elif megatron_version_minor >= 16:
            # 0.16+ 的 forward_data_store / loss reduction 形状在 PR sweep 中
            # 未被覆盖;ElasticMegatron 当前没有任何调用方走这条路径(apply_hetero_dp
            # 在 Phase B 全部实验里都没被调用)。如果将来要支持,需要重新对 0.16+
            # 下 loss_reduced 的实际类型确认后再加分支。
            raise NotImplementedError(
                f"hetero_dp not yet adapted to Megatron {megatron_version_minor}; "
                f"see hetero_dp.py docstring."
            )
        else:
            raise ValueError(
                f"Unsupported Megatron minor version: {megatron_version_minor}"
            )

        torch.distributed.all_reduce(
            loss_reduced_tensor, group=parallel_state.get_data_parallel_group()
        )
        if megatron_version_minor <= 11:
            forward_data_store[i] = {
                "lm loss": (loss_reduced_tensor[0], loss_reduced_tensor[1])
            }
        elif megatron_version_minor == 13:
            forward_data_store[i] = {"lm loss": loss_reduced_tensor.view(-1)}
        elif megatron_version_minor >= 16:
            # 0.16+ 的 forward_data_store / loss reduction 形状在 PR sweep 中
            # 未被覆盖;ElasticMegatron 当前没有任何调用方走这条路径(apply_hetero_dp
            # 在 Phase B 全部实验里都没被调用)。如果将来要支持,需要重新对 0.16+
            # 下 loss_reduced 的实际类型确认后再加分支。
            raise NotImplementedError(
                f"hetero_dp not yet adapted to Megatron {megatron_version_minor}; "
                f"see hetero_dp.py docstring."
            )
        else:
            raise ValueError(
                f"Unsupported Megatron minor version: {megatron_version_minor}"
            )
    return forward_data_store


class ConstantNumMicroBatchesCalculatorForHeteroDP(ConstantNumMicroBatchesCalculator):
    """Patch of megatron.core.num_microbatches_calculator.ConstantNumMicroBatchesCalculator.

    When the global batch size is not divisible by the micro batch size * data parallel size, set different num_microbatches for each data parallel group.
    """

    _alignment: int = 1

    @classmethod
    def set_alignment(cls, alignment: int):
        cls._alignment = alignment

    def __init__(
        self,
        global_batch_size: int,
        micro_batch_size: int,
        data_parallel_size: int,
        decrease_batch_size_if_needed: bool,
        rank: int,
    ) -> None:
        if global_batch_size % (micro_batch_size * data_parallel_size) == 0:
            super().__init__(
                global_batch_size,
                micro_batch_size,
                data_parallel_size,
                decrease_batch_size_if_needed,
                rank,
            )
            return

        assert global_batch_size % (micro_batch_size * self._alignment) == 0

        num_microbatches_with_alignment_sum = (
            global_batch_size // micro_batch_size // self._alignment
        )
        assert num_microbatches_with_alignment_sum >= data_parallel_size
        num_microbatches_with_alignment_per_dp = (
            num_microbatches_with_alignment_sum // data_parallel_size
        )

        if (
            parallel_state.get_data_parallel_rank()
            < num_microbatches_with_alignment_sum % data_parallel_size
        ):
            num_microbatches_with_alignment_per_dp += 1

        hetero_num_microbatches = (
            num_microbatches_with_alignment_per_dp * self._alignment
        )

        hetero_global_batch_size = (
            hetero_num_microbatches * micro_batch_size * data_parallel_size
        )
        super().__init__(
            hetero_global_batch_size,
            micro_batch_size,
            data_parallel_size,
            decrease_batch_size_if_needed,
            rank,
        )

    def update(self, consumed_samples, consistency_check, verbose=False) -> None:
        pass


def apply_hetero_dp(process_fwd_bwd_outputs: Callable = None, alignment: int = 1):
    """Apply hetero-dp for megatron training.

    Args:
        process_fwd_bwd_outputs (Callable): the func to execute dp-group sync after foward_backward_func is done. It accepts only one argument, which is the list of forward_backward_func outputs.
        alignment (int): the alignment of each batch size.

    Example:
        >>> apply_hetero_dp(process_fwd_bwd_outputs)
        def process_fwd_bwd_outputs(forward_backward_outputs):

            # Step-1 : get the max length of forward_backward_outputs across all data parallel groups.

            forward_data_store_len = len(forward_data_store)
            max_forward_data_store_len = torch.tensor(forward_data_store_len).cuda()
            torch.distributed.all_reduce(
                max_forward_data_store_len,
                group=parallel_state.get_data_parallel_group(),
                op=torch.distributed.ReduceOp.MAX,
            )

            # Step-2 : padding the forward_backward_outputs to the max length.

            default_forward_data = {
                "lm loss" : torch.tensor(0.0).cuda()
            }
            forward_data_store += [deepcopy(default_forward_data) for _ in range(int(max_forward_data_store_len.item()) - forward_data_store_len)]

            # Step-3 : execute dp-group sync for each forward_backward_output.

            for forward_backward_output in forward_backward_outputs:
                torch.distributed.all_reduce(
                    forward_backward_output['lm loss'], group=parallel_state.get_data_parallel_group()
                )

            return forward_backward_outputs
    """
    # Patch num_microbatches_calculator
    ConstantNumMicroBatchesCalculatorForHeteroDP.set_alignment(alignment)
    num_microbatches_calculator.ConstantNumMicroBatchesCalculator = (
        ConstantNumMicroBatchesCalculatorForHeteroDP
    )

    # Disable data-parallel-group sync in loss_func.
    origin_forward_step = schedules.forward_step
    schedules.forward_step = wrap_forward_step(origin_forward_step)

    # Process for forward_backward_func outputs.
    if process_fwd_bwd_outputs is None:
        process_fwd_bwd_outputs = process_fwd_bwd_outputs_for_pretrain_gpt

    origin_forward_backward_no_pipelining = schedules.forward_backward_no_pipelining
    schedules.forward_backward_no_pipelining = wrap_forward_backward_func(
        origin_forward_backward_no_pipelining, process_fwd_bwd_outputs
    )

    # Patch forward_backward_pipelining_without_interleaving: process outputs on last PP stage
    origin_forward_backward_pipelining_without_interleaving = (
        schedules.forward_backward_pipelining_without_interleaving
    )
    schedules.forward_backward_pipelining_without_interleaving = (
        wrap_forward_backward_func(
            origin_forward_backward_pipelining_without_interleaving,
            process_fwd_bwd_outputs,
        )
    )

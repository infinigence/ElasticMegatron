import inspect
from functools import wraps

import torch

from .elastic_process_group import ElasticProcessGroup
from .util import get_all_dist_functions_with_group


def dist_wrapper(fn, patch_group_arg: bool = True):
    """Wrapper for torch.distributed functions that accept group parameter

    1. If not group provided, use current global group.
    2. Converts ElasticProcessGroup to ProcessGroup.
    """
    parameters = list(inspect.signature(fn).parameters.values())
    group_index = None
    for index, param in enumerate(parameters):
        if param.name == "group":
            group_index = index
            break
    assert group_index is not None, (
        f"Function {fn.__name__} does not have a group parameter"
    )

    @wraps(fn)
    def wrapper(*args, **kwargs):
        # If grpup is provided and is not None, return origin func call
        if fn.__name__ == "isend" and "group_src" in kwargs:
            kwargs["group_dst"] = kwargs.pop("group_src")
        if len(args) > group_index:
            args = list(args)
            if isinstance(args[group_index], ElasticProcessGroup):
                args[group_index] = args[group_index].group
            return fn(*args, **kwargs)

        if kwargs.get("group") is not None or not patch_group_arg:
            if isinstance(kwargs.get("group"), ElasticProcessGroup):
                kwargs["group"] = kwargs["group"].group
            return fn(*args, **kwargs)

        # Use world_group from megatron_state
        # Note. It is necessary to execute patch_torch_distributed() before initializing megatron, so import megatron_manager.megatron_state when get_world_group is called for the first time.
        if not hasattr(dist_wrapper, "get_world_group"):
            from ..megatron_manager.megatron_state import get_world_group

            dist_wrapper.get_world_group = get_world_group
        group = dist_wrapper.get_world_group()

        # Unwrap ElasticProcessGroup
        if isinstance(group, ElasticProcessGroup):
            group = group.group
        kwargs["group"] = group
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            print(
                f"Warning: Failed to patch {fn.__name__}, args: {args}, kwargs: {kwargs}, error: {e}"
            )
            raise e

    return wrapper


def patch_torch_distributed_c10():
    """Patch func or class in torch.distributed.distributed_c10."""

    # Original isend/irecv is wrapped by dist_wrapper
    def _check_op(op) -> None:
        """Check that the ``op`` is either isend or irecv."""
        if op not in [torch.distributed.isend, torch.distributed.irecv]:
            raise ValueError(
                "Invalid ``op``. Expected ``op`` "
                "to be of type ``torch.distributed.isend`` or "
                "``torch.distributed.irecv``."
            )

    torch.distributed.distributed_c10d._check_op = _check_op


def patch_torch_distributed():
    """Patch all torch.distributed functions that accept group parameter.

    1. If not group provided, use current global group.
    2. Converts ElasticProcessGroup to ProcessGroup.
    """
    excluded_functions = ["get_rank"]
    functions_to_patch = get_all_dist_functions_with_group()

    for module_name, func_name, original_func in functions_to_patch:
        module = torch.distributed
        wrapped_func = dist_wrapper(
            getattr(module, func_name), func_name not in excluded_functions
        )
        setattr(module, func_name, wrapped_func)

    # Patcher for torch.distributed.distributed_c10
    patch_torch_distributed_c10()

"""Redirect Megatron's distributed-checkpoint save collective to the current
elastic world group on a scale-down (world-shrink) reshard.

Why this exists
---------------
Megatron's ``TorchDistSaveShardedStrategy.async_save`` calls
``save_state_dict_async_plan(state_dict, writer, process_group=None, coordinator, ...)``
with ``process_group=None``; PyTorch DCP's ``_DistWrapper`` then defaults to
``torch.distributed.group.WORLD`` (the full *physical* world). Megatron itself
flags this in-source: ``# This should be set differently if we run in a smaller
process group than the default``.

After a scale-down reshard (e.g. world 8->4) ElasticMegatron has swapped the
*logical* parallel-state world to the destination ranks, but the *physical* torch
WORLD still spans every rank. Only the destination ranks hold state and call save;
the scaled-out ranks contribute a ``None`` local SavePlan, and DCP's
``dedup_save_plans`` crashes with ``AttributeError: 'NoneType' object has no
attribute 'items'`` (the after_reshard save only; the before-reshard save is at the
full world and is unaffected). See the verification-discipline note in
docs/cpu_adam_reshard_optimization/baseline.md.

EM already redirects every ``group=None`` ``torch.distributed.*`` call to the
current world group via ``dist_patch``, but DCP's ``_DistWrapper`` holds an
explicit ``group.WORLD`` object, so that redirection cannot reach it. This module
closes that one gap: it wraps Megatron's ``save_state_dict_async_plan`` binding and,
when ``process_group`` is ``None`` AND the elastic world is a strict subset of the
physical world (i.e. a scale-down), fills in the current elastic world group.
Full-world saves (the common case, and every symmetric/scale-up reshard) are left
untouched -- behaviour is byte-identical to upstream there.

``coordinator_rank`` is left at Megatron's default of 0: EM destination ranks are
always ``range(dst_world_size)``, so global rank 0 is always a member of the
shrunk group and a valid coordinator.

Top-level imports are kept torch-free (torch / megatron are imported lazily inside
the functions that need them) so the pure argument-rewriting helpers can be
unit-tested without torch -- see tests/test_dist_ckpt_patch.py.
"""

from functools import wraps

# Megatron module whose `save_state_dict_async_plan` *binding* is invoked at the
# crash site (megatron/core/dist_checkpointing/strategies/torch.py).
_MEGATRON_TORCH_STRATEGY_MODULE = "megatron.core.dist_checkpointing.strategies.torch"

# process_group is the 3rd positional parameter of save_state_dict_async_plan:
#   save_state_dict_async_plan(state_dict, storage_writer, process_group=None,
#                              coordinator_rank=0, planner=None, ...)
# Megatron passes it positionally (as None).
_PROCESS_GROUP_POS = 2

_installed = False


def _current_process_group(args, kwargs):
    """Return the process_group argument as Megatron passes it (None if absent)."""
    if "process_group" in kwargs:
        return kwargs["process_group"]
    if len(args) > _PROCESS_GROUP_POS:
        return args[_PROCESS_GROUP_POS]
    return None


def _inject_process_group(args, kwargs, override):
    """Return (args, kwargs) with the process_group argument set to ``override``.

    Pure / torch-free so it can be unit-tested. Handles all three call shapes:
    process_group passed positionally (Megatron's shape), passed by keyword, or
    omitted entirely.
    """
    if "process_group" in kwargs:
        kwargs = dict(kwargs)
        kwargs["process_group"] = override
        return args, kwargs
    args = tuple(args)
    if len(args) > _PROCESS_GROUP_POS:
        args = args[:_PROCESS_GROUP_POS] + (override,) + args[_PROCESS_GROUP_POS + 1 :]
        return args, kwargs
    kwargs = dict(kwargs)
    kwargs["process_group"] = override
    return args, kwargs


def _shrunk_world_process_group():
    """torch ProcessGroup of the current elastic world iff it is a strict subset
    of the physical world (a scale-down); otherwise None (leave save on WORLD)."""
    from ..megatron_manager.megatron_state import get_world_group
    from .elastic_process_group import ElasticProcessGroupManager

    wg = get_world_group()
    # No elastic world set yet, or no group created yet -> nothing to redirect.
    if wg is None or not ElasticProcessGroupManager.world_group_initialized:
        return None
    # Compare against ElasticProcessGroupManager.world_size (the *physical* world);
    # torch.distributed.get_world_size() is patched by EM to return the logical one,
    # so it must not be used here.
    if len(wg.ranks_set) < ElasticProcessGroupManager.world_size:
        import torch

        if torch.distributed.get_rank() == 0:
            print(
                f"[ElasticDCP] scale-down save: redirecting the DCP global-plan "
                f"collective to the {len(wg.ranks_set)}-rank elastic world group "
                f"(physical world={ElasticProcessGroupManager.world_size})",
                flush=True,
            )
        return wg.group
    return None


def _wrap_save_state_dict_async_plan(orig):
    @wraps(orig)
    def wrapper(*args, **kwargs):
        if _current_process_group(args, kwargs) is None:
            override = _shrunk_world_process_group()
            if override is not None:
                args, kwargs = _inject_process_group(args, kwargs, override)
        return orig(*args, **kwargs)

    setattr(wrapper, "_elastic_dcp_patched", True)
    return wrapper


def ensure_dist_ckpt_save_patched():
    """Idempotently wrap Megatron's ``save_state_dict_async_plan`` binding so the
    DCP global-plan collective runs over the elastic world on a scale-down save.

    Called lazily from TrainingState.save_checkpoint, which only runs inside a live
    Megatron training process -- so the strategy module is importable here.
    Idempotent (the wrapped binding is tagged and skipped on re-entry).
    """
    global _installed
    if _installed:
        return
    import importlib

    mod = importlib.import_module(_MEGATRON_TORCH_STRATEGY_MODULE)
    orig = mod.save_state_dict_async_plan
    if not getattr(orig, "_elastic_dcp_patched", False):
        setattr(mod, "save_state_dict_async_plan", _wrap_save_state_dict_async_plan(orig))
    _installed = True

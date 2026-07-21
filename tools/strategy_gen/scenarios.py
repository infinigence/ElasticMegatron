# tools/strategy_gen/scenarios.py
"""Reshard-sequence rules: (model, hardware) -> (base layout, override-dict list).

Each scenario returns the base full-world layout plus a list of override-dicts (relative
to base) whose first element is {} (the strategy_inject convention: strategy[0] == launch
config). Registered in SCENARIOS by name; add a scenario = add a function + register it.

The elastic loop cycles the list: at reshard step i it applies strategies[i % len] (see
check_reshard in training_016.py), so the sequence RETURNS TO BASE automatically after the
last entry — a scenario must NOT append a trailing {} to "return to base". It must also not
emit any strategy twice: the manager registers each list entry into a uniqueness-checked
dict (megatron_state.init_parallel_strategy asserts the strategy is not already present),
so a duplicate aborts initialization. `_no_duplicates` enforces this at generation time.
"""
from __future__ import annotations

from dataclasses import dataclass

from .planner import LayoutPlanner
from .registry import ModelArch, RunShape


@dataclass(frozen=True)
class Hardware:
    gpus: int
    gpu_mem_gb: float


def _no_duplicates(overrides: list[dict]) -> list[dict]:
    """Guard: the elastic loop cycles strategies mod len and the manager rejects a repeated
    strategy, so a generated sequence must contain no duplicate override-dict (in particular
    no return-to-base {} after strategy[0])."""
    seen = set()
    for o in overrides:
        key = tuple(sorted(o.items()))
        assert key not in seen, (
            f"scenario emitted a duplicate strategy {o}; the elastic loop already cycles "
            f"strategies[i % len] back to earlier entries and the manager asserts each strategy "
            f"is unique — do not emit a return-to-base or any repeated entry")
        seen.add(key)
    return overrides


def rl_dp_resize(arch: ModelArch, hw: Hardware, planner: LayoutPlanner, rs: RunShape):
    """Train at full world; shrink world (~half) for rollout. The loop cycles back to base
    automatically, so the sequence is just [base, rollout] (NO trailing return-to-base)."""
    base = planner.best(arch, hw.gpus, rs, "auto")
    # Resolve the rollout with "auto" so it stays feasible at the reduced world (a
    # GPU-adam-at-full model may only fit cpu-adam at half world). But a reshard cannot flip
    # CPU_OFFLOAD mid-run — the override carries no optimizer-placement flag and the recipe
    # keys CPU_OFFLOAD off `base` only — so assert they agree rather than silently run the
    # rollout under the wrong optimizer placement.
    rollout = planner.best(arch, max(hw.gpus // 2, 1), rs, "auto")
    assert base.cpu_adam == rollout.cpu_adam, (
        f"rl_dp_resize: base cpu_adam={base.cpu_adam} != rollout cpu_adam={rollout.cpu_adam}; "
        f"a reshard cannot flip CPU_OFFLOAD mid-run")
    return base, _no_duplicates([{}, rollout.override(base)])


def verify_sweep(arch: ModelArch, hw: Hardware, planner: LayoutPlanner, rs: RunShape, max_steps: int = 6):
    """Cycle distinct feasible layouts at the fixed launch world (correctness coverage)."""
    base = planner.best(arch, hw.gpus, rs, "auto")
    cand = planner.feasible(arch, hw.gpus, rs, base.cpu_adam)
    overrides = [{}]
    for layout in cand:
        ov = layout.override(base)
        if ov and ov not in overrides:
            overrides.append(ov)
        if len(overrides) >= max_steps:
            break
    return base, _no_duplicates(overrides)


SCENARIOS = {
    "rl_dp_resize": rl_dp_resize,
    "verify_sweep": verify_sweep,
}

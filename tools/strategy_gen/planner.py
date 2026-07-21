# tools/strategy_gen/planner.py
"""Enumerate, filter (by memory), and rank feasible parallel layouts (torch-free)."""
from __future__ import annotations

from dataclasses import dataclass

from .registry import ModelArch, RunShape


def _divisors(n: int) -> list[int]:
    return [d for d in range(1, n + 1) if n % d == 0]


@dataclass(frozen=True)
class Layout:
    world: int
    tp: int
    pp: int
    cp: int
    ep: int
    dp: int
    cpu_adam: bool
    moe: bool = False

    def override(self, base: "Layout") -> dict:
        """Override-dict vs a base layout: only the dims that differ (strategy_inject form).

        For a MoE layout (``self.moe``) with a non-empty diff we ALSO pin
        ``expert_tensor_parallel_size: 1`` explicitly — the runtime
        (parallel_strategy.py) only supports ETP==1 for MoE resharding, and the user
        directive forbids relying on the implicit ``{**base}`` inheritance of the launch
        ETP. An empty diff (``self == base``) returns ``{}`` so ``overrides[0] == {}`` and
        the return-to-base convention are preserved (base's ETP=1 comes from the recipe's
        ``TPE=1``).
        """
        keys = {
            "world_size": self.world,
            "tensor_model_parallel_size": self.tp,
            "pipeline_model_parallel_size": self.pp,
            "context_parallel_size": self.cp,
            "expert_model_parallel_size": self.ep,
        }
        base_keys = {
            "world_size": base.world,
            "tensor_model_parallel_size": base.tp,
            "pipeline_model_parallel_size": base.pp,
            "context_parallel_size": base.cp,
            "expert_model_parallel_size": base.ep,
        }
        diff = {k: v for k, v in keys.items() if v != base_keys[k]}
        if diff and self.moe:
            diff["expert_tensor_parallel_size"] = 1
        return diff


class LayoutPlanner:
    def __init__(self, memory_model, gpu_mem_gb: float):
        self.mem = memory_model
        self.cap = gpu_mem_gb

    def _enumerate(self, arch: ModelArch, world: int, cpu_adam: bool):
        eps = _divisors(arch.num_experts) if arch.moe else [1]
        for tp in _divisors(world):
            for pp in _divisors(world // tp):
                for cp in _divisors(world // (tp * pp)):
                    dp = world // (tp * pp * cp)
                    for ep in eps:
                        if ep > world or (arch.moe and ep > arch.num_experts):
                            continue
                        # MoE runtime (parallel_strategy.py) pins ETP==1, so the expert
                        # region is ETP*EP*PP = EP*PP and must divide the world, else
                        # _init_moe raises. Attention TP is independent (ETP=1 != TP).
                        if arch.moe and world % (ep * pp) != 0:
                            continue
                        layout = Layout(world, tp, pp, cp, ep, dp, cpu_adam, moe=arch.moe)
                        if arch.moe:
                            assert layout.world % (layout.ep * layout.pp) == 0, (
                                f"planner emitted illegal MoE layout: world={layout.world} "
                                f"not divisible by ep*pp={layout.ep * layout.pp}")
                        yield layout

    def feasible(self, arch: ModelArch, world: int, run_shape: RunShape, cpu_adam: bool) -> list[Layout]:
        out = [layout for layout in self._enumerate(arch, world, cpu_adam)
               if self.mem.est(arch, layout, run_shape) <= self.cap]
        out.sort(key=lambda layout: (-layout.dp, layout.tp, layout.pp, layout.cp, layout.ep))
        return out

    def best(self, arch: ModelArch, world: int, run_shape: RunShape, cpu_adam: bool | str) -> Layout:
        modes = [False, True] if cpu_adam == "auto" else [bool(cpu_adam)]
        for mode in modes:
            cand = self.feasible(arch, world, run_shape, mode)
            if cand:
                return cand[0]
        raise RuntimeError(
            f"no feasible layout for {arch.name} at world={world} within {self.cap}GB "
            f"(cpu_adam={cpu_adam}); model too large for this world."
        )

# tools/strategy_gen/memory.py
"""Per-GPU peak-memory estimators (torch-free), behind a pluggable interface.

v1 is a deliberately rough heuristic (params + a crude activation term + optimizer +
overhead). It exists behind `MemoryModel.est()` so an accurate analytic backend (ported
from the apache-2.0 ISEEKYAN memory-estimator project) can replace it without touching
the planner or scenarios.
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING, Protocol

from .registry import ModelArch, RunShape

if TYPE_CHECKING:
    from .planner import Layout

_GIB = 2 ** 30
_ACT_BYTES = 128         # tunable activation multiplier (bytes per token x hidden x layer);
                         # NOT literal fp16 bytes — it folds in a ~Nx live-activation factor
_RECOMPUTE_FACTOR = 0.15  # full activation recompute keeps ~this fraction resident
_OVERHEAD_GB = 10.0


class MemoryModel(Protocol):
    def est(self, arch: ModelArch, layout: "Layout", run_shape: RunShape) -> float:
        """Estimated per-GPU peak memory in GB for one layout."""
        ...


class HeuristicMemoryModel:
    """Rough analytic heuristic. See module docstring."""

    def est(self, arch: ModelArch, layout: "Layout", run_shape: RunShape) -> float:
        p = arch.total_params_b
        weight_grad = 6.0 * p / (layout.tp * layout.pp * layout.ep)
        optimizer = 0.0 if layout.cpu_adam else 12.0 * p / (layout.dp * layout.cp)
        layers_per_stage = math.ceil(arch.layers / layout.pp)
        act_bytes = _ACT_BYTES * run_shape.mbs * run_shape.seq * arch.hidden * layers_per_stage / layout.tp
        activation = act_bytes / _GIB
        if run_shape.recompute_full:
            activation *= _RECOMPUTE_FACTOR
        return weight_grad + optimizer + activation + _OVERHEAD_GB

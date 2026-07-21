# Strategy Generator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a torch-free `tools/strategy_gen/` package that turns `(model, gpus, gpu-mem, scenario)` into an elastic reshard sequence (override-dict JSON) plus a directly-runnable launch recipe, replacing the hand-written `examples/strategies/*.json` proliferation.

**Architecture:** Five small units — `registry` (model arch table + config/HF override), `memory` (pluggable per-GPU peak estimator, v1 heuristic), `planner` (enumerate + filter + rank feasible `(TP,PP,CP,EP,DP)` layouts), `scenarios` (rule functions producing override-dict sequences), `cli` (orchestrate + emit JSON/recipe/table). Subsumes and replaces `tools/strategy_oom.py`.

**Tech Stack:** Python 3 stdlib only (dataclasses, argparse, json, itertools). No torch. Tests are plain `python3 tests/test_*.py` scripts with `assert` + a `__main__` runner (matching the repo's existing torch-free test style).

**Working dir:** the `em-ca-verify` worktree (`/Users/tonic/Code/dynamic_experiment/em-ca-verify`), branch `feat/cpu-adam-core-verify`. Local commits only — **never push**.

**Spec:** `docs/superpowers/specs/2026-06-25-strategy-generator-design.md`.

---

## File structure

```
tools/strategy_gen/
  __init__.py        # exports: load_model, HeuristicMemoryModel, LayoutPlanner, SCENARIOS, generate
  __main__.py        # python -m tools.strategy_gen -> cli.main()
  registry.py        # ModelArch, RunShape dataclasses; MODEL_REGISTRY; load_model()
  memory.py          # MemoryModel Protocol; HeuristicMemoryModel.est()
  planner.py         # Layout dataclass; LayoutPlanner.feasible()/best()
  scenarios.py       # SCENARIOS registry; rl_dp_resize(); verify_sweep()
  cli.py             # argparse, generate(), run-recipe + summary-table emit, main()
tools/__init__.py    # NEW empty file (makes `tools` a package for `-m` + test imports)
tests/
  test_strategy_gen_registry.py
  test_strategy_gen_memory.py
  test_strategy_gen_planner.py
  test_strategy_gen_scenarios.py
  test_strategy_gen_cli.py
examples/strategies/precision/dense_no_tp.json   # REGENERATED fixture + inputs header comment
examples/strategies/moe_30b.json                  # REGENERATED fixture
tests/test_strategy_fixtures_drift.py             # committed fixture == fresh generator output
```

**Import discipline (avoids circular imports):** `registry` is a leaf. `planner` imports `ModelArch, RunShape` from `registry` and defines `Layout`; it receives a memory model instance (duck-typed, no import of `memory`). `memory` imports `Layout` only under `TYPE_CHECKING`. `scenarios` imports `registry` + `planner`. `cli` imports all. Tests run from repo root: `sys.path.insert(0, <repo root>)` then `from tools.strategy_gen.X import Y`.

---

## Task 0: Package skeleton + `tools` as a package

**Files:**
- Create: `tools/__init__.py` (empty)
- Create: `tools/strategy_gen/__init__.py` (empty for now)

- [ ] **Step 1: Confirm making `tools` a package is safe**

Run: `cd /Users/tonic/Code/dynamic_experiment/em-ca-verify && grep -rnE '^\s*(from|import)\s+tools(\.|\s|$)' --include=*.py . | grep -v tests/`
Expected: no hits that would break (the existing `tools/*.py` are run as scripts, not imported as `tools.X`). If hits appear, note them; an empty `tools/__init__.py` only *adds* import capability, it does not change script execution.

- [ ] **Step 2: Create the package files**

```bash
touch tools/__init__.py tools/strategy_gen/__init__.py
```

- [ ] **Step 3: Verify the namespace resolves**

Run: `python3 -c "import tools.strategy_gen; print('pkg ok')"`
Expected: `pkg ok`

- [ ] **Step 4: Commit**

```bash
git add tools/__init__.py tools/strategy_gen/__init__.py
git commit -m "feat(strategy_gen): package skeleton"
```

---

## Task 1: `registry.py` — model arch table + loader

**Files:**
- Create: `tools/strategy_gen/registry.py`
- Test: `tests/test_strategy_gen_registry.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_strategy_gen_registry.py
"""Torch-free tests for the model registry + loader."""
import json, os, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.strategy_gen.registry import ModelArch, MODEL_REGISTRY, load_model


def test_label_resolves_to_arch():
    a = load_model("qwen3-30b")
    assert a.moe and a.num_experts == 128 and a.hidden == 2048 and a.launcher == "run_qwen3_30b.sh"


def test_dense_label():
    a = load_model("llama2-7b")
    assert not a.moe and a.num_experts == 0 and a.layers == 32 and a.launcher == "run_dense.sh"


def test_unknown_label_raises():
    try:
        load_model("nope-1t")
    except KeyError as e:
        assert "nope-1t" in str(e)
    else:
        raise AssertionError("expected KeyError")


def test_config_override():
    cfg = dict(name="custom", moe=False, total_params_b=1.0, hidden=512, layers=4,
               ffn=1024, heads=8, kv_heads=8, seq=2048, vocab=32000)
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(cfg, f); path = f.name
    a = load_model("ignored", config_path=path)
    os.unlink(path)
    assert a.name == "custom" and a.hidden == 512 and not a.moe


if __name__ == "__main__":
    for n, fn in sorted((n, f) for n, f in globals().items() if n.startswith("test_") and callable(f)):
        fn()
    print(f"PASS {os.path.relpath(__file__)}")
```

- [ ] **Step 2: Run to verify it fails**

Run: `python3 tests/test_strategy_gen_registry.py`
Expected: FAIL — `ModuleNotFoundError: No module named 'tools.strategy_gen.registry'`

- [ ] **Step 3: Implement `registry.py`**

```python
# tools/strategy_gen/registry.py
"""Model arch registry + loader (torch-free).

Single source of model architecture for memory estimation. Seeded from the launcher
arch tables (run_dense.sh MODEL_SIZE switch, run_qwen3_30b.sh defaults). `total_params_b`
is the authoritative weight/grad/optimizer sizing number (params in billions, total —
for MoE that counts all experts); the arch fields drive the activation estimate.
"""
from __future__ import annotations

import json
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelArch:
    name: str
    moe: bool
    total_params_b: float
    hidden: int
    layers: int
    ffn: int
    heads: int
    kv_heads: int
    seq: int
    vocab: int
    num_experts: int = 0
    moe_ffn: int = 0
    topk: int = 0
    launcher: str = "run_dense.sh"
    default_mbs: int = 1


@dataclass(frozen=True)
class RunShape:
    mbs: int
    seq: int
    recompute_full: bool = False


# label -> ModelArch. Dense arch from run_dense.sh's MODEL_SIZE table; MoE from
# run_qwen3_30b.sh defaults. total_params_b mirrors the prior MODEL_SCALES model_gb.
MODEL_REGISTRY: dict[str, ModelArch] = {
    "llama2-medium": ModelArch("llama2-medium", False, 2.0, 4096, 8, 11008, 32, 8, 4096, 32000),
    "llama2-7b": ModelArch("llama2-7b", False, 7.0, 4096, 32, 11008, 32, 32, 4096, 32000),
    "llama2-13b": ModelArch("llama2-13b", False, 13.0, 5120, 40, 13824, 40, 40, 4096, 32000),
    "moe-4b": ModelArch("moe-4b", True, 4.0, 2048, 12, 6144, 32, 4, 4096, 151936,
                        num_experts=128, moe_ffn=768, topk=8, launcher="run_qwen3_30b.sh"),
    "moe-15b": ModelArch("moe-15b", True, 15.0, 2048, 24, 6144, 32, 4, 4096, 151936,
                         num_experts=128, moe_ffn=768, topk=8, launcher="run_qwen3_30b.sh"),
    "qwen3-30b": ModelArch("qwen3-30b", True, 30.0, 2048, 48, 6144, 32, 4, 4096, 151936,
                           num_experts=128, moe_ffn=768, topk=8, launcher="run_qwen3_30b.sh"),
}


def load_model(spec: str, config_path: str | None = None) -> ModelArch:
    """Resolve a model: a registry label, or a --config JSON of ModelArch fields."""
    if config_path is not None:
        with open(config_path) as f:
            return ModelArch(**json.load(f))
    if spec not in MODEL_REGISTRY:
        raise KeyError(f"unknown model {spec!r}; known: {sorted(MODEL_REGISTRY)} (or pass --config)")
    return MODEL_REGISTRY[spec]
```

- [ ] **Step 4: Run to verify it passes**

Run: `python3 tests/test_strategy_gen_registry.py`
Expected: `PASS tests/test_strategy_gen_registry.py`

- [ ] **Step 5: Commit**

```bash
git add tools/strategy_gen/registry.py tests/test_strategy_gen_registry.py
git commit -m "feat(strategy_gen): model arch registry + config loader"
```

---

## Task 2: `memory.py` — pluggable per-GPU peak estimator

**Files:**
- Create: `tools/strategy_gen/memory.py`
- Test: `tests/test_strategy_gen_memory.py`

The heuristic (per-GPU peak GB), where `P = total_params_b`:
- `weight_grad = 6 * P / (TP*PP*EP)` — bf16 weight+grad+misc, sharded by model-parallel dims.
- `optimizer = 0 if cpu_adam else 12 * P / world` — fp32 master+m+v, distributed-optimizer sharded across all ranks.
- `activation = ACT_BYTES * mbs * seq * hidden * ceil(layers/PP) / TP / 2**30`, then `* 0.15` if `recompute_full` — a deliberately rough term (the new piece vs the old OOM heuristic). `ACT_BYTES = 16`.
- `overhead = 10`.

Constants are heuristic/tunable; the calibration test below pins the known-good behavior.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_strategy_gen_memory.py
import math, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.strategy_gen.registry import load_model, RunShape
from tools.strategy_gen.memory import HeuristicMemoryModel

M = HeuristicMemoryModel()


class L:  # minimal duck-typed layout
    def __init__(self, world, tp, pp, cp, ep, dp, cpu_adam=False):
        self.world, self.tp, self.pp, self.cp, self.ep, self.dp, self.cpu_adam = world, tp, pp, cp, ep, dp, cpu_adam


def test_more_tp_lowers_peak():
    a = load_model("llama2-7b"); rs = RunShape(1, 4096)
    hi = M.est(a, L(8, 1, 1, 1, 1, 8), rs)
    lo = M.est(a, L(8, 2, 1, 1, 1, 4), rs)
    assert lo < hi


def test_cpu_adam_zeroes_optimizer():
    a = load_model("llama2-7b"); rs = RunShape(1, 4096)
    gpu = M.est(a, L(8, 1, 1, 1, 1, 8, cpu_adam=False), rs)
    cpu = M.est(a, L(8, 1, 1, 1, 1, 8, cpu_adam=True), rs)
    assert cpu < gpu


def test_recompute_lowers_activation():
    a = load_model("qwen3-30b")
    full = M.est(a, L(8, 4, 1, 1, 4, 2), RunShape(1, 4096, recompute_full=True))
    none = M.est(a, L(8, 4, 1, 1, 4, 2), RunShape(1, 4096, recompute_full=False))
    assert full < none


def test_calibration_qwen3_30b():
    # 30B on 8x80GB: GPU-adam must NOT fit; cpu-adam must fit. (Matches reality.)
    a = load_model("qwen3-30b"); rs = RunShape(1, 4096, recompute_full=True)
    gpu = M.est(a, L(8, 4, 1, 1, 4, 2, cpu_adam=False), rs)
    cpu = M.est(a, L(8, 4, 1, 1, 4, 2, cpu_adam=True), rs)
    assert gpu > 80.0, f"expected GPU-adam 30B to exceed 80GB, got {gpu:.1f}"
    assert cpu <= 80.0, f"expected cpu-adam 30B to fit 80GB, got {cpu:.1f}"


if __name__ == "__main__":
    for n, fn in sorted((n, f) for n, f in globals().items() if n.startswith("test_") and callable(f)):
        fn()
    print(f"PASS {os.path.relpath(__file__)}")
```

- [ ] **Step 2: Run to verify it fails**

Run: `python3 tests/test_strategy_gen_memory.py`
Expected: FAIL — `No module named 'tools.strategy_gen.memory'`

- [ ] **Step 3: Implement `memory.py`**

```python
# tools/strategy_gen/memory.py
"""Per-GPU peak-memory estimators (torch-free), behind a pluggable interface.

v1 is a deliberately rough heuristic (params + a crude activation term + optimizer +
overhead). It exists behind `MemoryModel.est()` so an accurate analytic backend (ported
from the apache-2.0 ISEEKYAN/megatron_memory_estimator) can replace it without touching
the planner or scenarios.
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING, Protocol

from .registry import ModelArch, RunShape

if TYPE_CHECKING:
    from .planner import Layout

_GIB = 2 ** 30
_ACT_BYTES = 16          # rough fp16 activation bytes per (token x hidden) element per layer
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
        optimizer = 0.0 if layout.cpu_adam else 12.0 * p / layout.world
        layers_per_stage = math.ceil(arch.layers / layout.pp)
        act_bytes = _ACT_BYTES * run_shape.mbs * run_shape.seq * arch.hidden * layers_per_stage / layout.tp
        activation = act_bytes / _GIB
        if run_shape.recompute_full:
            activation *= _RECOMPUTE_FACTOR
        return weight_grad + optimizer + activation + _OVERHEAD_GB
```

- [ ] **Step 4: Run to verify it passes**

Run: `python3 tests/test_strategy_gen_memory.py`
Expected: `PASS tests/test_strategy_gen_memory.py`
If `test_calibration_qwen3_30b` fails, adjust `_ACT_BYTES` / `_RECOMPUTE_FACTOR` so the two assertions hold (GPU-adam 30B > 80, cpu-adam 30B <= 80), then re-run.

- [ ] **Step 5: Commit**

```bash
git add tools/strategy_gen/memory.py tests/test_strategy_gen_memory.py
git commit -m "feat(strategy_gen): pluggable per-GPU memory estimator (v1 heuristic)"
```

---

## Task 3: `planner.py` — feasible layout enumeration + ranking

**Files:**
- Create: `tools/strategy_gen/planner.py`
- Test: `tests/test_strategy_gen_planner.py`

`Layout` carries the resolved dims + `cpu_adam`. `LayoutPlanner(memory_model, gpu_mem_gb)`:
- `feasible(arch, world, run_shape, cpu_adam)`: enumerate `TP,PP,CP` dividing `world`; `DP = world//(TP*PP*CP)`; for MoE enumerate `EP` in divisors of `num_experts` with `EP<=world` (dense `EP=1`); keep layouts with `est <= gpu_mem`; ranked by `(-DP, TP, PP, CP)` (max DP, then minimal model-parallel).
- `best(...)`: first feasible, or `RuntimeError` if none (loud).
- `cpu_adam="auto"`: try GPU-adam; if no feasible layout, retry with cpu-adam.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_strategy_gen_planner.py
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.strategy_gen.registry import load_model, RunShape
from tools.strategy_gen.memory import HeuristicMemoryModel
from tools.strategy_gen.planner import LayoutPlanner, Layout

PL = LayoutPlanner(HeuristicMemoryModel(), gpu_mem_gb=80.0)


def test_divisibility_and_dp():
    outs = PL.feasible(load_model("llama2-7b"), world=8, run_shape=RunShape(1, 4096), cpu_adam=False)
    for l in outs:
        assert l.world == 8 and l.world % (l.tp * l.pp * l.cp) == 0
        assert l.dp == l.world // (l.tp * l.pp * l.cp) and l.ep == 1


def test_moe_ep_divides_experts_and_etp1():
    outs = PL.feasible(load_model("qwen3-30b"), world=8, run_shape=RunShape(1, 4096, True), cpu_adam=True)
    assert outs, "expected feasible MoE layouts under cpu-adam"
    for l in outs:
        assert 128 % l.ep == 0 and l.ep <= l.world


def test_ranking_prefers_max_dp():
    outs = PL.feasible(load_model("llama2-7b"), world=8, run_shape=RunShape(1, 4096), cpu_adam=False)
    assert outs[0].dp == max(l.dp for l in outs)


def test_no_feasible_raises():
    small = LayoutPlanner(HeuristicMemoryModel(), gpu_mem_gb=1.0)
    try:
        small.best(load_model("qwen3-30b"), world=8, run_shape=RunShape(1, 4096, True), cpu_adam=False)
    except RuntimeError as e:
        assert "no feasible" in str(e).lower()
    else:
        raise AssertionError("expected RuntimeError")


def test_auto_falls_back_to_cpu_adam():
    # 30B has no GPU-adam layout at world 8 but a cpu-adam one exists -> auto picks cpu-adam.
    l = PL.best(load_model("qwen3-30b"), world=8, run_shape=RunShape(1, 4096, True), cpu_adam="auto")
    assert l.cpu_adam is True


if __name__ == "__main__":
    for n, fn in sorted((n, f) for n, f in globals().items() if n.startswith("test_") and callable(f)):
        fn()
    print(f"PASS {os.path.relpath(__file__)}")
```

- [ ] **Step 2: Run to verify it fails**

Run: `python3 tests/test_strategy_gen_planner.py`
Expected: FAIL — `No module named 'tools.strategy_gen.planner'`

- [ ] **Step 3: Implement `planner.py`**

```python
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

    def override(self, base: "Layout") -> dict:
        """Override-dict vs a base layout: only the dims that differ (strategy_inject form)."""
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
        return {k: v for k, v in keys.items() if v != base_keys[k]}


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
                        yield Layout(world, tp, pp, cp, ep, dp, cpu_adam)

    def feasible(self, arch: ModelArch, world: int, run_shape: RunShape, cpu_adam: bool) -> list[Layout]:
        out = [l for l in self._enumerate(arch, world, cpu_adam)
               if self.mem.est(arch, l, run_shape) <= self.cap]
        out.sort(key=lambda l: (-l.dp, l.tp, l.pp, l.cp, l.ep))
        return out

    def best(self, arch: ModelArch, world: int, run_shape: RunShape, cpu_adam) -> Layout:
        modes = [False, True] if cpu_adam == "auto" else [bool(cpu_adam)]
        for mode in modes:
            cand = self.feasible(arch, world, run_shape, mode)
            if cand:
                return cand[0]
        raise RuntimeError(
            f"no feasible layout for {arch.name} at world={world} within {self.cap}GB "
            f"(cpu_adam={cpu_adam}); model too large for this world."
        )
```

- [ ] **Step 4: Run to verify it passes**

Run: `python3 tests/test_strategy_gen_planner.py`
Expected: `PASS tests/test_strategy_gen_planner.py`

- [ ] **Step 5: Commit**

```bash
git add tools/strategy_gen/planner.py tests/test_strategy_gen_planner.py
git commit -m "feat(strategy_gen): feasible layout planner (enumerate+filter+rank)"
```

---

## Task 4: `scenarios.py` — reshard-sequence rules

**Files:**
- Create: `tools/strategy_gen/scenarios.py`
- Test: `tests/test_strategy_gen_scenarios.py`

A scenario takes `(arch, hw, planner, run_shape)` and returns `(base_layout, [override_dict])`
with `overrides[0] == {}` (= base, the strategy_inject convention).
- `rl_dp_resize`: base = `best(world=gpus)`; rollout = `best(world=gpus//2)`; sequence `[base, rollout, base]` → overrides `[{}, rollout.override(base), {}]`.
- `verify_sweep`: base = `best(world=gpus)`; then each *distinct* feasible layout at the same world (from `feasible`, capped at `max_steps=6`) as a follow-on → overrides `[{}] + [l.override(base) for l in others]`.

`hw = Hardware(gpus, gpu_mem_gb)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_strategy_gen_scenarios.py
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.strategy_gen.registry import load_model, RunShape
from tools.strategy_gen.memory import HeuristicMemoryModel
from tools.strategy_gen.planner import LayoutPlanner
from tools.strategy_gen.scenarios import SCENARIOS, Hardware


def _run(scenario, model, gpus, mem, rs):
    pl = LayoutPlanner(HeuristicMemoryModel(), mem)
    return SCENARIOS[scenario](load_model(model), Hardware(gpus, mem), pl, rs)


def test_overrides0_is_empty():
    base, ov = _run("rl_dp_resize", "llama2-7b", 8, 80, RunShape(1, 4096))
    assert ov[0] == {}


def test_rl_dp_resize_shrinks_world():
    base, ov = _run("rl_dp_resize", "llama2-7b", 8, 80, RunShape(1, 4096))
    assert any(o.get("world_size") == 4 for o in ov), f"expected a world=4 step, got {ov}"
    assert ov[-1] == {}  # returns to base


def test_verify_sweep_all_same_world_and_distinct():
    base, ov = _run("verify_sweep", "llama2-7b", 8, 80, RunShape(1, 4096))
    assert ov[0] == {} and len(ov) >= 2
    assert all("world_size" not in o for o in ov)  # fixed world sweep
    assert len(ov) == len({tuple(sorted(o.items())) for o in ov})  # distinct


def test_moe_scenarios_feasible():
    base, ov = _run("rl_dp_resize", "qwen3-30b", 8, 80, RunShape(1, 4096, True))
    assert base.cpu_adam and ov[0] == {}


if __name__ == "__main__":
    for n, fn in sorted((n, f) for n, f in globals().items() if n.startswith("test_") and callable(f)):
        fn()
    print(f"PASS {os.path.relpath(__file__)}")
```

- [ ] **Step 2: Run to verify it fails**

Run: `python3 tests/test_strategy_gen_scenarios.py`
Expected: FAIL — `No module named 'tools.strategy_gen.scenarios'`

- [ ] **Step 3: Implement `scenarios.py`**

```python
# tools/strategy_gen/scenarios.py
"""Reshard-sequence rules: (model, hardware) -> (base layout, override-dict list).

Each scenario returns the base full-world layout plus a list of override-dicts (relative
to base) whose first element is {} (the strategy_inject convention: strategy[0] == launch
config). Registered in SCENARIOS by name; add a scenario = add a function + register it.
"""
from __future__ import annotations

from dataclasses import dataclass

from .planner import Layout, LayoutPlanner
from .registry import ModelArch, RunShape


@dataclass(frozen=True)
class Hardware:
    gpus: int
    gpu_mem_gb: float


def rl_dp_resize(arch: ModelArch, hw: Hardware, planner: LayoutPlanner, rs: RunShape):
    """Train at full world; shrink world (~half) for rollout; return to full."""
    base = planner.best(arch, hw.gpus, rs, "auto")
    rollout = planner.best(arch, max(hw.gpus // 2, 1), rs, base.cpu_adam)
    overrides = [{}, rollout.override(base), {}]
    return base, overrides


def verify_sweep(arch: ModelArch, hw: Hardware, planner: LayoutPlanner, rs: RunShape, max_steps: int = 6):
    """Cycle distinct feasible layouts at the fixed launch world (correctness coverage)."""
    base = planner.best(arch, hw.gpus, rs, "auto")
    cand = planner.feasible(arch, hw.gpus, rs, base.cpu_adam)
    overrides = [{}]
    for l in cand:
        ov = l.override(base)
        if ov and ov not in overrides:
            overrides.append(ov)
        if len(overrides) >= max_steps:
            break
    return base, overrides


SCENARIOS = {
    "rl_dp_resize": rl_dp_resize,
    "verify_sweep": verify_sweep,
}
```

- [ ] **Step 4: Run to verify it passes**

Run: `python3 tests/test_strategy_gen_scenarios.py`
Expected: `PASS tests/test_strategy_gen_scenarios.py`

- [ ] **Step 5: Commit**

```bash
git add tools/strategy_gen/scenarios.py tests/test_strategy_gen_scenarios.py
git commit -m "feat(strategy_gen): rl_dp_resize + verify_sweep scenario rules"
```

---

## Task 5: `cli.py` + `__main__.py` — orchestrate + emit

**Files:**
- Create: `tools/strategy_gen/cli.py`
- Create: `tools/strategy_gen/__main__.py`
- Modify: `tools/strategy_gen/__init__.py` (export `generate`)
- Test: `tests/test_strategy_gen_cli.py`

`generate(...)` returns `(overrides, recipe, table)`. The recipe sets the launcher knobs:
dense → `MODEL_SIZE` cannot carry arbitrary arch, so the recipe sets `TP/PP/CP` + `MODEL_SIZE`
(label maps to run_dense.sh's table) ; MoE → `NUM_LAYERS/NUM_EXPERTS/TP/EP/CP` for run_qwen3_30b.sh.
Both set `ELASTIC_ENABLED=1`, `ELASTIC_STRATEGY_LIST_FILE=<out>`, `ELASTIC_RESHARD_INTERVAL`, and
`CPU_OFFLOAD` from `base.cpu_adam`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_strategy_gen_cli.py
import importlib.util, json, os, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.strategy_gen.cli import generate


def _load_strategy_inject():
    p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "elastic_megatron", "strategy_inject.py")
    spec = importlib.util.spec_from_file_location("strategy_inject_ut", p)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def test_generate_overrides_parse_through_strategy_inject(monkeypatch=None):
    overrides, recipe, table = generate("llama2-7b", gpus=8, gpu_mem_gb=80,
                                        scenario="rl_dp_resize", mbs=1, seq=4096)
    assert overrides[0] == {}
    si = _load_strategy_inject()
    os.environ["ELASTIC_STRATEGY_LIST"] = json.dumps(overrides)
    base = dict(world_size=8, tensor_model_parallel_size=1, pipeline_model_parallel_size=1,
                context_parallel_size=1, expert_model_parallel_size=1,
                expert_tensor_parallel_size=None, num_distributed_optimizer_instances=1,
                sequence_parallel=False)
    strategies = si.build_strategy_list(base)
    del os.environ["ELASTIC_STRATEGY_LIST"]
    assert len(strategies) == len(overrides) and strategies[0]["world_size"] == 8


def test_recipe_picks_launcher_and_offload():
    _, recipe, _ = generate("qwen3-30b", gpus=8, gpu_mem_gb=80, scenario="rl_dp_resize",
                            mbs=1, seq=4096, recompute_full=True)
    assert "run_qwen3_30b.sh" in recipe and "CPU_OFFLOAD=1" in recipe
    assert "ELASTIC_ENABLED=1" in recipe and "ELASTIC_STRATEGY_LIST_FILE=" in recipe


def test_dense_recipe_uses_run_dense():
    _, recipe, _ = generate("llama2-7b", gpus=8, gpu_mem_gb=80, scenario="verify_sweep",
                            mbs=1, seq=4096)
    assert "run_dense.sh" in recipe and "MODEL_SIZE=" in recipe


if __name__ == "__main__":
    for n, fn in sorted((n, f) for n, f in globals().items() if n.startswith("test_") and callable(f)):
        fn()
    print(f"PASS {os.path.relpath(__file__)}")
```

- [ ] **Step 2: Run to verify it fails**

Run: `python3 tests/test_strategy_gen_cli.py`
Expected: FAIL — `No module named 'tools.strategy_gen.cli'`

- [ ] **Step 3: Implement `cli.py`**

```python
# tools/strategy_gen/cli.py
"""CLI + orchestration: (model, hardware, scenario) -> reshard sequence + run recipe."""
from __future__ import annotations

import argparse
import json

from .memory import HeuristicMemoryModel
from .planner import LayoutPlanner, Layout
from .registry import RunShape, load_model
from .scenarios import SCENARIOS, Hardware

# Dense registry label -> run_dense.sh MODEL_SIZE token.
_DENSE_MODEL_SIZE = {"llama2-medium": "medium", "llama2-7b": "7", "llama2-13b": "13"}


def _recipe(arch, base: Layout, out_path: str, interval: int) -> str:
    common = (f"ELASTIC_ENABLED=1 ELASTIC_STRATEGY_LIST_FILE={out_path} "
              f"ELASTIC_RESHARD_INTERVAL={interval} CPU_OFFLOAD={1 if base.cpu_adam else 0}")
    if arch.moe:
        return (f"{common} NUM_LAYERS={arch.layers} NUM_EXPERTS={arch.num_experts} "
                f"TP={base.tp} PP={base.pp} EP={base.ep} CP={base.cp} ./{arch.launcher}")
    size = _DENSE_MODEL_SIZE.get(arch.name, "medium")
    return (f"{common} MODEL_SIZE={size} TP={base.tp} PP={base.pp} CP={base.cp} "
            f"./{arch.launcher}")


def _table(arch, base: Layout, overrides: list[dict], mem, rs) -> str:
    lines = ["  step  world  TP PP CP EP  DP   ~peakGB  cpu-adam"]
    for i, ov in enumerate(overrides):
        merged = dict(world=base.world, tp=base.tp, pp=base.pp, cp=base.cp, ep=base.ep)
        for k, v in ov.items():
            merged[{"world_size": "world", "tensor_model_parallel_size": "tp",
                    "pipeline_model_parallel_size": "pp", "context_parallel_size": "cp",
                    "expert_model_parallel_size": "ep"}[k]] = v
        dp = merged["world"] // (merged["tp"] * merged["pp"] * merged["cp"])
        l = Layout(merged["world"], merged["tp"], merged["pp"], merged["cp"], merged["ep"], dp, base.cpu_adam)
        gb = mem.est(arch, l, rs)
        lines.append(f"  {i:>4}  {l.world:>5}  {l.tp:>2} {l.pp:>2} {l.cp:>2} {l.ep:>2}  {l.dp:>2}  {gb:>7.1f}  {l.cpu_adam}")
    return "\n".join(lines)


def generate(model, gpus, gpu_mem_gb, scenario, mbs=1, seq=4096, recompute_full=False,
             config_path=None, out_path="examples/strategies/generated.json", interval=1):
    arch = load_model(model, config_path)
    rs = RunShape(mbs, seq, recompute_full)
    mem = HeuristicMemoryModel()
    planner = LayoutPlanner(mem, gpu_mem_gb)
    if scenario not in SCENARIOS:
        raise KeyError(f"unknown scenario {scenario!r}; known: {sorted(SCENARIOS)}")
    base, overrides = SCENARIOS[scenario](arch, Hardware(gpus, gpu_mem_gb), planner, rs)
    recipe = _recipe(arch, base, out_path, interval)
    table = _table(arch, base, overrides, mem, rs)
    return overrides, recipe, table


def main(argv=None):
    ap = argparse.ArgumentParser(description="Generate an elastic reshard sequence + run recipe.")
    ap.add_argument("--model", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--gpus", type=int, required=True)
    ap.add_argument("--gpu-mem", type=float, required=True)
    ap.add_argument("--scenario", required=True, choices=sorted(SCENARIOS))
    ap.add_argument("--mbs", type=int, default=1)
    ap.add_argument("--seq", type=int, default=4096)
    ap.add_argument("--recompute-full", action="store_true")
    ap.add_argument("--interval", type=int, default=1)
    ap.add_argument("--out", default="examples/strategies/generated.json")
    a = ap.parse_args(argv)
    overrides, recipe, table = generate(a.model, a.gpus, a.gpu_mem, a.scenario, a.mbs, a.seq,
                                        a.recompute_full, a.config, a.out, a.interval)
    with open(a.out, "w") as f:
        json.dump(overrides, f, indent=2)
        f.write("\n")
    print(f"wrote {len(overrides)}-step sequence -> {a.out}\n")
    print("run recipe:\n  " + recipe + "\n")
    print(table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Create `__main__.py`**

```python
# tools/strategy_gen/__main__.py
from .cli import main

raise SystemExit(main())
```

- [ ] **Step 5: Update `__init__.py`**

```python
# tools/strategy_gen/__init__.py
from .cli import generate
from .memory import HeuristicMemoryModel
from .planner import LayoutPlanner
from .registry import load_model
from .scenarios import SCENARIOS

__all__ = ["generate", "HeuristicMemoryModel", "LayoutPlanner", "load_model", "SCENARIOS"]
```

- [ ] **Step 6: Run tests + the CLI smoke**

Run: `python3 tests/test_strategy_gen_cli.py`
Expected: `PASS tests/test_strategy_gen_cli.py`
Run: `python3 -m tools.strategy_gen --model qwen3-30b --gpus 8 --gpu-mem 80 --scenario rl_dp_resize --recompute-full --out /tmp/seq.json`
Expected: prints `wrote 3-step sequence`, a run recipe containing `run_qwen3_30b.sh` + `CPU_OFFLOAD=1`, and a table.

- [ ] **Step 7: Commit**

```bash
git add tools/strategy_gen/cli.py tools/strategy_gen/__main__.py tools/strategy_gen/__init__.py tests/test_strategy_gen_cli.py
git commit -m "feat(strategy_gen): CLI — emit sequence JSON + runnable recipe + summary table"
```

---

## Task 6: Migration — regenerate fixtures, drift test, remove old code/JSONs

**Files:**
- Regenerate: `examples/strategies/precision/dense_no_tp.json`, `examples/strategies/moe_30b.json`
- Create: `tests/test_strategy_fixtures_drift.py`
- Delete: `tools/strategy_oom.py`, `tests/test_strategy_oom.py`
- Delete: the bulk JSONs (see Step 4)
- Test: drift test

**Fixture canonical inputs** (record in the drift test so they regenerate deterministically):
- `dense_no_tp.json` = `verify_sweep`, model `llama2-medium`, gpus 8, gpu-mem 80, mbs 1, seq 4096.
- `moe_30b.json` = `verify_sweep`, model `qwen3-30b`, gpus 8, gpu-mem 80, mbs 1, seq 4096, recompute-full.

> Note: regenerated content may differ from the old hand-written entries. Confirm each
> still contains an 8→4 scale-down edge (the DCP-fix verify relies on it). If `verify_sweep`
> at fixed world does NOT produce a world-shrink (it sweeps fixed world), use `rl_dp_resize`
> for these two fixtures instead — pick whichever scenario yields a sequence that (a) is
> valid and (b) includes a scale-down, and record that choice in the drift test. (Decide at
> implementation time from the actual generator output; the drift test pins it.)

- [ ] **Step 1: Write the drift test (failing)**

```python
# tests/test_strategy_fixtures_drift.py
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.strategy_gen import generate

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# (fixture path, generate kwargs) — the canonical inputs that produce each committed file.
FIXTURES = [
    ("examples/strategies/precision/dense_no_tp.json",
     dict(model="llama2-medium", gpus=8, gpu_mem_gb=80, scenario="rl_dp_resize", mbs=1, seq=4096)),
    ("examples/strategies/moe_30b.json",
     dict(model="qwen3-30b", gpus=8, gpu_mem_gb=80, scenario="rl_dp_resize", mbs=1, seq=4096, recompute_full=True)),
]


def test_fixtures_match_generator():
    for rel, kw in FIXTURES:
        overrides, _, _ = generate(out_path=rel, **kw)
        committed = json.load(open(os.path.join(ROOT, rel)))
        assert committed == overrides, f"{rel} drifted from generator output"
        assert any(o.get("world_size") == 4 for o in overrides), f"{rel} lacks an 8->4 scale-down"


if __name__ == "__main__":
    test_fixtures_match_generator()
    print(f"PASS {os.path.relpath(__file__)}")
```

- [ ] **Step 2: Regenerate the two fixtures from the tool**

```bash
python3 -m tools.strategy_gen --model llama2-medium --gpus 8 --gpu-mem 80 --scenario rl_dp_resize \
  --out examples/strategies/precision/dense_no_tp.json
python3 -m tools.strategy_gen --model qwen3-30b --gpus 8 --gpu-mem 80 --scenario rl_dp_resize \
  --recompute-full --out examples/strategies/moe_30b.json
```
Inspect both files; confirm `overrides[0] == {}` and a `{"world_size": 4}` step is present. (If `rl_dp_resize` for medium-dense at world 8 picks DP8 base, the rollout step is `{"world_size": 4}` — good.)

- [ ] **Step 3: Run the drift test to verify it passes**

Run: `python3 tests/test_strategy_fixtures_drift.py`
Expected: `PASS tests/test_strategy_fixtures_drift.py`

- [ ] **Step 4: Delete the obsolete files**

```bash
git rm tools/strategy_oom.py tests/test_strategy_oom.py
git rm examples/strategies/llama2_medium.json examples/strategies/llama2_7b.json examples/strategies/llama2_13b.json \
       examples/strategies/moe_4b.json examples/strategies/moe_15b.json \
       examples/strategies/precision/dense_tp.json examples/strategies/precision/moe_no_tp.json examples/strategies/precision/moe_tp.json
git rm -r examples/strategies/rl
```
(Kept: `precision/dense_no_tp.json`, `moe_30b.json`, the two READMEs — updated in Task 7.)

- [ ] **Step 5: Confirm nothing references the deleted files**

Run: `grep -rnE 'strategy_oom|strategies/(llama2_|moe_4b|moe_15b|rl/|precision/(dense_tp|moe_)) ' --include=*.py --include=*.sh --include=*.md . ; echo done`
Expected: only historical doc mentions (fix those in Task 7); no live code/launcher references.

- [ ] **Step 6: Commit**

```bash
git add -A examples/strategies tools tests/test_strategy_fixtures_drift.py
git commit -m "refactor(strategy_gen): regenerate kept fixtures, drift test, remove strategy_oom + bulk JSONs"
```

---

## Task 7: Docs

**Files:**
- Modify: `examples/strategies/README.md`, `examples/strategies/precision/README.md`
- Modify: any doc referencing `strategy_oom.py` or the deleted JSONs (from Task 6 Step 5)

- [ ] **Step 1: Rewrite the strategies READMEs**

Replace the "hand-written per-scale/scenario JSON" description with: the sequences are now
*generated* by `tools/strategy_gen` (`python -m tools.strategy_gen --model … --gpus … --gpu-mem … --scenario …`);
`dense_no_tp.json` + `moe_30b.json` are committed fixtures regenerated by the tool (the
`tests/test_strategy_fixtures_drift.py` pins them); list the available scenarios + the registry
labels. Keep it short.

- [ ] **Step 2: Fix stragglers**

Update any doc hit from Task 6 Step 5 (e.g. `tools/README.md`, `docs/project/repo_layout.md`)
to point at `tools/strategy_gen/` instead of `tools/strategy_oom.py`.

- [ ] **Step 3: Full torch-free regression**

Run: `for t in registry memory planner scenarios cli; do python3 tests/test_strategy_gen_$t.py; done && python3 tests/test_strategy_fixtures_drift.py && python3 tests/test_strategy_injection.py && python3 tests/test_dist_ckpt_patch.py`
Expected: every line prints `PASS`.

- [ ] **Step 4: Commit**

```bash
git add examples/strategies/*.md tools/README.md docs/
git commit -m "docs(strategy_gen): document the generator; drop strategy_oom/per-scale-JSON references"
```

---

## Follow-ups (out of this plan, tracked separately)

- **platform-launcher `docs/megatron-agent-use.md`** (separate repo): update to describe the
  generator-based strategy flow — produce `seq.json` via `tools/strategy_gen` then run; note the
  YAML-referenced `dense_no_tp.json`/`moe_30b.json` are generated fixtures. (User-flagged.)
- **GPU re-verify**: the regenerated `moe_30b.json`/`dense_no_tp.json` differ in content from the
  GPU-verified ones; re-run the aries dense scale-down + MoE smoke against the regenerated fixtures
  to confirm they still drive a valid reshard (incl the 8→4 the DCP-fix verify needs).

---

## Self-review

- **Spec coverage:** Registry (Task 1) ✓; pluggable MemoryModel + heuristic incl. activation (Task 2) ✓; LayoutPlanner enumerate/filter/rank/loud-fail (Task 3) ✓; scenarios rl_dp_resize + verify_sweep (Task 4) ✓; CLI emitting seq.json + run recipe + summary table (Task 5) ✓; migration — subsume strategy_oom, regenerate the two fixtures + drift test, delete bulk (Task 6) ✓; docs (Task 7) ✓; torch-free tests for every unit ✓.
- **Open implementation decision (flagged in Task 6):** whether the two fixtures use `rl_dp_resize` (guarantees an 8→4 scale-down) or `verify_sweep` (fixed-world, may lack a shrink). The drift test asserts an `8→4` step exists, forcing the implementer to pick the scenario that yields one — recorded `rl_dp_resize` as the default. If a fixture must ALSO carry fixed-world PP/CP/EP variety, a future `combo` scenario can be added; not needed for v1.
- **Type consistency:** `Layout(world,tp,pp,cp,ep,dp,cpu_adam)` + `.override(base)` used consistently across planner/scenarios/cli; `RunShape(mbs,seq,recompute_full)`, `Hardware(gpus,gpu_mem_gb)`, `ModelArch` fields consistent; `generate(...)` kwargs match the CLI args and the drift-test call.
- **Placeholder scan:** none — every step has runnable code/commands.

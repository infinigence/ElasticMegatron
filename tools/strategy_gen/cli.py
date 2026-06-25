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


def _recipe(arch, base: Layout, rs: RunShape, out_path: str, interval: int) -> str:
    common = (f"ELASTIC_ENABLED=1 ELASTIC_STRATEGY_LIST_FILE={out_path} "
              f"ELASTIC_RESHARD_INTERVAL={interval} CPU_OFFLOAD={1 if base.cpu_adam else 0}")
    if arch.moe:
        # run_qwen3_30b.sh knobs: TPE (expert-tensor-parallel, must stay 1), RECOMPUTE
        # (default ON — emit explicit 0 to honor a non-recompute plan), MBS, SEQ_LEN. Pinning
        # them makes the launched shape equal the shape the planner sized for feasibility.
        return (f"{common} NUM_LAYERS={arch.layers} NUM_EXPERTS={arch.num_experts} "
                f"TP={base.tp} PP={base.pp} EP={base.ep} CP={base.cp} TPE=1 "
                f"RECOMPUTE={1 if rs.recompute_full else 0} MBS={rs.mbs} SEQ_LEN={rs.seq} "
                f"./{arch.launcher}")
    # run_dense.sh knobs: MODEL_SIZE, RECOMPUTE_FULL (default OFF — only emit when on),
    # MBS, MAX_SEQ_LEN.
    size = _DENSE_MODEL_SIZE[arch.name]
    recompute = "RECOMPUTE_FULL=1 " if rs.recompute_full else ""
    return (f"{common} MODEL_SIZE={size} TP={base.tp} PP={base.pp} CP={base.cp} "
            f"{recompute}MBS={rs.mbs} MAX_SEQ_LEN={rs.seq} ./{arch.launcher}")


def _table(arch, base: Layout, overrides: list[dict], mem, rs) -> str:
    # expert_tensor_parallel_size is pinned to 1 by the MoE override path; it does not move any
    # displayed dim (ETP=1 is implicit in the layout), so it is intentionally not in this map.
    _DIM = {"world_size": "world", "tensor_model_parallel_size": "tp",
            "pipeline_model_parallel_size": "pp", "context_parallel_size": "cp",
            "expert_model_parallel_size": "ep"}
    lines = ["  step  world  TP PP CP EP  DP   ~peakGB  cpu-adam"]
    for i, ov in enumerate(overrides):
        merged = dict(world=base.world, tp=base.tp, pp=base.pp, cp=base.cp, ep=base.ep)
        for k, v in ov.items():
            if k in _DIM:
                merged[_DIM[k]] = v
        dp = merged["world"] // (merged["tp"] * merged["pp"] * merged["cp"])
        layout = Layout(merged["world"], merged["tp"], merged["pp"], merged["cp"], merged["ep"],
                        dp, base.cpu_adam, moe=arch.moe)
        gb = mem.est(arch, layout, rs)
        lines.append(f"  {i:>4}  {layout.world:>5}  {layout.tp:>2} {layout.pp:>2} {layout.cp:>2} "
                     f"{layout.ep:>2}  {layout.dp:>2}  {gb:>7.1f}  {layout.cpu_adam}")
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
    recipe = _recipe(arch, base, rs, out_path, interval)
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

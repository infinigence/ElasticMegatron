# Strategy generator — design spec

*2026-06-25*

## Context

Elastic reshard sequences are currently hand-written as JSON override-lists under
`examples/strategies/` — main per-scale sweeps, `rl/` DP-resize, `precision/`
verification — validated after the fact by `tools/strategy_oom.py`. The set grows as
*model × size × scenario*, so it already has 16 files and would keep multiplying. The
arch knowledge is also split between the bash launchers (`run_dense.sh`'s `MODEL_SIZE`
table, `run_qwen3_30b.sh`'s defaults) and `strategy_oom.py`'s `MODEL_SCALES`.

**Goal:** replace the hand-written proliferation with a rule-based generator. Given a
model (type + size, or an arbitrary config) plus the hardware (GPU count + per-GPU
memory) plus a scenario, it derives a *feasible* reshard sequence and emits a directly
runnable recipe — choosing dense vs MoE and the parallel layout automatically, with a
memory model gating feasibility. It must work for a human giving fixed inputs (no agent
required), and stay torch-free so it runs anywhere.

## Goals / non-goals

- **Goal:** one CLI: `(model, gpus, gpu-mem, scenario) -> reshard sequence + run recipe`.
- **Goal:** encode the MoE-vs-dense + divisibility + memory rules **once**, not per file.
- **Goal:** torch-free; pluggable memory model so an accurate backend can replace the
  v1 heuristic later without touching the planner/scenarios.
- **Non-goal (YAGNI):** a general constraint solver; accurate activation memory in v1;
  pulling every model config from the network; generating the *launch* layout as a
  standalone artifact (the base layout is derived as the sequence's anchor, not a
  separate deliverable).

## Inputs & invocation

```
python tools/strategy_gen/cli.py \
    --model qwen3-30b            # registry label  (OR --config model.json / --hf <id>)
    --gpus 8 --gpu-mem 80        # hardware: world size + per-GPU GB cap
    --scenario rl_dp_resize      # rl_dp_resize | verify_sweep  (extensible)
    [--mbs 1 --seq 4096]         # run shape (defaults from the model registry)
    [--cpu-adam auto|on|off]     # optimizer placement (auto = on iff GPU-adam won't fit)
    [--out examples/strategies/<name>.json]   # default: stdout
```

## Architecture — `tools/strategy_gen/` (torch-free package)

Five small, independently-testable units. Subsumes and replaces `tools/strategy_oom.py`.

| Unit | File | Responsibility / interface |
|---|---|---|
| **Registry** | `registry.py` | `load_model(spec) -> ModelArch`. Built-in table keyed by type+size stores real arch (`hidden, layers, ffn, heads, kv_heads, num_experts, moe_ffn, topk, seq, vocab`) + `moe` flag + default launcher + default run-shape. `--config x.json` / `--hf <id>` override for arbitrary models. Seeded from the existing launcher arch tables (single source of arch for estimation). |
| **MemoryModel** | `memory.py` | Pluggable `est(arch, layout, run_shape) -> peak_gb` (per-GPU peak). v1 `HeuristicMemoryModel` (below). Future `PortedMegatronMemoryModel` / `HFEstimatorBackend` drop in behind the same interface. |
| **LayoutPlanner** | `planner.py` | `feasible(arch, world, hw, run_shape) -> [Layout]`: enumerate legal `(TP,PP,CP,EP,DP)` (world % (TP·PP·CP)==0; DP=world//(TP·PP·CP); MoE: `num_experts % EP == 0`, `EP<=world`, `ETP=1`; dense: `EP=1`), keep those with `est <= gpu_mem`, rank by objective (default: max DP, then min TP, then min PP). `best(...)` returns the top feasible; **raises loudly** if none fits even with cpu-adam (model too big for that world). |
| **Scenario** | `scenarios.py` | `sequence(arch, hw, planner) -> list[override_dict]`. Picks the base full-world layout (`overrides[0] == {}` by convention), then per-scenario follow-on layouts. Registered by name; adding a scenario = adding a function. |
| **CLI** | `cli.py` | Parse → load model → run scenario → write the sequence JSON, print the run recipe, and a summary table (per step: layout + estimated peak GB + cpu-adam needed?). |

Data flow: `cli` → `registry.load_model` → `scenarios[name].sequence(arch, hw, planner)`
where `planner` consults `memory.est`. Output = `(override_dict list, run recipe, table)`.

### MemoryModel v1 — heuristic (shape only; constants tuned in the plan)

Per-GPU peak ≈ **weights+grads** (`~6 B/param`, sharded by `TP·PP·EP`) + **optimizer**
(`0` if cpu-adam else `~12 B/param` sharded by `DP·CP`, the distributed-optimizer form)
+ **activation** (rough: `∝ mbs · seq · hidden · (layers/PP) / TP`, discounted by
recompute) + **overhead** (~10 GB). The activation term is new vs `strategy_oom.py`
(which ignored it and could green-light an activation-OOM layout); it is deliberately
crude and lives behind `est()` for later replacement by the ported analytic model.

## Scenarios (v1)

- **`rl_dp_resize`** (replaces `rl/*`): base full-world layout → a reduced-world layout
  for rollout (shrink DP, keep TP/PP/EP) → back to full. Pure capacity/DP change.
- **`verify_sweep`** (replaces `precision/*` + the main per-scale coverage sweeps):
  at fixed world, cycle through PP / CP / EP(MoE) / TP variants, each filtered to
  feasible by `est`. The bit-exact / correctness coverage pattern.
- Extensible: `scale_up` / `scale_down` etc. are future functions in the registry.

## Output (satisfies "fixed input → directly runnable")

1. **`seq.json`** — the override-dict list, consumed unchanged by `strategy_inject.py`
   via `ELASTIC_STRATEGY_LIST_FILE`.
2. **Run recipe** — the exact env + launcher line, directly runnable:
   `MODEL_SIZE=… / NUM_EXPERTS=… TP=… EP=… ELASTIC_ENABLED=1
   ELASTIC_STRATEGY_LIST_FILE=seq.json … ./run_{dense,qwen3_30b}.sh`.
3. **Summary table** — per step: resolved `(TP,PP,CP,EP,DP)` + estimated peak GB +
   whether cpu-adam is required.

## Migration

- `tools/strategy_gen/` **subsumes** `tools/strategy_oom.py`: its `MODEL_SCALES`
  registry moves into `registry.py`; its `validate_list` feasibility becomes the
  planner's feasibility check. `strategy_oom.py` + `tests/test_strategy_oom.py` are
  removed.
- Hand-written JSONs: **delete the bulk** (main per-scale ×6, `rl/` ×6, surplus
  `precision/`). **Keep the two paths the platform YAMLs reference** —
  `precision/dense_no_tp.json` + `moe_30b.json` — but their *content* becomes whatever the
  generator emits for that (model, scenario); it may differ from today's hand-written
  entries, as long as it still exercises the needed reshards (notably the 8→4 scale-down
  the DCP-fix verify relies on). The YAMLs keep pointing at them by path. A test asserts
  `committed fixture == fresh generator output` to prevent drift; the canonical (model,
  scenario, gpus, gpu-mem) inputs that produce each fixture are recorded next to it.
- `strategy_inject.py` (the runtime loader) is **unchanged** — the generator produces
  the files it already consumes.

## Testing (torch-free)

- `registry`: label resolution + `--config` override produce the expected `ModelArch`.
- `memory`: monotonicity (more sharding → less per-GPU) + a couple of known-value anchors.
- `planner`: feasibility filtering, divisibility, MoE rules (EP|experts, ETP=1), the
  no-feasible-layout loud failure, ranking objective.
- `scenarios`: each scenario's generated sequence matches a golden expectation for a
  fixed (model, hardware) input.
- `cli` end-to-end: generated `seq.json` parses through `strategy_inject.build_strategy_list`.
- drift: the committed `dense_no_tp.json` / `moe_30b.json` equal fresh generator output.

## Future (out of v1 scope)

- Replace `HeuristicMemoryModel` with a ported analytic model from the apache-2.0
  `ISEEKYAN/megatron_memory_estimator` (`moe_mem_estimator/`), or an optional HF backend
  when torch+megatron are present — both behind the existing `est()` interface.
- More scenarios (`scale_up`/`scale_down`, PP-rebalance).
- Unify the bash launchers' arch tables to read from `registry.py` (remove duplication).

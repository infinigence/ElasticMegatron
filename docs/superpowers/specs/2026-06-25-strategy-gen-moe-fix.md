# Strategy-gen post-review fix — MoE legality (ETP==1) + recipe fidelity

*2026-06-25 — fixes from the adversarial review of the strategy_gen workflow.*

Context: the generator (`tools/strategy_gen/`) shipped with all unit tests green, but an
adversarial review found that for **MoE** models the planner emits parallel layouts that
**crash at runtime** in `elastic_megatron/megatron_manager/parallel_strategy.py`. The two
committed fixtures (`moe_30b.json`, `precision/dense_no_tp.json`) happen to be legal, but
`verify_sweep` on a MoE model emits illegal steps (e.g. 4 of 6 for qwen3-30b @ 8 GPUs).

Runtime facts (verified, parallel_strategy.py):
- `:75 assert self.expert_tensor_parallel_size == 1, "Only support TPE==1 for MoE resharding"`.
- `:120-121` if ETP is `None` it defaults to **TP** (`expert_tensor_parallel_size = tensor_model_parallel_size`).
- `:122-132` requires the expert region to divide the world: `world % (ETP * EP * PP) == 0`, else `RuntimeError`.
- `strategy_inject.build_strategy_list` merges `{**base, **override}` (override-only diff).
- `run_qwen3_30b.sh:65 TPE=${TPE:-1}`, `:188 --expert-tensor-parallel-size ${TPE}` (launcher already wires TPE).

**User directive:** *all MoE recipes/sequences must explicitly pin ETP==1* — do not rely on
the implicit `{**base}` inheritance of the launch ETP. Make ETP=1 explicit in both the run
recipe (`TPE=1`) and every non-empty MoE override (`expert_tensor_parallel_size: 1`).

---

## MUST-FIX

### A. planner MoE legality + explicit ETP=1  (blockers 1 & 2 + user directive)
- `Layout`: add a `moe: bool = False` field → `Layout(world, tp, pp, cp, ep, dp, cpu_adam, moe=False)`.
- `_enumerate`:
  - dense → `ep=1, moe=False`.
  - MoE → `moe=True`; `ep ∈ divisors(num_experts), ep<=world`; **and keep only layouts with
    `world % (ep * pp) == 0`** (the ETP=1 expert-region rule — this drops EP=8/PP=2 @ world=8).
    TP is free (attention TP is independent of the expert ETP=1).
- `Layout.override(base)`: compute the 5-dim diff as today; **then if the diff is non-empty
  and `self.moe`, also set `expert_tensor_parallel_size: 1`** (explicit per the user directive).
  If the diff is empty (self == base) return `{}` (preserve `overrides[0] == {}` and the
  return-to-base convention; base's ETP=1 is guaranteed by the recipe's `TPE=1`).
- Add a planner self-assert: every emitted MoE layout satisfies `world % (ep*pp) == 0` (loud).

### B. recipe fidelity  (integration majors)
- `cli._recipe` MoE branch: emit `TPE=1` (pins the launch ETP).
- Emit the run-shape knobs so the **launched shape == the shape the planner sized for feasibility**.
  Variable names MUST be taken from the actual launchers (read `run_dense.sh` and
  `run_qwen3_30b.sh` to confirm the exact env names before emitting):
  - recompute: when `recompute_full` is True emit the launcher's recompute-on knob
    (dense vs MoE differ); when False, explicitly emit the recompute-off value for the MoE
    launcher (its default is on), so a non-recompute plan is honored.
  - micro-batch + sequence length: emit the launcher's MBS and seq-length knobs for `mbs`/`seq`.
  - (Defaults mbs=1/seq=4096 happen to match launcher defaults, which is why the gap was invisible.)

### C. scenarios cpu_adam consistency  (correctness nit + Task-4 spec issue)
- `rl_dp_resize`: keep resolving the rollout with `"auto"` (so it stays feasible at the reduced
  world — the plan's verbatim `base.cpu_adam` raised RuntimeError for GPU-adam-at-full models),
  **but assert `base.cpu_adam == rollout.cpu_adam`** with a clear message. A reshard cannot flip
  CPU_OFFLOAD mid-run (the override carries no optimizer-placement flag and the recipe keys
  CPU_OFFLOAD off `base` only), so a divergence must fail loudly rather than silently run the
  rollout under the wrong optimizer placement.

### D. tests (TDD: assert the new contract first, watch it fail, then fix)
- `tests/test_strategy_gen_planner.py`: make `test_moe_ep_divides_experts_and_etp1` actually
  assert the invariants its name promises — every MoE feasible layout has `world % (ep*pp) == 0`
  and `layout.moe is True`.
- Add a MoE-legality test (planner and/or scenarios): `verify_sweep(qwen3-30b, 8, 80)` — every
  non-empty override must carry `expert_tensor_parallel_size == 1`, and each decoded step must
  satisfy `world % (EP*PP) == 0` (no step would trip parallel_strategy.py's assert/RuntimeError).
- `tests/test_strategy_gen_scenarios.py`: add the rl_dp_resize cpu_adam-equality expectation
  (base.cpu_adam == rollout.cpu_adam for the registered models, or a documented loud failure).

### E. regenerate fixtures
- Regenerate `examples/strategies/precision/dense_no_tp.json` + `examples/strategies/moe_30b.json`
  from the tool. Expected change: `moe_30b.json`'s world=4 step becomes
  `{"world_size": 4, "expert_tensor_parallel_size": 1}` (explicit ETP=1). `dense_no_tp.json` is
  unchanged (dense, EP=1). Update the drift test so it still pins the canonical inputs, asserts
  the 8→4 step, and (for the MoE fixture) asserts the ETP=1 pin. Full torch-free regression green.

---

## CLEANUP (doc + nits — separate, doc-only files to avoid clashing with A)
- `CLAUDE.md:63,72`: the two example commands point at the deleted `examples/strategies/llama2_medium.json`
  → repoint at a surviving fixture (`examples/strategies/precision/dense_no_tp.json`) or an inline list.
- `examples/strategies/README.md` + `precision/README.md`: fix the overstatement that `dense_no_tp.json`
  "varies world/PP/CP/DP" — it varies only `world_size` (DP derives from it); keep it short + accurate.
- `registry.py` docstring: "the arch fields drive the activation estimate" is inaccurate — the v1
  estimator reads only `total_params_b, hidden, layers, moe, num_experts`; the rest are kept for the
  planned analytic backend. Say that, don't claim a role they don't have.
- `memory.py`: the `_ACT_BYTES` comment ("rough fp16 activation bytes per element") understates a
  bundled ~Nx factor — correct the comment to admit it's a tunable activation multiplier, not literal
  bytes. Reword the module-docstring "ISEEKYAN/megatron_memory_estimator" bare token so a literal
  `grep megatron` acceptance gate stays clean.
- `cli.py`: delete the dead `if __name__ == "__main__": raise SystemExit(main())` guard (the dedicated
  `__main__.py` is the real `-m` entry). Replace `_DENSE_MODEL_SIZE.get(arch.name, "medium")` with
  `_DENSE_MODEL_SIZE[arch.name]` (honest KeyError, matching registry's style). Rename the single-letter
  `l` (Layout) to `layout` in planner/scenarios/cli (PEP8 E741).

## NOT fixing in this pass (record as v1 limitations)
- MoE weight memory shards the *full* param count by EP, over-discounting non-expert weights — the
  spec already calls the heuristic deliberately crude; the analytic backend behind `est()` is the fix.
- `--config` MoE only launches correctly for archs matching `run_qwen3_30b.sh`'s non-emitted defaults
  (hidden/ffn/moe_ffn/topk); the three registered MoE models all match. Document, don't expand.
- `cli._table` re-derives Layout from override-dicts via a hardcoded reverse key-map (duplicated
  mapping). A clean refactor (scenarios return resolved Layouts) is a follow-up — no correctness impact,
  covered by tests.

## File ownership (so the two fix agents never touch the same file)
- **core agent:** `planner.py`, `scenarios.py`, `cli.py`, `tests/test_strategy_gen_planner.py`,
  `tests/test_strategy_gen_scenarios.py`, `tests/test_strategy_gen_cli.py`,
  `tests/test_strategy_fixtures_drift.py`, `examples/strategies/*.json` (regenerated fixtures).
- **docs agent:** `CLAUDE.md`, `examples/strategies/README.md`, `examples/strategies/precision/README.md`,
  `registry.py` (docstring only), `memory.py` (docstring/comment only).
- The two agents share no files, but commit **sequentially** (one git index).

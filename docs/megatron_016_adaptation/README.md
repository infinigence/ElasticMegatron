# Megatron-LM 0.16 adaptation — session record

May 2026 work that ported ElasticMegatron from Megatron-LM 0.11/0.13 to 0.16. Landed as a single squash commit on branch `squash-working`. Run `git log --oneline` in the repo root for the current commit hash.

## Summary

- **Coverage:** 22/22 reshard transitions × {model weight, distrib-optim state} verified bit-equal across TP / PP / CP / DP / EP dimensions at the ckpt level.
- **Long-run sanity:** 6 × 100-iter runs (medium 1.68B and 7B model sizes) all within noise floor of identical-config baselines.
- **Eval path:** post-fix, `--eval-iters > 0` works end-to-end (periodic eval + final validation + final test eval all green, no `setStorage size 0`).
- **Files touched:** ~40 — `+7500 / -200` lines (figures approximate after doc reorg).

## Files in this directory

| File | Read this for ... |
|---|---|
| [`changelog.md`](changelog.md) | Phase-by-phase change log (`what / why` per topic). Start here if you want a chronological picture. |
| [`phase_b_report.md`](phase_b_report.md) | Reviewer-facing summary of the Phase B sweep (8-GPU multi-dimensional verification): the 22/22 ckpt-level result, the iter-21 spike investigation, and the 7B re-test. |

## Reading order if you are diagnosing a 0.16-related issue

1. [`changelog.md`](changelog.md) — find the Phase that touched the file you suspect.
2. [`phase_b_report.md`](phase_b_report.md) — for the iter-21 spike pattern (noise floor + bf16 reduce-order), if you see suspicious loss `|Δ|` numbers.
3. [`../project/debugging.md`](../project/debugging.md) — the four reshard failure modes (A: `NoneType` optim tensor / B: `tp_attr` assert / C: NCCL hang / D: `setStorage size 0`). Most 0.16-era bugs were instances of these.
4. [`../project/invariants.md`](../project/invariants.md) — every rule was written with a concrete 0.16 failure in mind.

## Reading order if you are doing a 0.17 (or later) adaptation

1. [`../project/cross_repo.md`](../project/cross_repo.md) — the patching contract.
2. [`changelog.md`](changelog.md) → "Phase 0" / "Treat EP=1 expert as dense" — what was needed for the 0.16 API breaks (`initialize_model_parallel` kwargs, dataloader provider list-wrapping, TE `GroupedLinear` flipping `allreduce` at EP=1).
3. Brittle spots to re-check on a new version:
   - `resharding_dp.py::_FakeTensor` — does the new `_ParamAndGradBuffer.__init__` access any attributes the fake does not implement? (The `__getattr__` trip-wire will tell you loudly.)
   - `tools/ckpt/run_convert_patch_loader.py` — does the `margs.world_size =` anchor still match? (Look for the `[loader_patch] forced TP=…` print in `convert` output — its absence means the anchor moved.)
   - `tools/ckpt/compare_optim_logical.py` — does the FQN regex still match? (`compare_optim_logical.py` emits a stderr warning if 0 of N candidate keys match.)
4. The eval-time `setStorage` fix is a worked example of the *slice-assignment vs rebind* class of bug — if Megatron's `pretrain → train` data flow changes, the same pattern can recur. See [`../project/invariants.md`](../project/invariants.md) I-6.

## Key technical results worth remembering

- `expert_is_dense_bucketed()` (TE GroupedLinear flips `allreduce` when EP=1) — drove four sites in `resharding/`. See [`../project/invariants.md`](../project/invariants.md) I-1.
- The Phase 4 save-bug (passing `self.optimizers[0]` instead of `self.optimizer` to `save_checkpoint`) silently dropped expert distrib-optim state. Caught by `compare_optim_logical.py` because EP=2 ckpt's `flat buffer numel = 430M` vs EP=1 ckpt's `1.235B`. See [`../project/invariants.md`](../project/invariants.md) I-4.
- The Phase B EP-transition step-sync bug — `zip(src, dst)` truncates on length mismatch. The newly-created chained optimizer slots got `step = 0`, tripping mcore's `_synchronize_steps` assert on the *next* save. See [`../project/invariants.md`](../project/invariants.md) I-5.
- The iter-21 loss spike (`|Δ| ~ 0.3 – 1.5` in some `dense_mix_full` runs) is **expected behaviour** — bf16 reduce-order changes when DP topology flips. Verified by checkpointing at every reshard point: 9/9 bit-equal, but loss diverges post-reshard because micro-batch grouping into grad accumulation changed. See [`phase_b_report.md`](phase_b_report.md) §3.
- DCP-level offline verification (`compare_dcp.py` + `compare_optim_logical.py`) bypasses Megatron's partly-broken 0.16 convert chain. This is the verification path going forward.

## Open follow-ups (deferred from the May 2026 review)

These are technical follow-ups that came out of code review and are worth tracking but were not in scope for this session. None of them are blockers; some are quality-of-life, some are robustness investments for the next adaptation cycle.

**Resolved since initial review:**
- `Megatron-LM-custom` hardcoding removed from all scripts and tools. `run_e2e_demo.sh` / `run_moe.sh` now require `MEGATRON_PATH` to be set explicitly (`${MEGATRON_PATH:?...}`). `run_convert_patch_loader.py` reads `MEGATRON_PATH` first and emits a clear `RuntimeError` if no Megatron directory is found (previously `StopIteration` at import time).

- **`MODEL_PARAM_TO_OPT_PARAM_INDEX` is a module-level global** in [`elastic_megatron/resharding/resharding_metadata.py`](../../elastic_megatron/resharding/resharding_metadata.py) (lines ~188–205), reset by `init_model_to_optimizer_index_dict`. Functionally correct because `DistributedOptimizer`'s reshard path returns early without reading it, but it creates an implicit inter-process coupling. Long-term: promote to a class-attr on `TrainingState` or similar.
- **NCCL / Megatron timeouts are duplicated across runners.** [`run_e2e_demo.sh`](../../run_e2e_demo.sh#L13-L15) and [`run_moe.sh`](../../run_moe.sh#L15-L17) each carry `TORCH_NCCL_BLOCKING_WAIT` / `NCCL_TIMEOUT` / `TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC` and an inline `--distributed-timeout-minutes 1`. They are different semantic axes, so a single env-var collapse is not obviously correct, but the current split is easy to forget when bumping for production.
- **`compare_optim_logical.py` multi-GPU memory peak on very large models.** The current parallel path holds 3 kinds × 2 ckpts = 6 full-precision fp32 tensors on CPU simultaneously (~168 GB for 7B). Single-GPU path already streams. For 13B+ on machines without that much RAM, the parallel path should switch to load-on-demand per kind.
- **No CI fixture for the ckpt tools.** A tiny EP=2 / EP=1 fixture pair under `tools/ckpt/` plus a CI job running `verify_all.sh` would catch silent regressions in the comparators ahead of the next Megatron upgrade.
- **`TypedStorage` deprecation warning** in [`resharding_metadata.py:241`](../../elastic_megatron/resharding/resharding_metadata.py) — `.storage()` should become `.untyped_storage()`. Cosmetic, but will eventually be enforced.
- **`hetero_dp.py` has no 0.16 branch.** Currently raises `NotImplementedError` explicitly when called on Megatron 0.16+ — fine because Phase B never goes through `apply_hetero_dp`. If hetero-DP becomes required, this needs implementing.
- **Group-Zero switching at fixed world-size is unsupported.** `is_redundant_backup` requires `src_world_size != dst_world_size` — designed for scale up/down only. See [`../project/invariants.md`](../project/invariants.md) I-9.

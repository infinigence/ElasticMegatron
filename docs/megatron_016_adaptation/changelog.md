# Changelog

Organised by topic, in reverse-chronological order. Each entry describes one cohesive change with the "what" and the "why". For the *how-to-debug* counterpart, see [`../project/debugging.md`](../project/debugging.md).

---

## Eval-time `setStorage` / `do_test` bug (2026-05-18)

**Scope.** `Megatron-LM-custom/megatron/training/training.py`, `elastic_megatron/megatron_manager/dataloader_state.py`, `run_e2e_demo.sh` / `run_moe.sh` / `run_experiment.sh`.

The earlier Phase B configuration disabled eval (`--eval-iters 0`) in every runner so demos would finish quickly (see [`phase_b_report.md`](phase_b_report.md) §4a). This change makes eval work end-to-end when actually enabled.

### Bug A — final eval `setStorage size 0` (fixed)

[`Megatron-LM-custom/megatron/training/training.py`](../../../Megatron-LM-custom/megatron/training/training.py)

**Symptom.** With `--eval-iters > 0`, the post-train `evaluate_and_print_results` inside `pretrain()` triggers:

```
RuntimeError: setStorage: sizes [vocab/tp, hidden], strides [...], storage offset N,
and itemsize 2 requiring a storage size of M are out of bounds for storage of size 0
```

failing inside the embedding layer at `F.embedding(masked_input, self.weight)`.

**Root cause.** During reshard, `state_manager.reshard()` calls `release_model()` on src, which resizes the DDP `_ParamAndGradBuffer.param_data` storage to 0. The elastic loop inside `train()` did:

```python
model = training_state.model
```

This only rebinds the *local* variable in `train()` — `pretrain()`'s reference to the original `model` list still points at the very first model (which may be the src of some past reshard, with already-released buffers). `pretrain()`'s final `evaluate_and_print_results` uses that stale model, and the embedding view explodes.

**Fix.** Slice-assignment, mutating the list **in place**:

```python
model[:] = training_state.model
```

`pretrain()`'s reference now sees the same list, with its contents updated to the current dst model chunks (whose storage was repopulated by `update_model_weight()`).

### Bug B — `args.do_test=True` but `test_data_iterator=None` (fixed)

[`elastic_megatron/megatron_manager/dataloader_state.py::build_iterators`](../../elastic_megatron/megatron_manager/dataloader_state.py)

**Symptom.** With Bug A fixed, `--split 98,2,0` (no test portion) + `--eval-iters > 0` still hangs in the final test eval:

```
AssertionError at megatron/training/utils.py:532
    assert data_iterator is not None
```

**Root cause.** `build_iterators` unconditionally set:

```python
args.do_train = args.train_iters > 0
args.do_valid = args.eval_iters > 0
args.do_test  = args.eval_iters > 0
```

*before* constructing iterators. With no test portion in `--split`, mcore returns `None` for the test iterator. But `pretrain()` reads `args.do_test=True` at the end and calls `evaluate_and_print_results(test_data_iterator, ...)` — on `None`.

**Fix.** Mirror mcore's original semantics ([`training.py:3302-3316`](../../../Megatron-LM-custom/megatron/training/training.py)):

1. Construct iterators *first*.
2. Decide each `do_*` flag from `(iter is not None) and (corresponding_iter_count > 0)`.
3. `all_reduce(MAX)` across ranks — if any rank has an iterator, all ranks set the flag (since `evaluate_*` is collective).

### Bug C (not a bug, just a script default)

[`run_e2e_demo.sh`](../../run_e2e_demo.sh) / [`run_moe.sh`](../../run_moe.sh): `--eval-interval` / `--eval-iters` are now env-overridable (`${EVAL_INTERVAL:-...}` / `${EVAL_ITERS:-0}`), with defaults unchanged (0 = off). [`run_experiment.sh`](../../run_experiment.sh): `REAL_DATA_ARGS` accepts a caller override.

### Verification (2026-05-18)

| Scenario | `EVAL_ITERS` | split | exit | periodic eval | final valid | final test |
|---|---|---|---|---|---|---|
| dense_mix (4-GPU) | 2 | 98,2,0 | ✓ | ✓ | ✓ | (skip, no test) |
| dense_mix (4-GPU) | 2 | 90,5,5 | ✓ | ✓ | ✓ | ✓ |
| moe_mix (4-GPU) | 2 | 90,5,5 | ✓ | ✓ | ✓ | ✓ |

### Files touched

- `Megatron-LM-custom/megatron/training/training.py` — `model[:] = training_state.model`
- `elastic_megatron/megatron_manager/dataloader_state.py::build_iterators` — `do_*` derived from iterator state + `all_reduce(MAX)` sync
- `run_e2e_demo.sh` — `--eval-interval ${EVAL_INTERVAL:-10000} --eval-iters ${EVAL_ITERS:-0}`
- `run_moe.sh` — same
- `run_experiment.sh` — `REAL_DATA_ARGS` becomes `${REAL_DATA_ARGS:-default}`

---

## Phase B: 8-GPU multi-dimensional reshard sweep + ckpt-level optim verification (2026-05-15)

**Scope.** `Megatron-LM-custom/megatron/training/training.py` strategy modes; new entries in `run_experiment.sh`; new tool `tools/ckpt/compare_optim_logical.py`.

### Goal

Phase 4 (below) fixed the `save_checkpoint(..., optimizers[0], ...)` bug. Phase B's goal: cover **every reshard dimension ElasticMegatron supports** with both ckpt-level and loss-level verification, and complete the logical-level optim-state comparison tool.

### New strategy modes (`training.py::init_parallel_strategy_list`)

| Mode | Strategies | Purpose |
|---|---|---|
| `dense_mix_full` | base/TP=2/PP=2/CP=2/CP=4/TP=2_PP=2/TP=4/PP=4 (8) | Dense full-dimensional sweep incl. CP |
| `dense_cp_only` | CP=1/2/4/8 (4) | CP dimension isolation |
| `moe_mix_full` | base(EP=2)/EP=4/EP=8/CP=2_EP=2/TP=2_EP=2/PP=2_EP=2/TP=2_EP=4/TP=2_PP=2_EP=2 (8) | MoE full-dim sweep incl. CP |
| `moe_cp_only` | CP=1/2/4 (3, EP=2 fixed) | MoE × CP isolation |

**Excluded** (separate issue, not in Phase B scope):

- **Group-Zero** (`num_distributed_optimizer_instances > 1`). `is_redundant_backup` requires `world_size` to change (scale up/down); at fixed `world_size`, DGZ switching is not supported.

### Side-fix in Phase B: EP=1 ↔ EP>1 step synchronization

`training_state.py::update_optimizer_and_opt_param_scheduler` used `zip(src, dst)` over the two chained-optimizer lists. When the lengths differ, `zip` truncates:

- **EP=1 → EP=2.** `src = [dense_optim_with_expert]` (len 1), `dst = [dense_optim, expert_optim]` (len 2). The `zip` only pairs index 0 → **the new `dst[1]` (expert distrib_optimizer) never gets `step`**, defaulting to `step=0`. Inconsistent with `dst[0]`'s `step=N` → the next `save_checkpoint` trips mcore's `_synchronize_steps` assert (`assert len(steps) <= 1` over `{N, 0}`).

**Fix.** Position-based `zip` truncation only copies lr/betas etc. `step` is **explicitly pulled from any src chained's any param_group and broadcast to every dst chained's every non-empty param_group**. Called only on rank 0, but downstream `transfer_opt_param_scheduler` broadcasts the param_groups to other ranks, so the single-rank fix propagates correctly. After this fix, EP=1 ↔ EP>1 transitions re-entered the `moe_mix_full` sweep — 8/8 pass.

### New tool: `tools/ckpt/compare_optim_logical.py`

Compares the distrib-optim flat buffers of two `dist_ckpt`s as **logical multisets** — sort them and compare element-wise. Handles three layout differences:

1. **Padding zeros** (`gbuf_world_numel_unpadded` includes `BucketBuilder` intra-param padding). Filter out zero elements before comparing.
2. **Cross-`dp_group_idx` TP replicas** (layer-norm and other un-shardable 1D params each get a copy per TP rank at TP=k). Detection rule: same `(chained_prefix, gbuf_idx, bucket_idx, offset, size)` appears on multiple `dp_group_idx`'s **and** the actual tensor contents are bit-equal → treated as replicas; keep only the smallest `dp_g`. **Important:** under PP=k, different `dp_group`'s segments at the same offset have **different** content (different PP-stage fp32 data) — so the check must compare content, not just structure.
3. **DP padding tail** (every bucket end is padded to a DP-rank-integer multiple). Truncate via `per_bucket_numel_unpadded`.

### Performance

- `--device cuda` / `--devices 0,1,2`. GPU sort acceleration, with multi-GPU support (the three kinds — `param` / `exp_avg` / `exp_avg_sq` — one per GPU).
- Large-tensor sort uses chunking + multi-way merge. 1.68B-fp32 peaks at about 30 GB on a single GPU.
- CPU load → GPU sort → streaming CPU reduce (fp64), with fp32 as the default compute precision.

### New tool: `tools/ckpt/verify_all.sh`

Walks every `iter_*` subdirectory shared by `tools/ckpt/before_reshard/` and `tools/ckpt/after_reshard/`, running `compare_dcp.py` (weights) + `compare_optim_logical.py` (optim). Prints pass/fail counts; failure details land in `/tmp/verify_<iter>.log.{weight,optim}`.

### Results

#### Short (every transition, interval=1, all before/after ckpts saved)

| Mode | Transitions | Weight | Optim |
|---|---|---|---|
| `dense_mix_full` | 8 | **8/8 ✓** | **8/8 ✓** |
| `dense_cp_only` | 4 | **4/4 ✓** | **4/4 ✓** |
| `moe_mix_full` | 7 | **7/7 ✓** | **7/7 ✓** |
| `moe_cp_only` | 3 | **3/3 ✓** | **3/3 ✓** |
| **Total** | **22** | **22/22** | **22/22** |

Every `rel_rms < 1e-3` (most are exactly 0).

#### Long (each mode 100 iter, interval=10, vs corresponding baseline `|Δloss|`)

| Variant | P50 \|Δloss\| | P95 \|Δloss\| | max |
|---|---|---|---|
| dense_mix_full | 3.81e-2 | 6.39e-2 | 1.55 |
| dense_cp_only | 3.42e-2 | 1.56e-1 | 1.52 |
| moe_mix_full | 3.19e-2 | 8.20e-2 | 1.08e-1 |
| moe_cp_only | 7.82e-2 | 1.75e-1 | 2.26e-1 |

Reference: identical-config baseline noise floor `(P50=7.3e-3 / P95=3.8e-2 / max=1.14)`. Each variant's deviation is within the noise-floor order of magnitude; no reshard-induced systematic drift observed.

### 7B re-test (llama2-7B size, 32 layers)

On llama2-7B (`hidden=4096, layers=32, FFN=11008`) we repeated baseline vs `mix_full`. Requires `RECOMPUTE_FULL=1 RECOMPUTE_LAYERS=8` to run across the full `PP=1/2/4` range (`--recompute-num-layers` must be `≤ per-PP-stage layers`; `32/PP_max=4 → 8` works).

**Noise floor** (B1 vs B2, identical config):

| Model | P50 | P95 | max |
|---|---|---|---|
| medium 1.68B | 7.3e-3 | 3.8e-2 | 1.14 |
| **7B** | **9.5e-3** | **8.1e-2** | **0.38** |

7B max is significantly smaller (0.38 vs 1.14) — as the model grows, bf16 accumulation noise becomes small relative to the gradient magnitude.

**Reshard `|Δ|`** (two 100-iter `mix_full` runs vs B1):

| Run | P50 | P95 | max |
|---|---|---|---|
| M1 | 3.91e-2 | 1.15e-1 | 4.25e-1 |
| M2 | 4.92e-2 | 7.77e-2 | 3.96e-1 |

Same pattern as the medium model: post-reshard first-step `|Δ|` median around 4–5e-2, slightly above the noise-floor median but at the same order as the noise-floor max.

### iter-21 spike investigation (expected behaviour, not a bug)

`dense_mix_full` long-run sometimes had `|Δ|` jumping to `0.3 – 1.5` on the first step after a reshard (then settling back to `5e-2`). On medium 1.68B with 100 iter + `ELASTIC_SAVE_CKPT=1 INTERVAL=10`, every reshard point's before/after ckpt was saved and pairwise-verified:

- **All 9 reshard points pass weight + optim bit-equal** (9/9 + 9/9).
- Input dumps verified that post-reshard `mix_full` unique samples are a subset of the baseline's same-iter samples; sampler `consumed_samples` matches — no lost or duplicated samples.

**Where `|Δ|` comes from.** Reshard changes the DP topology → the per-DP-rank micro-batch count and order change → grad-accumulation order changes → bf16 accumulation differs → fp32 `main_param` after the next optimizer step is on a different trajectory. When the baseline itself has a grad spike at that iter, `mix_full` may "miss" the spike, making the single-point `|Δ|` look large.

**Conclusion.** Full ckpt-level bit-equality rules out any data corruption. The `|Δ|` is the expected product of "bf16 training is sensitive to reduce order" + "reshard necessarily changes reduce order". Fully eliminating it requires `--deterministic-mode` (perf cost 30–50%), inappropriate for dev.

---

## Phase 4: distrib_optimizer save-path bug fix + ckpt-level optim verification (2026-05-15)

**Scope.** `elastic_megatron/megatron_manager/training_state.py::TrainingState.save_checkpoint`.

### The bug

```python
optimizer = self.optimizers[0]   # WRONG: only the first chained sub-optimizer
save_checkpoint(..., optimizer, ...)
```

`self.optimizers` is the `ChainedOptimizer.chained_optimizers` list:

- **EP=1.** `list = [dense_optim_with_expert]`. `[0]` covers everything → save complete ✓
- **EP>1.** `list = [dense_optim, expert_optim]`. `[0]` is only the dense chain → **the entire expert side never lands on disk** ✗

### Symptom

The EP=2 ckpt's distrib_optim flat buffer total `numel = 430M` (dense only) while the model total `numel = 1.235B`. The EP=1 ckpt's flat is `1.235B` (correct). The gap equals the total expert parameter count.

### Fix

```python
save_checkpoint(..., self.optimizer, ...)   # the ChainedOptimizer itself
```

`save_checkpoint` handles `ChainedOptimizer` natively (it has `is_stub_optimizer` / `sharded_state_dict`).

### Impact

Before the fix, the in-memory optimizer state was complete (reshard copies momentum tensors directly), so **training loss looked fine** — but the saved ckpt was missing the expert side, breaking resume. Phase 2's long-run experiments converged stably (no resume), but the ckpts they produced were already broken.

### Verification

After the fix, `EP=2 → EP=1` reshard's logical optim-state comparison:

| | param | exp_avg | exp_avg_sq |
|---|---|---|---|
| numel | 1,235,324,928 | 1,235,324,928 | 1,235,324,928 |
| rel_rms (sorted) | **0.000e+00** | **0.000e+00** | **0.000e+00** |
| sum rel diff | 0.000e+00 | 2.6e-16 | 1.7e-16 |

Bit-exact.

---

## Phase 3: ckpt-level reshard correctness verification (2026-05-14 → 2026-05-15)

**Scope.** `tools/ckpt/`, `elastic_megatron/megatron_manager/training_state.py`, `Megatron-LM-custom/megatron/training/training.py` (the last is a Megatron-side hook; not in ElasticMegatron's repo, but must be applied to the target Megatron checkout).

### Background

The repo's existing `tools/ckpt/convert_and_compare.sh` is designed to use Megatron's `tools/checkpoint/convert.py` to convert a `before_reshard` ckpt to the target topology, then do a weight-by-weight comparison against `after_reshard` with tolerance `--thresh=1e-3` (rel_rms).

Megatron 0.16 has several incompatibilities:

- swiglu's sharded factory uses a closure → cannot be pickled into the legacy ckpt format.
- `loader_base.py` calls `model_provider()` missing the `model_builder` argument.
- `GPTModel.__init__` requires `pg_collection` by default; a single-process loader has no DP group.

We fixed the fixable parts (swiglu, loader patcher, path compatibility); the last is not easily reachable, so we **wrote** `compare_dcp.py` to compare the logical state_dicts of two `dist_ckpt`s directly, bypassing Megatron's convert chain.

### How to enable (runtime hook)

In `Megatron-LM-custom/megatron/training/training.py`'s `train()` loop, when the `ELASTIC_SAVE_CKPT=1` env var is set, pass `save_ckpt=True` into `elastic_megatron_manager.reshard()`. This triggers ckpt saves to `tools/ckpt/{before,after}_reshard/iter_<N>/` around every reshard.

### Changes

| File | Change |
|---|---|
| `elastic_megatron/megatron_manager/training_state.py::TrainingState.save_checkpoint` | Stop forcing `args.use_dist_ckpt=False` (0.16's swiglu closure factory cannot be pickled into legacy format); add `try/finally` to restore `args.save` — otherwise `pretrain()`'s closing block does `iteration % args.save_interval` and trips `TypeError` because `save_interval` is `None` |
| `tools/ckpt/run_convert_patch_loader.py` | Accept both `Megatron-LM-custom` and `Megatron-LM` paths; **on 0.16, patch `loader_base.py`** instead of `loader_core.py` (the `margs.world_size = ` line moved into base); copy `loader_core.py` as the plugin entry point; fix `loader_core.import_model_provider`'s `return model_provider` → `return self.model_provider` (0.16 dropped the `partial` wrap; `loader_base` calls `model_provider(pre_process=, post_process=)` and crashes on the missing `model_builder`) |
| `tools/ckpt/convert_and_compare.sh` | `MEGATRON_PATH` fallback for both names |
| `tools/ckpt/compare_dcp.py` *(new)* | Reads two `dist_ckpt`s via `dcp.load + FileSystemReader.read_metadata()`, obtains the logical tensor per key, compares by `rel_rms`. By default skips distrib-optim flat buffers (those are physically sharded per DP rank — different shapes across configs cannot be compared element-wise); `--include-optim-buffers` adds a coarse `sum/L2` sanity check |

### Results (4 reshard directions)

| Direction | Model | weight match | rel_rms |
|---|---|---|---|
| Dense TP=1/DP=4 → TP=2/DP=2 | medium 1.68B | **12/12** | **0.000e+00** (bitwise) |
| Dense TP=2/DP=2 → TP=1/DP=4 | medium 1.68B | **12/12** | **0.000e+00** (bitwise) |
| MoE EP=2/DP=2 → EP=1/DP=4 | small 0.97B | **10/10** | **0.000e+00** (bitwise) |
| MoE EP=1/DP=4 → EP=2/DP=2 | small 0.97B | **10/10** | **0.000e+00** (bitwise) |

Tolerance `1e-3`, actually 0 deviation across the board.

### What this phase did not cover (became Phase 4)

- **Distributed-optimizer fp32 `main_param` / `exp_avg` / `exp_avg_sq` alignment after reshard was not verified at the logical level.** A direct sum/L2 comparison of the flat buffers differed by ~18%, but that is an observation-bias issue (different buffer layouts), not real divergence. A decoder that reverses `model_param_group_index_map` to turn the flat buffer into a `{logical_param: (main, exp_avg, exp_avg_sq)}` dict is needed; that became Phase 4's task.
- Long-run 100-iter loss within noise floor is an indirect indicator that fp32 optim state is probably correct, but not direct ckpt-level evidence.

---

## Phase 2: long-run experiment framework + noise-floor quantification (2026-05-14)

**Scope.** `tools/{analyze_experiments,noise_floor}.py`, `run_experiment.sh`, `run_e2e_demo.sh`, `run_moe.sh`.

### Key finding: training itself is non-deterministic

Running two dense-baseline jobs with **identical config and seed** (TP=1/DP=4, 100 iter, medium model `hidden=4096 layers=8`), the pairwise `|Δloss|` distribution is:

| Statistic | \|Δloss\| |
|---|---|
| P50 | 7.3e-3 |
| P95 | 3.8e-2 |
| P99 | 4.2e-1 |
| max | 1.14 |

Iter 1, 2 match exactly (seed init agrees); from iter 3 bf16 rounding accumulates (`1e-6 → 1e-3`); after iter 11, grad spikes diverge the trajectories into `1e-1 ~ 1e0`.

Root cause: NCCL collective ordering non-determinism + cuBLAS heuristics + flash-attn kernel non-determinism. Eliminating it entirely requires `--deterministic-mode` (drop flash-attn, big perf hit).

### Framework

- **`run_experiment.sh`** *(new)* — six mode runners:
  - `dense_baseline_tp1` (TP=1/DP=4) / `dense_baseline_tp2` (TP=2/DP=2)
  - `moe_baseline_ep2` / `moe_baseline_ep1`
  - `dense_mix` (tp_flip) / `moe_mix` (ep_flip)
  - 4 GPUs per run by default; two groups can run in parallel (GPU 0-3 + 4-7, master_port 6000/6001).
- **`run_e2e_demo.sh` / `run_moe.sh`** refactor:
  - New `MODEL_SIZE=medium` (`hidden=4096, ffn=11008, layers=8/4, GQA=8`).
  - `GBS / MBS / LR / INIT_STD / MAX_SEQ_LEN / NUM_LAYERS / FFN_HIDDEN_SIZE` all configurable.
  - MoE defaults to `NUM_EXPERTS=4` (so EP=1's 4 experts fit on one card).
  - End-of-train eval disabled (`--eval-iters 0`) to work around the known `setStorage size 0` issue.
  - MoE adds `--seed 1234` to align baseline vs elastic starting points.

### Loss comparison tools

| Tool | Purpose |
|---|---|
| `tools/analyze_experiments.py` | Compare baseline vs elastic; precision-bucket distribution (`<1e-6, <1e-5, <1e-4, ≥1e-4`); reshard cost |
| `tools/noise_floor.py` | Pairwise `|Δloss|` across multiple baseline runs; P50/P95/P99 |

### Conclusions

- Two identical baseline runs' loss noise floor: **P50=7.3e-3, P95=3.8e-2**.
- Dense elastic vs baseline: P50=1.66e-2, max=1.14 (within noise floor).
- MoE elastic vs baseline: P50=2.62e-2, max=5.95e-2 (within noise floor, even tighter).

**Reshard introduces no deviation beyond the noise floor**, but this is only a loss-level indicator. Phase 3's ckpt-level zero-deviation is the direct evidence of reshard correctness.

### Performance data (2026-05-14, 100 iter `dense_mix` interval=10)

- **First reshard transfer: 1251 ms / 5.47 GB / 4.4 GB/s** (NCCL group first-time negotiation).
- **Subsequent 8 reshard transfers: 46-56 ms / 5.47 GB / 97-118 GB/s** (saturating NVLink).

**Matches the expected "only the first lazy init is slow, the rest are stable" pattern. No additional synchronisation barriers were introduced by the 0.16 adaptation** (audited every new piece of code in `transfer.py` / `dist_patch.py` / `p2p_to_collective.py` / `update_global_args` / `experts_are_dense_bucketed` — none contains a `barrier / synchronize / collective`).

### Known carry-overs (not fixed in this phase)

- Post-train eval path `setStorage size 0` (`rerun_state_machine` views weights that were `resize_(0)`'d by `release_optimizer`). Workaround: `--eval-iters 0`. Fixed in the May 2026-05-18 follow-up.
- `hetero_dp.py` has `<=11 / ==13 / else raise` branching with no 0.16 path.

---

## Treat EP=1 expert as dense across every reshard path (2026-05-13)

**Scope.** `elastic_megatron/resharding/` + `elastic_megatron/megatron_manager/parallel_strategy.py`.

### Background

In Megatron 0.16, TransformerEngine's `GroupedLinear` / `Linear` sets `expert_parallel=False` (i.e., `expert_model_parallel_size == 1`) → flips the expert params' `allreduce` back to `True`. Two downstream layers then treat experts as **dense**:

1. `DistributedDataParallel` buckets by `param.allreduce` — experts land in the *dense* `_ParamAndGradBuffer`, not in a separate MoE bucket.
2. `DistributedOptimizer`'s shard layout follows the dense bucket's `(tp, dp)` grid; the rank space falls back to using `tensor_model_parallel_size` (under `order="tp-cp-ep-dp-pp"`, dense and expert share one rank space) as the TP coefficient, not `expert_tensor_parallel_size`.

ElasticMegatron originally assumed experts always live on the `(etp, edp, ep)` grid. That assumption is correct at EP>1, but misaligned with Megatron's actual behaviour at EP=1, triggering two symptoms:

- **Symptom A** (`NoneType.create_padded_optimizer_tensor`): the simulated `dst_aligned_global_rank` points to the wrong rank; that rank's `DistributedOptimizer.model_param_group_index_map` does not contain this param; `get_main_weight` returns `None`. `moe_mix`'s 3rd reshard (`EP=8 → EP=1`) triggers it.
- **Symptom C** (NCCL timeout): in the `TP=2, ETP=1, EP=1` combination, `get_global_rank`'s computed aligned receiver does not match Megatron's actual shard owner. The sender's broadcast has no corresponding receive. `moe_mix`'s 4th reshard (`EP=1 → TP=2/EP=1`) triggers it — this is the first time `TP ≠ ETP` in the sweep (the first three reshards all stayed at `TP=1`, hiding the bug).

Both symptoms share one root cause: ElasticMegatron did not thread "EP=1 means expert is dense" through the simulation layer.

### Changes

**Unified entry point.** Introduce a semantic helper on `ParallelStrategy`:

```python
def expert_is_dense_bucketed(self) -> bool:
    """True iff expert params live in the dense DDP bucket under this strategy."""
    return self.expert_model_parallel_size == 1
```

All four decision points in the resharding layer call this helper rather than re-implementing the `== 1` check:

| Site | Before | After |
|---|---|---|
| `resharding_dp.py::get_params_dp_distribution` | Bucketed by `is_expert` | `expert_is_dense_bucketed()` → experts join `dense_params`, matching Megatron's actual DDP buckets |
| `resharding_dp.py::DataParallelReshardingInfo._init_dp_distribution_with_global_rank` | `tp_rank` / `ep_rank` always derived from `is_expert` | `expert_as_dense` branch: `tp_rank` follows dense, `ep_rank=None` |
| `resharding.py::ReshardPlan.__post_init__` send/recv | Always passed `self.src_ep_rank` / `dst_ep_rank` | Pre-compute `_src_ep_rank_for_gr` / `_dst_ep_rank_for_gr`; `None` on the EP=1 side; `_generate_global_send/recv_info` no longer recomputes |
| `virtual_param.py::VirtualParam._generate_reshard_plan` | Used TPE as TP axis | `expert_is_dense_bucketed()` → use dense `tp_size`, so `_resharding_without_split` fans out correctly to every dst TP rank holding a replica |

**Relax `ReshardPlan.__post_init__`'s single-entry assert:** only require expert TP send/recv plans to have exactly one entry each when *both* sides have EP>1; EP=1-as-dense at `tp_size > 1` legitimately produces multiple entries.

**`parallel_strategy.py::update_global_args` syncs `sequence_parallel`:** otherwise reshard to `TP>1` causes Megatron's MoE forward to `raise ValueError("MoE and tensor parallelism require sequence parallelism")`. The strategy's `sequence_parallel` was already coupled to TP in `__post_init__`, but if we don't push it back into Megatron's global `args`, only the Strategy knows.

**Demo-script defaults:** `run_e2e_demo.sh` / `run_moe.sh` default to `Megatron-LM-custom`, 8 GPUs, 60-second NCCL timeout (so hangs surface quickly; bump before production).

### Verification

- `dense_mix` on 8 GPUs, 18 iter, 8 reshards: all pass.
- `moe_mix` on 8 GPUs, 18 iter, 8 reshards: all pass. The 4th (`EP=1 → TP=2/EP=1`) went from hang to a successful ~11 ms transfer. Loss decreases monotonically `10.38 → 10.04`.
- Dense 2-GPU `tp_flip`: no regression.

### What this change does not do

- **Does not modify the Megatron side** (the `allreduce` flip in TE `GroupedLinear`). That is Megatron's contract; ElasticMegatron's job is to align with it.
- **Does not handle `hetero_dp.py`'s version branching** (`<=11 / ==13 / else raise`) — no 0.16 branch. Current tests do not exercise this path, but it remains a future hazard; deferred until something triggers it.

---

## Add reshard debug playbook (2026-05-13)

**File.** `docs/project/debugging.md` (previously `docs/DEBUGGING.md`).

Consolidates the four common reshard failure modes (A: `NoneType` optimizer tensor / B: `tp_attr` assert / C: NCCL timeout / D: `setStorage size 0`) — symptoms, hypotheses, probe sites, env-var cheat sheet, resolved-bug postmortems. The unresolved runtime bucket-reader experiment (the 2026-05-13 session) is also recorded as a negative result — it redirected attention from the bucket-slicing layer to the actual `global_rank` mapping layer.

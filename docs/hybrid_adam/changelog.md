# hybrid-adam changelog

Reverse-chronological. Each entry is one coherent change.

---

## R.1 — OptimizerAdapter: consolidate per-optimizer branches behind one seam (2026-06-01, refactor)

Pure refactor (no behaviour change), motivated by FP8 being the next optimizer to support.
After hybrid-adam landed, the "which optimizer implementation is this, and how do I read/write
its state" logic was scattered across `resharding_metadata.py` (`_is_hybrid_device_optimizer`,
`get_main_weight`, `init_empty_state_dict`, `ordered_optimizer_state_keys`, and the big
HDO/precision-aware/distrib branch in `get_optimizer_tensors_by_model_weight`) and
`training_state.py` (`_precision_aware_copy_main_to_model` + the precision-aware branch in
`update_model_weight`). Each new optimizer meant touching several `isinstance`/class-name/config
checks.

These now live behind **`OptimizerAdapter`** (new `resharding/optimizer_adapter.py`), with one
dispatch point `OptimizerAdapter.create()` and a subclass per implementation difference:

```
OptimizerAdapter
├── Float16OptimizerAdapter
└── DistributedOptimizerAdapter
    └── PrecisionAwareOptimizerAdapter
        └── HybridDeviceOptimizerAdapter   # dummy_step init
```

Interface: `get_main_weight` / `model_param_sub_range` / `ensure_state_initialized` /
`discover_states` / `copy_main_to_model`. `get_optimizer_tensors_by_model_weight()` is now a thin
assembly over the adapter; `update_model_weight()` calls `adapter.copy_main_to_model()`. `OptState`
moved into the adapter module (breaks the import cycle) and is re-exported from
`resharding_metadata`. The `MODEL_PARAM_TO_OPT_PARAM_INDEX` module-global (float16 index cache) is
gone — replaced by a per-adapter lazy map + a `WeakKeyDictionary` adapter cache. The DDP buffer
sync mechanics (`param_gather_handle` / `cached_param_buffer_shard_list` reset + forced
`start_param_sync`) stay in `training_state` — they belong to DDP, not the optimizer.

**FP8 seam:** an FP8 adapter overrides `discover_states` (non-param-shaped scale/amax still needs a
separate transport — I-15) + a `create()` branch; no other call site changes.

Verified: `python -c import` (no cycle) + offline unit checks (state-key ordering parity, factory
dispatch + memoization, class-name HDO detection, Float16 discovery order). Method bodies are
verbatim moves of the prior branches, so the bit-equal GPU regression net (8-GPU full sweep, with
and without CPU-adam, weight + optim) is expected to stay green — **to be re-run when GPU frees up.**

---

## H.8 — 8-GPU full sweep, both with and without CPU-adam (2026-05-29, GPU-verified)

Expanded coverage to a single **8-GPU** `dense_mix_full` run (full TP / PP / CP / DP / Group-Zero
sweep, 24 iters, 7 reshards in both directions), under both conditions:

| condition | result |
|---|---|
| **without** CPU-adam (`CPU_OFFLOAD` unset) | 7 reshards, no hang, **weight 7/7 + optim 7/7 ALL PASS** |
| **with** CPU-adam (`CPU_OFFLOAD=1 OFFLOAD_FRACTION=0.5`) | 7 reshards, no hang, **weight 7/7 + optim 7/7 ALL PASS** (6 pairs exercised the sub-multiset containment path for non-zero PA padding) |

Both run with `RERUN_MODE=disabled` (resharding breaks the rerun result-validation's bit-determinism
assumption). This covers PP / CP / Group-Zero reshard dimensions with CPU offload, not just the
4-GPU TP/DP flip. **Conclusion: reshard works in both conditions across the full dimension sweep.**

**Cleanup (post-verification).** The temporary debug instrumentation used during H.3–H.6 —
`EM_DEBUG_REBUILD` (per-rank release/rebuild/sync + instance-group prints) and `EM_DISABLE_PA_COPY`
(skip the H.4 copy for isolation) — has been removed from `training_state.py`. The functional fixes
(precision-aware master→model copy, rebuild size fallback, `param_gather_handle` +
`cached_param_buffer_shard_list` reset) are kept. The mentions of these env switches in the H.3/H.5
entries below are historical.

---

## H.7 — TP2→TP1 hang FIXED (stale shard-cache); optim "fail" is a comparator padding artifact (2026-05-29)

**Hang fixed (GPU-verified).** The H.6 `cached_param_buffer_shard_list` reset resolves the
deadlock: `CPU_OFFLOAD=1 OFFLOAD_FRACTION=0.5`, 12 iters, **3 reshards (TP1→TP2→TP1→TP2,
incl. the previously-hanging reverse one) complete with no hang**. So the root cause of the
TP2→TP1 deadlock was the re-entered chunk's stale `cached_param_buffer_shard_list` (shard views
into the pre-release param_data storage); `start_param_sync` rebuilds them now. The
`param_gather_handle` reset (H.5) is also kept.

**Verification (`verify_all.py` on the 3 pairs):** weight **3/3 bit-equal** (both directions,
incl. reverse), optim **2/3** — only the reverse pair (iter-6) "fails".

**The iter-6 optim "fail" is a comparator false-positive, not a reshard bug.** Breakdown:
- dst (TP1/DP4) flat buffer = 21,635,584 = exact real param count (nothing missing).
- src (TP2/DP2) = 2 dp_group buckets, drop-zeros total 21,640,192 = real + **4608 non-zero elements**.
- These 4608 are distributed-optimizer **intra-param padding**; `compare_optim_logical` assumed
  padding is zero (drop_zeros), but under `--use-precision-aware-optimizer` a *trained* source's
  padding is **non-zero**, so it isn't dropped → false NUMEL MISMATCH. Asymmetry confirms it:
  TP1→TP2 (dst = freshly-built TP2, zero padding) passes; TP2→TP1 (src = trained TP2, non-zero
  padding) "fails". Weights are bit-equal both ways, so the reshard is correct.

**Comparator fix.** `compare_optim_logical.compare_sorted`: on unequal numel, instead of a hard
FAIL, run a **sub-multiset containment check** (`_submultiset_check`, exact unique+counts via
searchsorted on the sorted tensors) — if the smaller side is fully contained in the larger, the
logical moment state is preserved and the residual is layout-dependent padding → PASS (with a
note); otherwise real values were lost → FAIL. CPU unit-tested for padding-residual / real-loss /
equal / count-violation cases. **Does not mask real bugs** (real value changes are not contained → FAIL).

**GPU re-verify — DONE.** `verify_all.py` on the 3 pairs → **weight 3/3 + optim 3/3 ALL PASS**.
iter-6 (reverse) param/exp_avg/exp_avg_sq each: "smaller multiset fully contained in larger,
residual=4608 (layout-dependent padding), logical state preserved". Confirms the reverse reshard's
moment state is correct and the earlier optim fail was the padding artifact.

**→ hybrid-adam reshard is functionally verified**: `CPU_OFFLOAD=1` multi-reshard (both directions),
no hang, weights bit-equal, optimizer moment multiset preserved.

---

## H.6 — TP2→TP1 hang: deeper analysis + stale-cache fix (2026-05-29, code-only, pending GPU verify)

H.5 (clearing `param_gather_handle`) bypassed the early-return `.wait()` but the reshard still
hangs at `start_param_sync` for the re-entered TP1 dst under precision-aware (offload 0.5 **and**
0.0; the H.4 master→model copy completes on all ranks first).

**Offline NCCL-trace analysis (per-rank logs).**
- All NCCL comms reach `Init COMPLETE` on every rank — so it is **not** a comm-init hang.
- **Comm counts differ across ranks** (rank0 built 4, rank2 built 5) — an asymmetry, though some of
  it is expected (each rank only joins the 2-rank p2p groups it belongs to, created by
  `create_p2p_collective_groups` during transfer).
- The iter-6 DP all-gather is **never issued to NCCL** (last AllGather logged is the iter-3 DP=2
  one); EM-SYNC prints BEFORE without AFTER and the faulthandler sits in
  `_coalescing_manager` → so `start_param_sync` is stuck **Python-side, before launching the
  collective**. The 2-rank {0,2}/{1,3} comms in the trace are the transfer's p2p groups, not the sync.
- `nccl_group_recreate` is never called in the reshard path, so the dst DP=4 comm from init is not
  torn down. The open question is whether the re-entered chunk's
  `intra_distributed_optimizer_instance_group` is consistent (DP=4 {0,1,2,3}) across all ranks at
  iter-6 — the prior EM-SYNC print read the wrong attribute (`data_parallel_group` → None).

**This change.**
1. **Stale-cache fix.** Alongside the `param_gather_handle` reset, also reset each re-entered bucket
   group's `cached_param_buffer_shard_list` to `[None]*N`. Those entries cache shard *views* into the
   param_data storage from the chunk's previous active period; `release_model`/`rebuild_model` change
   the storage, leaving the views stale. Forcing a rebuild makes `start_param_sync` re-shard the
   current `param_data`. (Real correctness bug on re-entry; may or may not be the deadlock cause.)
2. **Conclusive instrumentation.** `EM_DEBUG_REBUILD=1` now prints, per rank, each bucket group's
   `intra_distributed_optimizer_instance_group` rank set + bucket-group count before the sync. One GPU
   run will show definitively whether the group is a consistent DP=4 {0,1,2,3} on all ranks or a stale
   2-rank set — pinpointing stale-group vs coalescing-desync.

**Next GPU run:** `CPU_OFFLOAD=1 EM_DEBUG_REBUILD=1` to iter-6; read `instance_group_ranks` per rank.

---

## H.5 — fix the TP2→TP1 start_param_sync deadlock (2026-05-29, code-only, pending GPU verify)

**Scope.** `elastic_megatron/megatron_manager/training_state.py::update_model_weight`.

**Symptom.** After H.4 the forward reshard (TP1→TP2) is bit-equal, but the **reverse** reshard
(TP2→TP1, dst DP grows 2→4) hangs in `start_param_sync` under precision-aware (with offload 0.5
*and* 0.0; not the CPU path, not the H.4 copy — that completes on all ranks first).

**Root cause (from NCCL COLL trace).** All 4 ranks issue an identical 2422 AllGathers and the last
one is the iter-3 (DP=2) sync — **iter-6's all-gather is never issued by any rank**, yet EM-SYNC
prints BEFORE without AFTER (stuck inside, faulthandler → `_coalescing_manager`). Cause:
`ParamAndGradBucketGroup.start_param_sync(force_sync=True)` early-returns via
`if self.param_gather_handle is not None: self.param_gather_handle.wait(); return`
(`param_and_grad_buffer.py:243-247`). The iter-6 dst (TP1) is a **re-entered cached model chunk**
(used at iters 1-3, released at the iter-3 reshard). Under precision-aware it carries a **stale
pending `param_gather_handle`** from before that release; `force_sync` `.wait()`s on it and
deadlocks — the matching collective was abandoned when the reshard rebuilt the buffers/groups.
The non-precision path leaves the handle None, falls through, and issues a fresh all-gather (works).

**Fix.** Before the forced sync in `update_model_weight`, reset
`bucket_group.param_gather_handle = None` on every model chunk's bucket groups (dense + expert).
All ranks then deterministically fall through and issue a fresh, matched all-gather. No-op for the
non-precision-aware path (handle already None), so F1 is untouched.

**Status: pending GPU verify** — re-run `CPU_OFFLOAD=1` with ≥2 reshards (TP1→TP2→TP1) and confirm
both directions complete + `verify_all.py` passes weight+optim on each pair.

Debug aids retained (default off): `EM_DEBUG_REBUILD=1` (release/rebuild + sync per-rank prints),
`EM_DISABLE_PA_COPY=1` (skip the H.4 master→model copy for isolation).

---

## H.4 — fix the model-weight refill; forward reshard now bit-equal (2026-05-29)

**Scope.** `elastic_megatron/megatron_manager/training_state.py`, `run_e2e_demo.sh`.

Implements the §6 fix from [`megatron_hybrid_optimizer.md`](megatron_hybrid_optimizer.md):

- **`update_model_weight` now does an explicit master→model copy under precision-aware.** New
  helper `_precision_aware_copy_main_to_model(dist_optimizer)`: for every model param, pulls the
  fp32 master from `_get_main_param_and_optimizer_states(p)["param"]` and writes it into the model
  `param_data` buffer at the param's `gbuf_world_in_bucket` range (cast to the buffer's dtype/device;
  CPU-offloaded masters are moved to GPU). This replaces the reliance on
  `DistributedOptimizer._copy_main_params_to_model_params`, which no-ops under precision-aware. It
  uniformly covers the float16 body and the `shard_fp32` (LayerNorm/bias/output) group.
- **`run_e2e_demo.sh`: `RERUN_MODE` env** → `--rerun-mode`. Megatron's default `validate_results`
  re-runs steps to check bit-determinism, which resharding inherently breaks; set
  `RERUN_MODE=disabled` for elastic sweeps.

**Verified.** 4-GPU `CPU_OFFLOAD=1 OFFLOAD_FRACTION=0.5` (HDO + precision-aware), iter-3
TP1→TP2 reshard: `verify_all.py` → **weight 1/1 bit-equal (rel_rms=0) + optim 1/1**. Before this
fix the `shard_fp32` params were 1e34 garbage. Also confirmed: with correct weights the post-reshard
step is no longer anomalous, so the `rerun_state_machine` no longer trips at iter 3.

**New open issue (separate).** The **reverse** reshard (TP2→TP1) under offload **hangs in
`start_param_sync`** (clean NCCL hang, killed by the dev 60 s timeout) — even with
`RERUN_MODE=disabled`. The forward (TP1→TP2) reshard is fine, and TP2→TP1 works *without* offload,
so this is an offload-specific reverse-reshard param-sync problem (likely a union/DP-group rank-set
mismatch — triage via debugging.md Symptom C). Not yet fixed.

---

## H.3 — smoke run: reshard completes, model-weight refill is the remaining bug (2026-05-29)

First `CPU_OFFLOAD=1` reshard that runs end-to-end (after H.1 + H.2 + the `rebuild_model` size
fallback). Findings from the iter-3 before/after ckpt pair (`verify_all.py`):

- **Transfer + optimizer state: correct.** `optim` multiset matches (✓). The reshard plumbing
  (master extraction via `_get_main_param_and_optimizer_states`, CPU-state transport, dummy_step
  init) works under precision-aware/HDO.
- **Model weights: wrong for the `shard_fp32_groups` params.** `weight` compare fails on
  `decoder.final_layernorm.{weight,bias}`, `*.layer_norm_{weight,bias}`, `output_layer.weight`
  (1e34-scale garbage). With `OFFLOAD_FRACTION=0` (no CPU offload, still precision-aware) *more*
  params are garbage → **the bug is precision-aware-intrinsic, not the CPU-offload device path**.
- **Root cause.** `DistributedOptimizer._copy_main_params_to_model_params()` is a **no-op** under
  precision-aware (`distrib_optimizer.py:2431-2432`; the master→model copy normally happens inside
  `optimizer.step()`). ElasticMegatron's `update_model_weight()` calls it expecting a real copy, so
  the model `param_data` is never refilled from the transferred master; `rebuild_model`'s size
  fallback leaves it as uninitialized memory. `optim` multiset is a *global* set check, so it does
  not catch per-param model-weight corruption.
- **Also seen (separate).** `rerun_state_machine` requests "checkpoint + exit to diagnose" on the
  post-reshard step (resharding breaks bit-determinism) and crashes on `args.save=None`. A
  Megatron-diagnostic incompatibility with resharding, not a hybrid bug.

Full analysis + the fix design (explicit master→`param_data` copy under precision-aware) is in
[`megatron_hybrid_optimizer.md`](megatron_hybrid_optimizer.md) §6. **Not yet fixed.**

Debug aid: `EM_DEBUG_REBUILD=1` prints per-buffer release/rebuild storage state in
`training_state.py` (default off).

---

## H.2 — precision-aware state extraction (2026-05-29, code-only, UNVERIFIED)

**Diagnosis of the H smoke crash.** First `CPU_OFFLOAD=1` reshard smoke (4-GPU dense
tp_flip): training iters 1–3 fine, then the first reshard crashed with async
`CUDA error: invalid argument` at the transfer's self-send `tensor.clone()`
(`communicator.py:76`). Root cause is **not** the transport — it is that
`--use-precision-aware-optimizer` (which HDO *requires*) changes `DistributedOptimizer`'s
layout:

- `optimizer.param_groups[g]["params"][o]` returns the **bf16/fp16 model shard**, not the
  fp32 master (`distrib_optimizer.py:442-446`).
- The fp32 master lives in optimizer **state** (`state[shard]["master_param"]`), and moments
  may be scaled — the canonical read is `DistributedOptimizer._get_main_param_and_optimizer_states(model_param)`
  → `{"param": fp32 master, "exp_avg", "exp_avg_sq"}` (`distrib_optimizer.py:874-900`).

So `get_main_weight` was handing the transfer a **bf16 shard of a FusedAdam-managed fused
buffer** as if it were the fp32 master; viewing/cloning/sending it tripped the CUDA error.
The H.1 `master_param` exclusion was also wrong under this path (master_param *is* the master).

**Fix.** In `get_optimizer_tensors_by_model_weight`, when `use_precision_aware_optimizer`
is set, build the state set from `optimizer._get_main_param_and_optimizer_states(model_weight)`
instead of `param_groups` + raw `state`. For the **HDO** branch of that accessor the returned
tensors are the real state tensors (references), so ElasticMegatron's in-place reshard recv
still updates the optimizer. The verified non-precision F1/Adam path is untouched (gated).

**Status: UNVERIFIED** — needs the `CPU_OFFLOAD=1` smoke re-run when a GPU is free. Open risks:
- non-HDO precision-aware returns unscaled *copies* (would need `set_scaled_state` on write-back) — not handled; only HDO targeted.
- `store_param_remainders` / `main_params_dtype != fp32` variants not exercised.
- `update_model_weight` (`_copy_main_params_to_model_params`) under precision-aware is Megatron's path; assumed to work, unconfirmed.

---

## H.1 — HybridDeviceOptimizer integration (2026-05-29, code-only, UNVERIFIED)

**Scope.** `elastic_megatron/resharding/resharding_metadata.py`, `run_e2e_demo.sh`, `run_moe.sh`.

Target: Megatron's `HybridDeviceOptimizer` (HDO,
`megatron/core/optimizer/cpu_offloading/hybrid_optimizer.py`). HDO is a
`torch.optim.Optimizer` **wrapped by** `DistributedOptimizer`; it splits params per-param
(by `offload_fraction` × GPU numel) between a CPU sub-optimizer (pinned state) and a GPU
sub-optimizer, and its `.state` is a synced view over the two. The fp32 master stays on GPU
(the DO shard); the **moments of offloaded params live on CPU** — exactly the per-state
device heterogeneity F1+F2 were built for.

**Changes (all in `get_optimizer_tensors_by_model_weight` / helpers):**

- **Exclude `master_param` from discovered states.** HDO is built with
  `param_update_in_fp32=True`, so `state[param]["master_param"]` holds the fp32 master — it
  *is* `states[0]`, not a separate state. Added to `_NON_TRANSFER_STATE_KEYS` so discovery
  does not emit/transfer it twice.
- **`_is_hybrid_device_optimizer()`** — class-name-based detection (no hard import on the
  `cpu_offloading` module, so older Megatron still imports).
- **DST not-initialized path uses `dummy_step()` for HDO** instead of `init_empty_state_dict`
  / `init_state_fn` (which is `None` for HDO and would not connect to the sub-optimizers).
  `dummy_step()` allocates the real moments on each sub-optimizer's own device; on the dst
  both the values and the random-grad perturbation are overwritten by the reshard transfer.
  Only the master is offloaded (resized to 0); moments stay allocated and are received in place.
- **Launchers:** `CPU_OFFLOAD=1` (+ `OFFLOAD_FRACTION`) adds `--optimizer-cpu-offload
  --optimizer-offload-fraction <f> --use-precision-aware-optimizer` (Megatron requires the
  precision-aware path for HDO).

**The CPU moments ride F2's device-aware transport** (`communicator.py` GPU bounce buffer),
so no transport change was needed for H.

### Assumptions to verify in joint testing (NOT yet run)

1. With HDO, `DistributedOptimizer.model_param_group_index_map` + `optimizer.optimizer.param_groups[g]["params"][o]` still resolve the correct fp32 master (HDO's outer `param_groups` must not permute relative to the DO map).
2. `--use-precision-aware-optimizer` changes how DO stores main params / states (its own dtype-aware buffers); confirm `get_main_weight` and `_get_model_param_range_map` still return param-shaped, element-contiguous ranges under this path.
3. `dummy_step()` on the dst leaves the sub-optimizer state tensors as the very objects `HDO.state[main_weight]["exp_avg"]` references, so receiving into them populates the real optimizer state.
4. CPU-offloaded moments survive `release()`/`rebuild()` (storage `resize_` on pinned CPU storage) without losing correctness.
5. Reshard that *changes which params are offloaded* (offload split depends on per-rank GPU numel, which changes with DP/TP/EP) — src and dst may disagree on a param's device. Per-state device is read independently on each side, so values should still transfer; confirm no assumption that src.device == dst.device per state.
6. Meta-device HDO build is explicitly not handled (`dummy_step` needs real storage).

**Verification target.** `CPU_OFFLOAD=1 OFFLOAD_FRACTION=0.5` dense reshard, then ckpt-level
`compare_optim_logical.py` against the all-GPU baseline (logical multiset equal).

---

## F2 — device-aware transport (2026-05-29)

**Scope.** `elastic_megatron/transfer/communicator.py`.

NCCL only moves CUDA tensors, but after F1 an `OptState` may live on CPU (hybrid offload).
`Communicator.send` / `recv` / `broadcast` now stage non-CUDA tensors through a GPU bounce
buffer:

- `send`: if the tensor is not CUDA, `.to(cuda)` before `send_fn`.
- `recv`: if the destination is not CUDA, receive into a contiguous GPU buffer then
  `copy_` back (also subsumes the existing non-contiguous handling).
- `broadcast`: if not CUDA, broadcast a GPU copy and `copy_` the result back on every rank.

The self-copy path (`_send_self`/`_recv_self`) was already cross-device safe. Communication-
byte accounting is unchanged (`nbytes` is device-independent).

`release_optimizer` / `rebuild_optimizer` already iterate `optimizer_tensors` and call
`storage().resize_()`, which works on CPU storages — no change needed for F2. The
"offload-not-free" lifecycle for CPU-resident states is an H-phase concern.

**Verification.** ⏳ Pending (GPU busy): a forced-CPU-state smoke + ckpt-level compare
against the all-GPU baseline.

---

## F1 — optimizer state model generalization (2026-05-29)

**Scope.** `elastic_megatron/resharding/resharding_metadata.py` (main),
`elastic_megatron/transfer/transfer.py` (src/dst guard),
`elastic_megatron/transfer/ipc_manager.py` (note only).

**What changed.** `OptimizerTensorInfo` no longer hard-codes the
`(main_weight, exp_avg, exp_avg_sq)` triple. It now holds `states: list[OptState]`
(variable length; `states[0]` = master = geometry anchor), with these pieces:

- `OptState{name, tensor}`; device/dtype read from the tensor.
- `ordered_optimizer_state_keys(state, anchor_numel)` — discovers param-shaped state keys
  deterministically (Adam moments first; non-param-shaped entries like a scalar `step`
  dropped), so SRC (initialized) and DST (offload) enumerate states in the same order.
- `create_padded_optimizer_tensor` allocates each padded buffer with its **own** state's
  dtype/device instead of `range(3)` + `main_weight.dtype`.
- `init_empty_state_dict` builds the offload placeholders from `_ADAM_STATE_KEYS` (the one
  remaining spot that encodes "which states exist" for a not-yet-initialized optimizer).
- `get_optimizer_tensors_by_model_weight` constructs `states` generically from the
  optimizer's state dict.
- Read-only `.main_weight` / `.exp_avg` / `.exp_avg_sq` accessors retained for Adam-shaped
  call sites (`ipc_manager.py`).
- `transfer.py::_main_process` asserts `src.state_names == dst.state_names` when both sides
  are present on a rank (best-effort heterogeneity guard; never fires for Adam).
- A **param-shaped invariant** is enforced in `__post_init__` (see [invariants I-15](../project/invariants.md)).

For Adam this yields exactly `[main_weight, exp_avg, exp_avg_sq]`, so the transferred data
and order are unchanged.

`transfer/ipc_manager.py` (inter-process mode, no intra-process test coverage) still reads
the three Adam states via the compat accessors; a comment marks how to generalize it.

**Verification.** Unit smoke (CPU): construct/accessors, scalar-`step` drop, non-param-shaped
rejection, per-state dtype in padding — all pass. Regression: 8-GPU `dense_mix_full` with
`ELASTIC_SAVE_CKPT=1` → `tools/ckpt/verify_all.sh` = **7/7 weight bit-equal + 7/7 optim
multiset-equal**. `moe_mix_full` ran clean (loss converges, 7 ckpt pairs); its `verify_all.sh`
is ⏳ pending (GPU busy).

# Megatron HybridDeviceOptimizer & precision-aware optimizer — reference

Knowledge sink for the CPU+GPU hybrid-offload work. Characterizes Megatron's
`HybridDeviceOptimizer` (HDO) and the `--use-precision-aware-optimizer` code path it
rides on, and how both differ from the plain `DistributedOptimizer` that ElasticMegatron
was built against. File:line refs are into `Megatron-LM-custom/` as of the May 2026 work.

> **TL;DR of why this matters.** HDO **requires** `--use-precision-aware-optimizer`
> (`arguments.py:1235`). Under that flag the optimizer layout changes in three ways that
> break ElasticMegatron's assumptions: (a) `param_groups` holds the **bf16/fp16 shard**, not
> the fp32 master; (b) the fp32 master + moments live in optimizer **state** and may be on
> CPU; (c) `_copy_main_params_to_model_params()` is a **no-op** — the master→model copy is
> done inside `optimizer.step()`, not as a standalone call. See the empirical consequences in
> [`changelog.md`](changelog.md) (H.1/H.2) and the code review at the end of this doc.

## 1. HybridDeviceOptimizer structure

Defined in `megatron/core/optimizer/cpu_offloading/hybrid_optimizer.py:14`.

- **Class.** `HybridDeviceOptimizer(torch.optim.Optimizer)` — a first-class optimizer,
  **wrapped by** `DistributedOptimizer` (`distrib_optimizer.py:599-605` reconstructs it with
  the DO's sharded param groups). So in ElasticMegatron `optimizer` is the `DistributedOptimizer`
  and `optimizer.optimizer` is the HDO.
- **Two sub-optimizers.** A CPU optimizer (or one-per-param when
  `overlap_cpu_optimizer_d2h_h2d`) and a GPU optimizer (`hybrid_optimizer.py:203-214`).
  `sub_optimizers` property unifies them.
- **Per-param CPU/GPU split.** `_get_sub_optimizer_param_groups` (`:251-300`): offloads the
  first params (by numel) up to `offload_fraction × gpu_param_numel` to CPU
  (`.detach().clone().cpu().pin_memory()`), the rest stay on GPU. Split granularity is
  **per-param**, decided by iteration order + a cumulative numel threshold.
- **`.state` is a synced view.** `_sync_sub_optimizers_state_to_hdo` (`:302-321`) rebuilds
  `self.state[orig_param] = sub_optimizer.state[inner_param]` after each step, and adds
  `state[orig_param]["master_param"] = <fp32 inner param>` when `param_update_in_fp32=True`
  (the factory always sets this for HDO). So `state[p]` keys are `exp_avg`, `exp_avg_sq`,
  `master_param`; the moments live on the sub-optimizer's device (CPU for offloaded params).
- **No `init_state_fn`.** The factory sets it to `None` (`optimizer/__init__.py`). State is
  allocated by **`dummy_step()`** (`:453-463`): assigns random grads, `step()`, `zero_grad()`
  — used to materialize state before checkpoint load.
- **Construction.** Selected when `config.optimizer_cpu_offload` (`optimizer/__init__.py:278`),
  with `param_update_in_fp32=True`, `cpu_optimizer_cls`/`gpu_optimizer_cls` = CPUAdam/Adam.

## 2. The precision-aware optimizer layout (what HDO rides on)

`--use-precision-aware-optimizer` → `config.use_precision_aware_optimizer_no_fp8_or_ds_fp8`
internally. Branch sites in `distrib_optimizer.py`: `375, 437, 799, 886, 913, 2276, 2289,
2413(copy), 2525`.

- **`param_groups` holds the model SHARD, not the fp32 master** (`:437-446`):
  - normal: `orig_group["params"] = [*shard_fp32_params, *shard_fp32_from_float16_params]`
    (all fp32 — the float16 params get an fp32 *copy*).
  - precision-aware: `orig_group["params"] = [*shard_fp32_params, *shard_float16_params]`
    (the float16 params are the **bf16/fp16 shards themselves**; `shard_main_param = None` at `:404-405`).
- **The fp32 master lives in optimizer state** (`:799-805`): `state[p]["master_param"]`, dtype
  = `config.main_params_dtype` (default fp32) — or **int16 remainders** when
  `store_param_remainders=True` + bf16, reconstructed via `get_unscaled_state`.
- **moment dtypes are configurable**: `exp_avg_dtype` / `exp_avg_sq_dtype` (`arguments.py:3493`,
  default fp32) — may be scaled; the inner optimizer exposes `get_unscaled_state` / `set_scaled_state`.
- **Two param groups coexist**: `shard_fp32_groups` (params that were already fp32 — typically
  LayerNorm weights/biases) and `shard_float16_groups` (the bf16 body). They are refilled by
  *different* code paths (see §4) — this is the source of the H model-weight bug.

### Canonical accessors (use these, do not poke internals)

`DistributedOptimizer._get_main_param_and_optimizer_states(model_param)` (`:874-900`) returns
`{"param": fp32 master, "exp_avg", "exp_avg_sq"}` handling all cases:
- precision-aware **HDO** branch (`:890-892`): reads `state[shard][k]` directly (the real
  tensors — references, so in-place writes stick), renames `master_param`→`param`.
- precision-aware non-HDO (`:894`): uses `get_unscaled_state` → **copies** (in-place writes do
  NOT stick; needs `set_scaled_state` on write-back).
- normal (`:897-899`): `{"param": fp32 master from param_groups, **state}`.

`_set_main_param_and_optimizer_states(model_param, tensors)` (`:902-931`) is the symmetric writer.

## 3. main_weight ↔ model_weight interaction (the critical difference)

| | normal `DistributedOptimizer` | precision-aware / HDO |
|---|---|---|
| fp32 master location | `param_groups[g]["params"][o]` (a real fp32 shard tensor) | `state[shard]["master_param"]` (HDO: fp32 inner param; non-HDO: scaled/remainder, via get_unscaled_state) |
| model param (bf16) location | DDP `buffer.param_data` (managed by ElasticMegatron's release/rebuild) | DDP `buffer.param_data` — but **released early** (storage 0 by the time ElasticMegatron's dst release runs) |
| **master → model copy** | `_copy_main_params_to_model_params()` writes master shard → `param_data` buffer at the param's world range (`:2438-2463`, two groups: `shard_fp32_from_float16_groups` and `shard_fp32_groups`) | **`_copy_main_params_to_model_params()` is a NO-OP** (`:2431-2432` early-return). The master→model copy is done **inside `optimizer.step()`** (FusedAdam `master_weights=True` / HDO's H2D copy-back). |
| model→master copy | `_copy_model_params_to_main_params` | precision-aware branch at `:2542` |

**This is the crux for resharding.** ElasticMegatron's reshard moves the *master* (and moments)
between layouts, then calls `update_model_weight()` to rebuild the model `param_data` from the
master. Under normal DO that works because `_copy_main_params_to_model_params` actually copies.
Under precision-aware that method returns immediately, so unless ElasticMegatron does the copy
itself, the model `param_data` is never written from the transferred master.

## 4. Differences vs normal DistributedOptimizer — summary

| Aspect | normal DistributedOptimizer | HDO / precision-aware |
|---|---|---|
| Inner optimizer | FusedAdam/Adam, fp32 master params | HDO (CPU+GPU sub-opts) over precision-aware FusedAdam |
| State init when empty | `optimizer.init_state_fn(optimizer.optimizer)` | `init_state_fn is None` → `HDO.dummy_step()` |
| Per-param state tensors | `main_weight` (fp32, in param_groups) + `exp_avg` + `exp_avg_sq` (fp32) | `master_param` (in state; CPU or GPU; fp32/int16) + `exp_avg` + `exp_avg_sq` (configurable dtype/device) |
| State device | all GPU | mixed: offloaded params' master+moments on **pinned CPU**, rest on GPU |
| Read a param's master+state | `param_groups[...][o]` + `state[main]` | `_get_main_param_and_optimizer_states(model_param)` |
| `_get_model_param_range_map` | element offsets into flat buffer | unchanged (element offsets; master has same numel as the bf16 shard) |
| master → model copy | `_copy_main_params_to_model_params()` (real) | **no-op**; done in `optimizer.step()` |
| `model_param_group_index_map` | on the DO | on the DO (HDO does not expose it) |

## 5. Implications for ElasticMegatron (where the code touches this)

> **Status (as of changelog R.1).** Everything below is implemented and now lives behind
> **`OptimizerAdapter`** (`elastic_megatron/resharding/optimizer_adapter.py`), not scattered across
> call sites. Current locations: state extraction → `PrecisionAwareOptimizerAdapter.discover_states`;
> fresh-dst init → `HybridDeviceOptimizerAdapter.ensure_state_initialized` (`dummy_step`); master→model
> refill → `PrecisionAwareOptimizerAdapter.copy_main_to_model` (calls `_precision_aware_copy_main_to_model`).
> `rebuild_model`'s size fallback stays in `training_state.py`. The file::function refs below are the
> pre-refactor narrative — chase the adapter for the live code. See [`../project/optimizer_state_model.md`](../project/optimizer_state_model.md).

- **State extraction** (`resharding_metadata.py::get_optimizer_tensors_by_model_weight`): must
  read the master/moments via `_get_main_param_and_optimizer_states` under precision-aware (done
  in H.2), never treat `param_groups[...]` as the master. See [invariants I-15](../project/invariants.md)
  and [`../project/optimizer_state_model.md`](../project/optimizer_state_model.md).
- **State init for a fresh dst** (`get_optimizer_tensors_by_model_weight`): `dummy_step()` for HDO
  (done in H.1), since `init_state_fn is None`.
- **Model-weight rebuild** (`training_state.py::rebuild_model`): the dst model `param_data` is
  storage-0 at release under precision-aware → `param_data_size` was never recorded → rebuild
  needs the size fallback (done) — but see the code review below, the deeper issue is the copy.
- **Model-weight refill** (`training_state.py::update_model_weight`): **broken** — relies on
  `_copy_main_params_to_model_params`, which is a no-op under precision-aware. This is the
  remaining bug. See the code review.

---

## 6. Code review (step 2): can the current code handle the hybrid-opt update?

**No.** The reshard *transfer* layer is correct under precision-aware after H.1/H.2 (the
optimizer-state multiset matches before/after — verified on the iter-3 smoke pair). But the
**model-weight update path cannot reconstruct the model weights** from the transferred master,
for two coupled reasons:

1. **`update_model_weight()` → `optimizer._copy_main_params_to_model_params()` is a no-op under
   precision-aware** (`distrib_optimizer.py:2431-2432`). The model `param_data` buffer is never
   written from the transferred master, so it keeps whatever `rebuild_model` left in it.
2. **`rebuild_model()`'s size fallback fills the buffer with uninitialized memory.** Under
   precision-aware the dst `param_data` is storage-0 at release (weights live in the optimizer),
   so the fallback `resize_`s it to garbage; with (1) being a no-op, that garbage survives.
   Empirically: the `shard_fp32_groups` params (LayerNorm weight/bias, output_layer.weight) come
   out as `1e34`-scale garbage; with `OFFLOAD_FRACTION=0` *more* params are garbage — confirming
   the bug is **precision-aware-intrinsic, not the CPU-offload device path**.

### What needs to change

- **Primary — explicit master→model copy under precision-aware** (`training_state.py::update_model_weight`).
  After the transfer + `rebuild_model`, when the optimizer is precision-aware, ElasticMegatron
  must itself write each param's fp32 master shard into the model `param_data` buffer.
  **Do not** port `copy_group_params` verbatim: under precision-aware `shard_fp32_from_float16_groups`
  is empty (float16 params have no fp32 copy — their master is in `state["master_param"]`), so that
  loop would skip the bf16 body. Instead, iterate the model params owned by this DO and, per param:
    1. master = `optimizer._get_main_param_and_optimizer_states(model_param)["param"]` (the just-
       transferred fp32 master; for offloaded params it is on CPU);
    2. range = `_get_model_param_range_map(model_param)["gbuf_world_in_bucket"]`;
    3. `param_buffer = self.buffers[gbuf_index].buckets[bucket_id].param_data`;
       `param_buffer.view(-1)[range.start:range.end].copy_(master.to(param_buffer.device, param_buffer.dtype))`.
  This covers **both** the float16 body and the `shard_fp32_groups` LayerNorm/bias params uniformly.
  Then `start_param_sync(force_sync=True)` (already called) all-gathers the correct values.
  (Check first whether Megatron exposes a standalone "copy master→param_buffer" entry for the
  precision-aware path — e.g. via the inner optimizer — and prefer it over hand-rolling.)
- **Secondary — `rebuild_model` garbage.** Once (Primary) writes every param, the resized
  buffer is fully overwritten, so the garbage no longer matters. If any param is still not
  covered, zero-fill on rebuild instead of leaving uninitialized memory, so a miss is detectable
  (zeros) rather than silent garbage.
- **HDO device.** The master/moments for offloaded params are on CPU; the explicit copy must
  cast/move CPU fp32 → GPU `param_data` (the transport already stages CPU tensors; the copy here
  is local `param_data.copy_(master.to(device, dtype))`).
- **Out of scope / separate**: the `rerun_state_machine` requests "checkpoint + exit to diagnose"
  on the post-reshard step (resharding breaks bit-determinism, which it flags) and then crashes
  on `args.save=None`. This is a Megatron-diagnostic incompatibility with resharding, not a hybrid
  bug — disable it in the launcher (a `--no-...`/rerun flag) so offload runs can do multiple
  reshards. Non-HDO precision-aware write-back (`set_scaled_state`) is also out of scope; only HDO
  is targeted.

### Verification once fixed

Re-run the `CPU_OFFLOAD=1` smoke (and `OFFLOAD_FRACTION=0`) → the iter-3 before/after pair should
pass **both** weight (bit-equal) and optim (multiset) in `verify_all.py`. Today optim passes and
weight fails on the `shard_fp32_groups` params.

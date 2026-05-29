# Invariants

Rules that are not enforced by any single assert, but if you break them, things will fail in confusing places. Each one comes with the reason and the failure mode you should expect if it breaks.

## I-1. EP=1 → expert lives in the dense DDP bucket

**Rule.** When `expert_model_parallel_size == 1`, expert parameters are *not* placed into a separate MoE bucket; they go into the dense `_ParamAndGradBuffer`, share the dense optimizer's `(tp, dp)` grid, and (for `get_global_rank()` purposes) carry `ep_rank = None`.

**Why.** Megatron's TransformerEngine `GroupedLinear` / `Linear` flips `param.allreduce` to `True` when `expert_parallel=False`. mcore's `DistributedDataParallel` then puts those params in the dense bucket, and `DistributedOptimizer`'s shard layout follows. ElasticMegatron's plan must match what mcore actually does — otherwise the simulated `dst_aligned_global_rank` points to a rank that does not own the param, and the receiver `get_main_weight()` returns `None`.

**Where it lives.** `ParallelStrategy.expert_is_dense_bucketed()` (in `parallel_strategy.py`). Four sites in `resharding/` consult it:

- `resharding_dp.py::get_params_dp_distribution` — bucket routing
- `resharding_dp.py::DataParallelReshardingInfo._init_dp_distribution_with_global_rank` — TP-rank / EP-rank derivation
- `resharding.py::ReshardPlan.__post_init__` + `_ep_rank_for_global_rank` — what to pass to `get_global_rank()`
- `virtual_param.py::_effective_tp_size` (closure inside `_generate_reshard_plan`) — TP axis used when expanding to replicated dst TP ranks

**Failure mode if broken.** Symptom A (`NoneType.create_padded_optimizer_tensor`) on transitions touching EP=1, or NCCL hang (symptom C) when `TP != ETP`. Triage in [`debugging.md`](debugging.md).

---

## I-2. The two flavours of `ep_rank` are distinct

`ReshardPlan` stores `self.src_ep_rank` (and `dst_ep_rank`) — the **physical** expert-parallel rank — separately from `self._src_ep_rank_for_gr` (and `_dst_ep_rank_for_gr`) — the value to **pass into `get_global_rank()`**. The latter is `None` whenever I-1 says "treat expert as dense".

If you find yourself reading `self.src_ep_rank` from a new piece of code, you almost certainly want `self._src_ep_rank_for_gr` instead. See the docstring at the field assignment in [`resharding.py::ReshardPlan.__post_init__`](../../elastic_megatron/resharding/resharding.py).

---

## I-3. Expert classification is name-based, not attr-based

`resharding_metadata.py::_is_expert_param` decides "is this an expert param" by `".experts." in name`, **not** by `getattr(param, "allreduce")`. The `allreduce` attribute flips under I-1 and would misclassify the same logical param across EP=1 vs EP>1.

If you add a new model where experts have a different name pattern, update `_is_expert_param` accordingly.

---

## I-4. Distributed-optimizer state must be saved as the full `ChainedOptimizer`

In `TrainingState.save_checkpoint`, pass `self.optimizer` (the `ChainedOptimizer`), **not** `self.optimizers[0]` (its first chained slot). When `EP > 1`, `self.optimizers == [dense_optim, expert_optim]` — passing `[0]` silently drops the expert optimizer state on save.

**Failure mode.** Training looks correct (in-memory state is intact), but the saved ckpt is missing all expert main_param / exp_avg / exp_avg_sq. On resume you would lose all MoE optimizer state.

---

## I-5. `step` must be broadcast across all chained optimizers on EP transitions

`TrainingState.update_optimizer_and_opt_param_scheduler` uses `zip(src_optimizers, dst_optimizers)` to copy lr/betas/etc. On EP=1 → EP>1 (or the reverse), the lists have different lengths; `zip` truncates. The newly-created chained slots get `step = 0`, and the next `save_checkpoint` trips mcore's `_synchronize_steps` assertion (`assert len(steps) <= 1` over `{N, 0}` fails).

Therefore `step` is broadcast explicitly: pulled from any src chained's any param_group, written into every dst chained's every non-empty param_group. Do not unify this with the lr/betas copy unless you preserve that explicit broadcast.

---

## I-6. `model = training_state.model` (rebind), NOT `model[:] =` (slice assignment) — and disable eval/save in scripts

Inside the elastic loop in `train()`:

```python
model = training_state.model        # ← current convention (写法 A, rebind)
# model[:] = training_state.model   ← DO NOT USE: poisons cached TrainingState.model lists
```

**Why rebind, not slice-assignment.** `ElasticMegatronManager.__init__` stores the same model `list` object that `pretrain()` holds: `TrainingState.__init__` does `self.model = model` (no copy). Every cached `TrainingState` (one per parallel strategy) ends up keyed on the **same list reference**. A `model[:] = training_state.model` slice-assignment mutates that list **in place** → it overwrites the `.model` field of every cached TrainingState at once. The next time you reshard *back* to a previously-used strategy, that strategy's cached TrainingState finds its `.model` no longer matches its `.optimizer.buffers` (the list contents now point at a different strategy's DDP wrappers), and `update_model_weight()` blows up at `_copy_main_params_to_model_params` with `setStorage: storage of size 0`.

**Required script-side restriction.** Rebind has its own well-known footgun: `pretrain()` retains its local `model` list, and after `train()` returns it will use that list for `evaluate_and_print_results` / `save_checkpoint`. Under rebind, `pretrain()`'s list still points at the initial src model — whose DDP buffer storage was `resize_(0)`'d by an earlier reshard's `release_model()`. The eval/save path then trips `setStorage: out of bounds for storage of size 0`.

So under the rebind convention the launcher scripts **must disable end-of-train eval and `--save`** (e.g., `--eval-iters 0`, no `--save`). All `run_e2e_demo.sh` / `run_moe.sh` / `run_experiment.sh` modes ship with `--eval-iters ${EVAL_ITERS:-0}` and no `--save`; honor that. If you ever genuinely need a real eval pass, do not "fix it" by flipping back to slice-assignment — that re-introduces the cache-poisoning bug above. Instead, break the list aliasing first (e.g., `self.model = list(model)` in `TrainingState.__init__`), *then* slice-assign. Both halves of the fix are required together.

**Failure modes if you break this rule.**
- Slice-assignment **without** breaking the list aliasing: `setStorage size 0` from `_copy_main_params_to_model_params` on the **second** reshard back to a cached strategy (`tp_flip` / `ep_flip` reproduce this immediately at iter 6 with `interval=3`).
- Rebind **with** eval/save enabled: `setStorage size 0` from `F.embedding` in the post-train `evaluate_and_print_results`.

Same idea applies to *any* mutable container that the caller and callees share by reference — be deliberate about whether mutation should propagate.

---

## I-7. `do_train` / `do_valid` / `do_test` must be set from iterator presence + reduced across ranks

In `dataloader_state.py::build_iterators`, after building iterators, compute `do_train/do_valid/do_test` per-rank from `(iter is not None) and (corresponding_iter_count > 0)`, then `all_reduce(MAX)` across all ranks.

**Why.** Different ranks build different iterators (only TP-rank-0 owns the dataset, others get `None`). Megatron's `evaluate_and_print_results` is a collective — every rank must agree on whether to enter it. Setting `args.do_test = args.eval_iters > 0` unconditionally (as the pre-fix code did) breaks when `--split` has no test portion: TP-rank-0 has `test_iter = None`, other ranks think `do_test = True`, and the collective hangs or asserts.

---

## I-8. `update_global_args` must push `sequence_parallel` back into Megatron's `args`

When the new strategy implies `sequence_parallel = True` (e.g., MoE × TP>1 always requires it), `ParallelStrategy.update_global_args` must mirror it into Megatron's global `args.sequence_parallel`. The strategy's own `sequence_parallel` field is derived in `__post_init__`, but Megatron's MoE forward reads `args.sequence_parallel` and raises `ValueError("MoE and tensor parallelism require sequence parallelism")` otherwise.

---

## I-9. `is_redundant_backup` only fires when you explicitly switch Group-Zero on/off

`ParallelStrategy.is_redundant_backup(src, dst)` returns non-None **only** when `src.num_distributed_optimizer_instances != dst.num_distributed_optimizer_instances` (i.e. you are deliberately turning DGZ on or off across a reshard). Plain reshards — including ones that change `world_size` (e.g. `TP=4/DP=2 → TP=4/DP=1`), `TP`, `PP`, `EP`, or `CP` — go through the **normal** `transfer_params` path, *not* the redundant-backup path.

> **Common misreading (do not propagate).** Earlier notes in this repo at times implied "scale up/down ⇒ redundant_backup path". That is wrong. World-size changes alone do not select the redundant_backup path. A normal reshard that happens to drop ranks (`world_size` shrinking) just makes the dropped ranks fall outside the `union_world_group` after the transfer; they never see the redundant_backup code.
>
> Concrete rule of thumb: if no strategy in your `parallel_strategy_list` has `num_distributed_optimizer_instances > 1`, **none of your reshards ever exercise `is_redundant_backup` or `transfer_manager.redundant_backup`**. If you are debugging a non-DGZ test and find yourself reading those code paths, you are in the wrong place.

**When is_redundant_backup is exercised.** The current code further requires `src_world_size != dst_world_size` *in addition to* the DGZ flip, because that is the scale-up/down scenario it was designed for (one of the two sides has `dgz=1` and a smaller world; the other has `dgz>1` and a larger world that creates the redundant replicas). Flipping DGZ at a fixed `world_size` is therefore unsupported today; deferred until a real use case asks for it.

Phase B's `dense_cp_only` sweep avoids DGZ flips for this reason. Plain `TP=4/DP=2 ↔ TP=4/DP=1` (scale down without DGZ flip) is *not* a DGZ test — it goes through `transfer_params`, and ranks 4-7 simply fall outside `dst.world_size` on the down-step.

---

## I-10. Caches in `VirtualParam` are keyed by strategy alone

`_dp_distribution_cache` and `_reshard_plan_cache` use `(parallel_strategy, ...)` as the key — they do **not** include `self.is_expert` or `expert_is_dense_bucketed()`. This is safe because:

- `self.is_expert` is a per-VirtualParam constant.
- `expert_is_dense_bucketed()` is a pure function of `ParallelStrategy`, which is already in the key.

If you refactor caching (e.g., to a module-level dict), preserve this dependency or add it to the key explicitly.

---

## I-11. The mocked `_FakeTensor` only covers what mcore 0.16 touches

In `resharding_dp.py::mock_ddp_buffer_init`, `_FakeTensor` implements `nelement / numel / detach / copy_` and the `shape / dtype / device / requires_grad` attributes — the minimum mcore 0.16's `_ParamAndGradBuffer.__init__` actually accesses. Any other attribute access raises `NotImplementedError` (the `__getattr__` trip-wire).

If a future Megatron version adds a `.dim()` / `.untyped_storage()` / etc. call inside `_ParamAndGradBuffer.__init__`, you will get a clear failure here. Extend `_FakeTensor` — do not silence the trip-wire.

---

## I-12. Cross-repo patches must go through `examples/intra_process/training_016.py` (or its 0.11 sibling)

`Megatron-LM-custom/megatron/training/training.py` carries three patches: parallel-strategy modes / `ELASTIC_SAVE_CKPT` hook / the elastic-loop integration (currently `model = training_state.model` rebind — see I-6). Those patches do **not** live in ElasticMegatron's tree directly — they live in the example snapshot under `examples/intra_process/`. When you change the patches, update the example file too, otherwise downstream users running off the example will lag behind master. See [`cross_repo.md`](cross_repo.md).

---

## I-13. Simulated DDP bucket layout must match Megatron's real-run bucket layout

**Rule.** Whenever ElasticMegatron hands `VirtualParam`s to Megatron's `_ParamAndGradBuffer` constructor for bucket-layout simulation (today: inside `resharding_dp.py::get_params_dp_distribution`), the param-attribute state visible to Megatron must reproduce what the real `pretrain` model produces *for the strategy being simulated*. Diverging by even one attribute can split a real bucket into two simulated buckets (or vice versa); the resulting `dp_distribution` is no longer a function of the real `DistributedOptimizer`'s shard layout, and downstream transfer trips with the user-visible symptoms below.

**The one currently-known divergence.** `shared_embedding`. Megatron sets `weight.shared_embedding = True` only when `pipeline_model_parallel_size > 1` (`LanguageModule.setup_embeddings_and_output_layer` early-returns at PP=1). ElasticMegatron's `VirtualParam.shared_embedding` is set unconditionally to `share_embeddings_and_output_weights`, because other paths (e.g. `transfer.py`'s OUTPUT_LAYER orphan guard) need it as a model-level descriptor regardless of PP. So at PP=1 the simulated layout has to mask the attribute back to `False` to match real-run behaviour — that is what `resharding_dp.py::_mask_shared_embedding_for_pp1` does.

**How to extend.** If a future Megatron version makes `_does_param_require_new_bucket` (or any bucket-splitting predicate) read another param attribute, audit the corresponding `VirtualParam` field for the same kind of "unconditionally set on the VP but conditionally set on the real param" mismatch. If found, extend the contextmanager — do not change the VP's default attribute value, because other code paths may depend on it.

**Failure modes if broken.**
- `AssertionError: is_contain` inside `transfer.py`'s ZeRO-1 gather — the simulated dp_distribution and the real `DistributedOptimizer.model_param_group_index_map` reference different sub-ranges of the same param.
- The companion `AssertionError: dp_distribution is not None` at `virtual_param.py:168` is *not* an instance of this invariant — see I-14 below.

---

## I-14. Orphan `VirtualParam`s (`stage_id == -1` on both sides) must be skipped in every iteration over `all_virtual_params`

**Rule.** A `VirtualParam` whose `get_model_param_stage_id(pp_size)` returns `-1` is an "orphan" — it shares storage with another param and has nothing of its own to transfer. The only orphan today is the OUTPUT_LAYER under tied embedding + PP=1 (see `resharding_pp.py::_get_non_transformer_layer_stage_id`). `_build_stages_virtual_params` already filters orphans by stage_id, so their `_dp_distribution_cache` is never populated.

Any code that walks `VirtualParamSpace.all_virtual_params` and then calls `apply_reshard_plan` / `get_dp_distribution` / similar plan-generation paths on each VP must use the same filter. Use `VirtualParam.is_orphan_for(src_pp_size, dst_pp_size)` rather than re-implementing `stage_id == -1` checks at the call site.

**Failure mode if broken.** `AssertionError: dp_distribution is not None` at [virtual_param.py:168](../../elastic_megatron/resharding/virtual_param.py) for the orphan VP — fires on the first reshard with tied + PP=1.

**Where the filter must live (current sites).**

- `VirtualParamSpace._build_stages_virtual_params` — already filters by `stage_id != -1`.
- `VirtualParamSpace.register_reshard` — uses `is_orphan_for`.
- `TransferManager.transfer_optimizer_tensors` — already has the equivalent guard via `virtual_param.shared_embedding and layer_type == OUTPUT_LAYER` (`transfer.py:317-329`). If a future change introduces a non-OUTPUT_LAYER orphan, prefer the stage-id-based predicate over the attribute-based one.

---

## I-15. Every optimizer state of a param is param-shaped and shares the param's single reshard plan

**Rule.** `OptimizerTensorInfo` (`resharding_metadata.py`) holds an ordered, variable-length list of named `OptState`s; `states[0]` is the master weight and the geometry anchor. **Every** state tensor must have the same `numel` as the master. The reshard geometry (`dp_distribution` + `ReshardPlan`) is computed **once per param from the master** and reused for all states — this is only valid because they are param-shaped. Per-state `dtype` and `device` may differ (Adam: all equal); `create_padded_optimizer_tensor` allocates each padded buffer with its state's own dtype/device, and `transfer/communicator.py` stages non-CUDA tensors through a GPU bounce buffer.

**Corollaries.**
- State discovery is deterministic and name-based: `ordered_optimizer_state_keys()` keeps only param-shaped tensors (a scalar per-param `step`, if any, is dropped — synced via param_groups, see I-5) and orders Adam moments first, so the SRC (initialized) and DST (offload-allocated) sides enumerate states in the same positional order.
- src and dst must carry the **same** ordered state set per param; `transfer.py::_main_process` asserts `src.state_names == dst.state_names` when both are present on a rank.
- The DST offload path (`init_empty_state_dict`) still encodes "which states exist" for a not-yet-initialized optimizer — defaults to the Adam moments. A non-Adam optimizer (e.g. Muon's `momentum`) needs its own offload schema there.

**Not supported (guarded).** Non-param-shaped states — FP8/FP4 per-block `scale`/`amax`. `OptimizerTensorInfo.__post_init__` raises an `AssertionError` rather than silently mishandling them; supporting them needs a second transport path (replicate/broadcast) and block-vs-shard alignment.

**Failure mode if broken.** Adding a non-param-shaped state trips the `__post_init__` assert immediately. A src/dst state-set mismatch (heterogeneous optimizer without matching schema) trips the `_main_process` assert, or — if that rank holds only one side — desyncs the positional send/recv counts into an NCCL hang.

See [`optimizer_state_model.md`](optimizer_state_model.md) for the full contract and [`../hybrid_adam/`](../hybrid_adam/) for the work that introduced this model.

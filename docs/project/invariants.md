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

## I-6. `model[:] =` (slice assignment), not `model =` (rebind)

Inside the elastic loop in `train()`:

```python
model[:] = training_state.model  # ← correct
# model = training_state.model   ← WRONG: rebinds local var, pretrain()'s reference stays stale
```

**Why.** Megatron's `pretrain()` passes its local `model` list to `train()` and **also retains its own reference** to that same list (for the post-train `evaluate_and_print_results` and possibly `save_checkpoint`). `train()`'s elastic loop reaches the new `training_state.model`. A bare `model = ...` rebinds only `train()`'s local variable; `pretrain()`'s reference still points at the *initial* model — whose DDP buffer storage was resize-to-0'd by `state_manager.release_model()` on some past reshard.

**Failure mode.** `RuntimeError: setStorage: ... out of bounds for storage of size 0` on the embedding weight in the final `evaluate_and_print_results`. See `docs/megatron_016_adaptation/changelog.md` "Eval-time setStorage / do_test bug" for the worked example.

Same idea applies to *any* mutable container that the caller retains a reference to. If you replicate this pattern elsewhere, prefer slice-assignment or in-place mutation over rebind.

---

## I-7. `do_train` / `do_valid` / `do_test` must be set from iterator presence + reduced across ranks

In `dataloader_state.py::build_iterators`, after building iterators, compute `do_train/do_valid/do_test` per-rank from `(iter is not None) and (corresponding_iter_count > 0)`, then `all_reduce(MAX)` across all ranks.

**Why.** Different ranks build different iterators (only TP-rank-0 owns the dataset, others get `None`). Megatron's `evaluate_and_print_results` is a collective — every rank must agree on whether to enter it. Setting `args.do_test = args.eval_iters > 0` unconditionally (as the pre-fix code did) breaks when `--split` has no test portion: TP-rank-0 has `test_iter = None`, other ranks think `do_test = True`, and the collective hangs or asserts.

---

## I-8. `update_global_args` must push `sequence_parallel` back into Megatron's `args`

When the new strategy implies `sequence_parallel = True` (e.g., MoE × TP>1 always requires it), `ParallelStrategy.update_global_args` must mirror it into Megatron's global `args.sequence_parallel`. The strategy's own `sequence_parallel` field is derived in `__post_init__`, but Megatron's MoE forward reads `args.sequence_parallel` and raises `ValueError("MoE and tensor parallelism require sequence parallelism")` otherwise.

---

## I-9. Group Zero (`num_distributed_optimizer_instances > 1`) requires world-size change

`ParallelStrategy.is_redundant_backup()` returns non-None **only** when `src_world_size != dst_world_size`. The code path is for scale up/down — *not* for switching DGZ on/off at a fixed world size.

Phase B's `dense_cp_only` sweep avoids DGZ flips for this reason. If you need DGZ-flip-at-fixed-world-size, this is currently unsupported; another issue is required.

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

`Megatron-LM-custom/megatron/training/training.py` carries three patches: parallel-strategy modes / `ELASTIC_SAVE_CKPT` hook / `model[:]` slice-assignment / the elastic-loop integration. Those patches do **not** live in ElasticMegatron's tree directly — they live in the example snapshot under `examples/intra_process/`. When you change the patches, update the example file too, otherwise downstream users running off the example will lag behind master. See [`cross_repo.md`](cross_repo.md).

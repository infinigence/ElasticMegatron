# Architecture

How `ElasticMegatronManager.reshard()` actually works inside, end to end. Read [`README.md`](README.md) first for the concept names; read [`invariants.md`](invariants.md) for the design rules referenced below.

## Overview

```
┌──── train() loop (caller) ─────────────────────────────────────────┐
│                                                                    │
│   for iteration:                                                   │
│     new_strategy = check_reshard(iteration)                        │
│     if new_strategy is not None:                                   │
│       ts = elastic_megatron_manager.reshard(new_strategy)          │
│       model[:] = ts.model                                          │
│       optimizer = ts.optimizer                                     │
│       …                                                            │
│                                                                    │
│     forward / backward / step                                      │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
                       │
                       ▼
┌──── elastic_manager.reshard(new_strategy) ─────────────────────────┐
│                                                                    │
│  step 0: (optional) save_ckpt(before_reshard)  ── ELASTIC_SAVE_CKPT │
│  step 1: state_manager.reshard()  → (src_state, dst_state, union)  │
│            ├ release src.model (DDP buffers → storage 0)           │
│            └ build dst.state if first time, init metadata          │
│  step 2: transfer / redundant_backup  → data moves across NCCL     │
│  step 3: transfer_learning_rate (lr / step / param_group attrs)    │
│  step 4: src.release_optimizer (fp32 main/exp_avg/exp_avg_sq → 0)  │
│  step 5: dst.update_model_weight                                   │
│           ├ rebuild_model (DDP buffers ← storage restored)         │
│           ├ _copy_main_params_to_model_params                      │
│           └ start_param_sync(force_sync=True) across DP group      │
│  step 6: log perf  +  (optional) save_ckpt(after_reshard)          │
│                                                                    │
│  return dst.training_state                                         │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

## State lifecycle

```
ParallelStrategy --(key)--> MegatronStateManager._parallel_strategy_to_megatron_state
                                 │
                                 ▼
                            MegatronState
                                 │
              ┌──────────────────┼──────────────────┐
              ▼                  ▼                  ▼
         mpu_state         training_state       world_ranks
        (parallel-state    (model, optimizer,   (which global
         snapshot)          opt_param_scheduler, ranks belong)
                            params_to_resharding_metadata,
                            optimizer_tensor_info_list)
```

Each `MegatronState` is **lazily** materialised the first time you reshard *to* its strategy. Subsequent transitions reuse the cached `mpu_state` and `training_state` slots. `training_state.model` is a list of `DDP`-wrapped model chunks (always 1 element in current usage).

## The six-step pipeline in detail

The numbered steps below are in [`elastic_manager.py::reshard`](../../elastic_megatron/elastic_manager.py).

### Step 0 — Optional `before_reshard` checkpoint save

If `save_ckpt=True` (driven by `ELASTIC_SAVE_CKPT=1` env var), save the *current* `training_state` to `tools/ckpt/before_reshard/iter_N/`. This is the input to offline verification via `tools/ckpt/verify_all.sh`.

### Step 1 — Compute src/dst states and union world group

`MegatronStateManager.reshard(new_strategy)`:

1. Look up `src = current_megatron_state` and `dst = self._parallel_strategy_to_megatron_state[str(new_strategy)]`.
2. `self.apply(new_strategy)` rebinds Megatron's parallel-state globals to point at `dst.mpu_state`.
3. Compute `union_world_group, union_world_ranks` = the rank set involved in either strategy.
4. If this rank is not in the union, return `(None, None, None)` — the rank sits this transition out.
5. If `src` has a live `training_state`, call `src.training_state.release_model()` — the DDP buffer's `param_data` storage is `resize_(0)`'d to free GPU memory.
6. If `dst.training_state` is `None` (first visit), build it with `setup_model_and_optimizer(is_meta_device=is_meta_device)`, then `init_metadata(dst_strategy, offload_opt_tensors=True)` (this also releases the dst model immediately, leaving optimizer-tensor metadata cached).

After step 1, both src and dst have *metadata* but no live GPU storage for the model buffers — they will be re-realised by step 5.

### Step 2 — Transfer

Two paths, both in [`elastic_manager.py`](../../elastic_megatron/elastic_manager.py):

- **Redundant backup** (`is_redundant_sacle_up is not None`): only used for Group-Zero scale up/down, where `src_world_size != dst_world_size`. Copies the optimizer state to a redundant replica.
- **Direct transfer** (the common case): `TransferManager` in [`transfer/transfer.py`](../../elastic_megatron/transfer/transfer.py) walks `params_to_resharding_metadata`, for each `VirtualParam` uses its cached `ReshardPlan` (built lazily in `apply_reshard_plan`), and issues NCCL `broadcast_object_list` + p2p sends/recvs over the union world group.

Plans are computed at `(src_strategy, dst_strategy, with_ddp)` granularity, cached on the `VirtualParam`. The first reshard between any two strategies pays the planning cost; later reshards in the same direction reuse the cached plan.

### Step 3 — Transfer learning rate

[`TrainingState.update_optimizer_and_opt_param_scheduler`](../../elastic_megatron/megatron_manager/training_state.py) copies:

- `param_group` attrs (lr, betas, weight_decay, ...) via positional `zip(src_optimizers, dst_optimizers)` — only well-defined when both sides have the same chain length.
- `step` (the optimizer-step counter) explicitly broadcast from src to **every** dst chained optimizer's **every** non-empty param_group — see [`invariants.md`](invariants.md) I-5 for why the zip-truncation case must be handled separately.
- `opt_param_scheduler` state, with `num_steps` preserved.

### Step 4 — Release src optimizer

`src.release_optimizer()` walks `optimizer_tensor_info_list` and resizes each tensor's storage to 0 (main_weight, exp_avg, exp_avg_sq). After this, src is fully "evaporated" — both model buffers and optimizer state are at storage 0. The metadata is still in memory, ready if src is re-entered as a dst on a later reshard.

### Step 5 — Update dst model weight

`dst.update_model_weight()`:

1. `rebuild_model()` — DDP buffer's `param_data` storage is `resize_()`'d back to `param_data_size`.
2. `optimizer._copy_main_params_to_model_params()` — for every chained optim, copy the fp32 main_param (which step 2 has populated) into the bf16 model param.
3. `start_param_sync(force_sync=True)` across each model chunk's DP group — ensures all DP replicas see the same model param.

After step 5, dst's model is live and ready to forward.

### Step 6 — Logging + optional `after_reshard` save

Communication info (bytes, time, bandwidth) is logged. If `save_ckpt=True`, save dst to `tools/ckpt/after_reshard/iter_N/`. The result returned to the caller is `dst_megatron_state.training_state`.

## What the caller must do after `reshard()` returns

```python
training_state = elastic_megatron_manager.reshard(new_strategy)
if training_state is not None:
    # I-6: mutate model list in place so pretrain()'s reference stays live
    model[:] = training_state.model
    optimizer = training_state.optimizer
    opt_param_scheduler = training_state.opt_param_scheduler

    # Rebuild iterators for the new DP / TP layout
    train_data_iterator, valid_data_iterator, _ = \
        elastic_megatron_manager.build_iterators()

    # Re-fetch config and forward_backward_func (PP size may have changed)
    config = training_state.refresh_config(model, optimizer)
    forward_backward_func = get_forward_backward_func()
    num_microbatches = get_num_microbatches()
```

The full pattern is in [`examples/intra_process/training_016.py`](../../examples/intra_process/training_016.py).

## Why this design

The reshard is **symmetric**: src and dst are both `MegatronState` objects, just at different points in their lifecycle. There is no "main" state and "shadow" state. This is what makes round-trip transitions (TP=1 ↔ TP=2 ↔ ...) efficient: each strategy is built once and then alternately re-realised.

The `release_model()` / `release_optimizer()` ↔ `rebuild_model()` / `rebuild_optimizer()` symmetry is how memory is bounded: only the *currently active* strategy holds GPU storage for its buffers and optimizer tensors. The inactive strategies retain their metadata (cheap) but their storage is at 0.

## Background: how Megatron's training loop fits in

Megatron's `pretrain()` builds initial model/optimizer/dataloader, then calls `train()`. ElasticMegatron injects:

- `ElasticMegatronManager.register(...)` at the top of `pretrain()` — caches the setup_model_and_optimizer closure for later strategy builds.
- The elastic loop inside `train()` — the snippet above.
- The three patches in `examples/intra_process/training_016.py` (or its 0.11 sibling) wrap these calls into Megatron's existing structure.

See [`cross_repo.md`](cross_repo.md) for the patching contract.

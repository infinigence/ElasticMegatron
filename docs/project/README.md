# ElasticMegatron — project overview (for agents)

This file is the entry point. It covers **what the project does, the core concepts you need in your head, and where to read next.** It deliberately does not duplicate the top-level [`README.md`](../../README.md) (which is user-facing) or `architecture.md` (which is implementation-deep).

## What ElasticMegatron is

A side library that adds **online parallel-strategy switching** to Megatron-LM. During training, you can call

```python
training_state = elastic_megatron_manager.reshard(new_parallel_strategy)
```

and the model + optimizer state are *redistributed* under a different `(TP, PP, CP, DP, EP)` layout — without restarting the process, without losing optimizer state, in tens to hundreds of milliseconds.

There are two deployment modes:

- **intra-process** — one process group spans every reshard candidate; the world size is fixed; only the `(TP, PP, CP, DP, EP)` partition changes. This is the mode covered by every example in this repo, and the only mode currently exercised in tests.
- **inter-process** — distinct process groups per parallel strategy; supports node-level scale up/down. Out of scope for this README.

## Core concepts (5 minute mental model)

- **`ParallelStrategy`** — a dataclass describing one parallel layout: `(world_size, tp, pp, cp, dp, ep, etp, dgz, sequence_parallel)`. Hashable; used as a key in caches. Defined in [`elastic_megatron/megatron_manager/parallel_strategy.py`](../../elastic_megatron/megatron_manager/parallel_strategy.py).
- **`MegatronState`** — a per-strategy bundle of `{mpu_state, training_state, world_ranks}`. Cached in `MegatronStateManager._parallel_strategy_to_megatron_state` keyed by `str(strategy)`. Each strategy you reshard to gets its own slot, lazily initialized.
- **`TrainingState`** — for one strategy, owns `{model, optimizer, opt_param_scheduler, params_to_resharding_metadata, optimizer_tensor_info_list}`. Knows how to `release_model()` / `rebuild_model()` (DDP buffer storage resize to 0 / back) and `release_optimizer()` / `rebuild_optimizer()`. Defined in [`elastic_megatron/megatron_manager/training_state.py`](../../elastic_megatron/megatron_manager/training_state.py).
- **`VirtualParam`** — a model parameter described by its `param_position_attr` (PP stage, layer index, expert id, ...) and `tensor_parallel_attr` (whether sharded, partition_dim, ...). Carries caches for `dp_distribution` and `reshard_plan` keyed by parallel strategy. Defined in [`elastic_megatron/resharding/virtual_param.py`](../../elastic_megatron/resharding/virtual_param.py).
- **`ReshardPlan`** — for one `(src_strategy, dst_strategy, with_ddp)` tuple, the send/recv map at global-rank granularity. Built in [`elastic_megatron/resharding/resharding.py`](../../elastic_megatron/resharding/resharding.py).
- **`ElasticMegatronManager.reshard(new_strategy)`** — the public entry point. Returns the new `TrainingState`. Defined in [`elastic_megatron/elastic_manager.py`](../../elastic_megatron/elastic_manager.py).

The reshard itself is a six-step pipeline (`elastic_manager.py::reshard`); see [`architecture.md`](architecture.md).

## How ElasticMegatron plugs into Megatron-LM

Megatron's `pretrain() → train()` loop drives training. ElasticMegatron is invoked from inside `train()`, via three patches the user adds to `megatron/training/training.py`:

1. Construct `init_parallel_strategy_list()` + `check_reshard(iteration)`.
2. `ElasticMegatronManager.register(...)` at the top of `pretrain()`.
3. Inside the `train()` while-loop, before each step, call `check_reshard()` and, if non-None, `elastic_megatron_manager.reshard()`; on success, update local `model` / `optimizer` / `opt_param_scheduler` and rebind data iterators.

The full patched 0.16 file is at [`examples/intra_process/training_016.py`](../../examples/intra_process/training_016.py) (and `training_011.py` for 0.11). See [`cross_repo.md`](cross_repo.md) for the relationship between this repo and `Megatron-LM-custom/`.

## Where to read next

- [`repo_layout.md`](repo_layout.md) — every directory and important file in one place
- [`invariants.md`](invariants.md) — what NOT to break (assumptions wired through multiple files)
- [`architecture.md`](architecture.md) — the six-step reshard pipeline in detail, with state flow
- [`debugging.md`](debugging.md) — the four reshard failure modes (NCCL hang / `setStorage size 0` / optimizer NoneType / TP attr assert) and how to triage them
- [`cross_repo.md`](cross_repo.md) — ElasticMegatron vs Megatron-LM-custom: what each owns, what's the patching contract

## When you are about to change code

- If you are about to touch `resharding/`, `transfer/`, or `megatron_manager/` — read [`invariants.md`](invariants.md) first. Several rules are wired across multiple files and are not obvious from any single file.
- If you are about to add a new test or sanity check on ckpts — see the DCP-level tools in [`../../tools/ckpt/`](../../tools/ckpt/) and their README. Do not write yet another ad-hoc comparator.
- If you are about to add Megatron version compatibility (e.g., 0.17), the patterns from the 0.16 work in [`../megatron_016_adaptation/`](../megatron_016_adaptation/) are the closest precedent.

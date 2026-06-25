# Cross-repo relationship

ElasticMegatron is a *side library*: it expects to live next to a Megatron-LM checkout (any reasonably modern version) and patches a few hook points inside it. The result is that *some* of the working code lives in `ElasticMegatron/` (this repo) and *some* lives in `Megatron-LM-custom/` (a sibling checkout).

## Directory layout on disk

```
/mnt/hisys-data/tonic/
├── ElasticMegatron/                ← this repo (under git)
├── Megatron-LM-custom/             ← Megatron 0.16 source with our patches (NOT under git)
└── Megatron-LM-origin/             ← pristine 0.16 reference (NOT under git)
```

`Megatron-LM-custom` is the *target of integration* — we patch a handful of files in it but **do not version-control it**. The canonical source of truth for the patches lives here in ElasticMegatron, specifically:

- [`examples/intra_process/training_016.py`](../../examples/intra_process/training_016.py) — full patched snapshot of `megatron/training/training.py` for 0.16
- [`examples/intra_process/training_011.py`](../../examples/intra_process/training_011.py) — same, for 0.11
- [`examples/intra_process/README.md`](../../examples/intra_process/README.md) — the three patch points described in source form

When you change ElasticMegatron's contract with Megatron (e.g., introduce a new env var, change the elastic-loop signature), update **both** the example snapshot and the README. Otherwise downstream users running from the example file lag behind master.

## What is patched in Megatron-LM-custom for 0.16

Three files:

### 1. `megatron/training/training.py`

The bulk of the integration:

- `init_parallel_strategy_list()` — derives `base` (the launch config) from args, then merges the injected reshard sequence onto it via `elastic_megatron.strategy_inject.build_strategy_list` (`ELASTIC_STRATEGY_LIST` inline JSON override-dicts, or `ELASTIC_STRATEGY_LIST_FILE`; default `[{}]` = no reshard). The former hardcoded `ELASTIC_STRATEGY_MODE` sweeps now live as data under [`examples/strategies/`](../../examples/strategies/).
- `check_reshard(iteration)` — picks the next strategy when `iteration % ELASTIC_RESHARD_INTERVAL == 0`.
- `init_elastic_megatron_manager()` — wires the strategy list into `ElasticMegatronManager`.
- The elastic loop inside `train()` — the `if elastic_megatron_manager: ...` block, using plain rebind `model = training_state.model` (see [`invariants.md`](invariants.md) I-6). The launcher must run with `--eval-iters 0` and no `--save` under this convention.
- `ELASTIC_SAVE_CKPT=1` hook — passes `save_ckpt=True` to `reshard()` so before/after ckpts land in `tools/ckpt/{before,after}_reshard/iter_N/`.
- `ElasticMegatronManager.register(...)` at the top of `pretrain()`.

### 2. `pretrain_gpt.py`

`ELASTIC_DUMP_INPUTS=<path>:<iters>` hook in `forward_step` — dumps `{tokens, labels, loss_mask, attention_mask, position_ids}` on the specified iters. Used to verify that two parallel layouts see consistent input batches (an iter-21 spike investigation tool from Phase B).

### 3. (none others currently)

## The patching contract

The Megatron side is allowed to:

- Import `elastic_megatron` (this repo, must be importable on `PYTHONPATH`).
- Read env vars: `ELASTIC_ENABLED`, `ELASTIC_STRATEGY_LIST` / `ELASTIC_STRATEGY_LIST_FILE`, `ELASTIC_RESHARD_INTERVAL`, `ELASTIC_SAVE_CKPT`, `ELASTIC_DUMP_INPUTS`.
- Call `ElasticMegatronManager.register(...)` and `elastic_megatron_manager.reshard(...)` / `.build_iterators()`.

The Megatron side **must**:

- Update `model` in the elastic loop with **plain rebind** (`model = training_state.model`), and run the launcher with `--eval-iters 0` and no `--save` (I-6). Do **not** use `model[:] =` slice-assignment — it poisons the cached `TrainingState.model` lists.
- Re-derive `forward_backward_func` after reshard (PP size may have changed, which selects a different schedule).
- Re-derive `config = training_state.refresh_config(...)` after reshard.

ElasticMegatron does **not** touch Megatron internals beyond what's documented above. In particular, it does not:

- Override Megatron's parallel-state globals directly. It snapshots them and rebinds via `MegatronStateManager.apply()` — Megatron is then free to read them as usual.
- Patch `_ParamAndGradBuffer` permanently. The mocking in `mock_ddp_buffer_init` is a contextmanager — Megatron sees the original behaviour at training time.

## Why the patches don't go upstream

The patches are intentionally minimal and confined to a handful of clearly-named entry points. They could in principle live as upstream PRs, but:

- The strategy-list / `check_reshard` indirection only makes sense for users who want elastic training. Always-on it would be a behavioural change for everyone else.
- Tying ourselves to Megatron's PR timeline would slow the library down.
- Different Megatron versions have different `training.py` structures (0.11 has `loader_core.py`, 0.16 has `loader_base.py`, etc.), so a single upstream patch wouldn't cover them all.

So the contract is: **`examples/intra_process/training_*.py` is the source of truth for what Megatron needs to look like; copy/diff into your target Megatron checkout.**

> **Note (strategy injection / stale 0.11):** `training_016.py` (0.16) was refactored so the reshard
> strategy list is **injected from the launcher** (`ELASTIC_STRATEGY_LIST` / `ELASTIC_STRATEGY_LIST_FILE`
> → `elastic_megatron/strategy_inject.py`) instead of selected by a hardcoded `ELASTIC_STRATEGY_MODE`.
> `training_011.py` (0.11) is **stale** — it still carries the old `ELASTIC_STRATEGY_MODE` machinery and
> was intentionally not updated; port the same thinning if 0.11 is revived.

## Working with the customized Megatron in this repo

When working on changes that span both repos:

1. Make the changes in `Megatron-LM-custom/megatron/training/training.py` (and `pretrain_gpt.py` if needed).
2. Test end-to-end with `ELASTIC_ENABLED=1 ELASTIC_STRATEGY_LIST_FILE=examples/strategies/<name>.json ./run_dense.sh` (or `run_qwen3_30b.sh`).
3. Before commit, **copy** the new state of `Megatron-LM-custom/megatron/training/training.py` into `examples/intra_process/training_016.py`:
   ```bash
   cp /mnt/hisys-data/tonic/Megatron-LM-custom/megatron/training/training.py \
      /mnt/hisys-data/tonic/ElasticMegatron/examples/intra_process/training_016.py
   ```
4. Update [`examples/intra_process/README.md`](../../examples/intra_process/README.md) if the patch points changed.

This is how the 0.16 adaptation kept the example snapshot honest. See `docs/megatron_016_adaptation/changelog.md` for the worked example.

# CLAUDE.md — agent entry point

> Auto-loaded when an agent opens this repo. It is deliberately thin: it points
> you at the real docs, names the environment you need, and lists the rules that
> are expensive to learn by breaking them. **Read [`docs/README.md`](docs/README.md) next** —
> it has the full guided reading order.

## What this is

ElasticMegatron is a *side library* for Megatron-LM that adds **online parallel-strategy
switching**: during training you call `elastic_megatron_manager.reshard(new_strategy)` and the
model + optimizer state are redistributed under a different `(TP, PP, CP, DP, EP)` layout —
no process restart, no lost optimizer state, in tens to hundreds of milliseconds. The mode
exercised by every example and test here is **intra-process** (fixed world size, partition
changes only).

## Reading order (don't skip)

Do not start editing `resharding/`, `transfer/`, or `megatron_manager/` before reading these:

1. [`README.md`](README.md) — user-facing API.
2. [`docs/project/README.md`](docs/project/README.md) — 5-minute mental model + core concepts.
3. [`docs/project/repo_layout.md`](docs/project/repo_layout.md) — one line per file.
4. [`docs/project/invariants.md`](docs/project/invariants.md) — **the rules below in full; do not break them.**
5. [`docs/project/architecture.md`](docs/project/architecture.md) — the 6-step `reshard()` pipeline.
6. [`docs/project/cross_repo.md`](docs/project/cross_repo.md) — what lives here vs in `Megatron-LM-custom/`.
7. [`docs/project/debugging.md`](docs/project/debugging.md) — symptom → root-cause playbook (Symptoms A–E).

The 0.16 port itself is logged in [`docs/megatron_016_adaptation/`](docs/megatron_016_adaptation/) —
read it only when you need that history; it is **not** required to start.

## Environment & cross-repo layout

ElasticMegatron expects a **sibling Megatron-LM checkout** that carries a handful of patches.
That checkout is **not under git** — the canonical source of the patches lives *here*, in
[`examples/intra_process/training_016.py`](examples/intra_process/training_016.py) (0.11 sibling: `training_011.py`).

```
<workspace>/
├── ElasticMegatron(-clean)/   ← this repo (under git)
├── Megatron-LM-custom/        ← Megatron 0.16 + our patches  (NOT under git)  ← MEGATRON_PATH
└── Megatron-LM-origin/        ← pristine 0.16 reference      (NOT under git)
```

- `MEGATRON_PATH` **must** point at the patched `Megatron-LM-custom` checkout (the launchers
  `:?`-fail without it).
- `PYTHONPATH` defaults to *this script's own directory* + `MEGATRON_PATH`, so a clone tests
  itself rather than a sibling `ElasticMegatron/`. Override by exporting `PYTHONPATH` first.
- Dev convention: **NCCL / torchrun / Megatron timeouts are kept at ~60s** so hangs surface
  fast; bump them for long real runs (`--distributed-timeout-minutes`, `NCCL_TIMEOUT`).

## Running a smoke test / verifying reshard

```bash
# 4-GPU reshard sweep (quick functional check; loss should converge):
MEGATRON_PATH=/path/to/Megatron-LM-custom ./run_experiment.sh dense_mix     # dense
MEGATRON_PATH=/path/to/Megatron-LM-custom ./run_experiment.sh moe_mix       # MoE

# 8-GPU bit-exact ckpt-level verification (the trustworthy correctness signal):
#   1. run a *_full mode with ELASTIC_SAVE_CKPT=1 → before/after ckpts under tools/ckpt/
#   2. tools/ckpt/verify_all.sh compares them via compare_dcp.py + compare_optim_logical.py
```

- **Loss curves cannot validate reshard** — identical baselines already diverge by the NCCL/
  cuBLAS/flash-attn noise floor. Use the **ckpt-level** tools in [`tools/ckpt/`](tools/ckpt/),
  not an ad-hoc comparator. See `docs/project/repo_layout.md` → `tools/`.
- If switching demos, clear stale ckpts first: `rm -rf <BASE_PATH>/log/iter_* <BASE_PATH>/log/latest_checkpointed_iteration.txt`.

## Hard rules (full text + reasons in `docs/project/invariants.md`)

These are wired across multiple files and fail in confusing places if broken:

- **I-6 — elastic loop uses plain rebind** (`model = training_state.model`), **never** `model[:] =`
  slice-assignment (it poisons the cached `TrainingState.model` lists shared across strategies).
  The cost: launcher scripts **must** run with `--eval-iters 0` and no `--save`.
- **I-1 / I-2 / I-3 — EP=1 ⇒ experts live in the *dense* DDP bucket.** Expert-vs-dense
  classification is **name-based** (`".experts." in name`), not attr-based (`allreduce` flips).
- **I-13 / I-14 — tied embedding + PP=1** needs the `_mask_shared_embedding_for_pp1` bucket-sim
  mask *and* the `VirtualParam.is_orphan_for` orphan skip. Both are required; one alone fails differently.
- **I-9 — `is_redundant_backup` fires *only* on an explicit Group-Zero on/off flip**, not on any
  plain TP/PP/CP/EP/world-size change. Non-DGZ reshards never touch that path.
- **I-15 — optimizer state is a variable-length set of *param-shaped* named states**, each with its
  own device/dtype; the reshard plan is computed once per param and reused for all states. Adding a
  new optimizer? All implementation specifics live behind `OptimizerAdapter`
  (`resharding/optimizer_adapter.py`) — add a subclass + a `create()` branch, no call-site edits;
  read `docs/project/optimizer_state_model.md` first. Non-param-shaped states (FP8 scales) are
  unsupported (guarded by assert).
- **I-12 — cross-repo patches are mirrored, not symlinked.** After editing
  `Megatron-LM-custom/megatron/training/training.py`, copy it into
  `examples/intra_process/training_016.py` and update `examples/intra_process/README.md`.

## Convention when integrating into Megatron

Import `elastic_megatron` **before** `torch`/`megatron` (first import in the entry script), so it
can take over parallel-state setup. Adding a new Megatron version (e.g. 0.17)? The 0.16 work in
`docs/megatron_016_adaptation/` is the closest precedent — start from its `README.md`.

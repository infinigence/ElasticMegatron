# Precision-verification strategy lists

A reshard has **two** correctness signals:

- **ckpt bit-exact** (always valid): `tools/ckpt/verify_all.py` compares the before/after-reshard
  checkpoints (weight bit-equality + optimizer-state multiset-equality).
- **loss match** (valid **only when no TP-degree change**): a TP change reorders the tensor-parallel
  reductions, so loss diverges by the NCCL/cuBLAS/flash-attn noise floor (see `docs/project/invariants.md`).
  With **no TP change** the reduction order is preserved, so a resharding run's per-iteration loss
  matches a no-reshard baseline **within ~1e-6** — a cheap extra signal for non-TP reshards.

`dense_no_tp.json` is a **generated** no-TP sequence (committed as a stable fixture): it varies
only `world_size` (8→4→back to base; DP derives from the world) and never the TP degree, so it
carries **both** signals. Produced by `tools/strategy_gen`
(`--model llama2-medium --scenario rl_dp_resize`) and pinned to the generator output by
`tests/test_strategy_fixtures_drift.py`. For a TP-changing sweep (ckpt-only), generate one with
`--scenario verify_sweep`.

## Recipe

```bash
# no-TP: loss match (≤1e-6) + ckpt bit-exact
# 1) baseline loss (no reshard):
ELASTIC_ENABLED=0 MODEL_SIZE=medium TRAIN_ITERS=9 GPUS_PER_NODE=8 ./run_dense.sh   # record per-iter loss
# 2) resharding run — loss must match (1) within ~1e-6 at every iter, AND ckpt bit-exact:
ELASTIC_ENABLED=1 ELASTIC_RESHARD_INTERVAL=1 ELASTIC_SAVE_CKPT=1 MODEL_SIZE=medium TRAIN_ITERS=9 \
  ELASTIC_STRATEGY_LIST_FILE=examples/strategies/precision/dense_no_tp.json ./run_dense.sh
python3 tools/ckpt/verify_all.py            # -> ALL PASS
```

MoE: same flow via `run_qwen3_30b.sh` against a generated MoE no-TP sequence (`--model moe-15b`);
MoE ckpt save must be off-NFS (`/tmp`). Bit-exact is size-independent
(`docs/megatron_016_adaptation/phase_b_report.md` §5), so `MODEL_SIZE=tiny` / `NUM_LAYERS=8` works
identically and is faster for the ckpt check. The generator filters every entry against the
GPU-memory cap, so a listed sequence is already feasible.

# `tools/` — offline helpers

ElasticMegatron's offline tooling, split by purpose: **ckpt-level verification** (`tools/ckpt/`) and **loss-level experiment analysis** (`tools/`).

## Ckpt-level verification (`tools/ckpt/`)

See [`tools/ckpt/README.md`](ckpt/README.md) for the full guide. One-line index:

| Tool | Purpose | Reusable? |
|---|---|---|
| `compare_dcp.py` | Compare model weights between two `dist_ckpt` directories at the logical-tensor level (auto-skips distrib-optimizer flat buffers, which are physically sharded per-DP and not element-wise comparable across configs) | Yes — generic |
| `compare_optim_logical.py` | Compare distrib-optim flat buffers as logical multisets after sorting; handles padding zeros, cross-`dp_group_idx` TP-replica, and DP padding tail; GPU-parallel sort | Yes — generic |
| `verify_all.sh` | Batch runner that pairs both comparators across every `tools/ckpt/{before,after}_reshard/iter_*` subdir | Yes — coupled to the ckpt-save path convention |
| `convert_and_compare.sh` | Older path: Megatron `convert.py` → element-wise compare (partly broken on Megatron 0.16; prefer the DCP-level tools above) | Limited to specific scenarios |
| `run_convert_patch_loader.py` | 0.16 compatibility shim that string-patches `loader_base.py` so Megatron's `convert.py` accepts a forced TP/PP/EP | Limited (tied to convert chain) |
| `compare_ckpt.py` | Legacy `.pt` ckpt comparator — only called by `convert_and_compare.sh` | Limited |

## Loss-level experiment analysis (`tools/`)

| Tool | Purpose | Reusable? |
|---|---|---|
| `noise_floor.py` | Pairwise `|Δloss|` statistics (P50 / P95 / P99 / max + precision-bucket distribution) across N baseline runs | **Yes** — accepts any N log dirs as positional args |
| `plot_loss_curve.py` | Multi-curve loss plotter; auto-detects ElasticMegatron's plain-number `loss.txt` vs Megatron's standard log format | Yes |
| `analyze_experiments.py` | Phase 2 four-way comparator (`dense_baseline` / `dense_mix` / `moe_baseline` / `moe_mix`) — loss alignment + reshard cost | **Half-reusable** — see the hardcoding note below |

### Limits of `analyze_experiments.py`

This script was written for Phase 2's 4-way comparison and hardcodes:

- `EXP_ROOT = "/mnt/hisys-data/tonic/log/experiments"` as the search root.
- Four fixed experiment name suffixes (`dense_baseline`, `dense_mix`, `moe_baseline`, `moe_mix`). Phase B's sweep-style names (`dense_mix_full`, `dense_cp_only`, ...) do not match the auto-discovery fallback; you would need to pass all four `--*-baseline` / `--*-elastic` flags manually.
- The 4-way comparison structure itself is hardcoded into `main()`.

**Recommendation:** for any new long-term comparison need, prefer `noise_floor.py` (which accepts arbitrary log dirs as positional args) wrapped in a small custom script. Do not extend `analyze_experiments.py`.

## Typical workflow

Verifying one round of reshard transitions:

```bash
# 1. Train with ElasticMegatron saving before/after-reshard ckpts to
#    tools/ckpt/{before,after}_reshard/iter_N/
ELASTIC_SAVE_CKPT=1 TRAIN_ITERS=9 ELASTIC_RESHARD_INTERVAL=1 \
  ELASTIC_ENABLED=1 ELASTIC_STRATEGY_LIST_FILE=examples/strategies/precision/dense_no_tp.json \
  ./run_dense.sh
#    (a committed/generated sequence — see examples/strategies/README.md;
#     generate your own via: python3 -m tools.strategy_gen --model ... --scenario ...)

# 2. Offline batch-verify every (before, after) ckpt pair
DEVICES=0,1,2 bash tools/ckpt/verify_all.sh cuda
# Output: "22/22 pass" or similar summary

# 3. Detail on any failing case:
cat /tmp/verify_iter_0000005.log.weight
cat /tmp/verify_iter_0000005.log.optim
```

Loss-noise quantification (Phase 2 style):

```bash
# Noise floor between identical-config baseline runs
python tools/noise_floor.py /path/to/baseline_run_A /path/to/baseline_run_B

# Multi-curve plot
python tools/plot_loss_curve.py /path/to/run1/node0.log /path/to/run2/node0.log
```

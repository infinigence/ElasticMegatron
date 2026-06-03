# Checkpoint verification tools

This directory contains two families of verification tools:

- **DCP-level tools** (recommended on Megatron-LM 0.16+): `compare_dcp.py`, `compare_optim_logical.py`, `verify_all.sh`. These read `torch.distributed.checkpoint` metadata directly and do not depend on Megatron's `convert.py` chain.
- **Legacy convert-based path**: `convert_and_compare.sh`, `compare_ckpt.py`, `run_convert_patch_loader.py`. Uses Megatron's `tools/checkpoint/convert.py` to convert the source ckpt to the target topology and then compares element-wise. Partly broken on Megatron 0.16 — kept as a fallback.

## DCP-level tools (recommended)

### `compare_dcp.py` — model weight comparison

Compares the **logical** content of every model weight across two `dist_ckpt` directories. Automatically skips `distributed_optimizer` flat buffers (those are physically sharded per DP rank — their shapes differ across parallel configurations and cannot be compared element-wise).

```bash
python tools/ckpt/compare_dcp.py <before_reshard_dir> <after_reshard_dir> \
    [--thresh 1e-3] [--device cuda] [--include-optim-buffers]
```

| Flag | Effect |
|---|---|
| `--thresh` | `rel_rms` threshold (default `1e-3`). A correct reshard should be bit-equal (`rel_rms == 0`). |
| `--device cuda` | Run the comparison on GPU. fp64 precision, 1–2 orders of magnitude faster than CPU. |
| `--include-optim-buffers` | Do not skip flat buffers — also run a coarse `sum / L2` sanity check on them. |

### `compare_optim_logical.py` — distrib-optimizer flat buffer multiset comparison

The fp32 `main_param` / `exp_avg` / `exp_avg_sq` tensors stored in `distributed_optimizer` flat buffers form the same **logical** multiset before and after a reshard (reshard only redistributes — the numerical content is invariant). This script concatenates each bucket's non-padding elements, sorts them, and compares the sorted multisets across the two ckpts.

```bash
python tools/ckpt/compare_optim_logical.py <ckpt_A> <ckpt_B> \
    [--thresh 1e-3] [--device cuda] [--devices 0,1,2] [--fp64] [--keep-zeros]
```

| Flag | Effect |
|---|---|
| `--device cuda` | GPU-accelerated sort (1–2 orders of magnitude faster than CPU for large tensors). |
| `--devices 0,1,2` | Multi-GPU parallel sort: the three kinds (`param` / `exp_avg` / `exp_avg_sq`) are sorted concurrently, one kind per GPU. |
| `--fp64` | Use fp64 for comparison (default fp32 — the ckpts themselves are fp32). |
| `--keep-zeros` | Keep zero elements. Default drops them, because `BucketBuilder` inserts intra-param padding zeros whose **count** differs across TP/PP/EP layouts; without dropping, naive multiset comparison fails with `NUMEL MISMATCH`. The script also auto-detects and skips cross-`dp_group_idx` TP replicas by bit-equality check. |

### `verify_all.py` — batch verification (recommended; single-process, multi-GPU)

For every iteration in both `before_reshard/` and `after_reshard/`, runs the weight +
optim comparison. **Imports torch once** and runs the iteration pairs **concurrently across
GPUs** (one GPU per pair) via a thread pool — avoiding the per-pair `import torch` / CUDA-init
cost that dominates `verify_all.sh` for small/medium ckpts. Verbose output goes to a log;
the terminal shows one line per pair + a summary.

```bash
python tools/ckpt/verify_all.py                 # all GPUs, one pair per GPU
GPUS=0,1,2,3 JOBS=4 python tools/ckpt/verify_all.py --thresh 1e-3
python tools/ckpt/verify_all.py --before <dir> --after <dir> --log /tmp/v.log
```

It reuses `compare_dcp.compare()` and `compare_optim_logical.compare_optim()` directly, so
pass/fail semantics are identical to the per-process path.

### `verify_all.sh` — batch verification (shell; per-pair parallel)

Same comparisons via separate processes. Each iteration pair is pinned to one GPU and the
pairs run concurrently (`JOBS`, default = number of GPUs). Failure details in
`/tmp/verify_<iter>.log.{weight,optim}`.

```bash
bash tools/ckpt/verify_all.sh cuda                       # auto: all GPUs, one pair each
GPUS=0,1,2,3 JOBS=4 bash tools/ckpt/verify_all.sh cuda    # limit GPUs / concurrency
```

> Perf note: `compare_dcp.py` no longer calls `torch.cuda.empty_cache()` per key (that forced
> a full device sync every iteration and dominated runtime for many-key ckpts); freed blocks
> are reused by the caching allocator via `del`. Numerically a no-op.

## Legacy convert-based path

### Generating checkpoints

Pass `save_ckpt=True` to `reshard()` so ElasticMegatron persists before/after ckpts:

```python
training_state = elastic_megatron_manager.reshard(new_parallel_strategy, save_ckpt=True)
```

By default the ckpts land in:
- Before reshard: `.../ElasticMegatron/tools/ckpt/before_reshard`
- After reshard:  `.../ElasticMegatron/tools/ckpt/after_reshard`

The same env var `ELASTIC_SAVE_CKPT=1` is honoured by the example `training.py` files in `examples/intra_process/`.

### Running the comparison

```bash
# Example: TP=2/PP=1  →  TP=1/PP=2
export TP=2 PP=1 EP=1                       # Source strategy
export TARGET_TP=1 TARGET_PP=2 TARGET_EP=1   # Target strategy
export ITER=iter_0000005                    # Iteration directory name

bash tools/ckpt/convert_and_compare.sh
```

| Variable | Purpose |
|---|---|
| `TP` / `PP` / `EP` | Parallel strategy of the source ckpt (`before_reshard`) |
| `TARGET_TP` / `TARGET_PP` / `TARGET_EP` | Parallel strategy of the target ckpt (`after_reshard`) |
| `ITER` | Iteration subdir name to verify (default `iter_0000005`) |

The script first invokes `run_convert_patch_loader.py` to perform the Megatron-side conversion, then runs `compare_ckpt.py` on the converted output against the actual `after_reshard` ckpt.

> **Note on Megatron-LM 0.16 compatibility:** `convert_and_compare.sh` relies on `tools/checkpoint/convert.py`, which has several incompatibilities under 0.16 — swiglu's closure-based `ShardedTensor` factory cannot be pickled into the legacy format, and `loader_base.py` calls `model_provider()` without the `model_builder` argument that the partial wrap is supposed to inject. `run_convert_patch_loader.py` works around two of these via string patching, but some issues are not easily fixable. **On 0.16, prefer the DCP-level tools above.**

## Design notes

- **`compare_dcp.py` and `compare_optim_logical.py` are self-contained.** They depend only on `torch.distributed.checkpoint` metadata and the on-disk layout. They do **not** require a working Megatron `convert.py`. As long as mcore's `dist_ckpt` format stays stable (it has been stable across 0.11–0.16), they should keep working on future versions.
- **`compare_ckpt.py`** is a legacy `.pt` comparator, only called by `convert_and_compare.sh`. It is not the main path on 0.16.
- **`run_convert_patch_loader.py`** is a thin compatibility shim that string-patches `loader_base.py` (or `loader_core.py` on older versions). It injects an explicit `print('[loader_patch] forced TP=… PP=… EP=…')` so the patch is observable at runtime — if you do not see that line, the patcher's heuristic anchor (`margs.world_size =`) no longer matches and you need to update `build_patched_loader`.

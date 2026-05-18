#!/usr/bin/env python3
"""
Compare two Megatron dist_ckpt directories at the *logical* (un-sharded) level.

Mcore 的 torch_dist 格式存的是 ShardedTensor,加载时根据当前 mpu 配置自动重组。
但我们要做的是离线对比 —— 我用 torch.distributed.checkpoint 的 lower-level API,
手动把所有分片读出来,按 sharded tensor 的 metadata 还原成完整逻辑 tensor,
再两侧 key-by-key 对比。

不需要起 NCCL,不需要起进程组。只在主 rank 上跑就行。
"""
import argparse
import sys

import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint import FileSystemReader


def load_dcp_state_dict(ckpt_dir: str) -> dict[str, torch.Tensor]:
    """加载 torch_dist ckpt 为完整逻辑 state_dict。

    用 dcp.load 的 planner: 给 reader 一个 metadata-only 的 state_dict,reader 会
    把分片读出来填进去。ShardedTensor 在 0.16 mcore 里是 logical tensor,所以
    metadata 里能直接拿到 shape。
    """
    reader = FileSystemReader(ckpt_dir)
    md = reader.read_metadata()

    # md.state_dict_metadata: dict[str, BytesStorageMetadata|TensorStorageMetadata]
    # Explicit device='cpu' guards against torch.set_default_device('cuda') left
    # by a previous process — a 1.2B fp32 checkpoint would otherwise consume ~5 GB
    # of GPU memory per tensor.
    state_dict = {}
    for fqn, meta in md.state_dict_metadata.items():
        if hasattr(meta, "size"):  # TensorStorageMetadata
            shape = tuple(meta.size)
            dtype = meta.properties.dtype if hasattr(meta, "properties") else torch.float32
            state_dict[fqn] = torch.empty(shape, dtype=dtype, device='cpu')
        # BytesStorageMetadata 跳过(metadata.json 等)

    if not state_dict:
        return {}

    dcp.load(state_dict, storage_reader=reader)
    return state_dict


def _is_distrib_optim_flat_buffer(key: str) -> bool:
    """Distributed optimizer 的 flat buffer 是按 DP rank 物理切片的,EP/TP/DP 切分
    变化时形状会改变,即便逻辑参数完全一致。这些 key 不应参与逻辑等价比较;
    我们关心的是 model weight / fp32 main params / 优化器状态(已经按 sharded
    tensor 形式重组成 logical 的那部分)。"""
    return ".bucket_idx_" in key and ".gbuf_idx_" in key and key.endswith(
        (".param", ".exp_avg", ".exp_avg_sq")
    )


def compare(a_dir: str, b_dir: str, thresh: float = 1e-3, include_optim_buffers: bool = False,
            device: torch.device | None = None) -> bool:
    if device is None:
        device = torch.device('cpu')
    print(f'Compare device: {device}')
    print(f"Loading A from: {a_dir}")
    a = load_dcp_state_dict(a_dir)
    print(f"  -> {len(a)} keys")
    print(f"Loading B from: {b_dir}")
    b = load_dcp_state_dict(b_dir)
    print(f"  -> {len(b)} keys")

    # 在第一次 load 时就把 flat-buffer keys 暂存出来,避免下方 sanity check 重读整个 ckpt。
    flat_a: dict[str, torch.Tensor] = {}
    flat_b: dict[str, torch.Tensor] = {}
    if not include_optim_buffers:
        for k in [k for k in a if _is_distrib_optim_flat_buffer(k)]:
            flat_a[k] = a.pop(k)
        for k in [k for k in b if _is_distrib_optim_flat_buffer(k)]:
            flat_b[k] = b.pop(k)
        if flat_a or flat_b:
            print(f"  (skipping {len(flat_a)} distrib optim flat buffers in A, {len(flat_b)} in B)")

    keys_a = set(a.keys())
    keys_b = set(b.keys())
    only_a = sorted(keys_a - keys_b)
    only_b = sorted(keys_b - keys_a)

    if only_a:
        print(f"\nKeys only in A ({len(only_a)}):")
        for k in only_a[:30]:
            print(f"  {k}  shape={tuple(a[k].shape)}")
        if len(only_a) > 30:
            print(f"  ... {len(only_a) - 30} more")
    if only_b:
        print(f"\nKeys only in B ({len(only_b)}):")
        for k in only_b[:30]:
            print(f"  {k}  shape={tuple(b[k].shape)}")
        if len(only_b) > 30:
            print(f"  ... {len(only_b) - 30} more")

    common = sorted(keys_a & keys_b)
    print(f"\nComparing {len(common)} common keys (rel_rms thresh={thresh}):")
    mismatched = 0
    skipped = 0
    matched = 0
    max_seen = 0.0
    max_seen_key = None
    for k in common:
        va, vb = a[k], b[k]
        if not torch.is_tensor(va) or not torch.is_tensor(vb):
            skipped += 1
            continue
        if va.shape != vb.shape:
            print(f"  SHAPE MISMATCH {k}: {tuple(va.shape)} vs {tuple(vb.shape)}")
            mismatched += 1
            continue
        if va.numel() == 0:
            continue
        # bf16 → float64 for numerical stability,放到 device(GPU/CPU)上做。
        va_f = va.to(device=device, dtype=torch.float64, non_blocking=True)
        vb_f = vb.to(device=device, dtype=torch.float64, non_blocking=True)
        diff = (va_f - vb_f).abs()
        rms = (diff.pow(2).mean()).sqrt().item()
        vb_rms = (vb_f.pow(2).mean()).sqrt().item()
        rel_rms = rms / (vb_rms + 1e-12)
        max_abs = diff.max().item()
        if rel_rms > max_seen:
            max_seen = rel_rms
            max_seen_key = k
        if rel_rms > thresh:
            print(f"  MISMATCH {k}: max_abs={max_abs:.3e}, rms={rms:.3e}, rel_rms={rel_rms:.3e}")
            mismatched += 1
        else:
            matched += 1
        del va_f, vb_f, diff
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    # Optim flat buffers: shape 在不同 DP/EP 下不同,无法逐 element 对比。这里只看
    # 整体 sum / L2 作为 sanity check —— 真正的 logical multiset 比较请用
    # compare_optim_logical.py。flat_a / flat_b 已在上面 pop 出来,不需要重读 ckpt。
    if flat_a or flat_b:
        flat_keys_a = sorted(flat_a)
        flat_keys_b = sorted(flat_b)
        print("\n=== Optim flat buffer (sum / L2) sanity check ===")
        sum_a = sum(flat_a[k].to(torch.float64).sum().item() for k in flat_keys_a)
        sum_b = sum(flat_b[k].to(torch.float64).sum().item() for k in flat_keys_b)
        l2_a = sum(flat_a[k].to(torch.float64).pow(2).sum().item() for k in flat_keys_a) ** 0.5
        l2_b = sum(flat_b[k].to(torch.float64).pow(2).sum().item() for k in flat_keys_b) ** 0.5
        print(f"  total numel A: {sum(flat_a[k].numel() for k in flat_keys_a)}")
        print(f"  total numel B: {sum(flat_b[k].numel() for k in flat_keys_b)}")
        print(f"  sum A: {sum_a:.6e}    sum B: {sum_b:.6e}    rel diff: {abs(sum_a-sum_b)/(abs(sum_b)+1e-12):.3e}")
        print(f"  L2  A: {l2_a:.6e}    L2  B: {l2_b:.6e}    rel diff: {abs(l2_a-l2_b)/(abs(l2_b)+1e-12):.3e}")

    print()
    print("=== Summary ===")
    print(f"  Matched (rel_rms<{thresh}):  {matched} / {len(common)}")
    print(f"  Mismatched (rel_rms>={thresh}): {mismatched}")
    print(f"  Skipped (non-tensor):  {skipped}")
    print(f"  Only in A:  {len(only_a)}")
    print(f"  Only in B:  {len(only_b)}")
    print(f"  Worst rel_rms: {max_seen:.3e}  at {max_seen_key}")
    return (mismatched == 0 and len(only_a) == 0 and len(only_b) == 0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("a", help="ckpt dir A (e.g. before_reshard/iter_xxxxxxxx)")
    p.add_argument("b", help="ckpt dir B (e.g. after_reshard/iter_xxxxxxxx)")
    p.add_argument("--thresh", type=float, default=1e-3, help="rel_rms threshold per tensor")
    p.add_argument("--include-optim-buffers", action="store_true",
                   help="对比 distrib optim 的 flat buffer(默认跳过 —— 这些 buffer 按 DP rank 物理切分,跨并行配置 shape 不同)")
    p.add_argument("--device", default="cpu",
                   help='Device for compare ("cuda" / "cuda:0" / "cpu"). GPU 上比 CPU 快 1-2 数量级。')
    args = p.parse_args()
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        print('CUDA not available, falling back to CPU', file=sys.stderr)
        device = torch.device('cpu')
    ok = compare(args.a, args.b, args.thresh, args.include_optim_buffers, device)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

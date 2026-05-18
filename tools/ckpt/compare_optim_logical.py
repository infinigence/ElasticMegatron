#!/usr/bin/env python3
"""
Logical compare of distributed-optimizer flat buffers across reshard.

Cross EP/DP layouts the *physical* shape of each flat buffer differs (EP=2 has
multiple sub-buckets across two dp_groups, EP=1 has one big dense bucket), so
逐 element 对比无意义。但 reshard 只是 *重新分发* 同一组 fp32 main_param /
exp_avg / exp_avg_sq —— 把所有 flat buffer 的 non-padding 元素拼起来,sort 后
应该是同一个 multiset。

我们做三件事:
  1. 按 per_bucket_numel_unpadded 取每个 bucket 的有效部分(去掉 DP padding)
  2. 把所有 buckets 的 .param / .exp_avg / .exp_avg_sq 拼起来,sorted 比较
  3. 如果 sorted multisets 在 rel_rms<thresh 内一致 → 通过

注:padding 是 0,所以 *不* 截断也能工作(因为 sorted multiset 里 padding
zeros 在两侧都有)— 但为了精度,还是去掉 padding。
"""
import argparse
import io
import re
import sys

import torch
from torch.distributed.checkpoint import FileSystemReader


def _bytes_payload(ckpt_dir: str, fqn_substr: str):
    """Read a BytesStorageMetadata key by parsing storage_data offsets directly.
    Returns the unpickled python object, or None if not found."""
    reader = FileSystemReader(ckpt_dir)
    md = reader.read_metadata()
    for mi, info in md.storage_data.items():
        if not hasattr(mi, 'fqn'):
            continue
        if fqn_substr in mi.fqn:
            with open(f'{ckpt_dir}/{info.relative_path}', 'rb') as f:
                f.seek(info.offset)
                raw = f.read(info.length)
            return torch.load(io.BytesIO(raw), weights_only=False)
    return None


def load_flat_optim(ckpt_dir: str) -> dict[str, dict[str, torch.Tensor]]:
    """Return {bucket_full_key: {'param': T, 'exp_avg': T, 'exp_avg_sq': T, 'numel_unpadded': int,
    'replica_offsets': set[(start, size)] }}.

    bucket_full_key looks like
    'chained_0.optimizer.distributed.dp_group_idx_0.gbuf_idx_0.bucket_idx_0'
    or 'optimizer.distributed.dp_group_idx_0.gbuf_idx_0.bucket_idx_0' (EP=1).

    Tensors are loaded to CPU to guard against torch.set_default_device('cuda') left
    by a previous process. Sorting moves tensors to GPU one pair at a time via
    _sort_on_device, keeping peak GPU memory bounded (~25 GB single GPU).

    Under TP>1, distrib_optimizer creates an independent dp_group_idx per TP rank.
    Non-shardable 1D params (layer-norm, bias) appear as bit-equal replicas at the
    same offset in every dp_group's flat buffer. These replicas are identified by
    matching (gbuf_idx, bucket_idx, offset, size) across dp_group_idx values and
    stored in 'skip_segments' so multiset comparison can deduplicate them (TP rank
    changes would otherwise produce a false-positive numel mismatch).
    """
    reader = FileSystemReader(ckpt_dir)
    md = reader.read_metadata()

    sd = {}
    for fqn, m in md.state_dict_metadata.items():
        if not hasattr(m, 'size'):
            continue
        if 'gbuf_idx_' not in fqn or 'bucket_idx_' not in fqn:
            continue
        sd[fqn] = torch.empty(tuple(m.size), dtype=m.properties.dtype, device='cpu')

    if not sd:
        return {}

    import torch.distributed.checkpoint as dcp
    dcp.load(sd, storage_reader=reader)

    pat = re.compile(
        r'^(?P<prefix>(?:chained_\d+\.)?optimizer\.distributed\.dp_group_idx_\d+)'
        r'\.gbuf_idx_(?P<g>\d+)\.dtype_\([^)]+\)\.bucket_idx_(?P<b>\d+)'
        r'\.(?P<kind>param|exp_avg|exp_avg_sq)$'
    )
    grouped: dict[str, dict[str, torch.Tensor]] = {}
    n_candidate = 0
    for fqn, t in sd.items():
        n_candidate += 1
        m = pat.match(fqn)
        if not m:
            continue
        bucket_key = f"{m['prefix']}.gbuf_idx_{m['g']}.bucket_idx_{m['b']}"
        grouped.setdefault(bucket_key, {})[m['kind']] = t
    # 防 silent pass:flat-buffer 候选 key 存在但 regex 一个都不 match,大概率是 mcore
    # 改了 FQN 格式(比如 dtype 序列化改名),应立刻喊出来,而不是返回空 grouped 让
    # 上层显示 "0 buckets" + "全部 0 偏差 PASS"。
    if n_candidate > 0 and not grouped:
        print(f'[WARN] FQN regex matched 0 of {n_candidate} candidate keys in {ckpt_dir}. '
              f'mcore distrib-optim FQN 格式可能已变 → 请更新 pat 正则。', file=sys.stderr)

    # 拉 per_bucket_numel_unpadded(EP=2 时 chained_0/chained_1 各有一份)
    # storage_data 上每个 chained_X 都有独立 fqn,我们一次扫描全收集
    md = reader.read_metadata()
    for mi, info in md.storage_data.items():
        if not hasattr(mi, 'fqn'):
            continue
        fqn = mi.fqn
        if 'per_bucket_numel_unpadded' not in fqn:
            continue
        # fqn = '<prefix>.per_bucket_numel_unpadded/shard_0_1'
        prefix = fqn.split('.per_bucket_numel_unpadded')[0]
        with open(f'{ckpt_dir}/{info.relative_path}', 'rb') as f:
            f.seek(info.offset)
            raw = f.read(info.length)
        obj = torch.load(io.BytesIO(raw), weights_only=False)
        # obj 形如 [[{(bf16,fp32): [N, ...]}]] —— gbuf_idx -> dtype -> [bucket numels]
        for g_idx, dtype_dict_list in enumerate(obj):
            for dtype_dict in dtype_dict_list if isinstance(dtype_dict_list, list) else [dtype_dict_list]:
                for numels in dtype_dict.values():
                    for b_idx, numel in enumerate(numels):
                        bk = f"{prefix}.gbuf_idx_{g_idx}.bucket_idx_{b_idx}"
                        if bk in grouped:
                            grouped[bk]['numel_unpadded'] = int(numel)

    # ===== Replica detection (TP>1 时跨 dp_group_idx 的 layer-norm 等 replica) =====
    # 同 (chained_prefix, gbuf_idx, bucket_idx, offset, size) 在多个 dp_group_idx 上
    # 出现,且两份内容 bit-equal → 视为 replica。区分 PP/TP 维度的关键:
    #   TP=k:dp_group 间相同 offset 的 segments 内容 bit-equal(都是同一份 layer-norm)
    #   PP=k:dp_group 间相同 offset 的 segments 内容不同(各 PP stage 的不同 fp32 数据)
    # 所以必须 *实际比较* 内容,不能仅看 metadata。
    chunks_by_key: dict = {}  # (canon_prefix, g, b, offset, size) -> [(dp_g, bucket_key)]
    for fqn, m in md.state_dict_metadata.items():
        if not hasattr(m, 'size'):
            continue
        match = pat.match(fqn)
        if not match:
            continue
        if match['kind'] != 'param':
            continue
        prefix_full = match['prefix']
        prefix_canon = re.sub(r'\.dp_group_idx_\d+$', '', prefix_full)
        dp_g_match = re.search(r'\.dp_group_idx_(\d+)$', prefix_full)
        dp_g = int(dp_g_match.group(1)) if dp_g_match else 0
        g_idx = match['g']
        b_idx = match['b']
        bk = f"{prefix_full}.gbuf_idx_{g_idx}.bucket_idx_{b_idx}"

        for c in m.chunks:
            offset = int(c.offsets[0])
            size = int(c.sizes[0])
            seg_key = (prefix_canon, g_idx, b_idx, offset, size)
            chunks_by_key.setdefault(seg_key, []).append((dp_g, bk, offset, size))

    # 对跨 dp_g (>=2 个 dp_group) 的 segments,实际比较 'param' 内容 bit-equal 才视为 replica。
    # 只保留 dp_g 最小的那一份,其它 mask 掉。
    skip_segments: dict[str, list[tuple[int, int]]] = {}
    for seg_key, occurrences in chunks_by_key.items():
        dp_gs_present = sorted({occ[0] for occ in occurrences})
        if len(dp_gs_present) <= 1:
            continue
        # 拿到该 seg 在每个 dp_g 上的 (bucket_key, offset, size)
        by_dpg = {occ[0]: occ for occ in occurrences}
        keep_dp_g = dp_gs_present[0]
        keep_bk, keep_offset, keep_size = by_dpg[keep_dp_g][1:]
        keep_t = grouped[keep_bk]['param'][keep_offset:keep_offset + keep_size]
        for dp_g in dp_gs_present[1:]:
            this_bk, this_offset, this_size = by_dpg[dp_g][1:]
            this_t = grouped[this_bk]['param'][this_offset:this_offset + this_size]
            if torch.equal(keep_t, this_t):
                skip_segments.setdefault(this_bk, []).append((this_offset, this_size))
            # 不 bit-equal 就保留(PP 等不同 stage 的不同数据)

    for bk, segs in skip_segments.items():
        if bk in grouped:
            grouped[bk]['skip_segments'] = sorted(segs)

    return grouped


def collect_unpadded(grouped: dict[str, dict], kind: str,
                     compute_dtype: torch.dtype, drop_zeros: bool) -> torch.Tensor:
    """Return CPU tensor concatenating all unpadded buckets for `kind`.

    When drop_zeros=True, elements equal to zero are filtered out. BucketBuilder
    adds intra-param padding zeros between some layers under TP>1, and the count
    varies across TP/PP configurations, so they must be removed before multiset
    comparison. The risk of false filtering is negligible for trained fp32 weights.

    Replica segments stored in d['skip_segments'] (layer-norm / bias params that
    appear bit-equal across dp_group_idx values) are masked out so each logical
    parameter is counted only once in the multiset.
    """
    parts = []
    for d in grouped.values():
        t = d[kind]
        n = d.get('numel_unpadded', t.numel())
        skip = d.get('skip_segments', [])
        if skip:
            # 构造 mask 把 skip 区域置 False
            mask = torch.ones(n, dtype=torch.bool)
            for off, sz in skip:
                if off < n:
                    end = min(off + sz, n)
                    mask[off:end] = False
            x = t[:n][mask].to(dtype=compute_dtype)
        else:
            x = t[:n].to(dtype=compute_dtype)
        if drop_zeros:
            x = x[x != 0]
        parts.append(x)
    return torch.cat(parts) if parts else torch.empty(0, dtype=compute_dtype)


def _sort_on_device(cpu_tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Sort a (potentially huge) CPU tensor on `device`, returning sorted CPU copy.

    PyTorch sort needs ~3x the input size as workspace (sorted output + index +
    swap buffer). We do it in chunks so peak GPU memory stays bounded:
      1. split into chunks of CHUNK_NUMEL elements
      2. sort each chunk on GPU, copy back to CPU
      3. multi-way merge on CPU (cheap, since each chunk is already sorted)

    For sort sizes ≤ ~500M float32 we sort in one shot (faster than merge).
    """
    if cpu_tensor.numel() == 0:
        return cpu_tensor
    if device.type == 'cpu':
        out, _ = torch.sort(cpu_tensor)
        return out
    BIG = 500_000_000  # >500M numel → chunk
    if cpu_tensor.numel() <= BIG:
        try:
            gpu = cpu_tensor.to(device, non_blocking=True)
            sorted_gpu, _ = torch.sort(gpu)
            out = sorted_gpu.cpu()
            del gpu, sorted_gpu
            torch.cuda.empty_cache()
            return out
        except torch.cuda.OutOfMemoryError:
            # 与下方 chunked 路径对称的 fallback:GPU 装不下就用 CPU sort,慢但不死。
            torch.cuda.empty_cache()
            return torch.sort(cpu_tensor).values
    # Chunked sort + CPU merge
    n = cpu_tensor.numel()
    n_chunks = (n + BIG - 1) // BIG
    sorted_chunks = []
    for i in range(n_chunks):
        sl = slice(i * BIG, min((i + 1) * BIG, n))
        gpu = cpu_tensor[sl].to(device, non_blocking=True)
        s, _ = torch.sort(gpu)
        sorted_chunks.append(s.cpu())
        del gpu, s
        torch.cuda.empty_cache()
    # CPU multi-way merge using torch operations: simple approach — concat+cpu sort
    # 因为 chunked CPU sort 上 CPU 也是 mem-bound,我们直接把 sorted chunks merge:
    # merge 用 heapq 太慢(python overhead),用 torch 的 cat+sort —— 但 cat 后再 sort
    # 又回到 1.68B CPU sort 慢的老问题。所以用 torch.sort 在 GPU 上做 final 合并:
    full_cpu = torch.cat(sorted_chunks)
    del sorted_chunks
    # 这次 cat 后我们仍在 CPU,如果整段还能塞进 GPU 就再 sort 一次(快路径)。
    # 否则 fallback 到 CPU sort(慢但能跑)。
    try:
        gpu = full_cpu.to(device, non_blocking=True)
        sorted_full, _ = torch.sort(gpu)
        out = sorted_full.cpu()
        del gpu, sorted_full
        torch.cuda.empty_cache()
        return out
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return torch.sort(full_cpu).values


def compare_sorted(a_cpu: torch.Tensor, b_cpu: torch.Tensor, name: str, thresh: float,
                   device: torch.device) -> bool:
    """Compare A and B as sorted multisets. A/B are full CPU tensors (potentially big)."""
    if a_cpu.numel() != b_cpu.numel():
        print(f'  [{name}] NUMEL MISMATCH: {a_cpu.numel()} vs {b_cpu.numel()} '
              f'(diff={a_cpu.numel()-b_cpu.numel()})')
        return False
    a_s = _sort_on_device(a_cpu, device)
    b_s = _sort_on_device(b_cpu, device)

    # 流式 reduce(CPU,fp64),避免存整段 diff
    chunk = 64 * 1024 * 1024
    n = a_s.numel()
    sum_diff_sq = 0.0
    sum_b_sq = 0.0
    sum_a_sq = 0.0
    max_abs = 0.0
    sum_a = 0.0
    sum_b = 0.0
    for off in range(0, n, chunk):
        sl = slice(off, min(off + chunk, n))
        ac = a_s[sl].to(torch.float64)
        bc = b_s[sl].to(torch.float64)
        d = (ac - bc).abs()
        sum_diff_sq += d.pow(2).sum().item()
        sum_b_sq += bc.pow(2).sum().item()
        sum_a_sq += ac.pow(2).sum().item()
        sum_a += ac.sum().item()
        sum_b += bc.sum().item()
        m = d.max().item()
        max_abs = max(max_abs, m)
    rms = (sum_diff_sq / n) ** 0.5
    b_rms = (sum_b_sq / n) ** 0.5
    rel = rms / (b_rms + 1e-12)
    l2_a = sum_a_sq ** 0.5
    l2_b = sum_b_sq ** 0.5
    ok = rel < thresh
    flag = '✓' if ok else '✗'
    print(f'  [{name}] {flag} numel={n:,}  '
          f'rel_rms(sorted)={rel:.3e}  max_abs={max_abs:.3e}')
    print(f'         sum:  A={sum_a:.6e}  B={sum_b:.6e}  rel={abs(sum_a-sum_b)/(abs(sum_b)+1e-12):.3e}')
    print(f'         L2:   A={l2_a:.6e}  B={l2_b:.6e}  rel={abs(l2_a-l2_b)/(abs(l2_b)+1e-12):.3e}')
    return ok


def main():
    p = argparse.ArgumentParser()
    p.add_argument('a')
    p.add_argument('b')
    p.add_argument('--thresh', type=float, default=1e-3)
    p.add_argument('--device', default='cpu',
                   help='Device for sort/compare ("cuda" / "cuda:0" / "cpu"). '
                        'GPU 上 sort 大 tensor 比 CPU 快 1-2 个数量级,但需要 ~16GB 显存。')
    p.add_argument('--devices', default=None,
                   help='Comma-separated GPU ids for parallel sort across (param/exp_avg/exp_avg_sq), '
                        'e.g. "0,1,2". 覆盖 --device。3 个 kind 各分配一个 GPU 同时跑。')
    p.add_argument('--fp64', action='store_true',
                   help='Use fp64 for compare (default fp32, since ckpt itself is fp32). '
                        'Only useful when one side is bf16.')
    p.add_argument('--keep-zeros', action='store_true',
                   help='Do NOT drop zero elements before comparing. By default we drop '
                        'zeros because BucketBuilder inserts intra-param padding (0) whose '
                        'count differs across TP/PP/EP layouts, making naive multiset '
                        'comparison fail with NUMEL MISMATCH even when reshard is correct.')
    args = p.parse_args()

    if args.devices is not None:
        device_ids = [int(x) for x in args.devices.split(',') if x.strip()]
        assert len(device_ids) >= 1, '--devices needs at least one id'
        if not torch.cuda.is_available():
            print('CUDA not available, falling back to CPU', file=sys.stderr)
            devices = [torch.device('cpu')] * 3
        else:
            # 3 个 kind:param / exp_avg / exp_avg_sq → 轮转分配
            devices = [torch.device(f'cuda:{device_ids[i % len(device_ids)]}') for i in range(3)]
    else:
        device = torch.device(args.device)
        if device.type == 'cuda' and not torch.cuda.is_available():
            print('CUDA not available, falling back to CPU', file=sys.stderr)
            device = torch.device('cpu')
        devices = [device, device, device]

    compute_dtype = torch.float64 if args.fp64 else torch.float32
    drop_zeros = not args.keep_zeros
    print(f'Using devices (param/exp_avg/exp_avg_sq): {devices}, compute_dtype: {compute_dtype}, drop_zeros: {drop_zeros}')

    print(f'Loading A: {args.a}')
    A = load_flat_optim(args.a)
    print(f'  buckets in A: {len(A)}')
    for bk, d in A.items():
        n_pad = d['param'].numel()
        n_unpad = d.get('numel_unpadded', n_pad)
        print(f'    {bk}  numel={n_pad:,} (unpadded={n_unpad:,})')

    print(f'\nLoading B: {args.b}')
    B = load_flat_optim(args.b)
    print(f'  buckets in B: {len(B)}')
    for bk, d in B.items():
        n_pad = d['param'].numel()
        n_unpad = d.get('numel_unpadded', n_pad)
        print(f'    {bk}  numel={n_pad:,} (unpadded={n_unpad:,})')

    print('\n=== Cross-layout multiset compare (sorted) ===')

    # 多卡并行:把每个 kind 在独立 worker thread 上跑(GPU 操作可以从主线程外发起,
    # CUDA stream 自然并行)。需要的 sort/比较都在 device-local stream 上完成。
    import concurrent.futures
    kinds = ('param', 'exp_avg', 'exp_avg_sq')
    a_cats = {k: collect_unpadded(A, k, compute_dtype, drop_zeros) for k in kinds}
    b_cats = {k: collect_unpadded(B, k, compute_dtype, drop_zeros) for k in kinds}

    results = {}
    if len({d.index for d in devices if d.type == 'cuda'}) > 1:
        # 多卡:并行
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(kinds)) as pool:
            futs = {
                pool.submit(compare_sorted, a_cats[k], b_cats[k], k, args.thresh, devices[i]): k
                for i, k in enumerate(kinds)
            }
            for fut in concurrent.futures.as_completed(futs):
                k = futs[fut]
                results[k] = fut.result()
    else:
        # 单卡或 CPU:串行(避免重复抢同一 GPU 显存)
        for i, k in enumerate(kinds):
            results[k] = compare_sorted(a_cats[k], b_cats[k], k, args.thresh, devices[i])
            del a_cats[k], b_cats[k]
            if devices[i].type == 'cuda':
                torch.cuda.empty_cache()

    ok = all(results.values())
    print()
    print('PASS' if ok else 'FAIL')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()

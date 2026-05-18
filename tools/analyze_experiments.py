#!/usr/bin/env python3
"""
analyze_experiments.py

对比 baseline 与 elastic-reshard 两组 run 的 loss 曲线和 per-iter 耗时,检查:
  1. loss 对齐精度:reshard 只是重新分布 optimizer state / model weights,不改
     变数学上要计算的 gradient,所以 loss 应逐 iter 与 baseline 对齐(bf16
     累积舍入内)。输出小数点后 6 位,并按 10^-4 / 10^-5 / 10^-6 三个
     精度阶统计逐 iter 对齐的占比。
  2. reshard 耗时:在触发 reshard 的 iter(每 RESHARD_INTERVAL 一次),
     per-iter elapsed 会明显高于 steady-state;我们把超出 baseline avg 的
     部分算作 reshard overhead。

用法:
  python3 analyze_experiments.py                # 用 EXPERIMENTS 里默认映射
  python3 analyze_experiments.py <json_file>    # 传一个 {name: dirname} json
"""
import argparse
import json
import os
import re
from collections import OrderedDict

# 最新一次长程实验的映射(100 iter + interval 5)
DEFAULT_EXPERIMENTS = OrderedDict([
    ("dense_baseline", None),
    ("dense_mix",      None),
    ("moe_baseline",   None),
    ("moe_mix",        None),
])
EXP_ROOT = "/mnt/hisys-data/tonic/log/experiments"

ITER_RE = re.compile(
    r"iteration\s+(\d+)\/\s*\d+.*?elapsed time per iteration \(ms\):\s*([\d.]+).*?"
    r"lm loss:\s*([-+]?[\d.]+E[-+]?\d+)(?:.*?load_balancing_loss:\s*([-+]?[\d.]+E[-+]?\d+))?",
    re.DOTALL,
)

# consumed samples 单独匹配,避免 optional group 让整个 regex 失败
CONSUMED_RE = re.compile(r"consumed samples:\s*(\d+)")


def parse_log(log_path):
    """返回 list of (iter, ms, loss, lb_loss_or_None, consumed_samples)。"""
    iters = []
    with open(log_path) as f:
        for line in f:
            if "lm loss:" not in line:
                continue
            m = ITER_RE.search(line)
            if not m:
                continue
            it = int(m.group(1))
            ms = float(m.group(2))
            loss = float(m.group(3))
            lb = float(m.group(4)) if m.group(4) else None
            cm = CONSUMED_RE.search(line)
            consumed = int(cm.group(1)) if cm else None
            iters.append((it, ms, loss, lb, consumed))
    return iters


def latest_log_dir(name):
    """在 EXP_ROOT 下找最近一次 dirname 后缀为 _<name> 的目录。"""
    candidates = [
        d for d in os.listdir(EXP_ROOT)
        if os.path.isdir(os.path.join(EXP_ROOT, d)) and d.endswith("_" + name)
    ]
    if not candidates:
        return None
    return os.path.join(EXP_ROOT, sorted(candidates)[-1])


def summarize_one(name, iters):
    if not iters:
        print(f"[{name}] empty")
        return
    losses = [x[2] for x in iters]
    times  = [x[1] for x in iters]
    steady_times = times[1:] if len(times) > 1 else times
    print(f"\n[{name}] {len(iters)} iters")
    print(f"  loss:  first={losses[0]:.6f}, last={losses[-1]:.6f}, min={min(losses):.6f}")
    print(f"  time:  iter1={times[0]:.1f}ms, avg(iter2+)={sum(steady_times)/len(steady_times):.1f}ms, "
          f"max={max(times):.1f}ms")


def compare(baseline_iters, elastic_iters, label, reshard_interval):
    print(f"\n======== {label}:elastic vs baseline loss 对齐(精度至 10^-6)========")
    b = {row[0]: row for row in baseline_iters}
    e = {row[0]: row for row in elastic_iters}
    common = sorted(set(b) & set(e))

    # 统计精度阶占比
    bucket_4 = 0  # |Δ| < 1e-4
    bucket_5 = 0  # |Δ| < 1e-5
    bucket_6 = 0  # |Δ| < 1e-6
    worse = 0     # |Δ| >= 1e-4
    max_abs = 0.0
    max_abs_iter = -1
    max_reshard_abs = 0.0
    max_reshard_iter = -1

    print(f"  {'iter':>4}  {'baseline':>12}  {'elastic':>12}  {'Δloss':>12}  "
          f"{'|Δ|/|base|':>11}  {'cs_b':>7}  {'cs_e':>7}  reshard?")
    for it in common:
        _, _, bl, _, cs_b = b[it]
        _, _, el, _, cs_e = e[it]
        d = el - bl
        ad = abs(d)
        rel = ad / max(abs(bl), 1e-12)
        is_reshard = (it > 0 and it % reshard_interval == 0)
        mark = "*" if is_reshard else ""

        if ad < 1e-6:
            bucket_6 += 1
        elif ad < 1e-5:
            bucket_5 += 1
        elif ad < 1e-4:
            bucket_4 += 1
        else:
            worse += 1

        if ad > max_abs:
            max_abs = ad
            max_abs_iter = it
        if is_reshard and ad > max_reshard_abs:
            max_reshard_abs = ad
            max_reshard_iter = it

        cs_b_str = f"{cs_b}" if cs_b is not None else "?"
        cs_e_str = f"{cs_e}" if cs_e is not None else "?"
        print(f"  {it:>4}  {bl:>12.6f}  {el:>12.6f}  {d:>+12.3e}  {rel:>11.3e}  "
              f"{cs_b_str:>7}  {cs_e_str:>7}  {mark}")

    N = len(common)
    print(f"\n  总 {N} iter 的精度阶分布:")
    print(f"    |Δloss| < 1e-6:  {bucket_6:>3} ({100*bucket_6/N:.1f}%)")
    print(f"    |Δloss| < 1e-5:  {bucket_5:>3} ({100*bucket_5/N:.1f}%)  [5-位精度]")
    print(f"    |Δloss| < 1e-4:  {bucket_4:>3} ({100*bucket_4/N:.1f}%)  [4-位精度]")
    print(f"    |Δloss| ≥ 1e-4:  {worse:>3} ({100*worse/N:.1f}%)  [低于 4 位]")
    print(f"  最大 |Δloss| = {max_abs:.3e} at iter {max_abs_iter}")
    if max_reshard_iter > 0:
        print(f"  reshard iter 最大 |Δloss| = {max_reshard_abs:.3e} at iter {max_reshard_iter}")


def reshard_cost(iters, baseline_avg_ms, reshard_interval):
    print(f"\n======== reshard 开销(相对 baseline steady 均值 {baseline_avg_ms:.1f}ms)========")
    reshard_iters = [(it, ms) for (it, ms, *_rest) in iters if it > 0 and it % reshard_interval == 0]
    if not reshard_iters:
        print("  (没有 reshard iter)")
        return
    overheads = []
    for it, ms in reshard_iters:
        oh = ms - baseline_avg_ms
        overheads.append(oh)
        print(f"  iter {it:>3}:  {ms:>8.1f} ms   (overhead {oh:>+8.1f} ms)")
    print(f"  summary: {len(overheads)} reshard events, "
          f"overhead avg={sum(overheads)/len(overheads):.1f} ms, max={max(overheads):.1f} ms")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dense-baseline", default=None)
    parser.add_argument("--dense-elastic", default=None)
    parser.add_argument("--moe-baseline", default=None)
    parser.add_argument("--moe-elastic", default=None)
    parser.add_argument("--reshard-interval", type=int, default=5)
    args = parser.parse_args()

    resolved = {
        "dense_baseline": args.dense_baseline or latest_log_dir("dense_baseline"),
        "dense_mix":      args.dense_elastic  or latest_log_dir("dense_mix"),
        "moe_baseline":   args.moe_baseline   or latest_log_dir("moe_baseline"),
        "moe_mix":        args.moe_elastic    or latest_log_dir("moe_mix"),
    }

    parsed = {}
    for name, logdir in resolved.items():
        if logdir is None:
            print(f"[{name}] no log dir found under {EXP_ROOT}")
            continue
        candidates = [f for f in os.listdir(logdir) if f.endswith(".log")]
        if not candidates:
            print(f"[{name}] no .log in {logdir}")
            continue
        log_path = os.path.join(logdir, candidates[0])
        print(f"[{name}] reading {log_path}")
        iters = parse_log(log_path)
        parsed[name] = iters
        summarize_one(name, iters)

    # Dense
    if "dense_baseline" in parsed and "dense_mix" in parsed:
        bi = parsed["dense_baseline"]
        ei = parsed["dense_mix"]
        b_times = [row[1] for row in bi[1:]]
        b_avg = sum(b_times) / len(b_times) if b_times else 0
        compare(bi, ei, "dense", args.reshard_interval)
        reshard_cost(ei, b_avg, args.reshard_interval)

    # MoE
    if "moe_baseline" in parsed and "moe_mix" in parsed:
        bi = parsed["moe_baseline"]
        ei = parsed["moe_mix"]
        b_times = [row[1] for row in bi[1:]]
        b_avg = sum(b_times) / len(b_times) if b_times else 0
        compare(bi, ei, "moe", args.reshard_interval)
        reshard_cost(ei, b_avg, args.reshard_interval)


if __name__ == "__main__":
    main()

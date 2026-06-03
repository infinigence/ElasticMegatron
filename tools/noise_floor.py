#!/usr/bin/env python3
"""量化 baseline 之间的 noise floor。

输入:两组(或多组)配置完全相同的 baseline run 目录,输出每对 |Δloss| 的统计。
"""
import argparse, os, re, statistics

ITER_RE = re.compile(
    r"iteration\s+(\d+)\/\s*\d+.*?lm loss:\s*([-+]?[\d.]+E[-+]?\d+)",
    re.DOTALL,
)


def parse(log_dir):
    log = next(f for f in os.listdir(log_dir) if f.endswith(".log"))
    losses = {}
    with open(os.path.join(log_dir, log)) as f:
        for line in f:
            if "lm loss:" not in line:
                continue
            m = ITER_RE.search(line)
            if m:
                losses[int(m.group(1))] = float(m.group(2))
    return losses


def main():
    p = argparse.ArgumentParser()
    p.add_argument("dirs", nargs="+", help="baseline run dirs (>=2)")
    args = p.parse_args()
    parsed = [parse(d) for d in args.dirs]

    print(f"Comparing {len(parsed)} baseline runs:")
    for d in args.dirs:
        print(f"  {d}")

    common_iters = sorted(set.intersection(*[set(p.keys()) for p in parsed]))

    # Pairwise differences across all runs at each iter
    deltas = {it: [] for it in common_iters}
    for it in common_iters:
        vals = [p[it] for p in parsed]
        for i in range(len(vals)):
            for j in range(i + 1, len(vals)):
                deltas[it].append(abs(vals[i] - vals[j]))

    print(f"\n{'iter':>4}  {'losses (per run)':>50}  {'max |Δ|':>10}")
    for it in common_iters:
        vals = [p[it] for p in parsed]
        max_d = max(deltas[it]) if deltas[it] else 0.0
        v_str = " ".join(f"{v:.6f}" for v in vals)
        print(f"  {it:>3}  {v_str:>50}  {max_d:>10.3e}")

    print("\n=== Noise-floor summary across all iters ===")
    all_deltas = [d for it in common_iters for d in deltas[it]]
    print(f"  N pairs           = {len(all_deltas)}")
    print(f"  P50 |Δloss|       = {statistics.median(all_deltas):.3e}")
    p95 = sorted(all_deltas)[int(0.95 * (len(all_deltas) - 1))]
    p99 = sorted(all_deltas)[int(0.99 * (len(all_deltas) - 1))]
    print(f"  P95 |Δloss|       = {p95:.3e}")
    print(f"  P99 |Δloss|       = {p99:.3e}")
    print(f"  max |Δloss|       = {max(all_deltas):.3e}")
    # By precision bucket
    n6 = sum(1 for d in all_deltas if d < 1e-6)
    n5 = sum(1 for d in all_deltas if 1e-6 <= d < 1e-5)
    n4 = sum(1 for d in all_deltas if 1e-5 <= d < 1e-4)
    n3 = sum(1 for d in all_deltas if 1e-4 <= d < 1e-3)
    n2 = sum(1 for d in all_deltas if 1e-3 <= d < 1e-2)
    n1 = sum(1 for d in all_deltas if d >= 1e-2)
    N = len(all_deltas)
    print(f"  |Δ| < 1e-6:  {n6}/{N}  ({100*n6/N:.1f}%)")
    print(f"  |Δ| < 1e-5:  {n6+n5}/{N}  ({100*(n6+n5)/N:.1f}%)")
    print(f"  |Δ| < 1e-4:  {n6+n5+n4}/{N}  ({100*(n6+n5+n4)/N:.1f}%)")
    print(f"  |Δ| < 1e-3:  {n6+n5+n4+n3}/{N}  ({100*(n6+n5+n4+n3)/N:.1f}%)")
    print(f"  |Δ| < 1e-2:  {n6+n5+n4+n3+n2}/{N}  ({100*(n6+n5+n4+n3+n2)/N:.1f}%)")
    print(f"  |Δ| ≥ 1e-2:  {n1}/{N}  ({100*n1/N:.1f}%)")


if __name__ == "__main__":
    main()

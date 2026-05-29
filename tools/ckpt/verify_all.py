#!/usr/bin/env python3
"""Single-process, multi-GPU batch verifier for (before, after) reshard ckpt pairs.

Why this exists alongside verify_all.sh: the shell version spawns 2 fresh Python
processes per pair (7 pairs => 14× `import torch` + CUDA init). For tiny/medium
ckpts that startup cost dominates. This driver imports torch ONCE and loops every
pair in-process, running pairs concurrently across GPUs via a thread pool (torch
GPU ops and DCP file IO release the GIL, so threads give real parallelism). Each
pair is pinned to one GPU.

It reuses the exact comparison functions from compare_dcp.py / compare_optim_logical.py
(no reimplementation), so pass/fail semantics are identical to the per-process path.

Usage:
  tools/ckpt/verify_all.py [--before DIR] [--after DIR] [--thresh 1e-3]
                           [--gpus 0,1,2,3,4,5,6,7] [--jobs N] [--log PATH]

Verbose comparator output goes to --log (default /tmp/verify_all.log); the terminal
shows one line per pair plus a summary. Exit 0 iff every pair is weight bit-equal
and optim multiset-equal.
"""
import argparse
import os
import sys
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare_dcp import compare as compare_weights  # noqa: E402
from compare_optim_logical import compare_optim  # noqa: E402

# The real terminal — driver progress/summary go here, bypassing the stdout
# redirect that captures the (very verbose) comparator prints into the log file.
_TERM = sys.__stdout__
_PRINT_LOCK = threading.Lock()


def term(msg: str) -> None:
    with _PRINT_LOCK:
        print(msg, file=_TERM, flush=True)


def _iter_pairs(before_dir: str, after_dir: str) -> list[str]:
    b = {d for d in os.listdir(before_dir) if d.startswith("iter_") and d[5:].isdigit()}
    a = {d for d in os.listdir(after_dir) if d.startswith("iter_") and d[5:].isdigit()}
    return sorted(b & a)


def _verify_one(it: str, before_dir: str, after_dir: str, thresh: float,
                gpu: int) -> tuple[str, bool, bool]:
    """Run weight + optim compare for one iter pair, pinned to one GPU."""
    a = os.path.join(before_dir, it)
    b = os.path.join(after_dir, it)
    dev = torch.device(f"cuda:{gpu}") if torch.cuda.is_available() else torch.device("cpu")
    try:
        w_ok = compare_weights(a, b, thresh=thresh, device=dev)
    except Exception:
        traceback.print_exc()
        w_ok = False
    try:
        # single device for all 3 kinds => serial branch inside compare_optim
        # (no nested thread pool); cross-pair parallelism comes from our pool.
        o_ok = compare_optim(a, b, thresh=thresh, devices=[dev, dev, dev])
    except Exception:
        traceback.print_exc()
        o_ok = False
    term(f"[{it}] {'✓' if w_ok else '✗'} weight | {'✓' if o_ok else '✗'} optim")
    return it, w_ok, o_ok


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser()
    p.add_argument("--before", default=os.path.join(here, "before_reshard"))
    p.add_argument("--after", default=os.path.join(here, "after_reshard"))
    p.add_argument("--thresh", type=float, default=float(os.environ.get("THRESH", "1e-3")))
    p.add_argument("--gpus", default=os.environ.get("GPUS", ""),
                   help="comma-separated GPU ids (default: CUDA_VISIBLE_DEVICES, else all)")
    p.add_argument("--jobs", type=int, default=int(os.environ.get("JOBS", "0")),
                   help="max concurrent pairs (default: number of GPUs)")
    p.add_argument("--log", default=os.environ.get("VERIFY_LOG", "/tmp/verify_all.log"))
    args = p.parse_args()

    if not os.path.isdir(args.before) or not os.path.isdir(args.after):
        term(f"No ckpts at {args.before} / {args.after}")
        sys.exit(1)

    # GPU pool
    if args.gpus.strip():
        gpus = [int(x) for x in args.gpus.split(",") if x.strip() != ""]
    elif os.environ.get("CUDA_VISIBLE_DEVICES", "").strip():
        gpus = list(range(len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))))
    elif torch.cuda.is_available():
        gpus = list(range(torch.cuda.device_count()))
    else:
        gpus = [0]
    ngpu = max(1, len(gpus))
    jobs = args.jobs if args.jobs > 0 else ngpu

    pairs = _iter_pairs(args.before, args.after)
    if not pairs:
        term("No (before, after) iter pairs found.")
        sys.exit(1)

    term(f"Verifying {len(pairs)} pair(s) over GPUs {gpus}, up to {jobs} concurrent "
         f"(verbose log -> {args.log}) ...")

    # Capture the comparators' verbose stdout into one log file; driver lines go
    # to the real terminal via term().
    logf = open(args.log, "w")
    old_stdout = sys.stdout
    sys.stdout = logf
    results: list[tuple[str, bool, bool]] = []
    try:
        with ThreadPoolExecutor(max_workers=jobs) as pool:
            futs = [
                pool.submit(_verify_one, it, args.before, args.after, args.thresh,
                            gpus[i % ngpu])
                for i, it in enumerate(pairs)
            ]
            for fut in as_completed(futs):
                results.append(fut.result())
    finally:
        sys.stdout = old_stdout
        logf.close()

    total = len(results)
    weight_pass = sum(1 for _, w, _ in results if w)
    optim_pass = sum(1 for _, _, o in results if o)
    weight_fail = sorted(it for it, w, _ in results if not w)
    optim_fail = sorted(it for it, _, o in results if not o)

    term("\n================================================================")
    term("Summary")
    term(f"  total iter pairs: {total}")
    term(f"  weight pass:      {weight_pass}/{total}")
    term(f"  optim pass:       {optim_pass}/{total}")
    if weight_fail:
        term(f"  weight fail iters: {' '.join(weight_fail)}")
    if optim_fail:
        term(f"  optim fail iters:  {' '.join(optim_fail)}")
    if weight_pass == total and optim_pass == total:
        term("  → ALL PASS")
        sys.exit(0)
    sys.exit(1)


if __name__ == "__main__":
    main()

# cpu-adam reshard transport: pinned D2H bounce + the production-vs-profiled lesson

> **Status (2026-06-19):** implemented + A100-measured on `fix/transfer-flat-peak`.
> `ELASTIC_TRANSFER_PINNED_BOUNCE` (default **OFF**) routes the cpu-adam unpack (D2H)
> through a bounded pinned host bounce. **Bit-exact. Production win ~5%** — NOT the 3.79×
> the per-phase profiler suggested; that gap is the headline lesson below. This supersedes
> the stale 2026-06-03 [`cpu-adam-overlap.md`](cpu-adam-overlap.md) on this branch.

## What was done

The reshard main-process transfer (`communicator.py::BatchedTransfer.unpack`) stages received
GPU bytes back into the (cpu-offloaded) destination optimizer state. Under
`--use-precision-aware-optimizer` (hard-wired for the CPU-offload path) Megatron's HDO rebinds
the fp32 master to **pageable** host memory, so the per-slice `GPU → pageable` D2H runs at
pageable PCIe speed (~2.9 GB/s) and `non_blocking=True` is effectively synchronous.

The bounce (only on the D2H — H2D is ~41 ms, not worth it; gated to pageable-host dst, so a
no-op for GPU-adam and already-pinned dst):
- one bulk `GPU → PINNED` copy (~25 GB/s) into a reused pinned host buffer, then a CPU scatter
  `PINNED → pageable dst` (host memcpy);
- the pinned buffer is **bounded** (`ELASTIC_TRANSFER_PINNED_BOUNCE_BYTES`, default 256 MiB);
  the D2H+scatter loops through it in sub-chunks, so the page-locked footprint stays small
  regardless of chunk size (a multi-GB `cudaHostAlloc` is slow/synchronizing and contends with
  NCCL's own pinned pool — see the A/B below);
- **local-only**: same bytes, same NCCL op set/order ⇒ bit-exact and needs no cross-rank match
  (unlike `_pack` / the staging cap). `ELASTIC_TRANSFER_PINNED_BOUNCE` is per-process and may be
  set independently per rank.

## The A100 measurement (30B, moe_30b W8→W4 reshard, cpu-adam, 45.35 GB moved)

| metric | bounce OFF | bounce ON (bounded 256 MiB) |
|---|---|---|
| **per-phase profiled** (`ELASTIC_TRANSFER_LOG_LEVEL=1`) D2H/unpack | ~22.0 s (2.06 GB/s) | **~5.2 s (7.80 GB/s) — 3.79×** |
| per-phase profiled comm/NCCL | ~1.7 s | ~10–14 s (variable) |
| **production wall-clock** (`ELASTIC_TRANSFER_LOG_LEVEL=0`, the default) | **45.3 s (1.00 GB/s)** | **42.9 s (1.06 GB/s) — ~5%** |

bit-exact: `verify_all.py` 8/8 weight + 8/8 optim **ALL PASS** with the bounce on, at both the
256 MiB and a 1 MiB (many-sub-chunk) cap.

## The lesson: profiled D2H win ≫ production win, because production already overlaps

The per-phase profiler showed a 3.79× D2H drop, but the **production wall-clock improved only
~5%**. Why: `BatchedTransfer._phase()` brackets each phase with `torch.cuda.synchronize()`, but
**only when `ELASTIC_TRANSFER_LOG_LEVEL != 0`**. Those syncs *serialize* pack/comm/unpack into
cleanly-separable buckets, so the slow pageable D2H looks like a big standalone cost. In
production (`LOG_LEVEL=0`, no per-phase syncs) the D2H already overlaps the NCCL comm on the
async default stream, so making the D2H 3.8× faster mostly speeds up work that was already
hidden. The "comm 10–21 s, inverse to unpack" wobble was the **same sync-attribution artifact**
(no comm/unpack split exists at `LOG_LEVEL=0`). Bounding the bounce roughly halved the *profiled*
comm inflation (the multi-GB pinned alloc was real contention) but, again, the *production*
number is what matters.

**Takeaway for future perf work:** trust the unconditional outer transfer wall-clock, not the
`_phase` per-phase numbers — the latter carry serialization overhead that does not exist in
production. The doc-claimed "D2H-bound → pin the staging" is a *real but small* (~5%) production
lever, not a multi-× one.

## Why the bounce stays default OFF (and is kept as opt-in)

A bit-exact, no-downside ~5% win on cpu-adam reshards, gated to the offloaded path. Kept as the
EM-side implementation of the documented "pin the staging" lever, but **not flipped default-ON**:
5% does not justify new per-reshard pinned-allocation behaviour as a default, and the real
bottleneck is elsewhere.

## The real bottleneck + next levers

The reshard transfer is **~1 GB/s end-to-end** (45 GB / 45 s) — *below* any single phase's rate
(NCCL 6.4, H2D 5.6, D2H 2.9 GB/s profiled), i.e. the cost is the serial dependency chain plus
overhead (collect, `_pre`/`_post` DP gather-scatter, per-chunk Python, NCCL launch amortization),
not the D2H sub-phase. To find a bigger win:
1. **Production-faithful profiling** (CUDA events / coarse one-shot Timers, NOT the `_phase`
   per-phase syncs) of where the ~45 s actually goes.
2. A full **3-stream pipeline** is likely *also* small here — production already overlaps phases
   on the default stream, so an explicit pipeline mostly formalizes that.
3. Pinning the HDO master itself (Megatron `pin_cpu_params`, cross-repo) would speed H2D+D2H at
   the source, but the same production-overlap caveat applies.

(nsys would localize the profiled comm window definitively, but v2025.6.1 only finalizes its
report on a clean target exit, which the elastic 30B run never gives post-reshard — the
production A/B settles the decision without it.)

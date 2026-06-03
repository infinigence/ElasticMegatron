# Design note: overlap CPU-adam H2D/D2H staging with NCCL

> **Status: not implemented (2026-06-03).** This captures a discussion: whether to do it,
> how, the cost, and what to measure first. The current `BatchedTransfer.transfer` packed
> path does **not** overlap; `ELASTIC_MAX_INFLIGHT_BYTES` bucketing only bounds peak
> memory, not latency. Read the "Measure first" section before building this.

## Problem

When CPU offload (HybridDeviceOptimizer / CPU-adam) is on, offloaded optimizer state lives
on pinned CPU. NCCL only moves CUDA tensors, so the packed path does:
1. **H2D**: copy (possibly CPU) send slices into a GPU packed buffer (stage-in);
2. **NCCL**: `batch_isend_irecv` cross-rank transfer;
3. **D2H**: copy received GPU bytes back into (possibly CPU) destinations (stage-out).

Today these three run **serially on the default stream** (`pack all → wait → unpack all`).
Bucketing splits that into sequential windows — lower peak memory, but each window is still
`H2D → NCCL → D2H` serial, so total time ≈ `ΣH2D + ΣNCCL + ΣD2H`.

## Why there is overlap to exploit

The three costs use **three independent hardware paths**:
- **H2D (send-side stage-in)** and **D2H (recv-side stage-out)** go over PCIe, which is
  **full-duplex** → H2D and D2H can run at the same time (a rank usually both sends and
  receives in one reshard, so both directions have work).
- **NCCL p2p** (intra-node) goes over NVLink — separate hardware from PCIe.

For offload, PCIe (~tens of GB/s) is typically an order of magnitude slower than NVLink
(~hundreds of GB/s), so the **bottleneck is usually the PCIe H2D/D2H, not NCCL**. An ideal
pipeline collapses `ΣH2D + ΣNCCL + ΣD2H` to roughly `max(ΣH2D, ΣD2H) + a little exposed
NCCL + fill/drain`.

> **No win for the pure-GPU case**: without offload, packing is a D2D copy (fast, and it
> contends with NCCL for GPU resources). This is an **offload-only** optimization.

## Design: multi-stream double-buffered pipeline on the existing bucket boundaries

Keep the bucket boundaries (still step-aligned + all-reduced stride, cross-rank consistent)
and add, **locally**:
- **3 CUDA streams**: `h2d` (pack), `comm` (NCCL), `d2h` (unpack).
- **Event dependencies**: `NCCL_i` waits on `H2D_i` (send reads the packed buffer);
  `D2H_i` waits on `NCCL_i` (recv fills the packed buffer); reuse of a packed buffer waits
  on its `D2H`.
- **Double/triple-buffered packed-buffer pool**: at steady state, while bucket i is in
  NCCL, bucket i+1 does H2D (prefetch pack) and bucket i-1 does D2H (write-back).
- **One final sync**: offloaded state is read by the host afterwards; with async D2H,
  `transfer` must sync before returning (today D2H is blocking, which satisfies this
  implicitly; an async version must add the sync).

Cross-rank correctness is unaffected — the pipeline is pure local stream scheduling; each
bucket's NCCL send/recv pairing still relies on the step-aligned bucket boundaries.

A simpler **single-sided overlap** captures much of the benefit: overlap only "bucket i+1
H2D prefetch" or "bucket i-1 D2H write-back" with "bucket i NCCL", instead of a full
three-stage pipeline.

## Cost & pitfalls (why not just do it)

1. **Memory vs benefit**: double buffering raises peak from "1 bucket" to "2–3 buckets",
   working against bucketing's memory goal. Bucket too small → poor NCCL-launch
   amortization + high fill/drain; too large → no memory saving + few overlap stages. Needs
   a tuned sweet spot, ideally auto-sized (see [`README.md`](README.md) §5).
2. **Complexity**: streams / events / buffer lifetimes are much heavier than the current
   serial code; `batch_isend_irecv` semantics on a **non-default stream** and its
   interaction with the dist-patch (`distributed/dist_patch.py`) / `P2PToCollective` need
   dedicated validation.
3. **Send-only / recv-only ranks**: a rank that only sends (only H2D) or only receives
   (only D2H) in a reshard can't overlap H2D↔D2H internally, but can still overlap with
   NCCL.
4. **Async D2H host-read hazard**: must guarantee all D2H complete before the host reads
   offloaded state (the final sync), else stale reads.

## Measure first, then implement

After the refactor, pack / NCCL / unpack are timed together as one `"Transfer"` phase
(the old `_main_process_batch` split them). Before building overlap:
1. inside `transfer/communicator.py::BatchedTransfer.transfer`, time **pack (H2D) / NCCL /
   unpack (D2H) separately** (local accumulation, rank-0 print, same approach as
   `transfer/transfer.py::TransferManager._timed`, no extra collective);
2. on the A100 offload matrix ([`README.md`](README.md) §4 cells 3/4/5) measure the
   three-way split and the effect of offload fraction;
3. **only if the PCIe segment dominates**, build the pipeline — the bucket hook is already
   there; it just swaps the serial flush for multi-stream + double buffering.

Building multi-stream double buffering before there are numbers is premature: high
complexity, and a badly-sized bucket could be slower or OOM.

## Open questions (affect the win and the design)

- **Single-node NVLink vs multi-node IB**: single-node makes NCCL fast and PCIe the
  bottleneck (big win); multi-node NCCL also uses PCIe/NIC and contends with H2D/D2H
  (smaller win, different design).
- **Typical offload fraction**: larger fraction → more CPU-side slices → higher H2D/D2H
  share → more worthwhile.

## Code anchors

- Packing / staging / butterfly / bucketing:
  `transfer/communicator.py::BatchedTransfer.transfer` / `_enqueue_peer` / `_pack_from` /
  `_unpack_into` / `_flush_stride`.
- Per-tensor staging (no-pack path): `transfer/communicator.py::BatchP2P.isend/irecv`.
- Same staging idea per-tensor: [`../hybrid_adam/README.md`](../hybrid_adam/README.md) F2,
  [`../project/optimizer_state_model.md`](../project/optimizer_state_model.md).

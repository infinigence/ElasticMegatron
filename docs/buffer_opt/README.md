# buffer-opt (batched / packed reshard transport) — work log

Branch `ref/buffer-opt-integration` (PR #6 → `dev`). Integrates @Lin-xs's transport
optimization (PR #2, `feat/buffer-opt`) onto the Megatron-0.16 + HybridDeviceOptimizer
base (`feat/core_r0.16.0`), then refactors it to respect the transfer↔communicator
boundary and to support CPU-offloaded (CPU-adam) optimizer state.

Goal: cut the NCCL launch count during a reshard by **coalescing each peer's
optimizer-state slices into one buffer and issuing a single `batch_isend_irecv`**, while
keeping the per-tensor path available and without breaking CPU-offloaded state.

## 1. What the original branch (`feat/buffer-opt`, PR #2) did

Author @Lin-xs. The original reshard transfer moved optimizer state **one tensor at a
time**; NCCL p2p per-op launch overhead is significant when there are many params. The
branch added a "batch + pack" fast path:

- **`BatchP2P`** — accumulates many `isend`/`irecv` into one `dist.batch_isend_irecv`,
  handling self-rank (self-copy queue), non-contiguous tensors (shadow buffer + copy-back)
  and byte accounting.
- **`_main_process_batch`** — collects all cross-rank sends/recvs per peer, packs each
  peer's slices into one `uint8` buffer, orders peers via an XOR butterfly with a
  deadlock-safe enqueue order, then issues a single `batch_isend_irecv`.
- **Per-rank byte accounting** — `get_communication_bytes()` returns a `CommunicationBytes`
  dataclass (bucketed by dst/src) and `log_communication_info` prints a per-rank table.
- **`ELASTIC_USE_ASYNCBUFFER_P2P` gate** — `1` selects the batched path, `0` the original.
- **Timer instrumentation** of the transfer phases.

It predates Megatron-0.16 and assumed **all optimizer state lives on GPU** (no CPU
offload) — the two gaps the integration had to close.

## 2. Conflicts resolved during the merge onto `core_r0.16.0`

Resolution rule (agreed with the maintainer): **infrastructure / scripts / SAVE_PATH
follow `core_r0.16.0`; transport takes the union.** The `core ← buffer-opt` merge had five
conflicting files:

| File | Conflict | Resolution |
|---|---|---|
| `megatron_manager/training_state.py` | buffer-opt changed the ckpt save path to `os.environ["SAVE_PATH"]`; core uses `Path(__file__).parents[2]/tools/ckpt`, `self.optimizer` (ChainedOptimizer-aware), `try/finally` restoring `args.save` | **take core** (drop SAVE_PATH) |
| `run_e2e_demo.sh` | buffer-opt's local debug config vs core's rewrite (TIE_EMBED / CPU_OFFLOAD / RERUN_MODE / mock-data / NCCL timeouts / model sizes) | **take core** |
| `resharding/util.py` | both touched `Range`/`ParamRange` type hints + `__str__`/`__repr__` | **take core** (PEP585 + repr) |
| `transfer/communicator.py` | buffer-opt added `BatchP2P` + per-rank accounting; core added **device staging** (non-CUDA tensors bounce through GPU for NCCL — the CPU-offload enabler) | **union** (keep both) |
| `transfer/transfer.py` | buffer-opt's batch path vs core's type modernization + typo fixes | **union** (keep batch path; adopt core's imports; drop the unused `Timer` import) |

`elastic_manager.py` was changed by both but in different regions and auto-merged. Merge
commit `0e1defe`; @Lin-xs's five original commits are preserved as the merge's first
parent, so authorship/`git blame` stays with them.

## 3. Design & implementation of the refactor

The merge exposed two problems: **(a) the fast path was incompatible with CPU offload** —
`BatchP2P` / packing built `dist.P2POp` directly and allocated the packed buffer on the
slice's own device (CPU for offloaded state) → NCCL p2p requires CUDA → error/hang; and
**(b) transport mechanism had leaked into `transfer.py`** — packing, the butterfly
topology, `get_world_size()`, shadow buffers all sat in the reshard logic.

### 3.1 The boundary

- `transfer/transfer.py` holds reshard **logic** only: which optimizer-tensor slice of
  which param goes to/from which rank, the padded-tensor lifecycle, swiglu shuffle,
  dp-gather/scatter.
- `transfer/communicator.py` owns the **mechanism**, split into two cohesive classes
  (composition, not inheritance):
  - **`Communicator`** — the per-op primitive layer: `send`/`recv`/`broadcast` (with
    CPU↔GPU staging), self-copy, NCCL connection building (fake_transfer), byte
    accounting, and the `batch_p2p()` factory.
  - **`BatchedTransfer`** — the batched orchestration. One entry point:

    ```python
    BatchedTransfer(communicator).transfer(send_tasks, recv_tasks, *, pack, max_inflight_bytes)
    #   send_tasks/recv_tasks: dict[peer -> list[slice tensors]]; self-rank handled by the caller
    ```

    It composes `Communicator` (holds a reference, uses its primitives via `BatchP2P`),
    so `Communicator` stays a thin primitive layer.

Both the per-tensor default path and the packed fast path go through this one API
(`_main_process`, `_send_optimizer_tensors`, `_recv_optimizer_tensors` all build task
dicts and call it), removing the two parallel transport code paths the original carried.

### 3.2 Timer (commit `e1d42e2`)

The original used `megatron.core.timers.Timers` with `start()/stop()` everywhere and a
per-transfer `.log()`. That `.log()` runs a **global** all-reduce: it uses the patched
`get_world_size()` (= union-group size) but the unpatched global `get_rank()`, so when the
reshard's union group is a strict subset of the world, `rank_name_to_time[global_rank]`
goes out of bounds → `IndexError` / misreport. Replaced with a `with self._timed(...)`
context manager backed by the existing CUDA-synced `resharding.util.Timer`; per-phase
durations accumulate locally and print on rank 0 — **no cross-rank collective**.
`ELASTIC_TRANSFER_LOG_LEVEL=0` disables it at zero cost.

### 3.3 CPU-adam device staging (key insight, commit `f4e2f38`)

Under HDO a single peer's slices are device-heterogeneous (offloaded params' states on
pinned CPU, the rest on GPU). The fix: **allocate the packed buffer on the current CUDA
device**, so

- packing `packed[..].copy_(cpu_slice.view(uint8))` is itself a cross-device copy =
  CPU→GPU **stage-in**; and
- unpacking `cpu_dst.view(uint8).copy_(packed[..])` = GPU→CPU **stage-out**.

No per-tensor bounce buffer needed. `BatchP2P.isend/irecv` also gained the same CPU↔GPU
staging so the `pack=0` per-tensor path is offload-correct too. Same idea as hybrid-adam's
F2; see [`../hybrid_adam/README.md`](../hybrid_adam/README.md) and
[`../project/optimizer_state_model.md`](../project/optimizer_state_model.md).

### 3.4 Butterfly & deadlock avoidance

`BatchedTransfer.transfer` orders peers with an XOR butterfly:
`num_steps = 1 << ((world_size - 1).bit_length())`, `peer = rank ^ step`, skipping
`peer >= world_size`. For any `world_size` this covers every real pair (for
`a, b < world_size ≤ num_steps`, `a ^ b < num_steps` and `peer = b < world_size`; only
non-existent ranks are skipped), so the original plan's non-power-of-two fallback would be
unreachable dead code and was not added. Both ranks of a pair meet at the same step; the
lower rank enqueues sends first, the higher enqueues recvs first — within one
`batch_isend_irecv` group there is no deadlock.

### 3.5 Memory-aware byte chunking

The packed path caps the bytes staged per `batch_isend_irecv` round, so staging residency
is bounded by ~2x the cap regardless of model size. Each peer's ordered slice list is split
into byte chunks (`chunk_schedule.chunk_ranges` / `slice_spans`) and exchanged in rounds
reusing one send + one recv uint8 staging buffer. Both ends of a pair derive identical chunk
boundaries from the same `(slice sizes, cap)`, so this needs **no collective exchange**. The
cap is resolved per reshard by `ELASTIC_STAGING_CAP_MODE` (see §3.7) and is identical on
every rank.

### 3.6 Deleted / kept

- **Deleted**: the old per-param `_main_process`, `_main_process_batch`, the inline
  `_pack_tensors`/`_unpack_tensors` + butterfly, the dead `send_buffers` list, the unused
  `BatchP2P.has_ops`, the unused `Timer` import.
- **Thinned**: `_send/_recv_optimizer_tensors` are now lifecycle + collect + transfer
  wrappers, so `_pre_process`/`_post_process` (dp-gather/scatter) needed no change.
- **Kept**: `_resolve_transfer_layout`, `_should_skip_virtual_param`, all
  `OptimizerTensorInfo` lifecycle calls.

The byte-chunking layer (§3.5) later replaced the step-aligned `_flush_stride` flush (and
its all-reduce) with per-peer byte chunks: `_pack_from`/`_unpack_into`/`_enqueue_peer` gave
way to `_pack_chunk`/`_unpack_chunk`/`_enqueue_unpacked` + `chunk_schedule`.

### 3.7 Environment variables

| Var | Default | Effect |
|---|---|---|
| `ELASTIC_USE_ASYNCBUFFER_P2P` | `1` | `1` = packed fast path (per-peer slices coalesced into reused staging buffers, byte-chunked); `0` = one staged p2p op per slice (legacy fallback; `ELASTIC_MAX_INFLIGHT_BYTES` ignored). |
| `ELASTIC_STAGING_CAP_MODE` | `free` | Per-chunk staging cap source: `free` = clamp((min current free GPU mem across union ranks − 2 GiB reserve) // 2, 512 MiB, 8 GiB) via one MIN all-reduce (the 2 GiB reserve keeps 2× cap from filling the card; reflects real device memory, incl. co-tenant processes); `fixed` = 2 GiB constant. |
| `ELASTIC_MAX_INFLIGHT_BYTES` | unset | Overrides the mode: positive = exact per-chunk cap; `<=0` = one chunk per peer (legacy residency). **Must be identical on every rank** (both ends derive chunk counts from it). |
| `ELASTIC_TRANSFER_LOG_LEVEL` | `1` | `0` disables per-phase transfer timing (zero overhead); otherwise rank 0 prints per-phase ms. |

## 4. Verification

Method as in [`../megatron_016_adaptation/phase_b_report.md`](../megatron_016_adaptation/phase_b_report.md)
§5: a reshard sweep with `ELASTIC_SAVE_CKPT=1` writes before/after ckpts, then
`tools/ckpt/verify_all.py` checks **weight bit-equality + optim multiset-equality**
(`→ ALL PASS`). This change only alters *how* state is moved, so a correct reshard stays
bit-equal in every cell.

Matrix — 8-GPU `dense_mix_full`, `MODEL_SIZE=tiny NUM_LAYERS=8`, 8 reshard transitions per
cell, `verify_all.py --thresh 1e-3`, on the A100 box
(`/mnt/hisys-data/tonic/ElasticMegatron`, Megatron-LM-custom):

| Cell | `ELASTIC_USE_ASYNCBUFFER_P2P` | CPU offload | bucketing | weight | optim | Result |
|---|---|---|---|---|---|---|
| 1 baseline | 0 | off | – | 8/8 | 8/8 | ✅ ALL PASS |
| 2 packed | 1 | off | – | 8/8 | 8/8 | ✅ ALL PASS |
| 3 offload, no-pack | 0 | on | – | 8/8 | 8/8 | ✅ ALL PASS |
| 4 offload, packed (headline fix) | 1 | on | – | 8/8 | 8/8 | ✅ ALL PASS |
| 5 offload, packed, bucketed | 1 | on | `ELASTIC_MAX_INFLIGHT_BYTES=1048576` | 8/8 | 8/8 | ✅ ALL PASS |

**5/5 cells ALL PASS** (2026-06-03) — no FAIL / OOM / NCCL hang; `rel_rms` mostly 0. Every
reshard is bit-equal (weight) + multiset-equal (optim) across the unified pack/no-pack
paths, CPU-adam staging, and step-aligned bucketing. Logs in `_gpu_verify_logs/`.

Reproduce one cell:

```bash
cd /mnt/hisys-data/tonic/ElasticMegatron
BASE_PATH=/mnt/hisys-data/tonic MEGATRON_PATH=/mnt/hisys-data/tonic/Megatron-LM-custom \
ELASTIC_USE_ASYNCBUFFER_P2P=1 ELASTIC_SAVE_CKPT=1 TRAIN_ITERS=9 ELASTIC_RESHARD_INTERVAL=1 \
MODEL_SIZE=tiny NUM_LAYERS=8 \
GPUS_PER_NODE=8 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 MASTER_PORT=6000 \
  ./run_experiment.sh dense_mix_full
python3 tools/ckpt/verify_all.py --thresh 1e-3      # expect 8/8 weight + 8/8 optim → ALL PASS
```

## 5. Follow-up TODO

- **Overlap H2D/D2H with NCCL** for CPU-adam — design captured (not implemented) in
  [`cpu-adam-overlap.md`](cpu-adam-overlap.md); first step is to time pack/NCCL/unpack
  separately inside `BatchedTransfer.transfer` and measure whether PCIe staging dominates.
- **Auto-size bucketing** — done as `ELASTIC_STAGING_CAP_MODE=free` (default): cap =
  `clamp((torch.cuda.mem_get_info() whole-GPU free across union ranks − 2 GiB reserve) / 2,
  512 MiB, 8 GiB)` via one `all_reduce(MIN)` (whole-GPU free sees co-tenant processes —
  right for RL co-location). Possible refinement: also intersect with this process's
  `memory_reserved() - memory_allocated()` headroom.
- **Perf profiling** — pack vs no-pack benefit at different message sizes.
- **`log_communication_info`** does an unconditional `all_gather_object` + table print per
  reshard; gate it behind a verbosity flag (LOW).
- **dev → main** merge cadence once the integration settles.

# Reshard transfer flat-peak — Implementation Plan (v2)

> **For agentic workers:** REQUIRED SUB-SKILL: use superpowers:subagent-driven-development or
> superpowers:executing-plans to implement task-by-task. Steps use `- [ ]` checkboxes.
> Design + the regression it fixes: [`reshard-peak-memory.md`](reshard-peak-memory.md).

**Goal:** make the reshard transfer peak `= max(src_all, dst_all) + chunk_size` (vs the regression's
`~2×shard + 2×cap`), restoring the original per-param flat-peak guarantee at a **tunable chunk
granularity**, for GPU-adam AND cpu-adam, keeping intra-chunk NCCL coalescing.

**Architecture:** Process the global `all_virtual_params` in **rank-invariant chunks** (the
release/create unit). Within a chunk, use the **"approach 2" ordering** — pack this chunk's src into
a send staging buffer, **RELEASE the chunk's src**, comm, **CREATE the chunk's dst**, unpack staging
into dst — so a chunk's src and dst never coexist; the endpoints just swap `src_all → dst_all` across
chunks and the only resident overhead is the chunk's staging (`chunk_size`). `chunk_size` is a KNOB
decoupled from `cap`; `cap` stays the staging/comm group size WITHIN a chunk (the existing
communicator cap-loop), so a chunk can later hold multiple cap-groups to overlap gpu-copy/H2D with
comm. Branch `fix/transfer-flat-peak` (worktree `../em-pack-peak`, off `ref/buffer-opt-integration`).

**Peak model (per rank, main-process only):**

| | peak |
|---|---|
| baseline (`main`, per-param) | `max(src_all, dst_all) + max(one param's opt tensors)` |
| regression (`ref/buffer-opt`) | `~2×shard + 2×cap` (all dst built while all src held) |
| **target (this fix, approach 2)** | **`max(src_all, dst_all) + chunk_size`**  (chunk_size ≈ 2×cap baseline) |
| approach 3 (create dst before comm) | `max(src_all, dst_all) + 2×chunk_size` — acceptable fallback, (2) preferred |

`chunk_size` = the send + recv staging working set. `chunk=1 param` → the original flat peak;
larger chunk → fewer/larger comms (more coalescing) at higher staging cost.

## Critical correctness constraint
Chunk boundaries MUST be identical on every rank, or the per-peer butterfly p2p desyncs. Cut by the
**rank-invariant `virtual_param.size`** (global numel; mirrors virtual_param.py:40-42), not per-rank
ownership. Budget is rank-identical (derived from the rank-identical staging cap + an all-reduced
per-numel byte size). Task 1's unit test locks the determinism in.

## File structure
- **Modify** `elastic_megatron/transfer/transfer.py`: the chunker + derived per-numel (Task 1); the
  `_main_process` rewrite to the chunked approach-2 ordering (Task 3 — pending Task 2's mechanism).
- **Possibly modify** `elastic_megatron/transfer/communicator.py`: to expose pack/comm/unpack so the
  manager can interleave release-src (after pack) and create-dst (before unpack) — **the mechanism is
  Task 2's deliverable** (under sub-agent investigation; do not pre-decide).
- **Create** `tests/test_transfer_chunking.py` (Task 1).

---

### Task 1: rank-invariant chunker + derived per-numel size (READY — independent of the comm mechanism)

**Files:** Modify `transfer.py` (3 methods on `TransferManager`); Test `tests/test_transfer_chunking.py`.

- [ ] **Step 1: failing test** (`tests/test_transfer_chunking.py`)

```python
"""Rank-invariant chunking of the reshard transfer. Boundaries must depend only on
the global vparam list (vp.size), never per-rank ownership, or the per-peer NCCL
butterfly desyncs. Import needs torch+megatron -> SKIPS on the dev box, runs on A100."""

def _load():
    try:
        from elastic_megatron.transfer.transfer import TransferManager
        return TransferManager
    except Exception:
        return None

class _VP:
    def __init__(self, size): self.size = size

def test_chunking_deterministic_budget_bounded_and_degenerate():
    TransferManager = _load()
    if TransferManager is None:
        print("SKIP test_chunking (torch/megatron unavailable)"); return
    tm = TransferManager.__new__(TransferManager)
    vps = [_VP(100), _VP(100), _VP(100), _VP(50)]
    chunks = list(tm._chunk_virtual_params(vps, budget_numel=200))
    assert [[v.size for v in c] for c in chunks] == [[100, 100], [100, 50]]
    assert chunks == list(tm._chunk_virtual_params(vps, 200))          # deterministic
    big = [_VP(10_000), _VP(10)]
    assert [[v.size for v in c] for c in tm._chunk_virtual_params(big, 200)] == [[10_000], [10]]
    assert list(tm._chunk_virtual_params(vps, None)) == [vps]          # degenerate -> one chunk
    assert list(tm._chunk_virtual_params(vps, 0)) == [vps]

if __name__ == "__main__":
    test_chunking_deterministic_budget_bounded_and_degenerate()
    print("PASS tests/test_transfer_chunking.py")
```

- [ ] **Step 2: run, verify fails** — `PYTHONPATH=.:$MEGATRON_PATH python3 tests/test_transfer_chunking.py` → AttributeError (or SKIP on dev).

- [ ] **Step 3: implement the helpers** (in `TransferManager`)

```python
    @staticmethod
    def _virtual_param_global_numel(virtual_param) -> int:
        """Rank-invariant global element count (mirrors virtual_param.py:40-42).
        Used only to cut chunk boundaries identically on every rank."""
        return virtual_param.size

    def _resolve_state_bytes_per_numel(self, virtual_params) -> int:
        """Sum(element_size) over a param's optimizer states (master + moments),
        DERIVED from the actual OptimizerTensorInfo the adapter populated (never
        hardcoded; Adam-fp32 -> 12, mixed/other optimizers come out right). Read
        from whatever info THIS rank holds, then all-reduce MAX so every rank uses
        the same value (ranks holding no param contribute 0) — the chunk budget
        derives from it and boundaries must be rank-identical. Runs inside
        with_world_group(union) like _resolve_staging_cap."""
        local = 0
        for vp in virtual_params:
            info = vp.src_optimizer_tensor_info or vp.dst_optimizer_tensor_info
            if info is not None:
                local = sum(t.element_size() for t in info.optimizer_tensors)
                break
        t = torch.tensor([local], device=torch.cuda.current_device(), dtype=torch.int64)
        torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.MAX)
        return max(1, int(t.item()))

    def _chunk_virtual_params(self, virtual_params, budget_numel):
        """Yield consecutive chunks whose combined global numel stays under
        ``budget_numel``. Boundaries depend ONLY on rank-invariant vparam sizes.
        ``budget_numel`` None/<=0 => one chunk. An oversized single vparam forms its
        own chunk (never split)."""
        if not budget_numel or budget_numel <= 0:
            yield list(virtual_params)
            return
        chunk, acc = [], 0
        for vp in virtual_params:
            vp_numel = self._virtual_param_global_numel(vp)
            if chunk and acc + vp_numel > budget_numel:
                yield chunk
                chunk, acc = [], 0
            chunk.append(vp)
            acc += vp_numel
        if chunk:
            yield chunk
```

- [ ] **Step 4: run on box, verify PASS.** **Step 5: confirm `vp.size` accessor** (`grep -n "def size" elastic_megatron/resharding/virtual_param.py`; fix `_virtual_param_global_numel` if it differs). **Step 6: commit** (`feat(transfer): rank-invariant vparam chunker + derived per-numel`).

---

### Task 2: [RESOLVED by `transfer-order-investigator` + user] mechanism = phase-split, defer self-copy

**Confirmed deadlock-safe:** the butterfly comm touches ONLY the staging byte-buffers, never the
src/dst optimizer tensors (chain `src → pack → s_stage → [NCCL] → r_stage → unpack → dst`). So
releasing src after pack and creating dst after the comm is deadlock- and bit-exact-neutral.

**Decision (Option b):** split `BatchedTransfer.transfer` into `pack()` / `exchange()` / `unpack()`
phases the manager drives per chunk. `exchange()` keeps the butterfly (`peer = rank ^ step`,
send-first/recv-first by `rank < peer`, per-pair `batch_isend_irecv`) **verbatim** — the
deadlock-critical machinery is untouched. `transfer()` is KEPT for `_pre_process` / `_post_process`
(DP gather/scatter) and the no-pack legacy; only `_main_process` moves to the phases.

**Decision (self-copy):** survival-rank self-edges currently go through the self-copy queue at
collect time and `_recv_self` reads immediately — which needs dst before the comm. **Defer them to
the unpack phase:** stash each self-edge's src bytes during pack; apply them to dst during unpack
(after dst is created). So self-copies obey the same src-released-before-dst-created ordering.

### Task 3: chunked approach-2 `_main_process` (the fix)

**3a. recv-size-from-metadata helper.** To keep create-dst AFTER the comm, `exchange` must size the
recv staging WITHOUT the dst tensors. Add a helper that sums per-peer recv bytes from the chunk's
`reshard_plan.global_recv_info[rank]` ranges × the dst placeholder's `Σ(state element_size)` (the
dst `OptimizerTensorInfo` exists as storage-0 placeholders from dst setup — shape/dtype intact). No
allocation. Self-edge (`src == rank`) recv bytes are excluded from the NCCL recv size (handled by
the deferred self-copy).

**3b. phase methods on `BatchedTransfer`** (new; `transfer()` unchanged):
- `pack(send_tasks) -> {peer: send_staging}`: contiguous uint8 GPU buffer per peer from the src
  byte-views; reuse instance buffers across chunks. After `pack` the caller may release src.
- `exchange({peer: send_staging}, {peer: recv_nbytes}) -> {peer: recv_staging}`: the butterfly
  (verbatim ordering); allocate a recv staging per peer sized by `recv_nbytes`; one
  `batch_isend_irecv` per pair. Touches only staging.
- `unpack({peer: recv_staging}, recv_tasks) -> None`: scatter recv staging into the dst byte-views
  (dst now created), then apply the deferred self-copies.

**3c. `_main_process` per-chunk loop:**
```python
    def _main_process(self, virtual_params):
        budget_numel = (max(1, self._max_inflight_bytes // self._resolve_state_bytes_per_numel(virtual_params))
                        if self._max_inflight_bytes and self._max_inflight_bytes > 0 else None)
        for chunk in self._chunk_virtual_params(virtual_params, budget_numel):
            send_tasks, self_src = self._collect_chunk_sends(chunk)        # src views + stashed self-edge src
            recv_nbytes = self._chunk_recv_nbytes(chunk)                   # 3a, from metadata (no dst alloc)
            staged = self.batched_transfer.pack(send_tasks)
            self._release_chunk_src(chunk)                                 # ← before exchange
            received = self.batched_transfer.exchange(staged, recv_nbytes)
            self._create_chunk_dst(chunk)                                  # create_padded + rebuild ← after exchange
            recv_tasks, recv_copy_back = self._collect_chunk_recvs(chunk)  # dst views (dst now exists)
            self.batched_transfer.unpack(received, recv_tasks)
            for recv_slice, buf in recv_copy_back: recv_slice.data.copy_(buf)
            self._apply_self_copies(self_src, chunk)                       # deferred self-edges → dst
            self._release_chunk_src_padded(chunk)
```
Reuses the existing `_collect_send`/`_collect_recv` slice logic (refactored to per-chunk + self-edge
stashing). `chunk = cap` for the first landing (one staging group); the chunk-size knob + multi-cap
overlap is a later seam (do NOT hardcode chunk = 2×cap).

**3d. residual-risk guards** (from the investigation): keep padded-buffer release in `_post_process`
(don't release early); leave `_pre_process`/`_post_process` on `transfer()` (verify their self-copy
+ create_padded ordering is unchanged); preserve the I-16 same-stream reclaim.

**3e. bit-exact** preserved (same bytes/order; only release/create timing + batch boundary move).
Subsumes codex's cpu-adam-only `_main_process_streaming`.

---

### Task 4: peak-reserved-bytes probe (verification aid)

`ELASTIC_TRANSFER_PEAK_PROBE=1` around the main process logs `torch.cuda.max_memory_reserved()`
(rank 0) — the signal distinguishing `max(src,dst)+chunk_size` from the 2×shard baseline.

### Task 5: GPU verification (the binding gate; A100, off-NFS)
- bit-exact `verify_all` ALL PASS: **CPU_OFFLOAD=0 (GPU-adam — the regressed path)** + **=1 (cpu-adam)**, pack on/off.
- peak probe: small chunk (many chunks) peak ≈ `max(src,dst)+chunk_size` ≪ the one-chunk 2× baseline.

## Open / deferred
- **The within-chunk comm mechanism** (Task 2, sub-agent investigating; discuss with user after).
- chunk_size default + knob naming (after Task 2).
- Fold codex's `feat/hostmem-30b-hang` streaming onto this path once verified (follow-up).

## Self-review
- Spec coverage: flat peak GPU+cpu-adam (Tasks 3) ✔; rank-invariant cuts (Task 1 + test) ✔; derived
  per-numel (Task 1) ✔; chunk≠cap knob (architecture + Task 3) ✔; bit-exact + GPU gate (Task 5) ✔.
- The comm mechanism is correctly deferred to Task 2 rather than pre-specified (it was the source of
  the earlier muddled design); Task 1 is mechanism-independent and can land now.

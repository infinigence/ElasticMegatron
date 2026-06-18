# Reshard transfer flat-peak — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: use superpowers:subagent-driven-development or
> superpowers:executing-plans to implement this task-by-task. Steps use `- [ ]` checkboxes.
> Design rationale + the regression it fixes: [`reshard-peak-memory.md`](reshard-peak-memory.md).

**Goal:** Make the batched reshard transfer hold a **flat** memory peak (≈ 1× the local
optimizer-state shard + bounded staging) instead of the current ~2× (all-dst-built + all-src-held),
restoring the original `main`/`feat/core_r0.16.0` guarantee for **all** optimizers (GPU-adam and
cpu-adam), without losing buffer-opt's intra-batch NCCL coalescing.

**Architecture:** Split the global `all_virtual_params` list into **chunks** sized by a
rank-invariant budget (each vparam's GLOBAL numel `virtual_param.size`, so every rank cuts at the
same boundary → NCCL stays in lockstep). Process one chunk at a time: collect sends, build that
chunk's dst, transfer (coalesced within the chunk), copy back, **release that chunk's src before
the next chunk's dst is built**. The budget is the existing free-mode staging cap
(`_max_inflight_bytes`). `budget None/≤0` → one chunk = today's behavior; tiny budget → per-param.

**Tech stack:** Python, PyTorch, Megatron-LM 0.16, NCCL `batch_isend_irecv` (butterfly p2p).
Branch `fix/transfer-flat-peak` (worktree `../em-pack-peak`, off `ref/buffer-opt-integration`).

---

## Critical correctness constraint (read first)

**Chunk boundaries MUST be identical on every rank.** A chunk is one `batched_transfer.transfer`;
its butterfly p2p ops only pair up if both ends batch the same vparams. If rank A cuts after vp5 and
rank B after vp6, A's chunk-1 sends vp1-5 while B expects vp1-6 → desync/hang. Therefore the chunk
metric must be a pure function of the (globally-identical, identically-ordered) `all_virtual_params`
list — **not** of per-rank ownership. We use `virtual_param.size` (the param's global numel, set from
`tensor_parallel_attr.get_model_param_range(tp)`; mirrors `virtual_param.py:40-42`), which is
rank-invariant for a given reshard. The budget (`_resolve_staging_cap`) is already rank-identical
(one MIN all-reduce). Same list + same metric + same budget ⇒ identical cuts. A unit test (Task 1)
locks this in.

## File structure

- **Modify** `elastic_megatron/transfer/transfer.py`
  - add module const `_RESHARD_STATE_BYTES_PER_NUMEL`
  - add `TransferManager._virtual_param_global_numel(vp)` (rank-invariant size accessor)
  - add `TransferManager._chunk_virtual_params(vps, budget_bytes)` (generator)
  - extract `TransferManager._transfer_chunk(chunk, timings)` from the current `_main_process` body
  - rewrite `TransferManager._main_process(vps)` into the chunk loop
- **Create** `tests/test_transfer_chunking.py` (rank-invariance + budgeting + degenerate cases)
- No changes to `_pre_process` / `_post_process` (stay full passes — they already release per-param
  via `_send_optimizer_tensors`), to `communicator.py` (staging cap unchanged), or to `OptimizerTensorInfo`.

---

### Task 1: rank-invariant chunk helper + tests

**Files:**
- Modify: `elastic_megatron/transfer/transfer.py` (module const + 2 methods, after `_main_process`)
- Test: `tests/test_transfer_chunking.py`

- [ ] **Step 1: Write the failing test** (`tests/test_transfer_chunking.py`)

```python
"""Behavioral test for rank-invariant chunking of the reshard transfer.

Chunk boundaries MUST depend only on the global vparam list (vp.size), never on
per-rank ownership, or the per-peer NCCL butterfly desyncs. Import needs torch +
megatron, so this SKIPS where they are unavailable (runs on the A100 box)."""

def _load():
    try:
        from elastic_megatron.transfer.transfer import (
            TransferManager, _RESHARD_STATE_BYTES_PER_NUMEL,
        )
        return TransferManager, _RESHARD_STATE_BYTES_PER_NUMEL
    except Exception:
        return None, None


class _VP:
    def __init__(self, size): self.size = size


def test_chunking_deterministic_budget_bounded_and_degenerate():
    TransferManager, BPN = _load()
    if TransferManager is None:
        print("SKIP test_chunking (torch/megatron unavailable)"); return
    tm = TransferManager.__new__(TransferManager)  # bypass __init__ (no dist needed)
    vps = [_VP(100), _VP(100), _VP(100), _VP(50)]

    # budget = 200 numel worth of bytes -> chunks of <=200 numel
    budget = 200 * BPN
    chunks = list(tm._chunk_virtual_params(vps, budget))
    assert [[v.size for v in c] for c in chunks] == [[100, 100], [100, 50]]
    # deterministic: identical inputs -> identical boundaries (rank-invariance proxy)
    assert chunks == list(tm._chunk_virtual_params(vps, budget))
    # an oversized single vparam still goes alone (never dropped)
    big = [_VP(10_000), _VP(10)]
    assert [[v.size for v in c] for c in tm._chunk_virtual_params(big, budget)] == [[10_000], [10]]
    # degenerate: None / <=0 budget -> exactly one chunk (today's batched behavior)
    assert list(tm._chunk_virtual_params(vps, None)) == [vps]
    assert list(tm._chunk_virtual_params(vps, 0)) == [vps]


if __name__ == "__main__":
    test_chunking_deterministic_budget_bounded_and_degenerate()
    print("PASS tests/test_transfer_chunking.py")
```

- [ ] **Step 2: Run it, verify it fails**

Run: `PYTHONPATH=.:$MEGATRON_PATH python3 tests/test_transfer_chunking.py`
Expected: `AttributeError`/`ImportError` (`_chunk_virtual_params` / `_RESHARD_STATE_BYTES_PER_NUMEL`
not defined) — on the dev box it will instead print `SKIP` (megatron absent); run the real assertion
on the A100 box.

- [ ] **Step 3: Implement the const + helpers** (in `transfer.py`, module scope + in `TransferManager`)

```python
# module scope, near the top
# Optimizer-state bytes per param element for chunk budgeting: Adam keeps 3
# param-shaped fp32 states (master + exp_avg + exp_avg_sq), see I-15. This only
# converts the byte budget into a rank-invariant numel threshold; a non-Adam
# optimizer would scale it, but the value need only be identical on every rank.
_RESHARD_STATE_BYTES_PER_NUMEL = 3 * 4
```

```python
    @staticmethod
    def _virtual_param_global_numel(virtual_param) -> int:
        """Rank-invariant global element count of a vparam (mirrors
        virtual_param.py:40-42). Used only to cut chunk boundaries identically on
        every rank."""
        return virtual_param.size

    def _chunk_virtual_params(self, virtual_params, budget_bytes):
        """Yield consecutive chunks of ``virtual_params`` whose combined global
        optimizer-state bytes stay under ``budget_bytes``. Boundaries depend ONLY
        on the (rank-invariant) global vparam sizes, so every rank cuts identically
        and the per-chunk butterfly p2p stays in lockstep. ``budget_bytes`` None or
        <= 0 => a single chunk (today's all-at-once behaviour). A single vparam
        larger than the budget is never split — it forms its own chunk."""
        if not budget_bytes or budget_bytes <= 0:
            yield list(virtual_params)
            return
        budget_numel = max(1, budget_bytes // _RESHARD_STATE_BYTES_PER_NUMEL)
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

- [ ] **Step 4: Run test, verify it passes** (on the box)

Run: `PYTHONPATH=.:$MEGATRON_PATH python3 tests/test_transfer_chunking.py`
Expected: `PASS tests/test_transfer_chunking.py`

- [ ] **Step 5: Confirm `virtual_param.size` is the right accessor**

Run: `grep -n "def size" elastic_megatron/resharding/virtual_param.py`
Expected: a `size` property on `VirtualParam` returning `shape.size`/`shape.numel()`. If it is named
differently or lives on a sub-attr, update `_virtual_param_global_numel` to the correct
rank-invariant accessor and re-run Step 4.

- [ ] **Step 6: Commit**

```bash
git add elastic_megatron/transfer/transfer.py tests/test_transfer_chunking.py
git commit -m "feat(transfer): rank-invariant vparam chunker for flat-peak reshard"
```

---

### Task 2: extract `_transfer_chunk` (pure refactor, behavior identical)

**Files:** Modify `elastic_megatron/transfer/transfer.py` (`_main_process` body → `_transfer_chunk`)

- [ ] **Step 1: Move the current `_main_process` body into `_transfer_chunk`**

Rename the existing `_main_process(self, virtual_params)` body (transfer.py:335-424) to
`_transfer_chunk(self, virtual_params, timings)`: drop the local `timings = ... ` line (now a
parameter) and the final rank-0 summary print (moves to the new `_main_process` in Task 3). Keep
everything else byte-for-byte — the same collect (`create_padded_optimizer_tensor()`+`rebuild()` for
aligned receivers, `_collect_send`/`_collect_recv`), the single `batched_transfer.transfer(...)`,
the `recv_copy_back` scatter, and the `src_to_release` / `src_to_release_padded` release. Signature:

```python
    def _transfer_chunk(self, virtual_params, timings):
        send_tasks = defaultdict(list); recv_tasks = defaultdict(list)
        recv_copy_back = []; src_to_release = []; src_to_release_padded = []
        # ... (unchanged body from current _main_process: Collect / Transfer /
        #      Copy recv / Release) ...
```

- [ ] **Step 2: Temporary `_main_process` calls it with one chunk**

```python
    def _main_process(self, virtual_params):
        timings = {} if self._log_transfer_timing else None
        self._transfer_chunk(virtual_params, timings)
        if timings is not None and self._rank == 0:
            summary = "  ".join(f"{n}: {ms:.2f}ms" for n, ms in timings.items())
            print(f"[ElasticMegatron-Transfer] {summary}", flush=True)
```

- [ ] **Step 3: Verify no behavior change**

Run: `python3 -m py_compile elastic_megatron/transfer/transfer.py && ruff check elastic_megatron/transfer/transfer.py`
Expected: clean. (Bit-exact equivalence is verified on GPU in Task 5 — this step is a pure
extraction; one chunk == the old single batched call.)

- [ ] **Step 4: Commit**

```bash
git add elastic_megatron/transfer/transfer.py
git commit -m "refactor(transfer): extract _transfer_chunk from _main_process (no behavior change)"
```

---

### Task 3: chunked `_main_process` (the fix)

**Files:** Modify `elastic_megatron/transfer/transfer.py` (`_main_process`)

- [ ] **Step 1: Replace `_main_process` with the chunk loop**

```python
    def _main_process(self, virtual_params):
        """Transfer the cross-rank reshard plan in rank-invariant chunks, releasing
        each chunk's src before the next chunk's dst is built, so the resident
        optimizer-state stays flat (~1x shard) instead of all-dst + all-src (~2x).
        The chunk budget is the staging cap; budget None/<=0 => one chunk (today's
        batched behaviour). Coalescing is preserved within each chunk."""
        timings = {} if self._log_transfer_timing else None
        for chunk in self._chunk_virtual_params(virtual_params, self._max_inflight_bytes):
            self._transfer_chunk(chunk, timings)
        if timings is not None and self._rank == 0:
            summary = "  ".join(f"{n}: {ms:.2f}ms" for n, ms in timings.items())
            print(f"[ElasticMegatron-Transfer] {summary}", flush=True)
```

Why this is correct: `_transfer_chunk` releases this chunk's `src_to_release` at its end, BEFORE the
next iteration builds the next chunk's dst (`create_padded`+`rebuild`). So at any time the resident
endpoints ≈ (un-processed src) + (already-built dst) ≈ 1× shard, plus one chunk's dst overshoot, plus
`2×cap` staging. Bit-exactness holds: same per-peer butterfly, same deterministic vparam order, same
slices and release order — only the batch boundary moves (same argument as the original per-param path).

- [ ] **Step 2: Verify compile/lint**

Run: `python3 -m py_compile elastic_megatron/transfer/transfer.py && ruff check elastic_megatron/transfer/transfer.py`
Expected: clean.

- [ ] **Step 3: Commit**

```bash
git add elastic_megatron/transfer/transfer.py
git commit -m "fix(transfer): chunked flat-peak _main_process (release src per chunk before next dst)"
```

---

### Task 4: peak-reserved-bytes probe (verification aid)

**Files:** Modify `elastic_megatron/transfer/transfer.py` (`transfer_optimizer_tensors`)

- [ ] **Step 1: Add an opt-in peak probe around the main process**

```python
        # ELASTIC_TRANSFER_PEAK_PROBE=1 reports the GPU reserved-bytes high-water of
        # the main transfer (rank 0) — the signal that distinguishes flat vs 2x peak.
        _peak_probe = os.getenv("ELASTIC_TRANSFER_PEAK_PROBE", "0") == "1"
        if _peak_probe:
            torch.cuda.reset_peak_memory_stats()
        process_virtual_params(self._pre_process)
        self._main_process(virtual_param_space.all_virtual_params)
        if _peak_probe and self._rank == 0:
            print(f"[ElasticMegatron-Transfer] main-process peak reserved: "
                  f"{torch.cuda.max_memory_reserved() / (1 << 20):.0f} MiB "
                  f"(cap={self._max_inflight_bytes})", flush=True)
        process_virtual_params(self._post_process)
```

(Replaces the current bare `process_virtual_params(self._pre_process)` / `self._main_process(...)` /
`process_virtual_params(self._post_process)` at transfer.py:544-546.)

- [ ] **Step 2: compile/lint + commit**

```bash
python3 -m py_compile elastic_megatron/transfer/transfer.py && ruff check elastic_megatron/transfer/transfer.py
git add elastic_megatron/transfer/transfer.py
git commit -m "feat(transfer): opt-in main-process peak-reserved probe"
```

---

### Task 5: GPU verification (bit-exact + flat-peak), via the A100 runner

Not a code step — the binding acceptance gate. Run on the freed A100, ckpts off-NFS (`/tmp`).

- [ ] **Step 1: bit-exact, GPU-adam (the regressed path)** — `CPU_OFFLOAD=0`, pack on AND off:
  `ELASTIC_SAVE_CKPT=1 TRAIN_ITERS=9 ELASTIC_RESHARD_INTERVAL=1 MODEL_SIZE=tiny NUM_LAYERS=8`
  `ELASTIC_USE_ASYNCBUFFER_P2P=1` then `=0`, on `dense_mix_full` → `verify_all.py --thresh 1e-3`.
  Expected: `→ ALL PASS` both. (Chunking must not change which bytes land.)
- [ ] **Step 2: bit-exact, cpu-adam** — `CPU_OFFLOAD=1` on `moe_mix_full`, same as above → `ALL PASS`.
- [ ] **Step 3: flat-peak demonstration** — `ELASTIC_TRANSFER_PEAK_PROBE=1` with a model whose dst
  optimizer state ≫ cap, on `dense_mix_full` `CPU_OFFLOAD=0`: compare the printed main-process peak
  reserved for a tiny cap (`ELASTIC_MAX_INFLIGHT_BYTES=536870912`, 512 MiB → many chunks → flat) vs a
  huge cap (`ELASTIC_MAX_INFLIGHT_BYTES=0`, one chunk → 2× baseline). Expect the small-cap peak to be
  ≈ 1× the per-rank optimizer shard + ~1.5 GiB, well below the one-chunk 2× peak.

---

## Self-review

- **Spec coverage:** restores flat peak for GPU-adam (Tasks 2-3) ✔; budget = free-mode cap (Task 3,
  `self._max_inflight_bytes`) ✔; small cap → per-param (Task 1 budget_numel→1 chunks of 1) ✔;
  coalescing preserved within a chunk (Task 2 keeps the single batched call per chunk) ✔; bit-exact
  (Tasks 2/5) ✔; rank-invariant boundaries (Task 1 + its test) ✔.
- **Placeholder scan:** the one open item is the exact `vp.size` accessor — guarded by Task 1 Step 5
  + the unit test, not left as a TODO in shipped code.
- **Type consistency:** `_chunk_virtual_params` yields `list[VirtualParam]`; `_transfer_chunk` takes
  `(list[VirtualParam], timings|None)`; `_main_process` passes `self._max_inflight_bytes` (already an
  `int|None`). Consistent.

## Open questions for the reviewer (you)

1. **Budget = cap exactly** gives peak ≈ `shard + cap + 2×cap` = `shard + 3×cap` (you chose this).
   Confirm you don't want the tighter `shard + 2×cap` (would split the cap between chunk-dst and
   staging — one extra line in `_main_process`). Current plan = `shard + 3×cap`.
2. **`_RESHARD_STATE_BYTES_PER_NUMEL = 12`** (Adam fp32 ×3) is only a budget→numel scale; it need not
   be exact, only rank-identical. OK to hardcode (matches the repo's Adam-only I-15 assumption), or
   prefer deriving it from the optimizer config?
3. Once verified, this **supersedes codex's `_main_process_streaming`** on `feat/hostmem-30b-hang`
   (its cpu-adam-only chunk=1). Fold that branch onto this path in a follow-up, or keep separate?

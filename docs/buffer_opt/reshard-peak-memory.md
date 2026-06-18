# Reshard transfer peak memory — the flat-peak regression and the fix

> **Conclusion (2026-06-18):** the buffer-opt batched transfer **regressed** the original
> reshard-transfer memory guarantee. The original `_main_process` kept the transfer peak
> **flat** (≈ 1× the local optimizer-state shard) by allocating each param's dst and
> releasing each param's src **one param at a time**. The batched rewrite allocates **all**
> dst then releases **all** src, so the peak is **≈ 2×** the shard, plus `2×cap` staging.
> This doc records the finding, adjudicates the free-mode cap question, and specifies the
> fix (one chunked flat-peak path). Fix branch: `fix/transfer-flat-peak` (off
> `ref/buffer-opt-integration`).

## 1. The original guarantee (`main`, `feat/core_r0.16.0`)

`TransferManager._main_process(virtual_param)` ran **per param**, via
`process_virtual_params(self._main_process)`:

- **sender:** `_send_optimizer_tensors(...)` collects the send slice, transfers it, and
  **`src_optimizer_tensor_info.release()` immediately** (frees that param's 3 opt tensors).
- **receiver (aligned rank):** `create_padded_optimizer_tensor()` + `_recv_optimizer_tensors(...)`
  → `dst_optimizer_tensor_info.rebuild()` allocates that param's dst tensors **right before**
  receiving, then receives.

Because src is freed and dst is built one param at a time, the freed src block is reused for
the next dst (same CUDA stream pool, see invariant I-16). **At any instant the resident
endpoint memory ≈ the optimizer-state shard that already existed before the transfer — no peak
rise.** The user (transfer's designer) intended: *default pack; pack only adds ~`2×cap` of
staging on top of this flat endpoint; and when the cap drops below a threshold the transfer
degenerates to per-param.*

## 2. The regression — two stages

1. **`feat/buffer-opt`** (off `main`) added `_main_process_batch(virtual_params)`:
   one collect loop `rebuild()`s **every** dst param, **one** `batch_isend_irecv`, then
   **`release()`s every src at the end**. It routed the **default** path
   (`use_asyncbuffer_p2p=1`) through this method. The per-param `_main_process` was kept, but
   only as the `use_asyncbuffer_p2p=0` fallback. → flat peak lost on the default path.
2. **`ref/buffer-opt-integration`** (merging buffer-opt into `feat/core_r0.16.0`) **deleted**
   the per-param `_main_process(vp)`; the batched `_main_process(list)` became the **sole**
   path (no gate). `pack` only changes how `batched_transfer.transfer` moves bytes, not the
   `rebuild`-all (L396-397) / `release`-all (L418) in `_main_process`. → **flat peak lost
   for every config**, including `pack=0`.

`codex feat/hostmem-30b-hang` adds `_main_process_streaming` (a 3rd method) that
re-implements the original per-param flat-peak — but **only for cpu-adam** (gated by
`_should_stream_main_process`). So on that branch **GPU-adam still carries the 2× regression.**

## 3. The free-mode cap question — adjudicated

`_resolve_staging_cap()` (`ref` transfer.py **L511**, reads `torch.cuda.mem_get_info()` at
**L78**) runs **before** `_main_process` (**L545**) rebuilds dst (**L396-397**). So the
free-mode cap (`2×cap = min_free − 2 GiB`, clamped ≤ 16 GiB) is computed when **src opt is
live but dst is not yet allocated** — it does **not** reserve for the dst alloc that follows.

Peak = `src + dst + 2×cap`. Substituting `2×cap = min_free − 2 GiB`, the leftover at peak is
≈ `2 GiB − dst` → **GPU-adam with dst > ~2 GiB over-commits and OOMs.** The 2 GiB reserve is
for staging, not for dst.

**Why it has not bitten yet:** (a) every GPU-adam config tested is tiny/small → dst ≪ 2 GiB →
fits even at 2× endpoints + `2×cap`; (b) the 30B run is **cpu-adam → dst lives on host**, so
the GPU cap is not the binding constraint (GPU holds only `2×cap` staging) — instead the
**host** OOMs from the 2× host-resident endpoints. So the latent GPU OOM is masked by small
configs, and at 30B scale the OOM simply moved to host.

**The "small cap → per-param" half of the design was never wired.** The cap only chunks the
staging **bytes** inside `batched_transfer.transfer`; it does **not** chunk the param loop.
`_main_process` always collects all params (rebuild all dst) regardless of cap.

## 4. State across branches

| branch | main-process | dst alloc | src release | transfer peak |
|---|---|---|---|---|
| `main` / `feat/core_r0.16.0` | `_main_process(vp)` per-param | per param, pre-recv | per param, post-send | **flat ≈ 1× shard** |
| `feat/buffer-opt` | batch (default) / per-param (asyncbuffer=0) | batch: all in collect | batch: all at end | asyncbuffer=1 → **2×**; =0 → flat |
| `ref/buffer-opt-integration` | `_main_process(list)` only | all in collect (L397) | all at end (L418) | **2× (all configs)** |
| `codex feat/hostmem-30b-hang` | cpu-adam→streaming; GPU→batch | cpu-adam per-param; GPU all | cpu-adam per-param; GPU all | cpu-adam flat; **GPU 2×** |

## 5. The fix — one chunked flat-peak path

Replace the all-or-nothing batched `_main_process` with a **single chunked path** in which the
cap/budget drives **both** the staging bytes **and** the param-chunking:

```
budget = resolve_chunk_budget()                 # from the same free-mode cap
for chunk in chunk_params_by_dst_bytes(virtual_params, budget):
    collect_send(chunk)                         # src still resident
    for vp in chunk: build dst (create_padded + rebuild)
    batched_transfer.transfer(chunk send/recv)  # intra-chunk coalescing preserved
    copy_back(chunk)
    release_src(chunk)                          # ← free this chunk's src before next chunk's dst
```

Invariant restored: **release each chunk's src before allocating the next chunk's dst**, so the
endpoint stays ≈ 1× shard (src→dst swap), and the only overshoot is one chunk's dst.

- **peak = 1× shard + one chunk's dst + `2×cap` staging.**
- `chunk=1` (small budget) → **per-param**, the original flat peak (= what codex's streaming
  does). `chunk=∞` → today's 2× behaviour. The budget is the single knob.
- **Coalescing preserved within a chunk** (the buffer-opt throughput win), lost only at
  `chunk=1` — same tradeoff streaming makes, now tunable.
- **Subsumes codex's `_main_process_streaming`** (it becomes the `chunk=1` degenerate) and
  **fixes GPU-adam too** (not just cpu-adam).
- **Bit-exactness preserved:** same per-peer butterfly, same deterministic param order, same
  send/recv slices and release order — only the batch boundary moves. (Same argument that makes
  codex's streaming bit-exact.)
- **Cap-timing gap closed:** the peak no longer requires all-dst + all-src at once, so the
  cap-before-dst computation is no longer unsound — it now bounds the per-chunk transient,
  and the ~1× shard endpoint was already resident at cap time.

### Open decisions (to confirm before coding)
1. **Budget source.** Reuse the existing free-mode `_resolve_staging_cap()` value as the chunk
   budget (one knob, matches the original intent), or a separate "endpoint budget"? Proposed:
   reuse the cap.
2. **Chunk metric.** Accumulate by **dst rebuild bytes** per chunk (the thing that overshoots).
   Must be identical on every rank (deterministic param order + identical budget → identical
   chunk boundaries → NCCL stays in lockstep, like `_should_stream`'s all-reduced decision).
3. **No-pack path.** Keep `pack=0` routed through the same chunked loop (it already degrades
   inside `batched_transfer.transfer`).
4. **Verification.** `verify_all` bit-exact (GPU, off-NFS) with `CPU_OFFLOAD=0` (GPU-adam, the
   regressed path) **and** `=1` (cpu-adam), pack on/off, plus a peak-reserved-bytes probe to
   confirm flat peak vs the 2× baseline.

## 6. Archive — cpu-adam host-memory / 30B-hang work

The host-memory + 30B-hang work is **committed and preserved** on `feat/hostmem-30b-hang`
(worktree `../em-hostmem-30b`): codex `84d6d47` (reviewed approve-with-fixes) + the per-param
Adam-`step` repair `641fa2b`. **GPU verification is INCOMPLETE** — the A100 hung on the 30B
run (Cell 3), possibly this very host-OOM; the runner has completed; the box 30B `torchrun`
may still be hung holding the 8 GPUs (py-spy-vs-kill pending). That branch is set aside while
we fix the more fundamental flat-peak regression here; once this fix lands, it should **replace**
codex's cpu-adam-only streaming with the unified chunked path.

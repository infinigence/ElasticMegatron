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

Generalize the original per-param ordering to a tunable **chunk** granularity. Process the global
vparam list one rank-invariant chunk at a time; WITHIN a chunk use the **"approach 2" ordering**
that keeps a chunk's src and dst from coexisting:

```
for chunk in chunk_params(all_vps, budget):     # rank-invariant cut by vparam.size
    pack chunk's src slices -> send staging
    RELEASE chunk's src optimizer tensors         # ← before the comm
    exchange (cross-rank butterfly: send staging / recv staging)
    CREATE chunk's dst optimizer tensors           # ← after the comm
    unpack recv staging -> chunk's dst
```

Because each chunk's src is released before its dst is created, the chunk's src and dst never
coexist; the endpoints just swap `src_all → dst_all` across chunks, and the only resident overhead
is the chunk's staging working set (`chunk_size` = the send + recv staging tensors).

- **peak = max(src_all, dst_all) + chunk_size**  (baseline `chunk_size ≈ 2×cap`: one send + one recv
  tensor). Compare the regression's `~2×shard + 2×cap` and the original baseline's
  `max(src_all, dst_all) + max(one param)`.
- A less-careful variant (CREATE dst BEFORE the comm, don't release src early) gives
  `max(src_all, dst_all) + 2×chunk_size` — acceptable, but the release-before-create ordering is
  tighter and preferred.
- **`chunk_size` is a KNOB, decoupled from `cap`.** `cap` stays the staging/comm group size WITHIN a
  chunk (the existing communicator cap-loop); a chunk may hold MULTIPLE cap-groups to overlap
  gpu-copy/H2D with comm (Phase-2 overlap). Do NOT hardcode `chunk = 2×cap` — leave the interface.
- `chunk = 1 param` → the original per-param flat peak (= what codex's cpu-adam streaming does);
  larger chunk → fewer/larger comms (more coalescing) at higher staging cost. **Subsumes codex's
  `_main_process_streaming`** and fixes GPU-adam too.
- **Bit-exactness preserved** (same bytes/order; only the release/create timing + batch boundary
  move). The cap-before-dst over-commit (§3) is also gone — the peak no longer holds all-dst +
  all-src at once.

### The within-chunk comm mechanism is UNDER INVESTIGATION
The exact `pack → release-src → comm → create-dst → unpack` mechanism (single butterfly with
release/create around it, vs separate send/recv phases; deadlock-safety with the per-cap-round loop)
is being investigated by the `transfer-order-investigator` sub-agent — see the plan's Task 2. Chunk
boundaries are cut by the rank-invariant `vparam.size`; the per-numel byte size is DERIVED from the
actual `OptimizerTensorInfo` states (all-reduced for rank-identity), not hardcoded.

## 6. Archive — cpu-adam host-memory / 30B-hang work

The host-memory + 30B-hang work is **committed and preserved** on `feat/hostmem-30b-hang`
(worktree `../em-hostmem-30b`): codex `84d6d47` (reviewed approve-with-fixes) + the per-param
Adam-`step` repair `641fa2b`. **GPU verification is INCOMPLETE** — the A100 hung on the 30B
run (Cell 3), possibly this very host-OOM; the runner has completed; the box 30B `torchrun`
may still be hung holding the 8 GPUs (py-spy-vs-kill pending). That branch is set aside while
we fix the more fundamental flat-peak regression here; once this fix lands, it should **replace**
codex's cpu-adam-only streaming with the unified chunked path.

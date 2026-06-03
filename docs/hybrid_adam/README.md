# hybrid-adam (CPU+GPU mixed offload) — work log

Branch `feat/hybrid-adam`. Goal: support resharding when optimizer state is **partly
on CPU and partly on GPU** (hybrid offload Adam). FP8/FP4 and Muon are out of scope
for now (FP8 needs non-param-shaped state support; Muon is param-shaped and the
foundation already covers it, but is untested).

The work is split into a reusable **foundation** and the hybrid integration on top:

- **F1 — optimizer state model generalization.** Replace the hard-coded
  `(main_weight, exp_avg, exp_avg_sq)` triple with a variable-length, named,
  per-state device/dtype state set. Adam behaviour stays bit-equal. See
  [`../project/optimizer_state_model.md`](../project/optimizer_state_model.md) and
  [invariants I-15](../project/invariants.md).
- **F2 — device-aware transport.** `transfer/communicator.py` stages non-CUDA tensors
  through a GPU bounce buffer for NCCL send/recv/broadcast.
- **H — hybrid integration.** Wire the actual CPU+GPU offload optimizer; CPU-resident
  states keep an "offload, do not free" lifecycle (mirror `release_model(offload_weight=)`).

## Status (2026-05-29)

| Phase | Code | Verification |
|---|---|---|
| F1 state-model generalization | ✅ done | ✅ `dense_mix_full` **7/7+7/7** and `moe_mix_full` **7/7+7/7** ckpt bit-equal (8-GPU) |
| F2 device-aware transport | ✅ done | ✅ exercised by `CPU_OFFLOAD=1` (offloaded moments on pinned CPU ride the GPU-bounce-buffer transport); reshard verified |
| H HybridDeviceOptimizer integration | ✅ **verified** (H.1/H.2/H.4/H.6/H.7/H.8) | **8-GPU `dense_mix_full` full sweep (TP/PP/CP/DP/Group-Zero, 7 reshards) passes both with and without CPU-adam: weight 7/7 + optim 7/7 ALL PASS each.** Deadlock fix = reset stale `cached_param_buffer_shard_list` on re-entered chunk; optim compared via sub-multiset containment (non-zero PA padding). See `changelog.md` H.7/H.8 |

## Why a foundation first

The reshard layer assumed every param's optimizer state was exactly three GPU tensors of
the master's dtype. CPU+GPU offload breaks the *device* assumption; future optimizers
(Muon, FP8) break the *count* and *dtype* assumptions. F1 removes all three at once so each
later feature is a small, localized addition rather than another core refactor. See
[`changelog.md`](changelog.md) for the per-change detail.

## Files in this directory

- [`megatron_hybrid_optimizer.md`](megatron_hybrid_optimizer.md) — **reference**: how Megatron's
  `HybridDeviceOptimizer` + `--use-precision-aware-optimizer` work and differ from the plain
  `DistributedOptimizer` (init, state tensors, main↔model weight interaction), plus the code
  review of what ElasticMegatron must change to handle the hybrid-opt update.
- [`changelog.md`](changelog.md) — per-phase change detail and verification notes.

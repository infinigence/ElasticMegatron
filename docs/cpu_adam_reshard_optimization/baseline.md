# cpu-adam reshard() optimization — pre-optimization baseline + research

> **Purpose.** Frozen "before" record for the cpu-adam `reshard()` optimization workstream
> on branch **`feat/cpu-adam-core-verify`** (= `feat/core_r0.16.0` + the A/B correctness/host-mem
> ports, A100-verified). Captures (1) the measured baseline timing, (2) the honest cost framing,
> (3) the research verdicts on the two seed ideas, (4) the ranked levers + guardrails. Compare
> post-optimization results against §1.
>
> Date: 2026-06-23. Sources: optimization-research workflow `wf_0577049d` (5 agents); A100
> phase-timing run `reshard-phase-timing-20260623`. Cross-session memory:
> `cpuadam-reshard-opt-research`, `cpuadam-on-core-porting`, `reference-hdo-pageable-master-data-movement`,
> `gpu-adam-reshard-perf`. Logs: `_gpu_verify_logs/reshard-phase-timing-20260623/`.

## 0. Scope & assumptions

- A (`repair_per_param_step`) + B (POOL-B `release_offload_host_buffers`) are **correct** (A100-verified
  2026-06-23: no hang / per-param step continuity / host-mem bounded / 8-8 bit-exact / 0 NaN).
- Base transport = core's **per-op per-param-DIRECT** `Communicator.send/recv` (the fast path; the union
  chunked butterfly+packing is a proven 2-2.5x GPU-adam regression — see `gpu-adam-reshard-perf`). **Do
  NOT reintroduce it for speed.**
- Vehicle = Qwen3-30B-A3B `moe_30b` (TP4/PP1/CP1/EP4, world 8↔4 DP2↔DP1), cpu-adam (HDO, `CPU_OFFLOAD=1`).

## 1. Baseline timing (A100, 30B moe_30b 8↔4, 6 reshards, GBS64/seq4096)

Gated by `ELASTIC_RESHARD_PHASE_TIMING=1` (commit `1cbc24e`). Reshard phase timings are **independent of
gbs/seq** (they move the fixed optimizer+model state bytes). Per-reshard, ms:

| R# | dir | transfer | release_optimizer | update_model_weight | trim_total (3 calls) | update_mw % xfer |
|----|------|----------|-------------------|---------------------|----------------------|------------------|
| 1 | 8→4 | 65019 | 8420 | 2716 | 8397 | 4.2% (cold-start) |
| 2 | 4→8 | 33646 | 10380 | 995 | 10338 | 3.0% |
| 3 | 8→4 | 32282 | 5051 | 2076 | 5030 | 6.4% |
| 4 | 4→8 | 34362 | 9983 | 1116 | 9939 | 3.2% |
| 5 | 8→4 | 33282 | 5029 | 2177 | 5007 | 6.5% |
| 6 | 4→8 | 34471 | 10315 | 1018 | 10272 | 3.0% |

**Warm per-direction averages (R1 excluded as cold-start):**

| dir | transfer | release_optimizer | update_model_weight | trim_total | update_mw % xfer |
|------|----------|-------------------|---------------------|------------|------------------|
| 8→4 (warm) | **32782** | 5040 | 2127 | 5018 | **6.5%** |
| 4→8 | **34160** | 10226 | 1043 | 10183 | **3.1%** |

Run: clean exit, loss steady 10.78–10.80, 0 NaN, no OOM/hang, ~19 min wall.

**Findings:**
1. **transfer DOMINATES** — ~32–34 s steady (42.2 GB at ~1.2–1.3 GB/s). This is the big fish.
2. **`update_model_weight` is small** — ~3.0% (4→8) / ~6.5% (8→4) of transfer, i.e. ~1–2 s.
3. **`trim_host_memory` ≈ ALL of `release_optimizer`** (~5 s 8→4, ~10 s 4→8). The `resize_(0)` loops are
   cheap; the 3 trims/reshard (2 per-optimizer `release_offload_host_buffers` + 1 trailing
   `release_optimizer`) are the cost. **Direction-asymmetric**: scale-up (4→8) ~2x scale-down (dst is the
   larger world ⇒ more state materialized + trimmed).

## 2. Honest cost framing

End-to-end ~1 GB/s sits **BELOW every single-phase rate** (NCCL 6.4, H2D 5.6, D2H 2.9 GB/s profiled) ⇒
the bottleneck is the **serial dependency chain + overhead**, not any one phase, and production already
overlaps D2H behind NCCL on the default stream. The `_phase()` `cuda.synchronize()` profiler **distorts**
(pinned-bounce: 3.79x profiled → ~5% production). Trust only the unconditional outer wall-clock +
CUDA events, 80-reshard warm medians, production allocator (`expandable_segments`).

## 3. The two seed ideas — verdicts (both CONDITIONAL)

**(1) Merge the master→model H2D (`update_model_weight`/`copy_main_to_model`, elastic_manager.py:245) into
the transfer recv.** A true fusion (consume the still-on-GPU recv bytes — `communicator.py:101-106` lands
the master in a GPU bounce buffer before D2H'ing to the pageable CPU master — and downcast straight into
the bf16 model param) genuinely skips one pageable-source H2D pass. **But §1 shows it is only ~1–2 s
(3–6.5% of transfer)** — a trailing-phase shave, not the dominant exchange; it crosses
communicator/transfer/optimizer-adapter layers, must reassemble the post-DP-scatter + swiglu-unshuffled
final master geometry, and holding GPU buffers across params risks the I-16 flat-peak 2x regression.
→ **bounded win; do not do standalone** (only fold in opportunistically).

**(2) DP-size-only optimizations.** (a) Blanket "skip transfer for a retained shard" is **INFEASIBLE** —
src/dst are DISTINCT cached `MegatronState` objects with DISTINCT optimizer storage (dst freshly
`dummy_step()` + `master.resize_(0)`, holds dummy values until filled); only the `_send_self` clone can be
elided. (b) De-serializing the rank-0 gather/scatter funnel (`transfer.py` `_pre_process`/`_post_process`
P2P-through-aligned-rank → a DP-group `all_gather`, or survivor-absorb/split targeted P2P for integer
ratios) is **conditionally promising — but payoff scales with dp_size**, so the 30B DP2↔DP1 case (2 funnel
participants) is near-zero; worth building **only if a larger 8↔4 DP rescale is on the roadmap.**

## 4. Ranked levers (data-refined)

1. **Production-faithful CUDA-event profile of the ~33 s transfer internals** (collect / `_pre` DP-gather /
   `_main` per-op chain / `_post` DP-scatter) — *prerequisite*; both prior dirs (no-chunk gate,
   pinned-bounce 3.79x) were sync/noise artifacts. Decides where the multi-x headroom is. The §1
   instrumentation measures transfer as one block; this breaks it open.
2. **Coalesce per-(param×state) NCCL launches to the aligned peer in `_main_process`** — the **only multi-x
   lever** (attacks the serial chain). Adam = 3 blocking drains/edge to the SAME peer → fuse to 1. core's
   single-aligned-peer symmetric structure may be deadlock-safe where the union butterfly is not. On
   `feat/core_r0.16.0`.
3. **Trim consolidation 3→1/reshard** — **DONE** (commit `f6e8915`): dropped the per-optimizer trim from
   `release_offload_host_buffers`; `release_optimizer`'s single trailing `trim_host_memory()` covers all
   chained optimizers. Expected ~2–5 s/reshard (esp. 4→8). **Re-measure on A100 to confirm** (compare
   `release_optimizer`/`trim_total` against §1; bit-exact must be unchanged).
4. **Port `ELASTIC_TRANSFER_PINNED_BOUNCE` to core** — ~5% (D2H bounce), cheap opt-in, not the headline.
5. **Make Megatron's dead `pin_cpu_params` live (pin the HDO fp32 master)** — cross-repo (I-12 mirror);
   bigger payoff is per-STEP H2D copy-back overlap than reshard; raises pinned-host high-water (re-check
   budget). `update_model_weight` fusion (idea 1) ranks here too (~1–2 s).

## 5. Guardrails (do NOT)

- Reintroduce chunked/butterfly/packing transport for speed (2-2.5x GPU-adam regression, inherent).
- Build a 3-stream pipeline before profiling proves large exposed serial PCIe (production already overlaps).
- Trust any `_phase()` sync number to size a lever.
- Implement any skip/fuse as a **local per-rank** decision — must be a rank-identical collective verdict
  (else NCCL desync hang); validate on an asymmetric-world DP scale-down, not just symmetric.
- Skip `repair_per_param_step` for offloaded params in any fuse/skip (silent numeric bug; multiset verify
  is blind to per-param step).
- Add any async/batched host staging without the I-16 `released_host_bytes`-gated
  `current_stream().synchronize()` before src release (host-byte-tear).

## 6. Verification discipline

Every transport/host change: `tools/ckpt/verify_all.py` 8/8 weight + 8/8 optim multiset (MoE `--jobs 4`)
on a cpu-adam reshard — symmetric or scale-down. (A world-shrink `after_reshard` DCP save previously
crashed in PyTorch DCP `dedup_save_plans` on the scaled-out ranks' `None` SavePlans;
`elastic_megatron/distributed/dist_ckpt_patch.py` fixes it by redirecting the DCP global-plan collective
to the elastic world group on a scale-down, so scale-down bit-exact is now verifiable.) Plus NCCL liveness
on an asymmetric-world DP scale-down. Default-OFF env gates. Perf on the A100 box (80-reshard warm
medians), not aries.

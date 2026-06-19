# ElasticMegatron docs

> ## 接手 prompt(下个 session 直接照做)
>
> **任务:实现 Task 3(flat-peak reshard transfer)—— 代码开发。**
>
> **在 worktree `../em-pack-peak`(分支 `fix/transfer-flat-peak`)里干活,不是主 checkout。** 按本仓 `CLAUDE.md` onboard:读本文件下方的 **"Current state / handoff (2026-06-18)"** 块 → `buffer_opt/reshard-peak-memory.md`(设计)→ `buffer_opt/reshard-flat-peak-plan.md`(**Task 3 §3a–3e 就是下一步**)。耐用结论也在跨会话 memory 里。
>
> **现状(2026-06-19 更新):Task 3 已实现 + 对抗式 review + A100 bit-exact 验证全过 —— flat-peak 修复 DONE。** 传输峰值从 `max(src,dst)` 回归成 `~2×shard+2×cap`;修复 = chunked **approach-2**(`pack src → 释放 src → comm → 建 dst → unpack`),峰值 `= max(src,dst)+chunk_size`。**已落代码:** `8a128e5`(chunked approach-2 `_main_process` + `BatchedTransfer.pack/exchange/unpack` + `_chunk_recv_nbytes` 从 dst 占位符 metadata 定尺寸 + fake 模式单独 `_build_main_process_connections` 保原连接顺序)、`286d709`(`ELASTIC_TRANSFER_PEAK_PROBE` 探针)、`647b693`(review 发现:approach-2 在 comm 前 `release()` src,**pinned host(cpu-offload)src** 的异步 H2D pack copy 会和 host 端 free 竞态撕字节 → 加显式 `current_stream().synchronize()`,只在有 host src 时,GPU-adam 不付代价)。**A100 验证(2026-06-19):** dense_mix_full / 8-GPU / tiny-8层 / 9 iter,4 格(CPU_OFFLOAD=0 GPU-adam **和** =1 cpu-adam × pack on/off)全部 **`8/8 weight + 8/8 optim → ALL PASS`**,主进程峰值平在 428–516 MiB(cap 8192),无 OOM。**⚠️ 验证陷阱:** `ELASTIC_SAVE_CKPT=1` 的 bit-equal dump 写到 run dir 下的 `tools/ckpt/{before,after}_reshard`,该盘在机器上是 100% 满的共享 mount;`EXPERIMENTS_DIR=/tmp` **只**改弹性 manager 自己的 save、不改这个 dump → 必须**额外**把 `tools/ckpt` 软链到 `/tmp`,否则会在 iter≈5 假崩(torch DCP `unexpected pos` 短写,看着像代码 bug 其实是磁盘满)。
>
> **自验进度:** `git -C ../em-pack-peak log --oneline -6`;`_main_process` 现在是 dispatcher(fake → 连接预建分支;real → 逐 chunk approach-2);`PYTHONPATH=.:$MEGATRON_PATH python3 tests/test_transfer_chunking.py`(机器上 PASS,本机 SKIP)。
>
> **Task 3 已做完(§3a–3e 全落)** —— 不要重做。机制:`BatchedTransfer.pack()/exchange()/unpack()`(`exchange` butterfly 原样照搬;`transfer()` 仍服务 `_pre/_post`+no-pack);recv staging 用 metadata 定尺寸(`release()` 后 shape/dtype 还在);self-copy 复用已有队列延迟到 unpack;I-16 + padded 释放时序保持。
>
> **续接纪律:** 别重跑已定的设计/死锁调查(已结晶进 docs+memory);sub-agent 重新 spawn、别复用旧 ID;每次 commit 跑 py_compile + ruff + `test_transfer_chunking.py`。**唯一剩下的门 = A100 bit-exact `verify_all`**(CPU_OFFLOAD=0 GPU-adam **和** =1 cpu-adam,pack on/off)+ `ELASTIC_TRANSFER_PEAK_PROBE` 探针 —— **ckpt 走 off-NFS `/tmp`;A100 常驻空闲、有最高优先级使用权,直接用 `gpu-run` skill 起,不用先问。**
>
> **别混淆:** **30B 间歇性卡死**(`feat/hostmem-30b-hang`,卡在 shrink 后的 "building GPT model")是**另一个独立未决问题**(疑似 NCCL-group/c10d 竞态;需 3–5× 反复跑 + py-spy),**不是** flat-peak 这条线。
>
> **推荐 skill:** `systematic-debugging`(卡死/OOM/desync)、`verification-before-completion`、`graphify`(导航)、`gpu-run`(验证)。设计已定,无需 brainstorming。

Four areas:

- **[`project/`](project/)** — **start here.** Project-wide knowledge that survives across sessions: architecture, code layout, invariants you cannot break, the [optimizer state model](project/optimizer_state_model.md), debugging playbook, the cross-repo relationship with `Megatron-LM-custom`. Read these before changing code.
- **[`megatron_016_adaptation/`](megatron_016_adaptation/)** — record of the May 2026 session that adapted ElasticMegatron to Megatron-LM 0.16 (commit `593ec68`). Phase-by-phase changelog, the Phase B reviewer report, the external code review and our response. Useful as a worked example of how a non-trivial adaptation lands, but **not** required reading to start working on the project.
- **[`hybrid_adam/`](hybrid_adam/)** — ongoing work (branch `feat/hybrid-adam`) adding CPU+GPU mixed-offload optimizer support: the optimizer-state-model generalization (F1), device-aware transport (F2), and the hybrid integration (H). Read its `README.md` if you are extending optimizer support.
- **[`buffer_opt/`](buffer_opt/)** — batched / packed reshard transport (branch `ref/buffer-opt-integration`, PR #6): coalesce each peer's optimizer-state slices into one buffer + a single `batch_isend_irecv`, behind the `communicator.transfer()` boundary, with CPU-adam staging and optional memory bucketing. Read its `README.md` if you are touching `transfer/`. Current follow-ups in **this worktree**: [`reshard-peak-memory.md`](buffer_opt/reshard-peak-memory.md) (the flat-peak regression + fix design), [`reshard-flat-peak-plan.md`](buffer_opt/reshard-flat-peak-plan.md) (impl plan), [`cpu-adam-overlap.md`](buffer_opt/cpu-adam-overlap.md). (The cpu-adam host-memory analysis `cpu-adam-host-memory.md` and the post-measurement overlap update live on branch `feat/cpu-adam-transfer-opt`, **not** in this worktree.)

## Current state / handoff (2026-06-18) — flat-peak transfer fix in progress (3 branches)

This session reviewed codex's cpu-adam / 30B-hang branch, **discovered + designed a fix for a
reshard-transfer peak-memory regression**, and started implementing it. Work spans three
branches/worktrees:

1. **`fix/transfer-flat-peak`** (worktree `../em-pack-peak`, off `ref/buffer-opt-integration`) —
   **the ACTIVE work.** buffer-opt regressed the original **per-param FLAT-peak** reshard transfer
   (`main` / `feat/core_r0.16.0`: peak `≈ max(src_all, dst_all)`) into collect-all-dst /
   release-all-src (`~2×shard + 2×cap`), in two stages (buffer-opt routed the default path through
   the batched path; `ref/buffer-opt-integration` then **deleted the per-param fallback**, making
   the 2× peak universal). The free-mode staging cap is computed **before** dst is allocated, so it
   under-reserves (latent GPU-adam OOM, masked by small configs / cpu-adam being host-bound).
   - **Design** → [`buffer_opt/reshard-peak-memory.md`](buffer_opt/reshard-peak-memory.md): target
     peak `= max(src_all, dst_all) + chunk_size` via the **approach-2** per-chunk ordering (pack src
     → RELEASE src → comm → CREATE dst → unpack), so a chunk's src and dst never coexist;
     `chunk_size` is a knob decoupled from `cap`.
   - **Plan** → [`buffer_opt/reshard-flat-peak-plan.md`](buffer_opt/reshard-flat-peak-plan.md):
     mechanism **RESOLVED** (split `BatchedTransfer.transfer` into `pack()/exchange()/unpack()`,
     `exchange` keeps the butterfly verbatim; defer survival-rank self-copy to unpack — confirmed
     deadlock-safe by the `transfer-order-investigator` sub-agent + user). **Task 1 DONE** (`dbd6975`).
     **Task 3 DONE + adversarially reviewed** (`8a128e5` chunked approach-2 `_main_process` +
     `BatchedTransfer.pack/exchange/unpack` + `_chunk_recv_nbytes` from dst placeholder metadata +
     fake-mode `_build_main_process_connections`; `286d709` peak probe; `647b693` review fix =
     barrier the async H2D pack copy before releasing **pinned host** cpu-offload src). **Only the
     A100 bit-exact verify remains** (§Task 5).
   - Verify: adversarial-review workflow → A100 bit-exact `verify_all` (CPU_OFFLOAD=0 GPU-adam AND
     =1 cpu-adam, pack on/off) + the `ELASTIC_TRANSFER_PEAK_PROBE` peak probe.

2. **`feat/hostmem-30b-hang`** (worktree `../em-hostmem-30b`, codex's branch) — **REVIEWED
   (approve-with-fixes).** codex `84d6d47` (stabilize 30b cpu-adam reshard) + `dea0497` (POOL B
   pinned-grad release, ≡ the A100-verified `6ced76f`). Added **`641fa2b`** = per-param Adam `step`
   repair (codex's HDO dst-init seeded the CPU torch-AdamW per-param `step` to 0 → first
   post-reshard cpu-adam step used wrong bias correction; `verify_all` is blind to it — see the
   commit message). **GPU verify (with 641fa2b):** cpu-adam bit-exact `verify_all` **ALL PASS (3/3)**
   + step unit test PASS; **BUT the 30B asymmetric-world hang is INTERMITTENT** — froze once at the
   post-shrink "building GPT model" (dst model build), passed once to iter6 `EXIT_CODE=0` — **NOT
   reliably fixed** (suspected NCCL-group / c10d-store race at the shrink; needs 3–5× repeat +
   py-spy). The flat-peak fix (strand 1) will **subsume codex's cpu-adam-only
   `_main_process_streaming`**.

3. **`feat/cpu-adam-transfer-opt`** (the repo's main checkout, `../ElasticMegatron`) — earlier
   cpu-adam host-memory + transport-overlap work. Its design notes (`cpu-adam-host-memory.md` and
   the post-measurement `cpu-adam-overlap.md` with the single-sided-overlap A/B = **+14.3% on 30B**,
   D2H-bound, default OFF — from the `b2f423f` profiler run) live **on that branch**, NOT in this
   worktree (the `cpu-adam-overlap.md` here is the older, pre-measurement version). Holds entangled
   uncommitted docs; reconcile after the flat-peak fix lands.

> Worktree map (`git worktree list`): `../em-pack-peak` = `fix/transfer-flat-peak` (active),
> `../em-hostmem-30b` = `feat/hostmem-30b-hang`, `../ElasticMegatron` = `feat/cpu-adam-transfer-opt`.
> A 4th worktree `../em-buffer-opt` (`fix/transfer-staging-residency`) is a **separate / parked**
> staging-residency experiment — **not part of this flat-peak work**.

**Immediate next step: performance tuning of the reshard transfer.** The flat-peak fix AND the
host-memory fix are DONE + A100-verified on `fix/transfer-flat-peak`:
- Flat-peak (`8a128e5`/`286d709`/`647b693`): chunked approach-2, reviewed, bit-exact 8/8+8/8 ALL PASS
  (GPU-adam AND cpu-adam × pack on/off), flat peak 428–516 MiB, no OOM.
- Host-memory fold (`b2c31ed`/`166f3e9`/`aacc25c`): codex's POOL B pinned-grad release + 30B
  stabilization + Adam step repair folded in; codex's cpu-adam-only `_main_process_streaming`
  **subsumed** by the chunked path (dropped), its `empty_host_cache` reclaim **integrated** into
  `_main_process_chunk`. **30B cpu-adam host-mem PARITY verified (2026-06-19):** baseline W8 1006 GB =
  elastic pre-reshard W8 1006 GB (zero infra overhead); settled after W8→W4 reshard ~843 GB ≤ baseline
  (no leftover). Bit-exact regression re-passed.
- **Perf target:** the 30B cpu-adam reshard moved 45.35 GB in 44.2 s (**1.03 GB/s**, D2H/H2D-bound).
  Precedent to fold/adapt: the cpu-adam single-sided H2D-prefetch overlap (+14.3% on 30B, default OFF)
  on `feat/cpu-adam-transfer-opt` (`cpu-adam-overlap.md`, `b2f423f`), plus the multi-cap-per-chunk
  overlap seam left open in the flat-peak chunked design.
- Pre-existing, report-only (NOT this work): baseline `ELASTIC_ENABLED=0` PG-timeout assert in
  `update_pg_timeout_after_init` (only unwraps ElasticProcessGroup on the =1 branch); too-short 30B
  dev timeouts. The moe_30b asymmetric-world hang is a separate open issue (did not fire this run).
- Open follow-ups: a PR for `fix/transfer-flat-peak`; reconcile `feat/cpu-adam-transfer-opt`'s
  uncommitted docs. Durable findings live in cross-session memory.

## If you are a new agent picking up this repo

Read in this order:

1. **Top-level [`README.md`](../README.md)** — what ElasticMegatron is, public-facing API.
2. **[`project/README.md`](project/README.md)** — agent-facing project overview, concepts, where things live.
3. **[`project/repo_layout.md`](project/repo_layout.md)** — directory map with one-line file purposes.
4. **[`project/invariants.md`](project/invariants.md)** — design rules you must preserve.
5. **[`project/architecture.md`](project/architecture.md)** — how `reshard()` actually works inside.
6. **[`project/cross_repo.md`](project/cross_repo.md)** — what lives in `ElasticMegatron/` vs `Megatron-LM-custom/`, and why.
7. **[`project/debugging.md`](project/debugging.md)** — the four reshard failure modes and how to triage them.

Then, only if you need to understand the 0.16 work specifically, dip into `megatron_016_adaptation/` (start with its `README.md`).

# ElasticMegatron docs

> ## 接手 prompt(下个 session 直接照做)
>
> **任务:实现 Task 3(flat-peak reshard transfer)—— 代码开发。**
>
> **在 worktree `../em-pack-peak`(分支 `fix/transfer-flat-peak`)里干活,不是主 checkout。** 按本仓 `CLAUDE.md` onboard:读本文件下方的 **"Current state / handoff (2026-06-18)"** 块 → `buffer_opt/reshard-peak-memory.md`(设计)→ `buffer_opt/reshard-flat-peak-plan.md`(**Task 3 §3a–3e 就是下一步**)。耐用结论也在跨会话 memory 里。
>
> **现状:** 传输峰值从 `max(src,dst)` 回归成了 `~2×shard+2×cap`;修复 = chunked **approach-2**(`pack src → 释放 src → comm → 建 dst → unpack`),峰值 `= max(src,dst)+chunk_size`。**Task 1 已完成**(`dbd6975` chunker + `233317b` accessor 修正)。**机制已定**(`8913704`:phase-split `pack/exchange/unpack`、self-copy 延迟到 unpack —— 已证 deadlock-safe)。**Task 3 = 下一步。**
>
> **自验进度:** `git -C ../em-pack-peak log --oneline -10`;`elastic_megatron/transfer/transfer.py` 的 `_main_process` 仍是批量-全量版(Task 3 未落);`PYTHONPATH=.:$MEGATRON_PATH python3 tests/test_transfer_chunking.py`(机器上 PASS,本机 SKIP)。
>
> **做 Task 3(§3a–3e):** 给 `BatchedTransfer` 加 `pack()/exchange()/unpack()`(`exchange` 里的 butterfly **原样照搬**;`transfer()` 保留给 `_pre/_post` + no-pack);`_main_process` 改成逐 chunk 循环;**recv staging 尺寸用 dst 占位符 metadata + recv ranges 算,不用已分配的 dst**(这样 create-dst 才能留在 comm 之后);**self-copy 延迟到 unpack**。护栏:padded 释放仍留 `_post_process`;`_pre/_post` 不动;保持 I-16。
>
> **续接纪律:** 别重跑已定的设计/死锁调查(已结晶进 docs+memory);sub-agent 重新 spawn、别复用旧 ID;每次 commit 跑 py_compile + ruff + `test_transfer_chunking.py`。**Task 3 之后:** 对抗式 review workflow → **A100 bit-exact `verify_all`**(CPU_OFFLOAD=0 GPU-adam **和** =1 cpu-adam,pack on/off)+ `ELASTIC_TRANSFER_PEAK_PROBE` 探针 —— **ckpt 走 off-NFS `/tmp`,起 GPU 前先确认 A100 空闲。**
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
     deadlock-safe by the `transfer-order-investigator` sub-agent + user). **Task 1 DONE** (`dbd6975`:
     rank-invariant chunker + derived per-numel). **Task 3 (the `_main_process` rewrite) is the
     immediate NEXT step** (concrete §3a–3e; the one subtlety: recv staging is sized from dst
     placeholder metadata + recv ranges, NOT allocated dst, so create-dst stays after the comm).
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

**Immediate next step:** implement **Task 3** of `reshard-flat-peak-plan.md` on `fix/transfer-flat-peak`,
then adversarial-review workflow, then A100 bit-exact verify. The durable findings (codex review
verdict, the flat-peak regression, the intermittent hang) also live in cross-session memory.

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

# ElasticMegatron docs

Three areas:

- **[`project/`](project/)** — **start here.** Project-wide knowledge that survives across sessions: architecture, code layout, invariants you cannot break, the [optimizer state model](project/optimizer_state_model.md), debugging playbook, the cross-repo relationship with `Megatron-LM-custom`. Read these before changing code.
- **[`megatron_016_adaptation/`](megatron_016_adaptation/)** — record of the May 2026 session that adapted ElasticMegatron to Megatron-LM 0.16 (commit `593ec68`). Phase-by-phase changelog, the Phase B reviewer report, the external code review and our response. Useful as a worked example of how a non-trivial adaptation lands, but **not** required reading to start working on the project.
- **[`hybrid_adam/`](hybrid_adam/)** — ongoing work (branch `feat/hybrid-adam`) adding CPU+GPU mixed-offload optimizer support: the optimizer-state-model generalization (F1), device-aware transport (F2), and the hybrid integration (H). Read its `README.md` if you are extending optimizer support.

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

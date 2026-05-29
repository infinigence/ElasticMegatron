# Repo layout

One-line purpose per file. Read [`README.md`](README.md) first if you have not.

## Top level

| Path | What |
|---|---|
| [`README.md`](../../README.md) | User-facing project description, public API, examples |
| [`docs/`](..) | This documentation tree |
| [`elastic_megatron/`](../../elastic_megatron/) | The library proper |
| [`examples/`](../../examples/) | Drop-in patched `training.py` snapshots for Megatron 0.11 (`training_011.py`) and 0.16 (`training_016.py`), plus a README with the three required patch points |
| [`tools/`](../../tools/) | Offline helpers — ckpt comparators, loss noise-floor estimator, log plotters |
| [`run_e2e_demo.sh`](../../run_e2e_demo.sh) / [`run_moe.sh`](../../run_moe.sh) | Dense / MoE single-process pretrain launchers; env-var driven |
| [`run_experiment.sh`](../../run_experiment.sh) | Phase B sweep launcher — wraps the above two with named mode entries (`dense_mix_full`, `moe_mix_full`, etc.) |

## `elastic_megatron/`

The library is organised by concern, not by parallelism dimension:

| Sub-package | Concern |
|---|---|
| `megatron_manager/` | Mirrors Megatron's stateful globals (mpu, args, dataloader, training-state) and tracks one snapshot per parallel strategy |
| `resharding/` | Plans the send/recv map for a (src → dst) strategy transition: TP / PP / DP / EP analysis, plus the optimizer-tensor metadata |
| `transfer/` | Actually moves the data: NCCL groups, p2p-to-collective batching, IPC, process-group patching |
| `distributed/` | Lower-level process-group helpers (`ElasticProcessGroup`, dist_patch shim) |
| `elastic_manager.py` | The public `ElasticMegatronManager` facade |
| `hetero_dp.py` | Hetero-DP loss reduction helper (covers 0.11/0.13; 0.16+ raises `NotImplementedError`) |

### `megatron_manager/`

| File | What |
|---|---|
| `parallel_strategy.py` | `ParallelStrategy` dataclass + `is_redundant_backup()` + `update_global_args()` (push strategy back into Megatron `args`). Also defines `expert_is_dense_bucketed()`, the helper that decides "is EP=1 → expert lives in dense DDP bucket" — used by 4 sites in `resharding/` |
| `megatron_state.py` | `MegatronState` (per-strategy bundle of mpu / training_state / world_ranks) + `MegatronStateManager` (cache, builds union world group on transition) |
| `training_state.py` | `TrainingState`: owns model/optimizer/scheduler; `release_model()` / `rebuild_model()` (DDP buffer storage resize); `release_optimizer()`; `update_model_weight()` (copy main_param → model_param, sync across DP); `save_checkpoint()` (hook for offline verify); `update_optimizer_and_opt_param_scheduler()` (cross-chained step broadcast on EP transitions) |
| `mpu_state.py` | Snapshot of Megatron's `parallel_state` globals per strategy |
| `dataloader_state.py` | Per-strategy train/valid/test iterator factory; decides `args.do_train/do_valid/do_test` based on iterator presence + `all_reduce(MAX)` across ranks |
| `meta_device_context.py` | "Build a Megatron model on meta-device" context manager (used to build dst state before transfer without spending GPU memory) |
| `rank_generator.py` | Strategy-aware rank-id generator (mirrors mcore `RankGenerator`) |

### `resharding/`

| File | What |
|---|---|
| `resharding.py` | `ReshardPlan`: takes the TP/PP/DP/EP per-dim plans and turns them into a global-rank send/recv map. Includes the `get_global_rank()` dispatch and the `_ep_rank_for_global_rank()` helper (key invariant: EP=1 → pass `None` → dense-bucket dispatch) |
| `resharding_tp.py` | TP-axis plan: for each param, who-sends-to-whom across TP ranks. Has `force_unsharded=True` path for the EP=1 expert-as-dense case |
| `resharding_pp.py` | PP-axis plan: stage→stage param movement. Defines `ParamPositionAttr` |
| `resharding_dp.py` | DP-axis plan: which DP rank holds which slice. Mocks `_ParamAndGradBuffer.__init__` (via `mock_ddp_buffer_init` + `_FakeTensor`) to predict mcore's bucket layout without allocating |
| `resharding_metadata.py` | Per-param `OptimizerTensorInfo` (main_weight, exp_avg, exp_avg_sq) + `release/rebuild`; also the master `generate_resharding_metadata()` that walks the model and decides which params are expert vs dense (name-based: `".experts." in name`) |
| `virtual_param.py` | `VirtualParam` — per-param state + caches for `dp_distribution` and `reshard_plan` |
| `util.py` | `Range`, `ParamRange` helpers |

### `transfer/`

| File | What |
|---|---|
| `transfer.py` | Top-level `TransferManager` — orchestrates the transfer using the `ReshardPlan` + optimizer info. Includes redundant-backup path |
| `communicator.py` | Async NCCL sender/receiver |
| `ipc_manager.py` | Cross-process IPC (inter-process mode; unused in intra-process Phase B) |
| `p2p_to_collective.py` | Batches small p2p sends into one collective when receivers overlap |

### `distributed/`

| File | What |
|---|---|
| `elastic_process_group.py` | `ElasticProcessGroup` wraps `torch.distributed.ProcessGroup`, allows lazy NCCL group init for the (src ∪ dst) union world group |
| `dist_patch.py` | Monkey-patches `torch.distributed` to intercept group creation during state setup |
| `util.py` | Misc dist helpers |

## `examples/intra_process/`

| File | What |
|---|---|
| `README.md` | Three patch points to add into Megatron's `megatron/training/training.py` |
| `training_011.py` | Full snapshot for Megatron-LM 0.11 — drop-in replacement |
| `training_016.py` | Full snapshot for Megatron-LM 0.16 — drop-in replacement, includes Phase B sweep modes + `ELASTIC_SAVE_CKPT` hook. Elastic loop uses plain rebind (`model = training_state.model`); launcher must set `--eval-iters 0` and not pass `--save` (see invariants.md I-6). |

## `tools/`

See [`tools/README.md`](../../tools/README.md) for the full index. Highlights:

| File | What | Reusable? |
|---|---|---|
| `ckpt/compare_dcp.py` | Compare two dist_ckpt dirs at the *logical* tensor level (skips distrib-optim flat buffers) | yes, generic |
| `ckpt/compare_optim_logical.py` | Compare distrib-optim flat buffers as logical multisets (handles padding zeros, cross-dp_group TP-replica, DP padding tail); GPU-parallel sort | yes, generic |
| `ckpt/verify_all.sh` | Batch-run the two comparators over all `tools/ckpt/{before,after}_reshard/iter_*` pairs | yes (coupled to ckpt save path convention) |
| `ckpt/convert_and_compare.sh` + `compare_ckpt.py` + `run_convert_patch_loader.py` | Old comparison path through Megatron's `convert.py`. Partly broken on 0.16; kept as fallback | limited |
| `noise_floor.py` | P50/P95/P99 of pairwise \|Δloss\| between N baseline runs | yes, generic |
| `plot_loss_curve.py` | Plots multiple loss curves; auto-detects ElasticMegatron's plain-number `loss.txt` vs Megatron's regex log | yes, generic |
| `analyze_experiments.py` | Phase 2 four-way comparator (dense_baseline/dense_mix/moe_baseline/moe_mix); hardcoded `EXP_ROOT` and experiment names | **Phase 2 specific** — not a good base for new comparisons |

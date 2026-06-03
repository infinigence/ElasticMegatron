# Optimizer state model

How ElasticMegatron represents and moves a param's optimizer state during reshard.
Read this before touching `resharding_metadata.py` / `transfer/` for a new optimizer
(CPU-Adam, hybrid offload, Muon, FP8, ...).

## The contract

For each model param, the reshard layer holds an **`OptimizerTensorInfo`** — an
ordered, **variable-length** set of named states (`elastic_megatron/resharding/resharding_metadata.py`):

```
states = [ OptState("main_weight", <fp32 master>),
           OptState("exp_avg",     <moment 1>),
           OptState("exp_avg_sq",  <moment 2>),
           ... ]      # Adam => 3; another optimizer may differ
```

- `states[0]` is the **master weight** and the **geometry anchor**.
- `.optimizer_tensors` is the working list `[s.tensor for s in states]`; `.state_names`
  is the parallel name list. The transfer / release / rebuild / swiglu code iterate
  `optimizer_tensors` generically — they do not know or care which optimizer produced them.
- `.main_weight` / `.exp_avg` / `.exp_avg_sq` remain as read-only accessors for the
  Adam-shaped call sites (e.g. `transfer/ipc_manager.py`).

### Invariants the model relies on

1. **Every state is param-shaped.** `state.tensor.numel() == master.numel()`. This is
   what lets the reshard *geometry* (dp_distribution + `ReshardPlan`) be computed **once
   per param from the master** and reused for every state. Non-param-shaped states (e.g.
   FP8 per-block `scale`/`amax`) are **not supported** — `OptimizerTensorInfo.__post_init__`
   asserts this. See [invariants.md](invariants.md) I-15.
2. **Per-state `dtype` and `device` are independent.** `create_padded_optimizer_tensor`
   allocates each padded buffer with that state's own dtype/device. The transport
   (`transfer/communicator.py`) stages non-CUDA tensors through a GPU bounce buffer, so a
   CPU-resident state moves over NCCL transparently. (Adam: all states share the master's
   dtype/device, so behaviour is unchanged.)
3. **src and dst must carry the same ordered state set per param.** The transfer zips
   src/dst `optimizer_tensors` positionally. State discovery is therefore deterministic:
   `ordered_optimizer_state_keys()` filters to param-shaped tensors and orders Adam moments
   first (a scalar `step`, if stored per-param, is dropped — it is synced via param_groups,
   see I-5). `transfer/transfer.py::_main_process` asserts `src.state_names == dst.state_names`
   when both sides are present on a rank.

## Where states come from — the OptimizerAdapter

**All per-optimizer-implementation specifics live behind one class:
`OptimizerAdapter` (`elastic_megatron/resharding/optimizer_adapter.py`).** The reshard
pipeline never inspects the concrete optimizer; `get_optimizer_tensors_by_model_weight()`
and `training_state.update_model_weight()` just call adapter methods. The single dispatch
point is `OptimizerAdapter.create(optimizer)`.

The adapter answers the four questions that differ per optimizer:

| Method | What it hides |
|---|---|
| `get_main_weight(model_weight)` | where the param-group anchor lives (`DistributedOptimizer.model_param_group_index_map` vs non-distributed `fp32_from_float16_groups`); `None` ⇒ not owned by this optimizer |
| `model_param_sub_range(...)` | distrib flat-buffer sub-range vs whole-param |
| `ensure_state_initialized(..., offload)` | how to init a not-yet-stepped optimizer's state — `init_state_fn` / empty Adam placeholders (`init_empty_state_dict`) / HDO `dummy_step` — plus offloading the master |
| `discover_states(...)` | **SRC**: master + every param-shaped state in canonical order (`ordered_optimizer_state_keys`, no hard-coded names); precision-aware reads via `_get_main_param_and_optimizer_states` |
| `copy_main_to_model()` | master→model refill after transfer; precision-aware adds the explicit copy `optimizer.step()` would normally do |

Class hierarchy (each subclass = one implementation difference):

```
OptimizerAdapter
├── Float16OptimizerAdapter          # non-distributed
└── DistributedOptimizerAdapter      # standard distrib Adam
    └── PrecisionAwareOptimizerAdapter   # --use-precision-aware-optimizer
        └── HybridDeviceOptimizerAdapter # HDO (CPU+GPU offload): dummy_step init
```

### Adding a new optimizer

This adapter is the **only** extension point. To support a new optimizer:

1. Add a subclass overriding the methods whose behaviour differs (most need only
   `discover_states` and/or `ensure_state_initialized`).
2. Add one branch to `OptimizerAdapter.create()`.

No call site in `resharding_metadata.py` / `training_state.py` / `transfer/` changes.

**FP8 (the next target) hits the param-shaped wall.** Its per-block `scale`/`amax` are
*not* param-shaped, which `OptimizerTensorInfo` rejects (invariant I-15). An FP8 adapter
would override `discover_states` to carry those states **and** needs a separate transport
(see *Out of scope* below) — the adapter is where that seam belongs.

### HybridDeviceOptimizer (CPU+GPU offload)

Megatron's `HybridDeviceOptimizer` (HDO) is a `torch.optim.Optimizer` wrapped *inside*
`DistributedOptimizer` (so `optimizer.optimizer` is the HDO). It splits params per-param
between a CPU and a GPU sub-optimizer; the fp32 master stays on GPU, but **the moments of
CPU-offloaded params live on pinned CPU memory** — handled by per-state device (this model)
+ the GPU-bounce-buffer transport in `transfer/communicator.py`. Two HDO-specific points:

- `state[param]["master_param"]` (present because HDO uses `param_update_in_fp32=True`) is
  the master, not a separate state — excluded via `_NON_TRANSFER_STATE_KEYS`.
- HDO's `.state` is a synced view and `init_state_fn` is `None`; the not-initialized DST path
  calls `HDO.dummy_step()` to allocate real sub-optimizer state instead of
  `init_empty_state_dict` (`HybridDeviceOptimizerAdapter.ensure_state_initialized`). Enabled in
  launchers with `CPU_OFFLOAD=1` (needs `--use-precision-aware-optimizer`). Status + assumptions:
  [`../hybrid_adam/`](../hybrid_adam/).
- **`--use-precision-aware-optimizer` changes the layout**: `param_groups` holds the bf16/fp16
  shard (not the fp32 master), and the master + moments live in optimizer state. Read them via
  Megatron's `DistributedOptimizer._get_main_param_and_optimizer_states(model_param)` →
  `{"param", "exp_avg", "exp_avg_sq"}` (what `PrecisionAwareOptimizerAdapter.discover_states`
  does), **never** by treating `param_groups[...]` as the master.

## Out of scope (guarded, not handled)

- **Non-param-shaped states** (FP8/FP4 per-block scale/amax). Would need a second transport
  path (replicate/broadcast rather than reshard) and block-alignment with DP shard
  boundaries. The param-shaped assert stops these from being silently mishandled.
- **`transfer/ipc_manager.py`** (inter-process mode) still assumes Adam-shaped states; the
  intra-process path is state-agnostic after the F1 generalization.

See [`../hybrid_adam/`](../hybrid_adam/) for the work log of the CPU+GPU hybrid-offload effort
that this model was built for.

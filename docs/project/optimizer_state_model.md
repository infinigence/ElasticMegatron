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

## Where states come from

`get_optimizer_tensors_by_model_weight()` builds the state list two ways:

- **SRC (initialized optimizer):** discover generically from `optimizer.state[main_weight]`
  — master + every param-shaped state, in canonical order. No hard-coded key names.
- **DST (offload, state dict empty):** `init_empty_state_dict()` pre-creates the keys the SRC
  side will send (today: the Adam moments), storage-0, so the positional transfer lines up.
  **A non-Adam optimizer needs its own offload schema here** — this is the one place that
  still encodes "which states exist" for the not-yet-initialized side.

`get_main_weight()` resolves the master from either a `DistributedOptimizer`
(`model_param_group_index_map` → flat-buffer sub-range) or the non-distributed
`fp32_from_float16_groups`.

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
  `init_empty_state_dict`. Enabled in launchers with `CPU_OFFLOAD=1` (needs
  `--use-precision-aware-optimizer`). Status + assumptions: [`../hybrid_adam/`](../hybrid_adam/).
- **`--use-precision-aware-optimizer` changes the layout**: `param_groups` holds the bf16/fp16
  shard (not the fp32 master), and the master + moments live in optimizer state. Read them via
  Megatron's `DistributedOptimizer._get_main_param_and_optimizer_states(model_param)` →
  `{"param", "exp_avg", "exp_avg_sq"}` (what `get_optimizer_tensors_by_model_weight` does when
  precision-aware is on), **never** by treating `param_groups[...]` as the master.

## Out of scope (guarded, not handled)

- **Non-param-shaped states** (FP8/FP4 per-block scale/amax). Would need a second transport
  path (replicate/broadcast rather than reshard) and block-alignment with DP shard
  boundaries. The param-shaped assert stops these from being silently mishandled.
- **`transfer/ipc_manager.py`** (inter-process mode) still assumes Adam-shaped states; the
  intra-process path is state-agnostic after the F1 generalization.

See [`../hybrid_adam/`](../hybrid_adam/) for the work log of the CPU+GPU hybrid-offload effort
that this model was built for.

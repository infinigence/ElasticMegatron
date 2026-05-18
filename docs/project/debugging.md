# Debugging ElasticMegatron

A playbook for diagnosing reshard-related problems. Organised as symptom → hypothesis → quick triage → root cause → fix.

## Mental model

Three layers of "consistency" matter during reshard, and breaking any one of them blows up:

1. **Virtual-param construction** ([`virtual_param.py::build_virtual_model`](../../elastic_megatron/resharding/virtual_param.py)) — all ranks share one set of virtual params, built once based on the initial parallel strategy.
2. **Resharding-metadata registration** ([`resharding_metadata.py::generate_resharding_metadata`](../../elastic_megatron/resharding/resharding_metadata.py) + [`virtual_param.py::_register_metadata`](../../elastic_megatron/resharding/virtual_param.py)) — on every reshard, src and dst each re-attach "the optimizer state this rank actually owns" back onto the matching virtual param.
3. **Simulated vs real DP distribution** ([`resharding_dp.py::get_params_dp_distribution`](../../elastic_megatron/resharding/resharding_dp.py)) — ElasticMegatron replays Megatron's bucket layout via a fake DDP buffer, then uses the result to decide `dst_aligned_global_rank` (who owns the param on the P2P transfer's destination side). If this layer's simulation diverges from Megatron's actual `DistributedOptimizer.model_param_group_index_map`, you get the classic "aligned rank is notified but does not actually hold the optimizer state" contradiction → crash in Step-2.2 with a `None`.

Most reshard bugs reduce to a broken invariant at one of these layers.

---

## Symptoms → triage templates

### Symptom A — `AttributeError: 'NoneType' object has no attribute 'create_padded_optimizer_tensor'`

Crashes inside [`transfer/transfer.py:222`](../../elastic_megatron/transfer/transfer.py).

**Meaning.** Some rank was told it is the dst-aligned receiver for a virtual param (should hold that param's optimizer shard after the reshard), but it never registered a `dst_optimizer_tensor_info` — meaning Megatron's `DistributedOptimizer` did not actually give this rank a shard for that param.

**Quick triage:**

1. Add a counter at the end of [`resharding_metadata.py::generate_optimizer_tensor_info`](../../elastic_megatron/resharding/resharding_metadata.py), printing `total / with_state / missing` per rank:

   ```python
   if os.environ.get("EM_DEBUG_OPTTENSOR", "0") == "1":
       total = len(params_to_optimizer_tensor_info)
       with_state = sum(1 for v in params_to_optimizer_tensor_info.values() if v is not None)
       print(f"[EM-OPTSUM rank={rank}] total={total} with_state={with_state}", flush=True)
   ```

   Compare `with_state` across ranks. If it is wildly uneven (e.g., 2 ranks have almost all of the optimizer state and the others have a handful), the DP-bucket split has clumped one whole group of params onto a few ranks.

2. Add a set-diff at the end of `virtual_param.py::_register_metadata`, grouped by `is_expert` + `expert_id`, printing which VPs failed to get `dst_optimizer_tensor_info`.

3. In `resharding_metadata.py::get_param_position_attr`'s expert branch, print `expert_id_offset` / `global_expert_id` to verify the sim's expert-id assignment matches the model side.

**Confirmed root-cause examples:**

- **EP=1 puts expert params in the dense DDP bucket** (because TE `GroupedLinear` flips `allreduce=True` when EP=1). ElasticMegatron's old simulation kept expert params in a separate MoE bucket via name-based `_is_expert_param`. The bucket layouts disagreed → `dst_aligned_global_rank` pointed at the wrong rank. Fixed in `resharding_dp.py` via the `experts_are_dense_bucketed` branch (2026-05-13). See [invariants.md](invariants.md) I-1.

### Symptom B — `AssertionError` inside `_register_metadata`

Usually `tensor_parallel_attr` mismatch.

**Meaning.** The same logical parameter, on src vs dst strategies, has different `TensorParallelAttr` values (`tensor_model_parallel` / `partition_dim` / `stride`). Reshard requires the two sides to describe "the same param" — they can differ in sharding but not in metadata identity.

**Quick triage:** Print all three fields right before `find_transformer_layer_param` returns, and again right before `assert virtual_param.tensor_parallel_attr == resharding_metadata.tensor_parallel_attr`.

**Confirmed root-cause examples:**

- **The `allreduce` attribute of MoE experts flips between EP=1 and EP>1** (TE `GroupedLinear`'s `expert_parallel` flag). Using `not getattr(param, "allreduce", True)` as the `is_expert` predicate classifies the same param differently across EPs. **Fix:** name-based classification (`".experts." in name`) — see [invariants.md](invariants.md) I-3.
- **`partition_dim` of MoE experts depends on `explicit_expert_comm`.** **Fix:** force `force_unsharded=True` for `TPE=1` so the param's `partition_dim` attribute is not consulted.

### Symptom C — NCCL hang / `Watchdog caught collective operation timeout`

Happens during the transfer phase, or in the first allreduce right after reshard.

**Meaning.** Some rank is in the collective's group but never enqueued the matching op (or vice versa). Usually this means "ElasticMegatron's planned `union_world_group` / send-recv plan" disagrees with what the ranks actually do.

**Triage steps:**

1. **Turn on NCCL debug logging to see which comm / op is stuck:**

   ```bash
   NCCL_DEBUG=INFO \
   NCCL_DEBUG_SUBSYS=COLL,INIT,P2P \
   NCCL_DEBUG_FILE=/tmp/nccl_rank_%h_%p.log \
   ELASTIC_ENABLED=1 ... bash run_moe.sh
   ```

   `NCCL_DEBUG_FILE` with `%p` splits one file per process so you can immediately see "which rank's log stopped at which point". `NCCL_DEBUG_SUBSYS=ALL` is noisy; `COLL,INIT,P2P` is enough.

2. **Use the watchdog message's `SeqNum` + `NumelIn` / `NumelOut` to locate the op:**

   PyTorch prints:

   ```
   Watchdog caught collective operation timeout: WorkNCCL(
       SeqNum=35, OpType=COALESCED, NumelIn=0, NumelOut=0, Timeout(ms)=60000
   )
   ```

   - `NumelIn/Out=0` means an empty op — usually a `torch.distributed.barrier()`.
   - Matching `opCount=35 sendbuff=... nelem=...` in the NCCL debug log tells you which group / rank started waiting.

3. **Inject prints into `torch.distributed` itself** (the highest-value probe sites):

   - `torch/distributed/distributed_c10d.py::all_reduce` / `broadcast` / `send` / `recv` entry — print `rank, group name/size, op shape`.
   - `torch/distributed/distributed_c10d.py::barrier` — print `rank, group`.

   Example patch on `/usr/local/lib/python3.12/dist-packages/torch/distributed/distributed_c10d.py`:

   ```python
   def broadcast(tensor, src, group=None, async_op=False, ...):
       import os
       if os.environ.get("EM_DEBUG_TDIST", "0") == "1":
           r = get_rank() if is_initialized() else -1
           g = group.group_name if group is not None else "WORLD"
           print(f"[EM-TDIST rank={r}] broadcast src={src} group={g} shape={tuple(tensor.shape)}", flush=True)
       ...
   ```

   Diff the per-rank print sequences — the trace will stop at the hung op and reveal which rank is one collective ahead or behind.

4. **Probe ElasticMegatron's transfer side too**, where the prints carry business semantics directly tied to the transfer plan:

   - `transfer/communicator.py::send` / `recv` — `rank, peer_rank, shape, dtype`.
   - `transfer/p2p_to_collective.py::send` — the set of ranks participating in each p2p→collective conversion.
   - `elastic_manager.py::transfer_params` entry — barrier then print the rank set.

   If you find "rank X says it is a sender, but rank Y's receiver list doesn't include this peer", the transfer plan generation is wrong — chase up into `virtual_param.py` / `resharding_dp.py::DataParallelReshardingInfo`.

5. **Check `union_world_group`.** `state_manager.reshard` computes a union world group (rank set of src ∪ dst). If it is missing a rank, that rank won't participate in transfers, but the existing collectives may still be waiting for it. Print `src_megatron_state.mpu_state.world_ranks`, `dst_megatron_state.mpu_state.world_ranks`, and `union_world_ranks` and check for asymmetry.

**Rule of thumb.** Any `collective timeout` with `NumelIn=0/Out=0` is, nine times out of ten, a barrier or a rebuilt EP/DP group whose rank count is off.

### Symptom D — `setStorage: ... out of bounds for storage of size 0`

**Meaning.** Some tensor's storage was previously `storage().resize_(0)`'d (typically by an optimizer offload), but downstream code tries to view it.

**Common scenarios:**

- **Training start-up.** Stale ckpts from a previous run, where `resume` reads a shape that does not match the actually-zero storage (because it was offloaded). **Fix:** before switching demos, clear the ckpt dir: `rm -rf /mnt/hisys-data/tonic/log/iter_* /mnt/hisys-data/tonic/log/latest_checkpointed_iteration.txt`.

- **Post-train eval forward (fixed 2026-05-18).** After reshard, `state_manager.reshard()` calls `release_model()` on src, which resizes the DDP buffer's `param_data` storage to 0. The elastic loop inside `train()` did `model = training_state.model`, but that only rebinds `train()`'s *local* variable — `pretrain()`'s reference to the original `model` list still points at the very first model (possibly the src of some past reshard, whose storage was released). The post-train `evaluate_and_print_results` uses that stale model → the embedding layer's `F.embedding(input, weight)` triggers `setStorage size 0`. **Fix:** slice-assignment, `model[:] = training_state.model`, so the list's contents are mutated in place and `pretrain()`'s reference observes the new model chunks. See [megatron_016_adaptation/changelog.md](../megatron_016_adaptation/changelog.md) "Eval-time setStorage / do_test bug" and [invariants.md](invariants.md) I-6.

---

## General probe-site cheat-sheet

| What to verify | Probe site | Key fields to print |
|---|---|---|
| Which params are classified as expert | `resharding_metadata.py::_is_expert_param` return | `name`, return value, `.allreduce` attr |
| `tensor_parallel_attr` matches across src/dst | `virtual_param.py::_register_metadata` assert site | `is_src`, the three `tensor_parallel_attr` fields, `module_name` |
| Which rank owns which optimizer state | `resharding_metadata.py::generate_optimizer_tensor_info` end | `rank`, `total`, `with_state` |
| DP distribution sim vs reality | `resharding_dp.py::get_ddp_buffer_distribution` return | per-param `dp_rank → Range` mapping |
| Reshard plan send/recv members | `transfer/transfer.py::_block_and_print` (built-in, enable with `use_block_and_print=True`) | sender → receiver mapping |
| `union_world_group` members | `megatron_manager/megatron_state.py::reshard` return | `src.world_ranks`, `dst.world_ranks`, `union_world_ranks` |
| NCCL collectives themselves | `torch.distributed.distributed_c10d` source | `rank`, `group`, op shape |

**Convention:** prefix debug prints with `[EM-XXX rank=R]` for grep-friendliness. Before shipping, clean up: `grep -rn "EM-DEBUG\|\[EM-" elastic_megatron/`.

---

## Useful environment variables

```bash
# NCCL layer
NCCL_DEBUG=INFO                               # Print NCCL init + collective steps
NCCL_DEBUG_SUBSYS=COLL,INIT,P2P              # Limit subsystems; ALL is too noisy
NCCL_DEBUG_FILE=/tmp/nccl_rank_%h_%p.log     # One file per process
NCCL_BLOCKING_WAIT=1                          # Surface NCCL errors immediately, no watchdog delay

# Torch distributed layer
TORCH_NCCL_BLOCKING_WAIT=1
TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=60
NCCL_TIMEOUT=60
TORCH_DISTRIBUTED_DEBUG=DETAIL                # Torch prints collective args
TORCH_CPP_LOG_LEVEL=INFO

# Megatron layer (CLI flag)
--distributed-timeout-minutes 1               # ProcessGroup timeout 1 minute
```

We keep the default at 1 minute during development so hangs surface fast. Bump it back up before production.

---

## Postmortems for resolved bugs

### EP=1 reshard crash (2026-05-13)

**Symptom.** Symptom A. `moe_mix` strategy, `EP=8 → EP=1` reshard crashes.

**Diagnosis path:**

1. Added a print right before `transfer.py:222` — found `dst_optimizer_tensor_info is None` for an expert param.
2. Printed `present_experts` at the end of `_register_metadata` — rank 0 DST had registered 0 of the expected 8 experts.
3. Dumped the expert param list in `generate_resharding_metadata` — Megatron's model on rank 0 actually had 32 expert params (8 experts × 4 weights). The model was fine.
4. Counted `with_state` in `generate_optimizer_tensor_info` — rank 0 had 1 param with optimizer state, ranks 3 and 4 had 23. **This was the signal.**
5. Traced back through `DistributedOptimizer.model_param_group_index_map` — found Megatron buckets by the `allreduce` attribute, and at EP=1 all experts have `allreduce=True`, so they fall into the dense bucket.
6. Compared against ElasticMegatron's sim (`get_params_dp_distribution`): it was routing by name-based `is_expert` and giving experts their own MoE bucket. Bucket layouts diverged → fix.

**Lesson.** Severely uneven `with_state` distribution across ranks almost always means "DP-bucket sim ≠ reality". Start at the DP-bucket layer; don't waste time chasing the downstream aligned-rank logic first.

### `moe_mix` 4th reshard hang, `EP=1 → TP=2/EP=1` (resolved 2026-05-13)

- **Symptom.** Symptom C, NCCL timeout, `OpType=COALESCED NumelIn=2909504 NumelOut=11638016`.
- **Root cause.** `get_global_rank` ([`elastic_megatron/resharding/resharding.py`](../../elastic_megatron/resharding/resharding.py)) used `expert_tensor_parallel_size` as the TP-axis coefficient for expert params. Megatron's rank layout always uses `tensor_model_parallel_size` (`order="tp-cp-ep-dp-pp"` puts tp innermost — dense and expert share one rank space). When `TP ≠ ETP` (this case: `TP=2, ETP=1, EP=1`), the simulated aligned global rank pointed at the wrong physical rank — e.g., the real global for `(dp=2, tp=0)` is 4, but sim computed 2.
- **Why earlier reshards didn't trigger it.** The first three reshards all stayed at `TP=1` (the elastic strategy passed `tp=1`), making `tensor_model_parallel_size = expert_tensor_parallel_size = 1` — the bug stayed dormant. The 4th reshard's dst was `TP=2/EP=1/TPE=1`, the first combination where `TP ≠ ETP`.
- **Fix.** When `ep_size == 1`, Megatron treats experts as dense (TE `GroupedLinear` / `Linear` sets `allreduce = not (is_expert and expert_parallel)`; at EP=1, `allreduce=True` → dense bucket). ElasticMegatron mirrors this with five coordinated changes:
  1. **`resharding_dp.py::_init_dp_distribution_with_global_rank`** — for an expert VP when that side has `ep_size==1`, call `get_global_rank` with `expert_model_parallel_rank=None` and the **dense `tp_rank`**, so the `dp_rank → global_rank` mapping matches Megatron's dense rank layout.
  2. **`resharding.py::ReshardPlan._generate_global_send_info/_recv_info`** — same policy: when src or dst has `ep_size==1`, pass `expert_model_parallel_rank=None` (dense branch).
  3. **`virtual_param.py::_generate_reshard_plan` (Step-1)** — for an expert VP when that side has `ep_size==1`, use the **dense `tp_size`** (not `etp_size`) as the TP-resharding `parallel_group_size`, so `_resharding_without_split` correctly fans out a TP-1→2 send/recv to both dst TP ranks.
  4. **`parallel_strategy.py::update_global_args`** — also push `sequence_parallel` back into Megatron's global `args`; otherwise reshard to `TP>1` raises `ValueError("MoE and tensor parallelism... without sequence parallelism")`.
  5. **`resharding.py::ReshardPlan.__post_init__`** — relax the "expert's TP send/recv plan must be exactly one entry each" assert to "only required when both sides have EP>1", since EP=1-as-dense legitimately produces multiple entries.
- **Verification.** All 8 reshards of `moe_mix` pass on 8 GPUs; all 8 reshards of `dense_mix` pass. The 4th reshard goes from "hang" to "11 ms success". Loss decreases monotonically `10.38 → 10.04`.

### (Historical) runtime bucket-layout reader experiment — wrong direction

An early hypothesis was "sim and real DDP bucket distributions disagree". I wrote a `_get_params_dp_distribution_from_runtime` that read the layout from a live `DistributedDataParallel.buffers` and overrode the sim. Result: **sim and runtime `dp_distribution` agreed 100%** (mapping by `dp_rank → Range`) and the hang persisted. **The negative result was useful — it redirected attention to the real bug (the `get_global_rank` mapping layer) instead of the bucket-slicing layer.** Not merged.

- Symptom D candidate (older note): hypothesis was that `release_optimizer()` released `main_weight` storage and `copy_main_params_to_model_params` repointed the weight without realizing the storage. Next step (if it ever recurs): print every param's `storage().size()` inside `training_state.py::update_model_weight` and check whether reshard accidentally resizes any storage.

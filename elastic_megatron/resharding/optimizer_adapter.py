"""OptimizerAdapter — isolate per-optimizer-implementation details for reshard.

The reshard pipeline needs four things from an optimizer, and every one of them
is implemented differently across Megatron's optimizer flavours:

  * where the fp32 master of a model param lives (param_groups vs the
    precision-aware optimizer state vs a non-distributed fp32_from_float16 group);
  * how to initialize a not-yet-stepped optimizer's per-param state on the dst
    (``init_state_fn`` vs empty Adam placeholders vs HybridDeviceOptimizer's
    ``dummy_step``);
  * which named, param-shaped states to transfer and in what order;
  * how to refill the model ``param_data`` from the master after the transfer
    (a plain copy vs the explicit precision-aware copy that ``optimizer.step``
    would normally do).

Before this module those branches were scattered across ``resharding_metadata``
and ``training_state`` as ad-hoc ``isinstance`` / class-name / config-flag checks,
so each new optimizer meant editing several call sites. They now live behind one
uniform interface: the reshard pipeline talks to an ``OptimizerAdapter`` and never
inspects the concrete optimizer. The single dispatch point is ``OptimizerAdapter.create``.

Adding a new optimizer (the next target is FP8) = add one subclass + one branch in
``create``; no call site changes. FP8's extra wrinkle is *non-param-shaped* states
(per-block scale / amax), which ``OptimizerTensorInfo`` does not yet support (see
invariant I-15) — an FP8 adapter would override :meth:`discover_states` (and likely
:meth:`ensure_state_initialized`) to carry those, plus a dedicated transport.
"""

from dataclasses import dataclass
from weakref import WeakKeyDictionary

import torch
from megatron.core.optimizer import DistributedOptimizer, MegatronOptimizer

from .util import ParamRange, Range

# Canonical ordering of per-param optimizer-state keys. The SRC side discovers
# states from an already-initialized optimizer.state dict; the DST side allocates
# empty placeholders during offload. The transfer zips src/dst optimizer_tensors
# positionally, so both sides MUST enumerate states in the same order. Known Adam
# moments come first in a fixed order; any other param-shaped states are appended
# alphabetically. Non-param-shaped entries (e.g. a scalar ``step``) are dropped —
# they are not transferred here (step is synced via param_groups, see I-5).
_ADAM_STATE_KEYS = ("exp_avg", "exp_avg_sq")

# State-dict entries that must never be transferred as a separate state:
#   - "master_param": Megatron's HybridDeviceOptimizer (param_update_in_fp32=True)
#     stores the fp32 master copy here. It IS the master (already states[0]); emitting
#     it again would transfer the master twice / corrupt the multiset.
#   - "step": a per-param scalar; synced via param_groups (see invariants I-5). It is
#     also dropped by the param-shaped filter, but we exclude it by name for clarity.
_NON_TRANSFER_STATE_KEYS = ("master_param", "step")


def ordered_optimizer_state_keys(state: dict, anchor_numel: int) -> list[str]:
    """Param-shaped state keys of one param, in a deterministic transfer order."""

    def is_param_shaped(v) -> bool:
        return torch.is_tensor(v) and v.numel() == anchor_numel

    keys = [
        k
        for k in state
        if k not in _NON_TRANSFER_STATE_KEYS and is_param_shaped(state[k])
    ]
    known = [k for k in _ADAM_STATE_KEYS if k in keys]
    extra = sorted(k for k in keys if k not in _ADAM_STATE_KEYS)
    return known + extra


def init_empty_state_dict(optimizer, main_weight: torch.nn.Parameter):
    """Allocate empty (storage-0) placeholder states for the offload path.

    Default schema = Adam moments. The DST optimizer is freshly built so its
    state dict is empty; we pre-create the keys the SRC side will send so the
    positional transfer lines up, then resize their storage to 0. A non-Adam
    optimizer (e.g. Muon's ``momentum``) needs its own offload schema here; the
    already-initialized SRC path discovers keys generically and needs no change.
    """
    for key in _ADAM_STATE_KEYS:
        optimizer.state[main_weight][key] = torch.zeros_like(main_weight.data)
        optimizer.state[main_weight][key].storage().resize_(0)


def _is_hybrid_device_optimizer(inner_optimizer) -> bool:
    """True if the inner torch optimizer is Megatron's HybridDeviceOptimizer.

    Detected by class name to avoid a hard import dependency on Megatron versions
    that lack the cpu_offloading module. HDO keeps its authoritative state inside
    device-specific sub-optimizers; its ``.state`` is a synced view, ``init_state_fn``
    is None, and the fp32 master/CPU-offloaded moments may live on different devices
    (see docs/project/optimizer_state_model.md and docs/hybrid_adam/).
    """
    for klass in type(inner_optimizer).__mro__:
        if klass.__name__ == "HybridDeviceOptimizer":
            return True
    return False


def _precision_aware_copy_main_to_model(dist_optimizer: DistributedOptimizer) -> None:
    """Refill the model param_data buffers from the (just-transferred) fp32 masters.

    Needed only under --use-precision-aware-optimizer, where
    DistributedOptimizer._copy_main_params_to_model_params() early-returns (the
    master->model copy is normally done inside optimizer.step()). During a reshard no
    step runs, so we replicate that copy here: for every model param owned by this
    optimizer, write its fp32 master shard into the model param_data buffer at the
    param's gbuf-world range. Mirrors Megatron's copy_group_params
    (distrib_optimizer.py:2438-2463) but pulls the master from
    _get_main_param_and_optimizer_states (the master lives in optimizer state under
    precision-aware, not in param_groups), so it uniformly covers both the float16 body
    and the shard_fp32 (LayerNorm/bias) group. See docs/hybrid_adam/megatron_hybrid_optimizer.md §6.
    """
    for model_param in dist_optimizer.model_param_group_index_map:
        master = dist_optimizer._get_main_param_and_optimizer_states(model_param)["param"]
        world_range = dist_optimizer._get_model_param_range_map(model_param)[
            "gbuf_world_in_bucket"
        ]
        gbuf_index, _, bucket_id = dist_optimizer.model_param_gbuf_map[model_param]
        param_buffer = dist_optimizer.buffers[gbuf_index].buckets[bucket_id].param_data
        shard_model = param_buffer.view(-1)[world_range.start : world_range.end]
        assert shard_model.numel() == master.numel(), (
            f"param_data world range {world_range.size} != master numel {master.numel()}"
        )
        shard_model.copy_(
            master.reshape(-1).to(device=param_buffer.device, dtype=param_buffer.dtype)
        )


@dataclass
class OptState:
    """One named optimizer-state tensor of a param (master copy or a moment).

    ``device`` / ``dtype`` are read from ``tensor``. The reshard geometry
    (dp_distribution / reshard plan) is computed once per param from the master
    and reused for every state, which requires every state to be *param-shaped*
    (same numel as the master). Non-param-shaped states (e.g. FP8 per-block
    scales) are not supported — see invariants I-15.
    """

    name: str
    tensor: torch.Tensor


# Memoize one adapter per optimizer object: adapters are stateless except for the
# non-distributed float16 index map (built lazily, derived from the optimizer), so
# reuse across the per-param metadata loop avoids rebuilding that map O(num_params)
# times. WeakKeyDictionary auto-evicts when a TrainingState's optimizer is dropped.
_ADAPTER_CACHE: "WeakKeyDictionary[MegatronOptimizer, OptimizerAdapter]" = (
    WeakKeyDictionary()
)


class OptimizerAdapter:
    """Uniform, optimizer-implementation-agnostic view used by the reshard pipeline.

    Wraps ONE non-chained ``MegatronOptimizer`` (ChainedOptimizer is unwrapped by the
    caller, one adapter per inner optimizer). Subclasses encode the per-implementation
    differences; :meth:`create` is the single place that decides which one to use.
    """

    def __init__(self, optimizer: MegatronOptimizer):
        self.optimizer = optimizer

    # ---- dispatch -----------------------------------------------------------
    @staticmethod
    def create(optimizer: MegatronOptimizer) -> "OptimizerAdapter":
        cached = _ADAPTER_CACHE.get(optimizer)
        if cached is not None:
            return cached

        if isinstance(optimizer, DistributedOptimizer):
            precision_aware = getattr(
                getattr(optimizer, "config", None),
                "use_precision_aware_optimizer",
                False,
            )
            if precision_aware:
                if _is_hybrid_device_optimizer(optimizer.optimizer):
                    adapter = HybridDeviceOptimizerAdapter(optimizer)
                else:
                    adapter = PrecisionAwareOptimizerAdapter(optimizer)
            else:
                adapter = DistributedOptimizerAdapter(optimizer)
        else:
            adapter = Float16OptimizerAdapter(optimizer)

        _ADAPTER_CACHE[optimizer] = adapter
        return adapter

    # ---- interface (defaults = standard, generically-discovered Adam) -------
    def get_main_weight(
        self, model_weight: torch.nn.Parameter
    ) -> torch.Tensor | None:
        """The param-group anchor param (state-dict key / offload-resize target).

        Returns ``None`` if this param is not owned by this optimizer. Note: under
        precision-aware this is the bf16/fp16 model SHARD, not the fp32 master — the
        real master is resolved separately by :meth:`discover_states`.
        """
        raise NotImplementedError

    def model_param_sub_range(
        self, model_weight: torch.nn.Parameter, main_weight: torch.Tensor
    ) -> ParamRange:
        """Sub-range of the model param that this data-parallel rank owns.

        Default (non-distributed): the rank owns the whole param.
        """
        return ParamRange(param_shape=model_weight.shape)

    def ensure_state_initialized(
        self,
        model_weight: torch.nn.Parameter,
        main_weight: torch.Tensor,
        offload: bool,
    ) -> None:
        """Make sure ``optimizer.state[main_weight]`` exists; offload the master if asked.

        Default: real init via ``init_state_fn`` (non-offload) or empty Adam
        placeholders (offload). Idempotent — no-op once the state is populated.
        """
        if len(self.optimizer.optimizer.state[main_weight]) != 0:
            return
        if not offload:
            self.optimizer.init_state_fn(self.optimizer.optimizer)
        else:
            init_empty_state_dict(self.optimizer.optimizer, main_weight)
            main_weight.storage().resize_(0)

    def discover_states(
        self, model_weight: torch.nn.Parameter, main_weight: torch.Tensor
    ) -> list[OptState]:
        """Ordered named states to transfer: master + param-shaped state tensors.

        Default reads ``optimizer.state[main_weight]`` generically. For Adam this is
        exactly ``[main_weight, exp_avg, exp_avg_sq]`` (a scalar ``step`` is dropped),
        so the Adam transfer is unchanged.
        """
        state = self.optimizer.optimizer.state[main_weight]
        states = [OptState("main_weight", main_weight)]
        for key in ordered_optimizer_state_keys(state, main_weight.numel()):
            states.append(OptState(key, state[key]))
        return states

    def copy_main_to_model(self) -> None:
        """Refill model param_data from the (transferred) masters after a reshard."""
        self.optimizer._copy_main_params_to_model_params()

    def release_offload_host_buffers(self) -> None:
        """Free per-optimizer host-side offload buffers not covered by the
        param-shaped optimizer-state release (``OptimizerTensorInfo.release``).

        Default: nothing. Only the CPU-offload optimizer (HybridDeviceOptimizer)
        holds such buffers; see :class:`HybridDeviceOptimizerAdapter`.
        """
        return


class Float16OptimizerAdapter(OptimizerAdapter):
    """Non-distributed float16 optimizer: master lives in fp32_from_float16_groups."""

    def __init__(self, optimizer: MegatronOptimizer):
        super().__init__(optimizer)
        self._index_map: dict[torch.nn.Parameter, tuple[int, int]] | None = None

    def get_main_weight(
        self, model_weight: torch.nn.Parameter
    ) -> torch.Tensor | None:
        if self._index_map is None:
            index_map: dict[torch.nn.Parameter, tuple[int, int]] = {}
            for i, group in enumerate(self.optimizer.float16_groups):
                for j, param in enumerate(group):
                    index_map[param] = (i, j)
            self._index_map = index_map
        i, j = self._index_map.get(model_weight, (-1, -1))
        return self.optimizer.fp32_from_float16_groups[i][j]


class DistributedOptimizerAdapter(OptimizerAdapter):
    """Standard DistributedOptimizer (no precision-aware): generic Adam discovery."""

    def get_main_weight(
        self, model_weight: torch.nn.Parameter
    ) -> torch.Tensor | None:
        if model_weight in self.optimizer.model_param_group_index_map:
            group_index, group_order = self.optimizer.model_param_group_index_map[
                model_weight
            ]
            return self.optimizer.optimizer.param_groups[group_index]["params"][
                group_order
            ]
        return None

    def model_param_sub_range(
        self, model_weight: torch.nn.Parameter, main_weight: torch.Tensor
    ) -> ParamRange:
        sub_range = self.optimizer._get_model_param_range_map(model_weight)["param"]
        # Convert megatron.core.optimizer.distrib_optimizer.Range to our ParamRange.
        sub_range_in_model_param = ParamRange(
            ranges=[Range(sub_range.start, sub_range.end)]
        )
        assert sub_range_in_model_param.size == main_weight.numel()
        return sub_range_in_model_param


class PrecisionAwareOptimizerAdapter(DistributedOptimizerAdapter):
    """DistributedOptimizer with --use-precision-aware-optimizer.

    The fp32 master + (unscaled) moments live in optimizer state and must be read via
    Megatron's accessor; ``param_groups`` only holds the bf16/fp16 model shard. The
    master->model copy is also not done by _copy_main_params_to_model_params (a no-op
    here — normally done inside optimizer.step()), so we add the explicit refill.
    """

    def discover_states(
        self, model_weight: torch.nn.Parameter, main_weight: torch.Tensor
    ) -> list[OptState]:
        # Reading param_groups directly (the base get_main_weight) yields the bf16
        # shard of a FusedAdam-managed buffer and crashes the transfer with an async
        # "CUDA error: invalid argument"; the accessor returns the real state tensors
        # (references), so the in-place reshard recv updates the optimizer.
        # NOTE: verified-by-design for the HDO branch of _get_main_param_and_optimizer_
        # states; non-HDO precision-aware returns unscaled COPIES (set_scaled_state
        # needed on write-back) and is not exercised here.
        td = self.optimizer._get_main_param_and_optimizer_states(model_weight)
        master = td["param"]
        anchor_numel = master.numel()
        states = [OptState("main_weight", master)]
        for key in ("exp_avg", "exp_avg_sq"):
            if key in td:
                states.append(OptState(key, td[key]))
        for key, val in td.items():
            if key in ("param", "exp_avg", "exp_avg_sq"):
                continue
            if torch.is_tensor(val) and val.numel() == anchor_numel:
                states.append(OptState(key, val))
        return states

    def copy_main_to_model(self) -> None:
        # Base call is a no-op under precision-aware; follow with the explicit refill.
        super().copy_main_to_model()
        _precision_aware_copy_main_to_model(self.optimizer)


class HybridDeviceOptimizerAdapter(PrecisionAwareOptimizerAdapter):
    """HybridDeviceOptimizer (CPU+GPU offload), wrapped by a precision-aware DO.

    Its ``.state`` is a view synced from device-specific sub-optimizers and
    ``init_state_fn`` is None, so empty placeholders would not connect to the real
    sub-optimizer tensors. ``dummy_step()`` allocates the real state on each
    sub-optimizer's own device (CPU for offloaded params, GPU otherwise) and syncs it
    into ``.state``. On the dst this is safe: both the state values and the dummy-grad
    param perturbation are overwritten by the reshard transfer. The CPU-resident
    moments then ride the device-aware transport in communicator.py. We offload only
    the master here; the moments stay allocated and are received in place.
    NOTE: meta-device HDO build is not handled (dummy_step needs real storage).
    """

    def ensure_state_initialized(
        self,
        model_weight: torch.nn.Parameter,
        main_weight: torch.Tensor,
        offload: bool,
    ) -> None:
        if len(self.optimizer.optimizer.state[main_weight]) != 0:
            return
        self.optimizer.optimizer.dummy_step()
        if offload:
            main_weight.storage().resize_(0)

    def release_offload_host_buffers(self) -> None:
        """Free the HDO's pinned CPU grad buffers (``cpu_copy_map_grad``).

        These pinned fp32 grad buffers (one per offloaded param, ~one master's
        worth per rank) are allocated lazily on the HDO's first step and held for
        the life of the HDO. EM's optimizer-state release only resize_(0)'s the
        param-shaped master + moments (POOL A) and never reaches these (POOL B), so
        a strategy cached across reshards keeps its pinned grad buffers resident ->
        N cached strategies hold N x this pinned block -> host OOM on long
        multi-strategy cpu-adam runs.

        Called on the *becoming-dormant* (src) gear during release_optimizer. The
        HDO re-creates these lazily on the gear's next step (the
        ``if param not in self.cpu_copy_map_grad`` path in
        ``_set_sub_optimizer_grads``), so reshard-back is unaffected: POOL A is
        refilled by rebuild()+transfer, POOL B by that lazy step path.
        """
        hdo = self.optimizer.optimizer
        for param, grad in hdo.cpu_copy_map_grad.items():
            # The owning sub-optimizer param holds .grad referencing this buffer; drop
            # that reference, then resize_(0) frees the pinned block in place (returned
            # to torch's pinned caching pool for the next gear to reuse). resize_(0),
            # rather than relying on GC, keeps the free deterministic and is the same
            # idiom EM uses to free POOL A.
            if param.grad is grad:
                param.grad = None
            grad.untyped_storage().resize_(0)
        # Clear the map so the HDO's lazy path re-creates fresh pinned buffers on the
        # gear's next step (it keys on `param not in self.cpu_copy_map_grad`).
        hdo.cpu_copy_map_grad.clear()

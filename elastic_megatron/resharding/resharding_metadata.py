import re
from dataclasses import dataclass

import torch
from megatron.core.distributed import DistributedDataParallel
from megatron.core.optimizer import (
    ChainedOptimizer,
    DistributedOptimizer,
    MegatronOptimizer,
)
from megatron.training import get_args

from ..megatron_manager.parallel_strategy import ParallelStrategy
from .resharding_pp import LayerType, ParamPositionAttr
from .resharding_tp import TensorParallelAttr
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


@dataclass
class OptimizerTensorInfo:
    # Ordered, variable-length set of named states. ``states[0]`` is the master
    # weight and serves as the geometry anchor. Adam => [main_weight, exp_avg,
    # exp_avg_sq]; other optimizers may carry a different count/dtype per state.
    states: list[OptState]
    optimizer_tensor_range_in_model_param: (
        ParamRange  # sub-range of the param that this data-parallel rank owns.
    )
    model_param_range: ParamRange

    def __post_init__(self):
        assert len(self.states) >= 1, "need at least the master-weight state"
        # Working list; create_padded swaps this to padded buffers and back.
        self.optimizer_tensors = [s.tensor for s in self.states]
        self.state_names = [s.name for s in self.states]
        self.optimizer_tensor_shape = self.states[0].tensor.shape

        anchor_numel = self.optimizer_tensor_shape.numel()
        for s in self.states:
            assert s.tensor.numel() == anchor_numel, (
                f"optimizer state '{s.name}' is not param-shaped "
                f"({s.tensor.numel()} != {anchor_numel} elems). Non-param-shaped "
                "states (e.g. FP8 per-block scales) are unsupported; see I-15."
            )
        assert (
            self.optimizer_tensor_range_in_model_param.size
            <= self.model_param_range.size
        )
        assert anchor_numel == self.optimizer_tensor_range_in_model_param.size

    # --- read-only accessors; main_weight/exp_avg/exp_avg_sq are the real (never
    # padded) state tensors, kept for Adam-shaped call sites (e.g. ipc_manager). ---
    @property
    def main_weight(self) -> torch.Tensor:
        return self.states[0].tensor

    def state_tensor(self, name: str) -> torch.Tensor:
        return self.states[self.state_names.index(name)].tensor

    @property
    def exp_avg(self) -> torch.Tensor:
        return self.state_tensor("exp_avg")

    @property
    def exp_avg_sq(self) -> torch.Tensor:
        return self.state_tensor("exp_avg_sq")

    def update_optimizer_tensors(self, new_optimizer_tensors: list[torch.Tensor]):
        """Update meta-device optimizer tensors by new optimizer tensors."""
        assert self.states[0].tensor.device == torch.device("meta"), (
            "Only support update optimizer tensors on meta device"
        )
        assert len(new_optimizer_tensors) == len(self.optimizer_tensors)
        for old_tensor, new_tensor in zip(
            self.optimizer_tensors, new_optimizer_tensors
        ):
            assert old_tensor.shape == new_tensor.shape
            assert old_tensor.dtype == new_tensor.dtype
        for s, new_tensor in zip(self.states, new_optimizer_tensors):
            s.tensor = new_tensor
        self.optimizer_tensors = new_optimizer_tensors

    def rebuild(self):
        for tensor in self.optimizer_tensors:
            if tensor.storage().size() == 0:
                tensor.storage().resize_(self.optimizer_tensor_shape.numel())

    def release(self):
        for tensor in self.optimizer_tensors:
            tensor.storage().resize_(0)

    def create_padded_optimizer_tensor(self):
        assert self.optimizer_tensor_range_in_model_param.is_normalized, (
            "We assume the aligned_dp_rank is the first dp rank in the dp-distribution"
        )
        if (
            self.optimizer_tensor_range_in_model_param.size
            == self.model_param_range.size
        ):
            return

        model_param_shape: torch.Size = self.model_param_range.to_torch_size()

        # One padded buffer per state, each with that state's OWN dtype/device
        # (Adam: all match the master; this generalizes to per-state dtype and to
        # CPU-resident states without special-casing).
        self.padded_optimizer_tensors = [
            torch.empty(
                model_param_shape,
                dtype=t.dtype,
                device=t.device,
            ).view(-1)
            for t in self.optimizer_tensors
        ]
        self.origin_optimizer_tensors = self.optimizer_tensors
        self.optimizer_tensors = self.padded_optimizer_tensors

    def release_padded_optimizer_tensor(self):
        if not hasattr(self, "padded_optimizer_tensors"):
            return
        assert self.optimizer_tensor_range_in_model_param.is_normalized
        assert self.optimizer_tensors[0].storage().size() == 0
        self.optimizer_tensors = self.origin_optimizer_tensors
        del self.padded_optimizer_tensors

    def shuffle_swiglu(self, scale_up_ratio: int):
        """Shuffle origin W/V Tensor to multiple W/V Tensors.
        [W | V] -> [W1 | V1, W2 | V2, ..., W(scale_up_ratio) | V(scale_up_ratio)]
        """
        if scale_up_ratio <= 1:
            return

        optimizer_tensor_shape = self.optimizer_tensors[0].shape
        optimizer_tensor_numel = optimizer_tensor_shape.numel()
        assert optimizer_tensor_numel % 2 == 0
        origin_w_tensor_numel = optimizer_tensor_numel // 2

        assert origin_w_tensor_numel % scale_up_ratio == 0
        split_w_tensor_numel = origin_w_tensor_numel // scale_up_ratio

        for optimizer_tensor in self.optimizer_tensors:
            w_tensor, v_tensor = torch.chunk(optimizer_tensor.view(-1), 2, dim=0)
            w_tensor_splits = torch.chunk(w_tensor, scale_up_ratio, dim=0)
            v_tensor_splits = torch.chunk(v_tensor, scale_up_ratio, dim=0)

            new_optimizer_tensor = torch.empty(
                optimizer_tensor_shape,
                dtype=optimizer_tensor[0].dtype,
                device=optimizer_tensor[0].device,
            ).view(-1)

            start = 0
            for split_w, split_v in zip(w_tensor_splits, v_tensor_splits):
                new_optimizer_tensor[start : start + split_w_tensor_numel].data.copy_(
                    split_w
                )
                start += split_w_tensor_numel
                new_optimizer_tensor[start : start + split_w_tensor_numel].data.copy_(
                    split_v
                )
                start += split_w_tensor_numel

            assert start == optimizer_tensor_numel

            assert optimizer_tensor.shape == new_optimizer_tensor.shape
            optimizer_tensor.data.copy_(new_optimizer_tensor)

    def unshuffle_swiglu(self, scale_down_ratio: int):
        """Unshuffle multiple W/V Tensors to origin W/V Tensor.
        [W1 | V1, W2 | V2, ..., W(scale_down_ratio) | V(scale_down_ratio)] -> [W | V]
        """
        if scale_down_ratio <= 1:
            return

        optimizer_tensor_shape = self.optimizer_tensors[0].shape
        optimizer_tensor_numel = optimizer_tensor_shape.numel()
        assert optimizer_tensor_numel % 2 == 0
        origin_w_tensor_numel = optimizer_tensor_numel // 2

        assert origin_w_tensor_numel % scale_down_ratio == 0
        split_w_tensor_numel = origin_w_tensor_numel // scale_down_ratio

        for optimizer_tensor in self.optimizer_tensors:
            new_optimizer_tensor = torch.empty(
                optimizer_tensor_shape,
                dtype=optimizer_tensor[0].dtype,
                device=optimizer_tensor[0].device,
            ).view(-1)
            w_tensor, v_tensor = torch.chunk(new_optimizer_tensor.view(-1), 2, dim=0)
            w_tensor_splits = torch.chunk(w_tensor, scale_down_ratio, dim=0)
            v_tensor_splits = torch.chunk(v_tensor, scale_down_ratio, dim=0)

            start = 0
            for split_w, split_v in zip(w_tensor_splits, v_tensor_splits):
                split_w.data.copy_(
                    optimizer_tensor[start : start + split_w_tensor_numel]
                )
                start += split_w_tensor_numel
                split_v.data.copy_(
                    optimizer_tensor[start : start + split_w_tensor_numel]
                )
                start += split_w_tensor_numel

            assert start == optimizer_tensor_numel
            assert optimizer_tensor.shape == new_optimizer_tensor.shape
            optimizer_tensor.data.copy_(new_optimizer_tensor)


@dataclass
class ParamReshardingMetaData:
    tensor_parallel_attr: TensorParallelAttr
    param_position_attr: ParamPositionAttr
    optimizer_tensor_info: OptimizerTensorInfo


MODEL_PARAM_TO_OPT_PARAM_INDEX = None


def set_model_to_optimizer_index_dict(
    model_to_optimizer_index_dict: dict[torch.nn.Parameter, tuple[int, int]],
):
    global MODEL_PARAM_TO_OPT_PARAM_INDEX
    MODEL_PARAM_TO_OPT_PARAM_INDEX = model_to_optimizer_index_dict


def get_model_to_optimizer_index_dict():
    global MODEL_PARAM_TO_OPT_PARAM_INDEX
    return MODEL_PARAM_TO_OPT_PARAM_INDEX


def init_model_to_optimizer_index_dict():
    """Empty MODEL_PARAM_TO_OPT_PARAM_INDEX at the beginning of the generate_model_and_optimizer_metadata()."""
    set_model_to_optimizer_index_dict({})


def get_main_weight(
    optimizer: MegatronOptimizer,
    model_weight: torch.nn.Parameter,
) -> torch.Tensor | None:
    if isinstance(optimizer, DistributedOptimizer):
        if model_weight in optimizer.model_param_group_index_map:
            group_index, group_order = optimizer.model_param_group_index_map[
                model_weight
            ]
            main_weight = optimizer.optimizer.param_groups[group_index]["params"][
                group_order
            ]
            return main_weight
        return None

    model_to_optimizer_index_dict: dict[torch.nn.Parameter, tuple[int, int]] = (
        get_model_to_optimizer_index_dict()
    )
    if not model_to_optimizer_index_dict:
        for i, group in enumerate(optimizer.float16_groups):
            for j, param in enumerate(group):
                model_to_optimizer_index_dict[param] = (i, j)
        set_model_to_optimizer_index_dict(model_to_optimizer_index_dict)

    i, j = model_to_optimizer_index_dict.get(model_weight, (-1, -1))
    return optimizer.fp32_from_float16_groups[i][j]


def init_empty_state_dict(
    optimizer: MegatronOptimizer, main_weight: torch.nn.Parameter
):
    """Allocate empty (storage-0) placeholder states for the offload path.

    F1 default schema = Adam moments. The DST optimizer is freshly built so its
    state dict is empty; we pre-create the keys the SRC side will send so the
    positional transfer lines up, then resize their storage to 0. A non-Adam
    optimizer (e.g. Muon's ``momentum``) needs its own offload schema here; the
    already-initialized SRC path discovers keys generically and needs no change.
    """
    for key in _ADAM_STATE_KEYS:
        optimizer.state[main_weight][key] = torch.zeros_like(main_weight.data)
        optimizer.state[main_weight][key].storage().resize_(0)


def get_optimizer_tensors_by_model_weight(
    optimizer: MegatronOptimizer,
    model_weight: torch.nn.Parameter,
    offload_opt_tensors: bool,
) -> OptimizerTensorInfo | None:
    """Get OptimizerTensorInfo by model weight."""

    main_weight = get_main_weight(optimizer, model_weight)
    if main_weight is None:
        return None

    opt_tensor_shape = torch.Size(main_weight.shape)
    state_initialized = len(optimizer.optimizer.state[main_weight]) != 0

    # Init optimizer.state. If offload_opt_tensors is True, offload the optimizer tensors.
    if not state_initialized:
        if _is_hybrid_device_optimizer(optimizer.optimizer):
            # HDO's .state is a view synced from device-specific sub-optimizers, and
            # init_state_fn is None — writing init_empty_state_dict would not connect
            # to the real sub-optimizer tensors. dummy_step() allocates the real state
            # (exp_avg/exp_avg_sq) on each sub-optimizer's own device (CPU for offloaded
            # params, GPU otherwise) and syncs it into .state. On the dst this is safe:
            # both the state values and the dummy-grad param perturbation are overwritten
            # by the reshard transfer. The CPU-resident moments then ride the device-aware
            # transport in communicator.py. We offload only the master here; the moments
            # stay allocated and are received in place (rebuild() is a no-op for them).
            # NOTE: meta-device HDO build is not handled (dummy_step needs real storage).
            optimizer.optimizer.dummy_step()
            if offload_opt_tensors:
                main_weight.storage().resize_(0)
        elif not offload_opt_tensors:
            optimizer.init_state_fn(optimizer.optimizer)
        else:
            init_empty_state_dict(optimizer.optimizer, main_weight)
            main_weight.storage().resize_(0)

    # Get the sub-range of the optimizer tensor in the model parameter.
    if isinstance(optimizer, DistributedOptimizer):
        sub_range = optimizer._get_model_param_range_map(model_weight)["param"]
        # Convert this megatron.core.optimizer.distrib_optimizer.Range to ParamRange.
        sub_range_in_model_param = ParamRange(
            ranges=[Range(sub_range.start, sub_range.end)]
        )
        assert sub_range_in_model_param.size == opt_tensor_shape.numel()
    else:
        sub_range_in_model_param = ParamRange(param_shape=model_weight.shape)

    use_precision_aware = isinstance(optimizer, DistributedOptimizer) and getattr(
        getattr(optimizer, "config", None), "use_precision_aware_optimizer", False
    )

    if use_precision_aware:
        # Under --use-precision-aware-optimizer (required by HybridDeviceOptimizer),
        # optimizer.param_groups holds the bf16/fp16 model SHARD, not the fp32 master —
        # the master + (unscaled) moments live in optimizer state and must be read via
        # Megatron's own accessor, which returns {"param": fp32 master, "exp_avg",
        # "exp_avg_sq", ...}. For HDO those are the real state tensors (references), so
        # the in-place reshard recv updates the optimizer. (Reading param_groups directly
        # — as get_main_weight does — yields the bf16 shard of a FusedAdam-managed buffer
        # and crashes the transfer with an async "CUDA error: invalid argument".)
        # NOTE: only verified-by-design for the HDO branch of _get_main_param_and_
        # optimizer_states; non-HDO precision-aware returns unscaled COPIES (set_scaled_
        # state needed on write-back) and is not handled here.
        td = optimizer._get_main_param_and_optimizer_states(model_weight)
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
    else:
        # Discover states generically: master + every param-shaped state in the
        # optimizer's state dict, in canonical order. For Adam this yields exactly
        # [main_weight, exp_avg, exp_avg_sq] (a scalar `step`, if present, is dropped
        # by the param-shaped filter), so the Adam transfer is unchanged.
        state = optimizer.optimizer.state[main_weight]
        states = [OptState("main_weight", main_weight)]
        for key in ordered_optimizer_state_keys(state, opt_tensor_shape.numel()):
            states.append(OptState(key, state[key]))

    return OptimizerTensorInfo(
        states=states,
        optimizer_tensor_range_in_model_param=sub_range_in_model_param,
        model_param_range=ParamRange(param_shape=model_weight.shape),
    )


def get_optimizer_tensors(
    optimizer: MegatronOptimizer,
    model_weight: torch.nn.Parameter,
    offload_opt_tensors: bool,
    is_expert: bool,
) -> OptimizerTensorInfo | None:
    if isinstance(optimizer, ChainedOptimizer):
        optimizers = optimizer.chained_optimizers
    else:
        optimizers = [optimizer]

    for optimizer in optimizers:
        optimizer_tensor_info = get_optimizer_tensors_by_model_weight(
            optimizer, model_weight, offload_opt_tensors
        )
        if optimizer_tensor_info is not None:
            return optimizer_tensor_info
    return None


def get_tensor_parallel_attr(
    param: torch.nn.Parameter,
    tensor_model_parallel_size: int,
    is_expert: bool = False,
) -> TensorParallelAttr:
    tensor_model_parallel: bool = param.tensor_model_parallel
    if not tensor_model_parallel:
        return TensorParallelAttr(model_param=param)
    # MoE experts with TPE==1 are not actually TP-split, yet 0.16's TE
    # GroupedLinear sets `partition_dim` to an implementation-defined default
    # that differs between EP configs (EP>1 disables TE's parallel_mode, EP==1
    # keeps it). Force the attr to "unsharded" for this specific case so src
    # and dst metadata match during reshard. Dense params keep their real
    # partition_dim/stride even at TP=1, since TP>1 peers need that info.
    if is_expert and tensor_model_parallel_size == 1:
        return TensorParallelAttr(model_param=param, force_unsharded=True)

    return TensorParallelAttr(
        model_param=param,
        parallel_group_size=tensor_model_parallel_size,
        parallel_group_str="",
    )


def get_param_position_attr(
    module_name: str, parallel_strategy: ParallelStrategy, is_expert: bool
) -> ParamPositionAttr:
    """Get the ParamPositionAttr of this param.

    Args:
        module_name (str): The name of the module.
        parallel_strategy (ParallelStrategy): The parallel strategy of the model.

    Returns:
        ParamPositionAttr: The ParamPositionAttr of param.
    """

    non_transformer_layer_keys = [
        "word_embedding",
        "position_embedding",
        "output_layer",
        "final_layernorm",
    ]
    for key in non_transformer_layer_keys:
        if key in module_name:
            return ParamPositionAttr(
                layer_type=LayerType(key),
                layer_index=0,
                module_name=module_name,
            )

    args = get_args()
    is_swiglu_fc = False
    if args.swiglu and "linear_fc1" in module_name and "mlp" in module_name:
        is_swiglu_fc = True

    transformer_layer_pattern = re.compile(r".*?layers\.(\d+).*?")

    expert_pattern0 = re.compile(
        r".*?experts\.(\d+).*?"
    )  # module.module.decoder.layers.0.mlp.experts.local_experts.0.linear_fc1.weight
    expert_pattern1 = re.compile(
        r".*?weight(\d+)$"
    )  # module.decoder.layers.3.mlp.experts.linear_fc2.weight0

    local_transformer_layer_id = int(
        transformer_layer_pattern.match(module_name).group(1)
    )
    global_transformer_layer_id = (
        parallel_strategy.transformer_layer_id_offset + local_transformer_layer_id
    )

    if is_expert:
        match = expert_pattern0.match(module_name)
        if match is None:
            match = expert_pattern1.match(module_name)
        assert match, f"can't find expert id from {module_name=}"
        local_expert_id = int(match.group(1))
        global_expert_id = parallel_strategy.expert_id_offset + local_expert_id
    else:
        global_expert_id = None

    return ParamPositionAttr(
        layer_type=LayerType.TRANSFORMER_LAYER,
        transformer_layer_id=global_transformer_layer_id,
        module_name=module_name,
        is_swiglu_fc=is_swiglu_fc,
        expert_id=global_expert_id,
    )


def _is_expert_param(name: str, param: torch.nn.Parameter) -> bool:
    """Decide whether a parameter belongs to a MoE expert.

    The historical check ``not getattr(param, "allreduce", True)`` is unreliable
    on Megatron 0.16: TE's GroupedLinear sets ``allreduce`` based on whether
    ``expert_parallel`` is enabled (EP>1), so the same expert param flips from
    ``False`` at EP=2 to ``True`` at EP=1. The module-path check below stays
    consistent across EP sizes, which matters when we compare src and dst
    reshard metadata for the same logical parameter.
    """
    return ".experts." in name


def generate_optimizer_tensor_info(
    model: list[DistributedDataParallel],
    optimizer: MegatronOptimizer,
    offload_opt_tensors: bool = False,
) -> dict[torch.nn.Parameter, OptimizerTensorInfo]:
    """Generate optimizer tensor info for each parameter."""
    init_model_to_optimizer_index_dict()
    params_to_optimizer_tensor_info: dict[torch.nn.Parameter, OptimizerTensorInfo] = (
        dict()
    )
    for model_chunk in model:
        for name, param in model_chunk.named_parameters():
            if not param.requires_grad:
                assert param.data.nelement() == 0
                continue

            is_expert = _is_expert_param(name, param)

            optimizer_tensor_info: OptimizerTensorInfo = get_optimizer_tensors(
                optimizer, param, offload_opt_tensors, is_expert
            )
            params_to_optimizer_tensor_info[param] = optimizer_tensor_info

    return params_to_optimizer_tensor_info


def generate_resharding_metadata(
    model: list[DistributedDataParallel],
    optimizer: MegatronOptimizer,
    parallel_strategy: ParallelStrategy,
    offload_opt_tensors: bool = False,
) -> dict[torch.nn.Parameter, ParamReshardingMetaData]:
    """Generate resharding metadata.

    Args:
        model (List[DistributedDataParallel]): Megatron DDP model. FSDP-2 is not supported yet.
        optimizer (MegatronOptimizer | None): Megatron optimizer.
        parallel_strategy (ParallelStrategy): The parallel strategy of the model.
        offload_opt_tensors (bool): Whether to offload the optimizer tensors.

    Returns:
        params_to_resharding_metadata (Dict[torch.nn.Parameter, ParamReshardingMetaData]): The metadata of the model.
    """
    params_to_optimizer_tensor_info: dict[torch.nn.Parameter, OptimizerTensorInfo] = (
        generate_optimizer_tensor_info(model, optimizer, offload_opt_tensors)
    )

    params_to_resharding_metadata: dict[torch.nn.Parameter, ParamReshardingMetaData] = (
        dict()
    )

    cur_transformer_layer_id = -1
    layer_index = 0
    for model_chunk in model:
        for name, param in model_chunk.named_parameters():
            if not param.requires_grad:
                assert param.data.nelement() == 0
                continue

            is_expert = _is_expert_param(name, param)

            # Get the position attributes of the parameter.
            param_position_attr: ParamPositionAttr = get_param_position_attr(
                module_name=name,
                parallel_strategy=parallel_strategy,
                is_expert=is_expert,
            )

            # Set the layer index for the transformer layer.
            if param_position_attr.layer_type == "transformer_layer":
                if cur_transformer_layer_id != param_position_attr.transformer_layer_id:
                    cur_transformer_layer_id = param_position_attr.transformer_layer_id
                    layer_index = 0
                else:
                    layer_index += 1
                param_position_attr.layer_index = layer_index

            # Get the tensor parallel attributes of the parameter.
            tensor_parallel_attr: TensorParallelAttr = get_tensor_parallel_attr(
                param,
                parallel_strategy.get_tensor_model_parallel_size(is_expert),
                is_expert=is_expert,
            )

            # Get the optimizer tensor info of the parameter.
            optimizer_tensor_info: OptimizerTensorInfo = (
                params_to_optimizer_tensor_info[param]
            )

            resharding_metadata = ParamReshardingMetaData(
                tensor_parallel_attr=tensor_parallel_attr,
                param_position_attr=param_position_attr,
                optimizer_tensor_info=optimizer_tensor_info,
            )
            params_to_resharding_metadata[param] = resharding_metadata

    update_layer_index(params_to_resharding_metadata)
    return params_to_resharding_metadata


def update_layer_index(
    params_to_resharding_metadata: dict[torch.nn.Parameter, ParamReshardingMetaData],
):
    # Key = tuple(LayerType, transformer_layer_id)
    dense_layer_index_dict: dict[tuple[LayerType, int], int] = {}

    # Key = tuple(transformer_layer_id, expert_id)
    moe_layer_index_dict: dict[tuple[int, int], int] = {}

    # Set layer_index for dense_params
    for _, resharding_metadata in params_to_resharding_metadata.items():
        param_position_attr: ParamPositionAttr = resharding_metadata.param_position_attr

        if param_position_attr.is_expert:
            key = (
                param_position_attr.transformer_layer_id,
                param_position_attr.expert_id,
            )
            param_position_attr.expert_param_layer_id = moe_layer_index_dict.setdefault(
                key, 0
            )
            moe_layer_index_dict[key] += 1
        else:
            key = (
                param_position_attr.layer_type,
                param_position_attr.transformer_layer_id,
            )
            param_position_attr.layer_index = dense_layer_index_dict.setdefault(key, 0)
            dense_layer_index_dict[key] += 1

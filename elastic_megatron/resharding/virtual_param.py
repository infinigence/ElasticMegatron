from copy import deepcopy

import torch
from megatron.core.tensor_parallel import set_tensor_model_parallel_attributes
from megatron.training.global_vars import get_args

from ..megatron_manager.megatron_state import MegatronState
from ..megatron_manager.parallel_strategy import ParallelStrategy
from .resharding import ReshardPlan
from .resharding_dp import (
    DataParallelReshardingInfo,
    ExpertParallelReshardingInfo,
    get_params_dp_distribution,
)
from .resharding_metadata import (
    OptimizerTensorInfo,
    ParamReshardingMetaData,
    get_tensor_parallel_attr,
)
from .resharding_pp import LayerType, ParamPositionAttr, PipelineParallelReshardingInfo
from .resharding_tp import TensorParallelAttr, TensorParallelReshardingInfo
from .util import (
    ParamRange,
    Range,
)


class VirtualTensor:
    def __init__(
        self, shape: torch.Size | ParamRange, dtype: torch.dtype = torch.float32
    ) -> None:
        self.shape = shape
        self.dtype = dtype
        self._base = None

    def resize(self, shape: torch.Size | ParamRange) -> None:
        self.shape = shape

    def nelement(self) -> int:
        if isinstance(self.shape, ParamRange):
            return self.shape.size
        return self.shape.numel()

    def element_size(self) -> int:
        return self.dtype.itemsize

    @property
    def nbytes(self) -> int:
        return self.nelement() * self.element_size()


class VirtualParam:
    """Virtual Parameter in Global Parameter Space.

    It's ParamPositionAttr and param_range is built by assuming of "all sizes of parallel strategy are equal to 1".
    """

    def __init__(
        self,
        param_position_attr: ParamPositionAttr,
        tensor_parallel_attr: TensorParallelAttr = None,
        shape: torch.Size = None,
        tensor_model_parallel: bool = False,
        partition_dim: int = -1,
        partition_stride: int = 1,
    ):
        self.param_position_attr = param_position_attr
        self.requires_grad = True
        self.shared_embedding = param_position_attr.shared_embedding

        # Attr of the parameter
        self.data = None
        if tensor_parallel_attr is None:
            self.data = VirtualTensor(shape=shape)
            set_tensor_model_parallel_attributes(
                self.data, tensor_model_parallel, partition_dim, partition_stride
            )
            tensor_parallel_attr = get_tensor_parallel_attr(
                param=self.data, tensor_model_parallel_size=1
            )
        else:
            self.data = VirtualTensor(
                shape=tensor_parallel_attr.get_model_param_range(1)
            )
        self.tensor_parallel_attr = tensor_parallel_attr

        self.is_swiglu_fc = (
            param_position_attr.is_swiglu_fc
            and self.tensor_parallel_attr.tensor_model_parallel
        )

        # Cache
        # _reshard_plan_cache 的 key 不含 self.is_expert / expert_is_dense_bucketed():
        # 缓存是 per-VirtualParam 的(self.is_expert 是常量),而
        # expert_is_dense_bucketed() 是 ParallelStrategy 的纯函数,已跟随
        # (src_strategy, dst_strategy) 一起进入 key。重构 key 时请保留这条假设。
        self._dp_distribution_cache: dict[
            tuple[ParallelStrategy, bool], dict[int, Range]
        ] = {}
        self._reshard_plan_cache: dict[
            tuple[ParallelStrategy, ParallelStrategy, bool], ReshardPlan
        ] = {}

        # Optimizer Info
        self.src_optimizer_tensor_info: OptimizerTensorInfo = None
        self.dst_optimizer_tensor_info: OptimizerTensorInfo = None

    @property
    def name(self) -> str:
        layer_info = str(self.param_position_attr.layer_type)
        if self.param_position_attr.layer_type == LayerType.TRANSFORMER_LAYER:
            layer_info = "transformer_layer=" + str(
                self.param_position_attr.transformer_layer_id
            )
        return f"global_index={self.param_position_attr.global_index} : {layer_info} : {self.param_position_attr.layer_index}"

    @property
    def is_expert(self) -> bool:
        return self.param_position_attr.expert_id is not None

    def set_optimizer_tensor_info(
        self, optimizer_tensor_info: OptimizerTensorInfo, is_src: bool
    ):
        if is_src:
            self.src_optimizer_tensor_info = optimizer_tensor_info
        else:
            self.dst_optimizer_tensor_info = optimizer_tensor_info

    def get_optimizer_tensor_info(self, is_src: bool) -> OptimizerTensorInfo:
        if is_src:
            return self.src_optimizer_tensor_info
        return self.dst_optimizer_tensor_info

    def numel(self) -> int:
        return self.data.nelement()

    def clear_optimizer_tensor_info(self):
        self.src_optimizer_tensor_info = None
        self.dst_optimizer_tensor_info = None

    def get_model_param_stage_id(self, pipeline_model_parallel_size: int) -> int:
        """Given a PP Size, get the stage id of the model weight."""
        return self.param_position_attr.get_model_param_stage_id(
            pipeline_model_parallel_size
        )

    def is_orphan_for(self, src_pp_size: int, dst_pp_size: int) -> bool:
        """True if this VP has no stage on either side of the reshard.

        Only the tied OUTPUT_LAYER under PP=1 satisfies this: it shares storage
        with WORD_EMBEDDING and its stage_id is -1 (see
        resharding_pp.py::_get_non_transformer_layer_stage_id). _build_stages_virtual_params
        already filters orphans out, so their _dp_distribution_cache is empty
        and apply_reshard_plan would trip get_dp_distribution's assert. The
        downstream transfer layer also skips them (transfer.py shared_embedding
        guard), so skipping plan generation here is a no-op for actual data flow.
        """
        return (
            self.get_model_param_stage_id(src_pp_size) == -1
            and self.get_model_param_stage_id(dst_pp_size) == -1
        )

    def set_model_param_range(
        self,
        parallel_strategy: ParallelStrategy,
    ):
        """Given a TP Size, set the param range of the model weight."""
        tensor_model_parallel_size = parallel_strategy.get_tensor_model_parallel_size(
            self.is_expert
        )
        self.data.resize(
            self.tensor_parallel_attr.get_model_param_range(tensor_model_parallel_size)
        )

    def get_dp_distribution(
        self,
        parallel_strategy: ParallelStrategy,
        with_ddp: bool,
        check_initialized: bool = False,
    ) -> dict[int, Range]:
        """Given a TP+PP+DP, get the DP distribution of the parameter."""
        dp_distribution = self._dp_distribution_cache.get((parallel_strategy, with_ddp))
        if not check_initialized:
            assert dp_distribution is not None, (
                f"parallel_strategy={parallel_strategy}, with_ddp={with_ddp}, {self}"
            )
        return dp_distribution

    def set_dp_distribution(
        self,
        parallel_strategy: ParallelStrategy,
        with_ddp: bool,
        dp_distribution: dict[int, Range],
    ):
        """Given a TP+PP+DP, set the DP distribution of the parameter."""
        self._dp_distribution_cache[(parallel_strategy, with_ddp)] = dp_distribution

    def apply_reshard_plan(
        self,
        src_parallel_strategy: ParallelStrategy,
        dst_parallel_strategy: ParallelStrategy,
        with_ddp: bool,
    ):
        key = (src_parallel_strategy, dst_parallel_strategy, with_ddp)
        if self._reshard_plan_cache.get(key) is not None:
            self.current_reshard_plan = self._reshard_plan_cache[key]
            return
        reshard_plan: ReshardPlan = self._generate_reshard_plan(
            src_parallel_strategy, dst_parallel_strategy, with_ddp
        )
        self._reshard_plan_cache[key] = reshard_plan
        self.current_reshard_plan = reshard_plan

    @property
    def reshard_plan(self) -> ReshardPlan:
        return self.current_reshard_plan

    def _generate_reshard_plan(
        self,
        src_parallel_strategy: ParallelStrategy,
        dst_parallel_strategy: ParallelStrategy,
        with_ddp: bool,
    ) -> ReshardPlan:
        """Reshard the parameter from src_parallel_strategy to dst_parallel_strategy."""

        # Step-1 : TP / TPEP
        # When EP=1, Megatron replicates expert params across the dense TP ranks,
        # so the effective TP size for planning must be the dense tp_size rather
        # than TPE. Without this, recv ranks at tp_rank>0 get no broadcast entry
        # and the transfer plan has holes.
        def _effective_tp_size(ps: ParallelStrategy) -> int:
            if self.is_expert and ps.expert_is_dense_bucketed():
                return ps.get_tensor_model_parallel_size(is_expert=False)
            return ps.get_tensor_model_parallel_size(self.is_expert)

        src_tensor_model_parallel_size = _effective_tp_size(src_parallel_strategy)
        dst_tensor_model_parallel_size = _effective_tp_size(dst_parallel_strategy)
        tensor_parallel_resharding_info: TensorParallelReshardingInfo = (
            self.tensor_parallel_attr.reshard(
                src_tensor_model_parallel_size, dst_tensor_model_parallel_size
            )
        )

        # Step-2 : PP
        src_pipeline_model_parallel_size = (
            src_parallel_strategy.pipeline_model_parallel_size
        )
        dst_pipeline_model_parallel_size = (
            dst_parallel_strategy.pipeline_model_parallel_size
        )
        pipeline_parallel_resharding_info: PipelineParallelReshardingInfo = (
            self.param_position_attr.reshard(
                src_pipeline_model_parallel_size, dst_pipeline_model_parallel_size
            )
        )

        # Step-3 : EP
        expert_parallel_resharding_info = None
        if self.is_expert:
            expert_parallel_resharding_info = ExpertParallelReshardingInfo(
                self.param_position_attr.expert_id,
                src_parallel_strategy,
                dst_parallel_strategy,
            )

        # Step-4 : DP
        data_parallel_resharding_info: DataParallelReshardingInfo = (
            DataParallelReshardingInfo(
                src_dp_distribution=self.get_dp_distribution(
                    src_parallel_strategy, with_ddp
                ),
                dst_dp_distribution=self.get_dp_distribution(
                    dst_parallel_strategy, with_ddp
                ),
                src_parallel_strategy=src_parallel_strategy,
                dst_parallel_strategy=dst_parallel_strategy,
                pipeline_parallel_resharding_info=pipeline_parallel_resharding_info,
                expert_parallel_resharding_info=expert_parallel_resharding_info,
            )
        )

        return ReshardPlan(
            tensor_parallel_resharding_info=tensor_parallel_resharding_info,
            pipeline_parallel_resharding_info=pipeline_parallel_resharding_info,
            data_parallel_resharding_info=data_parallel_resharding_info,
            expert_parallel_resharding_info=expert_parallel_resharding_info,
            src_parallel_strategy=src_parallel_strategy,
            dst_parallel_strategy=dst_parallel_strategy,
        )


class VirtualParamSpace:
    """Global Virtual Parameter Space. (Virutal Model of all parallel sizes equal to 1)."""

    def __init__(
        self,
        embedding_layer_virtual_params: list[VirtualParam],
        transformer_layers_virtual_params: list[list[VirtualParam]],
        output_layer_virtual_params: list[VirtualParam],
    ):
        self.embedding_layer_virtual_params: list[VirtualParam] = (
            embedding_layer_virtual_params
        )
        self.transformer_layers_virtual_params: list[list[VirtualParam]] = (
            transformer_layers_virtual_params
        )
        self.output_layer_virtual_params: list[VirtualParam] = (
            output_layer_virtual_params
        )

        self.all_virtual_params: list[VirtualParam] = []
        self.all_virtual_params += self.embedding_layer_virtual_params
        for virtual_params in self.transformer_layers_virtual_params:
            self.all_virtual_params += virtual_params
        self.all_virtual_params += self.output_layer_virtual_params

        assert all(
            param.param_position_attr.global_index == i
            for i, param in enumerate(self.all_virtual_params)
        ), "Global index of virtual params is not correct"

        # Cache
        self._pp_to_stages_virtual_params: dict[int, list[list[VirtualParam]]] = {}

    def _build_stages_virtual_params(
        self, pipeline_model_parallel_size: int
    ) -> list[list[VirtualParam]]:
        if pipeline_model_parallel_size in self._pp_to_stages_virtual_params:
            return self._pp_to_stages_virtual_params[pipeline_model_parallel_size]

        stages_virtual_params: list[list[VirtualParam]] = [
            None
        ] * pipeline_model_parallel_size
        for virtual_param in self.all_virtual_params:
            stage_id = virtual_param.get_model_param_stage_id(
                pipeline_model_parallel_size
            )
            if stage_id != -1:
                if stages_virtual_params[stage_id] is None:
                    stages_virtual_params[stage_id] = []
                stages_virtual_params[stage_id].append(virtual_param)

        self._pp_to_stages_virtual_params[pipeline_model_parallel_size] = (
            stages_virtual_params
        )
        return stages_virtual_params

    def is_virtual_space_initlized(self, parallel_strategy: ParallelStrategy) -> bool:
        first_param_dp_distribution = self.all_virtual_params[0].get_dp_distribution(
            parallel_strategy, with_ddp=True, check_initialized=True
        )
        return first_param_dp_distribution is not None

    def build_model_virtual_params(
        self, parallel_strategy: ParallelStrategy, ddp_config
    ):
        """Given a parallel_strategy, build the model virtual params, and get param's DP distribution.

        1. According PP-Size, split all_virtual_params into stages
        2. Set virtual_param range
            2.1 According TP-Size, set Dense virtual_param range
            2.2 According TPE-Size, set MoE virtual_param range
        3. Get param's DP distribution.
            3.1 According DP(CP, Group-zero)-Size, get the Dense virtual_param's DP distribution.
            3.2 According DP-EP-Size, get the MoE virtual_param's DP distribution.
        """
        args = get_args()
        with_ddp = args.use_distributed_optimizer
        assert with_ddp, "non-zero DP is not supported yet"
        if self.is_virtual_space_initlized(parallel_strategy):
            return

        # Get stages virtual params by PP
        stages_virtual_params: list[list[VirtualParam]] = (
            self._build_stages_virtual_params(
                parallel_strategy.pipeline_model_parallel_size
            )
        )

        # Set model param range by TP or TPEP
        for virtual_param in self.all_virtual_params:
            virtual_param.set_model_param_range(
                parallel_strategy,
            )

        # Get DP distribution
        for model_chunk_idx, stage_virtual_params in enumerate(stages_virtual_params):
            overlap_param_gather_with_optimizer_step = getattr(
                args, "overlap_param_gather_with_optimizer_step", False
            )
            disable_bucketing = (
                model_chunk_idx > 0
            ) or overlap_param_gather_with_optimizer_step
            stage_dp_distribution: dict[VirtualParam, dict[int, Range]] = (
                get_params_dp_distribution(
                    params=stage_virtual_params,
                    parallel_strategy=parallel_strategy,
                    disable_bucketing=disable_bucketing,
                    ddp_config=ddp_config,
                )
            )
            for virtual_param in stage_virtual_params:
                virtual_param.set_dp_distribution(
                    parallel_strategy, with_ddp, stage_dp_distribution[virtual_param]
                )

    def register_reshard(
        self,
        src_megatron_state: MegatronState,
        dst_megatron_state: MegatronState,
        with_ddp: bool,
    ):
        if src_megatron_state.training_state is not None:
            ddp_config = src_megatron_state.training_state.model[0].ddp_config
        elif dst_megatron_state.training_state is not None:
            ddp_config = dst_megatron_state.training_state.model[0].ddp_config

        self.build_model_virtual_params(
            src_megatron_state.parallel_strategy, ddp_config
        )
        self.build_model_virtual_params(
            dst_megatron_state.parallel_strategy, ddp_config
        )
        src_pp_size = src_megatron_state.parallel_strategy.pipeline_model_parallel_size
        dst_pp_size = dst_megatron_state.parallel_strategy.pipeline_model_parallel_size
        for virtual_param in self.all_virtual_params:
            virtual_param.clear_optimizer_tensor_info()
            # Skip orphan tied OUTPUT_LAYER (only appears for tied+PP=1 on both
            # sides). It has no dp_distribution and no parameter to transfer;
            # see VirtualParam.is_orphan_for and transfer.py shared_embedding guard.
            if virtual_param.is_orphan_for(src_pp_size, dst_pp_size):
                continue
            virtual_param.apply_reshard_plan(
                src_megatron_state.parallel_strategy,
                dst_megatron_state.parallel_strategy,
                with_ddp,
            )

        if src_megatron_state.training_state is not None:
            self._register_metadata(
                src_megatron_state.training_state.get_metadata(), is_src=True
            )

        if dst_megatron_state.training_state is not None:
            self._register_metadata(
                dst_megatron_state.training_state.get_metadata(), is_src=False
            )

    def _register_metadata(
        self,
        params_to_resharding_metadata: dict[
            torch.nn.Parameter, ParamReshardingMetaData
        ],
        is_src: bool,
    ):
        def find_non_transformer_layer_param(param_position_attr) -> VirtualParam:
            if param_position_attr.layer_type == LayerType.WORD_EMBEDDING:
                return self.embedding_layer_virtual_params[0]
            elif param_position_attr.layer_type == LayerType.POSITION_EMBEDDING:
                return self.embedding_layer_virtual_params[1]
            elif param_position_attr.layer_type == LayerType.FINAL_LAYERNORM:
                return self.output_layer_virtual_params[param_position_attr.layer_index]
            elif param_position_attr.layer_type == LayerType.OUTPUT_LAYER:
                return self.output_layer_virtual_params[-1]
            return None

        def find_transformer_layer_param(param_position_attr) -> VirtualParam:
            transfer_layer_params = self.transformer_layers_virtual_params[
                param_position_attr.transformer_layer_id
            ]

            if param_position_attr.is_expert:
                for param in transfer_layer_params:
                    if (
                        param.param_position_attr.expert_id
                        == param_position_attr.expert_id
                    ) and (
                        param.param_position_attr.expert_param_layer_id
                        == param_position_attr.expert_param_layer_id
                    ):
                        return param
                raise ValueError(f"{param_position_attr=} can't find virtual param")

            return transfer_layer_params[param_position_attr.layer_index]

        for param, resharding_metadata in params_to_resharding_metadata.items():
            param_position_attr: ParamPositionAttr = (
                resharding_metadata.param_position_attr
            )

            virtual_param = find_non_transformer_layer_param(param_position_attr)
            if virtual_param is None:
                virtual_param = find_transformer_layer_param(param_position_attr)

            assert (
                virtual_param.param_position_attr.layer_type
                == param_position_attr.layer_type
            )
            assert (
                virtual_param.tensor_parallel_attr
                == resharding_metadata.tensor_parallel_attr
            )
            virtual_param.set_optimizer_tensor_info(
                resharding_metadata.optimizer_tensor_info, is_src
            )


def get_embedding_layer_virtual_params(
    global_param_offset: int = 0,
) -> list[VirtualParam]:
    """Get embedding layer virtual params.

    According to the definitions of these parameters, calculate the torch.Size of Params when the TP Size is 1,

    Note.
    1. The shape of word_embedding is relevant to the padded_vocab_size, which is related to the TP Size and args.make_vocab_size_divisible_by.
    Therefore, when the size of TP is changing, the value of padded_vocab_size needs to remain unchanged.
    2. Set share_embeddings_and_output_weights attribute for word_embedding params.

    Returns:
        embedding_layer_virtual_params (list[VirtualParam]): word_embedding and position_embedding
    """
    embedding_layer_virtual_params: list[VirtualParam] = []
    args = get_args()
    share_embeddings_and_output_weights = not args.untie_embeddings_and_output_weights
    padded_vocab_size = args.padded_vocab_size
    hidden_size = args.hidden_size

    # Step1 : word_embedding
    word_embeddings_param_shape = torch.Size([padded_vocab_size, hidden_size])
    word_embeddings_virtual_param = VirtualParam(
        param_position_attr=ParamPositionAttr(
            global_index=global_param_offset,
            layer_type=LayerType.WORD_EMBEDDING,
            layer_index=0,
            shared_embedding=share_embeddings_and_output_weights,
        ),
        shape=word_embeddings_param_shape,
        tensor_model_parallel=True,
        partition_dim=0,
        partition_stride=1,
    )
    embedding_layer_virtual_params.append(word_embeddings_virtual_param)

    # Step2 : position_embedding
    if args.position_embedding_type == "learned_absolute":
        position_embeddings_param_shape = torch.Size(
            [args.max_position_embeddings, hidden_size]
        )
        position_embeddings_virtual_param = VirtualParam(
            param_position_attr=ParamPositionAttr(
                global_index=global_param_offset + 1,
                layer_type=LayerType.POSITION_EMBEDDING,
                layer_index=1,
            ),
            shape=position_embeddings_param_shape,
            tensor_model_parallel=False,
            partition_dim=-1,
            partition_stride=1,
        )
        embedding_layer_virtual_params.append(position_embeddings_virtual_param)
    return embedding_layer_virtual_params


def get_output_layer_virtual_params(global_param_offset: int) -> list[VirtualParam]:
    """Get output layer virtual params.

    According to the definitions of these parameters, calculate the torch.Size of Params when the TP Size is 1,

    Note.
    1. The shape of output_layer is relevant to the padded_vocab_size, which is related to the TP Size and args.make_vocab_size_divisible_by.
    Therefore, when the size of TP is changing, the value of padded_vocab_size needs to remain unchanged.
    2. Set share_embeddings_and_output_weights attribute for output_layer params.

    Returns:
        output_layer_virtual_params (list[VirtualParam]): final_layernorm and output_layer
    """
    output_layer_virtual_params: list[VirtualParam] = []
    args = get_args()
    share_embeddings_and_output_weights = not args.untie_embeddings_and_output_weights
    padded_vocab_size = args.padded_vocab_size
    hidden_size = args.hidden_size

    # Step1 : final_layernorm weight
    out_layer_param_index = 0
    final_layernorm_param_shape = torch.Size([hidden_size])
    final_layernorm_weight_virtual_param = VirtualParam(
        param_position_attr=ParamPositionAttr(
            global_index=global_param_offset + out_layer_param_index,
            layer_type=LayerType.FINAL_LAYERNORM,
            layer_index=out_layer_param_index,
        ),
        shape=final_layernorm_param_shape,
        tensor_model_parallel=False,
        partition_dim=-1,
        partition_stride=1,
    )
    output_layer_virtual_params.append(final_layernorm_weight_virtual_param)
    out_layer_param_index += 1

    # Step2 : final_layernorm bias
    if args.normalization != "RMSNorm":
        final_layernorm_bias_virtual_param = VirtualParam(
            param_position_attr=ParamPositionAttr(
                global_index=global_param_offset + out_layer_param_index,
                layer_type=LayerType.FINAL_LAYERNORM,
                layer_index=out_layer_param_index,
            ),
            shape=final_layernorm_param_shape,
            tensor_model_parallel=False,
            partition_dim=-1,
            partition_stride=1,
        )
        output_layer_virtual_params.append(final_layernorm_bias_virtual_param)
        out_layer_param_index += 1

    # Step3 : output layer
    output_layer_param_shape = torch.Size([padded_vocab_size, hidden_size])
    output_layer_virtual_param = VirtualParam(
        param_position_attr=ParamPositionAttr(
            global_index=global_param_offset + out_layer_param_index,
            layer_type=LayerType.OUTPUT_LAYER,
            layer_index=out_layer_param_index,
            shared_embedding=share_embeddings_and_output_weights,
        ),
        shape=output_layer_param_shape,
        tensor_model_parallel=True,
        partition_dim=0,
        partition_stride=1,
    )
    output_layer_virtual_params.append(output_layer_virtual_param)

    return output_layer_virtual_params


def get_single_transformer_layer_moe_params(
    moe_params: list[VirtualParam],
) -> list[VirtualParam]:
    """Get single transformer layer moe params.

    Args:
        moe_params (list[VirtualParam]): moe params of one transformer layer in Parallel strategy.

    Returns:
        single_transformer_layer_moe_param (list[VirtualParam]): moe params of one transformer layer in Virtual strategy(EP=1)

    There are two kinds of order of moe params:
    order case-0 (expert_id_order):
        expert.0.fc1
        expert.0.fc2
        ...
        expert.N.fc1
        expert.N.fc2

    order case-1 (grouped_gemm_order):
        expert.0.fc1
        ...
        expert.N.fc1
        expert.0.fc2
        ...
        expert.N.fc2
    """

    args = get_args()
    if not moe_params:
        return []

    moe_params_order_type = ["expert_id_order", "grouped_gemm_order"]
    first_expert_id = moe_params[0].param_position_attr.expert_id
    last_expert_id = moe_params[-1].param_position_attr.expert_id

    # Determine the order of moe params
    if last_expert_id == 0:
        order = (
            moe_params_order_type[1]
            if args.moe_grouped_gemm
            else moe_params_order_type[0]
        )
    else:
        if first_expert_id == moe_params[1].param_position_attr.expert_id:
            order = moe_params_order_type[0]
        else:
            order = moe_params_order_type[1]

    # Generate expert params
    single_expert_params = [
        param
        for param in moe_params
        if param.param_position_attr.expert_id == first_expert_id
    ]
    expert_to_params: dict[int, list[VirtualParam]] = {}

    for expert_id in range(args.num_experts):
        expert_params = deepcopy(single_expert_params)
        expert_param_layer_id = 0
        for expert_param in expert_params:
            expert_param.param_position_attr.expert_id = expert_id
            expert_param.param_position_attr.expert_param_layer_id = (
                expert_param_layer_id
            )
            expert_param_layer_id += 1
        expert_to_params[expert_id] = expert_params

    # Generate moe params with EP=1 strategy
    # expert_id_order
    if order == moe_params_order_type[0]:
        single_transformer_layer_moe_param: list[VirtualParam] = []
        for expert_params in expert_to_params.values():
            single_transformer_layer_moe_param += expert_params
        return single_transformer_layer_moe_param

    # grouped_gemm_order
    expert_params_len = len(single_expert_params)
    single_transformer_layer_moe_param: list[VirtualParam] = (
        [None] * expert_params_len * args.num_experts
    )
    for expert_id, expert_params in expert_to_params.items():
        for expert_param_layer_id, expert_param in enumerate(expert_params):
            index = expert_param_layer_id * args.num_experts + expert_id
            single_transformer_layer_moe_param[index] = expert_param

    return single_transformer_layer_moe_param


def get_single_transformer_layer_virtual_params(
    params_to_resharding_metadata: dict[torch.nn.Parameter, ParamReshardingMetaData],
) -> list[VirtualParam]:
    """Get single transformer layer virtual params.

    Note.
    1. There is at least one transformer layer on each rank.
    2. The params of each transformer layer are the same.
    3. The final_layernorm of the transformer block is put in the output_layer.
    4. In MoE model, return value

    Returns:
        single_transformer_layer_params (list[VirtualParam]) : virtual params in single transformer layer.

    """

    dense_params: list[VirtualParam] = []
    moe_params: list[VirtualParam] = []

    first_transformer_layer_id = -1

    for param, resharding_metadata in params_to_resharding_metadata.items():
        param_position_attr: ParamPositionAttr = resharding_metadata.param_position_attr
        tensor_parallel_attr: TensorParallelAttr = (
            resharding_metadata.tensor_parallel_attr
        )

        if param_position_attr.layer_type != LayerType.TRANSFORMER_LAYER:
            continue

        transformer_layer_id = param_position_attr.transformer_layer_id
        if first_transformer_layer_id == -1:
            first_transformer_layer_id = transformer_layer_id
        elif first_transformer_layer_id != transformer_layer_id:
            break

        Virtual_param = VirtualParam(
            param_position_attr=param_position_attr,
            tensor_parallel_attr=tensor_parallel_attr,
        )

        if param_position_attr.is_expert:
            moe_params.append(Virtual_param)
        else:
            dense_params.append(Virtual_param)

    assert len(dense_params) > 0

    single_transformer_layer_params = (
        dense_params + get_single_transformer_layer_moe_params(moe_params)
    )
    return single_transformer_layer_params


def get_transformer_layers_virtual_params(
    params_to_resharding_metadata: dict[torch.nn.Parameter, ParamReshardingMetaData],
    global_param_offset: int,
) -> list[list[VirtualParam]]:
    args = get_args()

    single_transformer_layer_params = get_single_transformer_layer_virtual_params(
        params_to_resharding_metadata
    )

    transformer_layers_params: list[VirtualParam] = []
    global_param_index = global_param_offset

    for transformer_layer_id in range(args.num_layers):
        transformer_layer_params: list[VirtualParam] = deepcopy(
            single_transformer_layer_params
        )
        for param in transformer_layer_params:
            param.param_position_attr.global_index = global_param_index
            param.param_position_attr.transformer_layer_id = transformer_layer_id
            global_param_index += 1

        transformer_layers_params.append(transformer_layer_params)
    return transformer_layers_params


def build_virtual_model(
    params_to_resharding_metadata: dict[torch.nn.Parameter, ParamReshardingMetaData],
) -> VirtualParamSpace:
    """Build the virtual parameter space."""
    # Step1: Generate embedding layer virtual params
    global_param_offset = 0
    embedding_layer_virtual_params = get_embedding_layer_virtual_params(
        global_param_offset=global_param_offset
    )
    global_param_offset += len(embedding_layer_virtual_params)

    # Step2: Generate transformer layers virtual params
    transformer_layers_virtual_params: list[list[VirtualParam]] = (
        get_transformer_layers_virtual_params(
            params_to_resharding_metadata,
            global_param_offset=global_param_offset,
        )
    )
    global_param_offset += len(transformer_layers_virtual_params) * len(
        transformer_layers_virtual_params[0]
    )

    # Step3: Generate output layer virtual params
    output_layer_virtual_params = get_output_layer_virtual_params(
        global_param_offset=global_param_offset
    )

    # Step4: Generate virtual param space
    virtual_param_space = VirtualParamSpace(
        embedding_layer_virtual_params=embedding_layer_virtual_params,
        transformer_layers_virtual_params=transformer_layers_virtual_params,
        output_layer_virtual_params=output_layer_virtual_params,
    )
    return virtual_param_space

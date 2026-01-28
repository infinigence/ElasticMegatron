from typing import Tuple, Dict
from megatron.training.global_vars import get_args
from enum import Enum
from dataclasses import dataclass


@dataclass
class PipelineParallelReshardingInfo:
    src_pp_rank: int
    dst_pp_rank: int


class LayerType(Enum):
    WORD_EMBEDDING = "word_embedding"
    POSITION_EMBEDDING = "position_embedding"
    TRANSFORMER_LAYER = "transformer_layer"
    FINAL_LAYERNORM = "final_layernorm"
    OUTPUT_LAYER = "output_layer"


class ParamPositionAttr:
    """Position attributes of a parameter in the global parameter space."""

    def __init__(
        self,
        layer_type: LayerType,
        module_name: str = "",
        global_index: int = -1,
        transformer_layer_id: int = -1,
        layer_index: int = -1,
        is_swiglu_fc: bool = False,
        shared_embedding: bool = False,
        expert_id: int = None,
        expert_param_layer_id: int = -1,
    ):
        self.layer_type = layer_type
        self.module_name = module_name  # Debug purpose

        self.global_index = global_index
        self.transformer_layer_id = transformer_layer_id
        self.layer_index = layer_index

        # Attr of some special layers
        self.is_swiglu_fc = is_swiglu_fc
        self.shared_embedding = shared_embedding

        self.expert_id = expert_id
        self.expert_param_layer_id = expert_param_layer_id

        self._check_initlalized = False

        # Cache
        self._resharding_info_cache: Dict[
            Tuple[int, int], PipelineParallelReshardingInfo
        ] = {}

    def _check_init(self):
        """Check the initialization of the parameter."""
        if self._check_initlalized:
            return
        self._check_initlalized = True

        assert self.global_index != -1

        if self.is_expert:
            assert self.expert_param_layer_id != -1
        else:
            assert self.layer_index != -1

        if self.layer_type == LayerType.TRANSFORMER_LAYER:
            assert self.transformer_layer_id != -1
        else:
            assert self.transformer_layer_id == -1

        self._pp_size_to_stage_id: Dict[int, int] = {}

    def get_model_param_stage_id(self, pipeline_model_parallel_size: int) -> int:
        """Given a PP Size, get the stage id of the model param."""
        self._check_init()
        assert pipeline_model_parallel_size >= 1
        if self._pp_size_to_stage_id.get(pipeline_model_parallel_size) is not None:
            return self._pp_size_to_stage_id[pipeline_model_parallel_size]

        if self.layer_type != LayerType.TRANSFORMER_LAYER:
            stage_id = self._get_non_transformer_layer_stage_id(
                pipeline_model_parallel_size
            )
        else:
            stage_id = self._get_transformer_layer_stage_id(
                pipeline_model_parallel_size
            )

        self._pp_size_to_stage_id[pipeline_model_parallel_size] = stage_id
        return stage_id

    def _get_non_transformer_layer_stage_id(
        self, pipeline_model_parallel_size: int
    ) -> int:
        assert isinstance(self.layer_type, LayerType), (
            f"{self.layer_type=}, type={type(self.layer_type)}"
        )
        if self.layer_type in [LayerType.WORD_EMBEDDING, LayerType.POSITION_EMBEDDING]:
            return 0

        if self.layer_type == LayerType.FINAL_LAYERNORM:
            return pipeline_model_parallel_size - 1

        # If shared_embedding is True and pipeline_model_parallel_size == 1, should ignore the output_layer
        assert self.layer_type == LayerType.OUTPUT_LAYER, f"{self}"
        if self.shared_embedding and pipeline_model_parallel_size == 1:
            return -1
        return pipeline_model_parallel_size - 1

    def _get_transformer_layer_stage_id(self, pipeline_model_parallel_size: int) -> int:
        args = get_args()
        assert self.transformer_layer_id != -1
        assert args.num_layers % pipeline_model_parallel_size == 0

        transformer_layer_num_per_stage = (
            args.num_layers // pipeline_model_parallel_size
        )
        return self.transformer_layer_id // transformer_layer_num_per_stage

    def reshard(
        self,
        src_pipeline_model_parallel_size: int,
        dst_pipeline_model_parallel_size: int,
    ) -> PipelineParallelReshardingInfo:
        """Reshard the parameter from src_pp_rank to dst_pp_rank."""
        if (
            self._resharding_info_cache.get(
                (src_pipeline_model_parallel_size, dst_pipeline_model_parallel_size)
            )
            is not None
        ):
            return self._resharding_info_cache[
                (src_pipeline_model_parallel_size, dst_pipeline_model_parallel_size)
            ]

        src_stage_id = self.get_model_param_stage_id(src_pipeline_model_parallel_size)
        dst_stage_id = self.get_model_param_stage_id(dst_pipeline_model_parallel_size)
        resharding_info = PipelineParallelReshardingInfo(
            src_pp_rank=src_stage_id, dst_pp_rank=dst_stage_id
        )
        self._resharding_info_cache[
            (src_pipeline_model_parallel_size, dst_pipeline_model_parallel_size)
        ] = resharding_info

        return resharding_info

    @property
    def is_expert(self) -> bool:
        return self.expert_id is not None

    def __str__(self):
        if self.transformer_layer_id == -1:
            return f"{self.layer_type=}, {self.global_index=}, {self.layer_index=}"
        assert self.layer_type == LayerType.TRANSFORMER_LAYER, f"{self.layer_type=}"
        if self.is_expert:
            return f"{self.global_index=}, {self.transformer_layer_id=}, {self.expert_id=}, {self.expert_param_layer_id=}"
        return (
            f"{self.global_index=}, {self.transformer_layer_id=}, {self.layer_index=}"
        )

    def __repr__(self):
        return str(self)

    def __hash__(self):
        return hash(str(self))

    def __eq__(self, other):
        assert isinstance(other, ParamPositionAttr)
        return hash(self) == hash(other)

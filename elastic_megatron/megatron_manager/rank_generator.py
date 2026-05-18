from typing import TYPE_CHECKING

from megatron.core.parallel_state import RankGenerator

if TYPE_CHECKING:
    from .parallel_strategy import ParallelStrategy


class ElasticRankGenerator:
    def __init__(self, parallel_strategy: "ParallelStrategy"):
        self.dense_generator = RankGenerator(
            tp=parallel_strategy.tensor_model_parallel_size,
            ep=1,
            dp=parallel_strategy.data_parallel_size,
            pp=parallel_strategy.pipeline_model_parallel_size,
            cp=parallel_strategy.context_parallel_size,
            order=parallel_strategy.order,
            rank_offset=0,
        )

        # TODO: support expert specific ordering
        self.moe_generator = RankGenerator(
            tp=parallel_strategy.expert_tensor_parallel_size,
            ep=parallel_strategy.expert_model_parallel_size,
            dp=parallel_strategy.expert_data_parallel_size,
            pp=parallel_strategy.pipeline_model_parallel_size,
            cp=1,
            order=parallel_strategy.order,
            rank_offset=0,
        )

        assert self.dense_generator.get_ranks("pp") == self.moe_generator.get_ranks(
            "pp"
        ), (
            f"Pipeline parallel groups are expected to be the same for Non-Expert and Expert part, \
        but got {self.dense_generator.get_ranks('pp')} and {self.moe_generator.get_ranks('pp')}"
        )

    def generator_wrapper(self, group_type, is_expert=False, **kwargs):
        """The `RankGenerator` class produces a hyper-rectangle for a given set of
        tensor, pipeline, data, expert, and context parallelism. If we have an encoder,
        in addition to the default decoder, we essentially instantiate two `RankGenerator`
        classes to construct the parallelism for each module separately, and we then have
        to stitch them together for the right groups. For now, this means pp and tp-pp."""
        if is_expert:
            d_ranks = self.moe_generator.get_ranks(group_type, **kwargs)
        else:
            d_ranks = self.dense_generator.get_ranks(group_type, **kwargs)

        for x in d_ranks:
            yield x

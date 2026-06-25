"""Launcher-injected reshard strategy list (torch-free).

The entry script derives ``base`` (the launch ``ParallelStrategy`` config) from Megatron
args, then asks for the full strategy list = ``base`` plus the reshard targets injected
from the launcher. Kept dependency-free (only ``json`` / ``os``) so it imports without
CUDA/Megatron and is unit-testable.
"""

from __future__ import annotations

import json
import os


def load_strategy_overrides() -> list[dict]:
    """Reshard sequence as a list of override-dicts, injected from the launcher.

    ``ELASTIC_STRATEGY_LIST`` (inline JSON) takes priority; otherwise
    ``ELASTIC_STRATEGY_LIST_FILE`` (path to a JSON file); otherwise ``[{}]`` (a single
    strategy = no reshard). Each entry overrides any ``ParallelStrategy`` field
    (``world_size``, ``tensor_model_parallel_size``, ``pipeline_model_parallel_size``,
    ``context_parallel_size``, ``expert_model_parallel_size``,
    ``expert_tensor_parallel_size``, ``num_distributed_optimizer_instances``,
    ``sequence_parallel``) onto ``base``.

    Convention: ``overrides[0]`` should be ``{}`` so ``strategy[0]`` equals the launch
    config (ElasticMegatron requires the first strategy to match the launcher args).
    """
    raw = os.environ.get("ELASTIC_STRATEGY_LIST", "").strip()
    if not raw:
        path = os.environ.get("ELASTIC_STRATEGY_LIST_FILE", "").strip()
        raw = open(path).read() if path else "[{}]"
    overrides = json.loads(raw)
    assert isinstance(overrides, list) and all(
        isinstance(o, dict) for o in overrides
    ), "ELASTIC_STRATEGY_LIST must be a JSON list of override objects"
    return overrides or [{}]


def build_strategy_list(base: dict) -> list[dict]:
    """Merge each injected override-dict onto ``base`` (= ``strategy[0]``, the launch config).

    Invalid combinations are not validated here; they fail fast in
    ``ParallelStrategy.__post_init__`` when the manager instantiates each strategy.
    """
    return [{**base, **override} for override in load_strategy_overrides()]

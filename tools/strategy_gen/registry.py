# tools/strategy_gen/registry.py
"""Model arch registry + loader (torch-free).

Single source of model architecture for memory estimation. Seeded from the launcher
arch tables (run_dense.sh MODEL_SIZE switch, run_qwen3_30b.sh defaults). `total_params_b`
is the authoritative weight/grad/optimizer sizing number (params in billions, total —
for MoE that counts all experts).

The v1 (heuristic) estimator reads only `total_params_b`, `hidden`, `layers`, `moe`, and
`num_experts` (the latter two drive EP enumeration in the planner). The remaining arch
fields (`ffn`, `heads`, `kv_heads`, `seq`, `vocab`, `moe_ffn`, `topk`) are unused by v1 —
they are kept for the planned analytic backend behind `MemoryModel.est()`.
"""
from __future__ import annotations

import json
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelArch:
    name: str
    moe: bool
    total_params_b: float
    hidden: int
    layers: int
    ffn: int
    heads: int
    kv_heads: int
    seq: int
    vocab: int
    num_experts: int = 0
    moe_ffn: int = 0
    topk: int = 0
    launcher: str = "run_dense.sh"
    default_mbs: int = 1


@dataclass(frozen=True)
class RunShape:
    mbs: int
    seq: int
    recompute_full: bool = False


# label -> ModelArch. Dense arch from run_dense.sh's MODEL_SIZE table; MoE from
# run_qwen3_30b.sh defaults. total_params_b mirrors the prior MODEL_SCALES model_gb.
MODEL_REGISTRY: dict[str, ModelArch] = {
    "llama2-medium": ModelArch("llama2-medium", False, 2.0, 4096, 8, 11008, 32, 8, 4096, 32000),
    "llama2-7b": ModelArch("llama2-7b", False, 7.0, 4096, 32, 11008, 32, 32, 4096, 32000),
    "llama2-13b": ModelArch("llama2-13b", False, 13.0, 5120, 40, 13824, 40, 40, 4096, 32000),
    "moe-4b": ModelArch("moe-4b", True, 4.0, 2048, 12, 6144, 32, 4, 4096, 151936,
                        num_experts=128, moe_ffn=768, topk=8, launcher="run_qwen3_30b.sh"),
    "moe-15b": ModelArch("moe-15b", True, 15.0, 2048, 24, 6144, 32, 4, 4096, 151936,
                         num_experts=128, moe_ffn=768, topk=8, launcher="run_qwen3_30b.sh"),
    "qwen3-30b": ModelArch("qwen3-30b", True, 30.0, 2048, 48, 6144, 32, 4, 4096, 151936,
                           num_experts=128, moe_ffn=768, topk=8, launcher="run_qwen3_30b.sh"),
}


def load_model(spec: str, config_path: str | None = None) -> ModelArch:
    """Resolve a model: a registry label, or a --config JSON of ModelArch fields."""
    if config_path is not None:
        with open(config_path) as f:
            return ModelArch(**json.load(f))
    if spec not in MODEL_REGISTRY:
        raise KeyError(f"unknown model {spec!r}; known: {sorted(MODEL_REGISTRY)} (or pass --config)")
    return MODEL_REGISTRY[spec]

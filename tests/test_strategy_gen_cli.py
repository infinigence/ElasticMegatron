# tests/test_strategy_gen_cli.py
import importlib.util, json, os, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.strategy_gen.cli import generate


def _load_strategy_inject():
    p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "elastic_megatron", "strategy_inject.py")
    spec = importlib.util.spec_from_file_location("strategy_inject_ut", p)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def test_generate_overrides_parse_through_strategy_inject():
    # llama2-13b keeps a consistent optimizer placement across the 8->4 DP-resize.
    overrides, recipe, table = generate("llama2-13b", gpus=8, gpu_mem_gb=80,
                                        scenario="rl_dp_resize", mbs=1, seq=4096)
    assert overrides[0] == {}
    si = _load_strategy_inject()
    os.environ["ELASTIC_STRATEGY_LIST"] = json.dumps(overrides)
    base = dict(world_size=8, tensor_model_parallel_size=1, pipeline_model_parallel_size=1,
                context_parallel_size=1, expert_model_parallel_size=1,
                expert_tensor_parallel_size=None, num_distributed_optimizer_instances=1,
                sequence_parallel=False)
    strategies = si.build_strategy_list(base)
    del os.environ["ELASTIC_STRATEGY_LIST"]
    assert len(strategies) == len(overrides) and strategies[0]["world_size"] == 8


def test_recipe_picks_launcher_and_offload():
    _, recipe, _ = generate("qwen3-30b", gpus=8, gpu_mem_gb=80, scenario="rl_dp_resize",
                            mbs=1, seq=4096, recompute_full=True)
    assert "run_qwen3_30b.sh" in recipe and "CPU_OFFLOAD=1" in recipe
    assert "ELASTIC_ENABLED=1" in recipe and "ELASTIC_STRATEGY_LIST_FILE=" in recipe


def test_moe_recipe_pins_etp1_and_shape_knobs():
    # The launched shape must equal the planner-sized shape: ETP=1 (TPE), recompute (RECOMPUTE),
    # micro-batch (MBS) and seq-length (SEQ_LEN) all pinned with the run_qwen3_30b.sh var names.
    _, recipe, _ = generate("qwen3-30b", gpus=8, gpu_mem_gb=80, scenario="rl_dp_resize",
                            mbs=2, seq=2048, recompute_full=True)
    assert "TPE=1" in recipe
    assert "RECOMPUTE=1" in recipe and "MBS=2" in recipe and "SEQ_LEN=2048" in recipe


def test_moe_recipe_recompute_off_emits_explicit_zero():
    # MoE launcher defaults recompute ON; a non-recompute plan must explicitly turn it off.
    _, recipe, _ = generate("qwen3-30b", gpus=8, gpu_mem_gb=80, scenario="rl_dp_resize",
                            mbs=1, seq=4096, recompute_full=False)
    assert "RECOMPUTE=0" in recipe


def test_dense_recipe_uses_run_dense():
    _, recipe, _ = generate("llama2-7b", gpus=8, gpu_mem_gb=80, scenario="verify_sweep",
                            mbs=1, seq=4096)
    assert "run_dense.sh" in recipe and "MODEL_SIZE=" in recipe
    assert "TPE=" not in recipe  # dense never pins expert-tensor-parallel


def test_dense_recipe_recompute_and_shape_knobs():
    # Dense launcher recompute var is RECOMPUTE_FULL (default off); only emit it when on.
    _, on, _ = generate("llama2-7b", gpus=8, gpu_mem_gb=80, scenario="verify_sweep",
                        mbs=2, seq=2048, recompute_full=True)
    assert "RECOMPUTE_FULL=1" in on and "MBS=2" in on and "MAX_SEQ_LEN=2048" in on
    _, off, _ = generate("llama2-7b", gpus=8, gpu_mem_gb=80, scenario="verify_sweep",
                         mbs=1, seq=4096, recompute_full=False)
    assert "RECOMPUTE_FULL" not in off


if __name__ == "__main__":
    for n, fn in sorted((n, f) for n, f in globals().items() if n.startswith("test_") and callable(f)):
        fn()
    print(f"PASS {os.path.relpath(__file__)}")

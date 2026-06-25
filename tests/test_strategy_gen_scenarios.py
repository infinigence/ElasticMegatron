# tests/test_strategy_gen_scenarios.py
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.strategy_gen.registry import load_model, RunShape
from tools.strategy_gen.memory import HeuristicMemoryModel
from tools.strategy_gen.planner import LayoutPlanner
from tools.strategy_gen.scenarios import SCENARIOS, Hardware, verify_sweep


def _run(scenario, model, gpus, mem, rs):
    pl = LayoutPlanner(HeuristicMemoryModel(), mem)
    return SCENARIOS[scenario](load_model(model), Hardware(gpus, mem), pl, rs)


def test_overrides0_is_empty():
    # llama2-13b: base and rollout both use cpu-adam at 80GB, so the DP-resize is legal
    # (llama2-7b would flip CPU_OFFLOAD across 8->4 -> rl_dp_resize raises by design, see
    # test_rl_dp_resize_rejects_cpu_adam_flip).
    base, ov = _run("rl_dp_resize", "llama2-13b", 8, 80, RunShape(1, 4096))
    assert ov[0] == {}


def test_rl_dp_resize_shrinks_world():
    base, ov = _run("rl_dp_resize", "llama2-13b", 8, 80, RunShape(1, 4096))
    # [base, rollout] only — the elastic loop cycles strategies[i % len] back to base, so there
    # is NO trailing return-to-base {} (a repeat would trip the registry's uniqueness assert).
    assert len(ov) == 2 and ov[0] == {}
    assert ov[1].get("world_size") == 4, f"expected a world=4 rollout step, got {ov}"


def test_verify_sweep_all_same_world_and_distinct():
    base, ov = _run("verify_sweep", "llama2-7b", 8, 80, RunShape(1, 4096))
    assert ov[0] == {} and len(ov) >= 2
    assert all("world_size" not in o for o in ov)  # fixed world sweep
    assert len(ov) == len({tuple(sorted(o.items())) for o in ov})  # distinct


def test_moe_scenarios_feasible():
    base, ov = _run("rl_dp_resize", "qwen3-30b", 8, 80, RunShape(1, 4096, True))
    assert base.cpu_adam and ov[0] == {}


def test_rl_dp_resize_cpu_adam_consistent():
    # For models where base and rollout agree on optimizer placement, rl_dp_resize succeeds.
    for model, rs in [("llama2-13b", RunShape(1, 4096)), ("qwen3-30b", RunShape(1, 4096, True))]:
        base, ov = _run("rl_dp_resize", model, 8, 80, rs)
        assert ov[0] == {} and len(ov) == 2  # [base, rollout]; loop returns to base by cycling


def test_rl_dp_resize_rejects_cpu_adam_flip():
    # A reshard cannot flip CPU_OFFLOAD mid-run. llama2-7b fits GPU-adam at world=8 but only
    # cpu-adam at world=4 (80GB), so rl_dp_resize must fail loudly rather than silently run the
    # rollout under the wrong optimizer placement.
    try:
        _run("rl_dp_resize", "llama2-7b", 8, 80, RunShape(1, 4096))
    except AssertionError as e:
        assert "cpu_adam" in str(e).lower() and "cpu_offload" in str(e).lower()
    else:
        raise AssertionError("expected rl_dp_resize to reject a CPU_OFFLOAD flip")


def test_verify_sweep_moe_steps_runtime_legal():
    # qwen3-30b @ 8 GPUs, 80-step sweep: every non-empty override pins ETP==1 and every
    # decoded step satisfies world % (EP*PP) == 0 (no step trips parallel_strategy.py).
    pl = LayoutPlanner(HeuristicMemoryModel(), 80)
    base, ov = verify_sweep(load_model("qwen3-30b"), Hardware(8, 80), pl, RunShape(1, 4096, True), max_steps=80)
    assert ov[0] == {}
    base_ep, base_pp, base_world = base.ep, base.pp, base.world
    for o in ov[1:]:
        assert o.get("expert_tensor_parallel_size") == 1, f"MoE override missing ETP=1: {o}"
        ep = o.get("expert_model_parallel_size", base_ep)
        pp = o.get("pipeline_model_parallel_size", base_pp)
        world = o.get("world_size", base_world)
        assert world % (ep * pp) == 0, f"illegal decoded step (world={world}, ep={ep}, pp={pp}): {o}"


def test_no_duplicate_strategy_in_sequence():
    # The elastic loop cycles strategies[i % len] and the manager asserts each strategy is
    # unique (megatron_state.init_parallel_strategy), so no scenario may emit a repeated
    # strategy — notably no trailing return-to-base {}. This is the bug that crashed init at
    # megatron_state.py:115 when rl_dp_resize emitted [{}, rollout, {}].
    cases = [("rl_dp_resize", "llama2-13b", RunShape(1, 4096)),
             ("rl_dp_resize", "qwen3-30b", RunShape(1, 4096, True)),
             ("verify_sweep", "llama2-7b", RunShape(1, 4096)),
             ("verify_sweep", "qwen3-30b", RunShape(1, 4096, True))]
    for scen, model, rs in cases:
        base, ov = _run(scen, model, 8, 80, rs)
        keys = [tuple(sorted(o.items())) for o in ov]
        assert len(keys) == len(set(keys)), f"{scen}/{model} emitted a duplicate strategy: {ov}"


if __name__ == "__main__":
    for n, fn in sorted((n, f) for n, f in globals().items() if n.startswith("test_") and callable(f)):
        fn()
    print(f"PASS {os.path.relpath(__file__)}")

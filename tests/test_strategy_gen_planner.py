# tests/test_strategy_gen_planner.py
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.strategy_gen.registry import load_model, RunShape
from tools.strategy_gen.memory import HeuristicMemoryModel
from tools.strategy_gen.planner import LayoutPlanner, Layout

PL = LayoutPlanner(HeuristicMemoryModel(), gpu_mem_gb=80.0)


def test_divisibility_and_dp():
    outs = PL.feasible(load_model("llama2-7b"), world=8, run_shape=RunShape(1, 4096), cpu_adam=False)
    for layout in outs:
        assert layout.world == 8 and layout.world % (layout.tp * layout.pp * layout.cp) == 0
        assert layout.dp == layout.world // (layout.tp * layout.pp * layout.cp)
        assert layout.ep == 1 and layout.moe is False


def test_moe_ep_divides_experts_and_etp1():
    # Every feasible MoE layout must be runtime-legal under parallel_strategy.py's
    # ETP==1 expert region: ep divides num_experts, ep<=world, AND world % (ep*pp) == 0
    # (else _init_moe raises). layout.moe must be flagged True for the recipe/override path.
    outs = PL.feasible(load_model("qwen3-30b"), world=8, run_shape=RunShape(1, 4096, True), cpu_adam=True)
    assert outs, "expected feasible MoE layouts under cpu-adam"
    for layout in outs:
        assert 128 % layout.ep == 0 and layout.ep <= layout.world
        assert layout.world % (layout.ep * layout.pp) == 0, (
            f"illegal MoE region (world={layout.world}, ep={layout.ep}, pp={layout.pp})")
        assert layout.moe is True


def test_moe_override_pins_etp1_and_empty_diff_stays_empty():
    # Non-empty MoE diff must explicitly carry expert_tensor_parallel_size==1 (user directive:
    # never rely on {**base} inheritance); an empty diff (self==base) must stay {} so
    # overrides[0]=={} is preserved.
    base = Layout(world=8, tp=1, pp=1, cp=1, ep=4, dp=8, cpu_adam=True, moe=True)
    rollout = Layout(world=4, tp=1, pp=1, cp=1, ep=4, dp=4, cpu_adam=True, moe=True)
    ov = rollout.override(base)
    assert ov.get("world_size") == 4 and ov.get("expert_tensor_parallel_size") == 1
    assert base.override(base) == {}  # empty MoE diff carries no ETP key


def test_dense_override_has_no_etp_key():
    base = Layout(world=8, tp=1, pp=1, cp=1, ep=1, dp=8, cpu_adam=False, moe=False)
    rollout = Layout(world=4, tp=1, pp=1, cp=1, ep=1, dp=4, cpu_adam=False, moe=False)
    ov = rollout.override(base)
    assert ov == {"world_size": 4}
    assert "expert_tensor_parallel_size" not in ov


def test_ranking_prefers_max_dp():
    outs = PL.feasible(load_model("llama2-7b"), world=8, run_shape=RunShape(1, 4096), cpu_adam=False)
    assert outs[0].dp == max(l.dp for l in outs)


def test_no_feasible_raises():
    small = LayoutPlanner(HeuristicMemoryModel(), gpu_mem_gb=1.0)
    try:
        small.best(load_model("qwen3-30b"), world=8, run_shape=RunShape(1, 4096, True), cpu_adam=False)
    except RuntimeError as e:
        assert "no feasible" in str(e).lower()
    else:
        raise AssertionError("expected RuntimeError")


def test_auto_falls_back_to_cpu_adam():
    # 30B has no GPU-adam layout at world 8 but a cpu-adam one exists -> auto picks cpu-adam.
    l = PL.best(load_model("qwen3-30b"), world=8, run_shape=RunShape(1, 4096, True), cpu_adam="auto")
    assert l.cpu_adam is True


if __name__ == "__main__":
    for n, fn in sorted((n, f) for n, f in globals().items() if n.startswith("test_") and callable(f)):
        fn()
    print(f"PASS {os.path.relpath(__file__)}")

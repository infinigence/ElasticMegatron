# tests/test_strategy_gen_memory.py
import math, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.strategy_gen.registry import load_model, RunShape
from tools.strategy_gen.memory import HeuristicMemoryModel

M = HeuristicMemoryModel()


class L:  # minimal duck-typed layout
    def __init__(self, world, tp, pp, cp, ep, dp, cpu_adam=False):
        self.world, self.tp, self.pp, self.cp, self.ep, self.dp, self.cpu_adam = world, tp, pp, cp, ep, dp, cpu_adam


def test_more_tp_lowers_peak():
    a = load_model("llama2-7b"); rs = RunShape(1, 4096)
    hi = M.est(a, L(8, 1, 1, 1, 1, 8), rs)
    lo = M.est(a, L(8, 2, 1, 1, 1, 4), rs)
    assert lo < hi


def test_cpu_adam_zeroes_optimizer():
    a = load_model("llama2-7b"); rs = RunShape(1, 4096)
    gpu = M.est(a, L(8, 1, 1, 1, 1, 8, cpu_adam=False), rs)
    cpu = M.est(a, L(8, 1, 1, 1, 1, 8, cpu_adam=True), rs)
    assert cpu < gpu


def test_recompute_lowers_activation():
    a = load_model("qwen3-30b")
    full = M.est(a, L(8, 4, 1, 1, 4, 2), RunShape(1, 4096, recompute_full=True))
    none = M.est(a, L(8, 4, 1, 1, 4, 2), RunShape(1, 4096, recompute_full=False))
    assert full < none


def test_calibration_qwen3_30b():
    # 30B on 8x80GB: GPU-adam must NOT fit; cpu-adam must fit. (Matches reality.)
    a = load_model("qwen3-30b"); rs = RunShape(1, 4096, recompute_full=True)
    gpu = M.est(a, L(8, 4, 1, 1, 4, 2, cpu_adam=False), rs)
    cpu = M.est(a, L(8, 4, 1, 1, 4, 2, cpu_adam=True), rs)
    assert gpu > 80.0, f"expected GPU-adam 30B to exceed 80GB, got {gpu:.1f}"
    assert cpu <= 80.0, f"expected cpu-adam 30B to fit 80GB, got {cpu:.1f}"


if __name__ == "__main__":
    for n, fn in sorted((n, f) for n, f in globals().items() if n.startswith("test_") and callable(f)):
        fn()
    print(f"PASS {os.path.relpath(__file__)}")

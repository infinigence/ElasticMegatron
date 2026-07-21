# tests/test_strategy_fixtures_drift.py
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.strategy_gen import generate

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# (fixture path, generate kwargs, is_moe) — canonical inputs that produce each committed file.
FIXTURES = [
    ("examples/strategies/precision/dense_no_tp.json",
     dict(model="llama2-medium", gpus=8, gpu_mem_gb=80, scenario="rl_dp_resize", mbs=1, seq=4096), False),
    ("examples/strategies/moe_30b.json",
     dict(model="qwen3-30b", gpus=8, gpu_mem_gb=80, scenario="rl_dp_resize", mbs=1, seq=4096, recompute_full=True), True),
]


def test_fixtures_match_generator():
    for rel, kw, is_moe in FIXTURES:
        overrides, _, _ = generate(out_path=rel, **kw)
        committed = json.load(open(os.path.join(ROOT, rel)))
        assert committed == overrides, f"{rel} drifted from generator output"
        assert overrides[0] == {}, f"{rel} overrides[0] must be {{}}"
        assert any(o.get("world_size") == 4 for o in overrides), f"{rel} lacks an 8->4 scale-down"
        scaledown = next(o for o in overrides if o.get("world_size") == 4)
        if is_moe:
            # MoE scale-down step must explicitly pin ETP=1 (runtime-legal, user directive).
            assert scaledown.get("expert_tensor_parallel_size") == 1, (
                f"{rel} MoE scale-down step missing explicit ETP=1: {scaledown}")
        else:
            # Dense fixture stays dense: no expert-tensor-parallel key anywhere.
            assert all("expert_tensor_parallel_size" not in o for o in overrides), (
                f"{rel} dense fixture leaked an ETP key")


if __name__ == "__main__":
    test_fixtures_match_generator()
    print(f"PASS {os.path.relpath(__file__)}")

"""Torch-free tests for the model registry + loader."""
import json, os, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.strategy_gen.registry import ModelArch, MODEL_REGISTRY, load_model


def test_label_resolves_to_arch():
    a = load_model("qwen3-30b")
    assert a.moe and a.num_experts == 128 and a.hidden == 2048 and a.launcher == "run_qwen3_30b.sh"


def test_dense_label():
    a = load_model("llama2-7b")
    assert not a.moe and a.num_experts == 0 and a.layers == 32 and a.launcher == "run_dense.sh"


def test_unknown_label_raises():
    try:
        load_model("nope-1t")
    except KeyError as e:
        assert "nope-1t" in str(e)
    else:
        raise AssertionError("expected KeyError")


def test_config_override():
    cfg = dict(name="custom", moe=False, total_params_b=1.0, hidden=512, layers=4,
               ffn=1024, heads=8, kv_heads=8, seq=2048, vocab=32000)
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(cfg, f); path = f.name
    a = load_model("ignored", config_path=path)
    os.unlink(path)
    assert a.name == "custom" and a.hidden == 512 and not a.moe


if __name__ == "__main__":
    for n, fn in sorted((n, f) for n, f in globals().items() if n.startswith("test_") and callable(f)):
        fn()
    print(f"PASS {os.path.relpath(__file__)}")

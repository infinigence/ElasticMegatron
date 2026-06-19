"""Static probe for scale-down/up experiment entries.

The A100 verification matrix must include explicit world-size-changing
reshards through both launcher paths: dense -> run_e2e_demo.sh and MoE ->
run_moe.sh. This test is intentionally torch-free and catches accidental
regression back to in-place-only strategy sweeps.
"""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_training_016_declares_scale_world_modes():
    src = (ROOT / "examples" / "intra_process" / "training_016.py").read_text()
    assert 'mode == "dense_scale_world"' in src
    assert 'mode == "moe_scale_world"' in src
    assert 'down["world_size"] = 4' in src
    assert "_PARALLEL_STRATEGY_LIST = [base, down]" in src


def test_run_experiment_exposes_both_scale_paths():
    src = (ROOT / "run_experiment.sh").read_text()
    assert "dense_scale_world)" in src
    assert "ELASTIC_STRATEGY_MODE=dense_scale_world" in src
    assert 'bash "$(dirname "$0")/run_e2e_demo.sh"' in src
    assert "moe_scale_world)" in src
    assert "ELASTIC_STRATEGY_MODE=moe_scale_world" in src
    assert 'bash "$(dirname "$0")/run_moe.sh"' in src


def test_qwen3_30b_declares_exact_scale_world_path():
    training = (ROOT / "examples" / "intra_process" / "training_016.py").read_text()
    assert 'mode == "moe_30b"' in training
    assert "moe_30b base must be TP4/PP1/CP1/EP4" in training
    assert 'dp1["world_size"] = 4' in training
    assert "_PARALLEL_STRATEGY_LIST = [base, dp1]" in training

    launcher = (ROOT / "run_qwen3_30b.sh").read_text()
    assert "ELASTIC_ENABLED=${ELASTIC_ENABLED:-0}" in launcher
    assert "TP=${TP:-4}" in launcher
    assert "EP=${EP:-4}" in launcher
    assert "TPE=${TPE:-1}" in launcher
    assert "--optimizer-cpu-offload" in launcher
    assert "--overlap-cpu-optimizer-d2h-h2d" in launcher


if __name__ == "__main__":
    test_training_016_declares_scale_world_modes()
    test_run_experiment_exposes_both_scale_paths()
    test_qwen3_30b_declares_exact_scale_world_path()
    print("PASS tests/test_scale_world_modes_static.py")

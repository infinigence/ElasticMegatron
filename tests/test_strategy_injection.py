"""Torch-free tests for launcher-injected reshard strategy lists (strategy_inject.py).

Loads strategy_inject.py directly by path: it is torch-free, but importing it through the
elastic_megatron package would trigger the package __init__ (which imports torch).
"""

import importlib.util
import os
import pathlib

_MOD_PATH = (
    pathlib.Path(__file__).resolve().parents[1]
    / "elastic_megatron"
    / "strategy_inject.py"
)
_spec = importlib.util.spec_from_file_location("strategy_inject", _MOD_PATH)
strategy_inject = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(strategy_inject)
load_strategy_overrides = strategy_inject.load_strategy_overrides
build_strategy_list = strategy_inject.build_strategy_list

_BASE = {
    "world_size": 8,
    "tensor_model_parallel_size": 4,
    "pipeline_model_parallel_size": 1,
    "context_parallel_size": 1,
    "num_distributed_optimizer_instances": 1,
    "expert_model_parallel_size": 4,
    "expert_tensor_parallel_size": None,
    "sequence_parallel": True,
}


def _clear_env():
    os.environ.pop("ELASTIC_STRATEGY_LIST", None)
    os.environ.pop("ELASTIC_STRATEGY_LIST_FILE", None)


def test_default_is_single_no_reshard():
    _clear_env()
    assert load_strategy_overrides() == [{}]
    assert build_strategy_list(_BASE) == [_BASE]  # strategy[0] = launch config, no reshard


def test_inline_json_overrides_merge_onto_base():
    _clear_env()
    os.environ["ELASTIC_STRATEGY_LIST"] = '[{}, {"world_size": 4}]'
    assert load_strategy_overrides() == [{}, {"world_size": 4}]
    out = build_strategy_list(_BASE)
    assert out[0] == _BASE  # strategy[0] unchanged = launch config
    assert out[1]["world_size"] == 4  # only world_size overridden
    assert out[1]["tensor_model_parallel_size"] == 4  # inherited from base
    _clear_env()


def test_file_used_when_inline_absent(tmp_path=None):
    import tempfile

    _clear_env()
    d = str(tmp_path) if tmp_path is not None else tempfile.mkdtemp()
    p = os.path.join(d, "s.json")
    with open(p, "w") as f:
        f.write('[{}, {"context_parallel_size": 2}]')
    os.environ["ELASTIC_STRATEGY_LIST_FILE"] = p
    out = build_strategy_list(_BASE)
    assert len(out) == 2 and out[1]["context_parallel_size"] == 2
    _clear_env()


def test_inline_beats_file():
    _clear_env()
    os.environ["ELASTIC_STRATEGY_LIST"] = "[{}]"
    os.environ["ELASTIC_STRATEGY_LIST_FILE"] = "/nonexistent/should-not-be-read.json"
    assert load_strategy_overrides() == [{}]  # file never opened
    _clear_env()


def test_non_list_payload_rejected():
    _clear_env()
    os.environ["ELASTIC_STRATEGY_LIST"] = '{"not": "a list"}'
    raised = False
    try:
        load_strategy_overrides()
    except AssertionError:
        raised = True
    assert raised, "expected an AssertionError for a non-list payload"
    _clear_env()


if __name__ == "__main__":
    test_default_is_single_no_reshard()
    test_inline_json_overrides_merge_onto_base()
    test_file_used_when_inline_absent()
    test_inline_beats_file()
    test_non_list_payload_rejected()
    print("PASS tests/test_strategy_injection.py")

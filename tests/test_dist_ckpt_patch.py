"""Torch-free unit tests for the DCP scale-down save patch's pure helpers.

The argument-rewriting logic (process_group passed positionally at index 2, by
keyword, or omitted) is the bug-prone part and is exercised here without torch.
The torch/megatron-dependent shrink detection + collective behaviour is covered by
the GPU scale-down verify (ELASTIC_SAVE_CKPT=1), not by this test.

Loaded by file path so importing it does not trigger elastic_megatron's package
__init__ (which imports torch).
"""

import importlib.util
import pathlib

_MOD_PATH = (
    pathlib.Path(__file__).resolve().parents[1]
    / "elastic_megatron"
    / "distributed"
    / "dist_ckpt_patch.py"
)
_spec = importlib.util.spec_from_file_location("dist_ckpt_patch_under_test", _MOD_PATH)
m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m)


def test_current_pg_positional_none():
    # Megatron's call shape: (state_dict, writer, None, coordinator), pg at index 2.
    assert m._current_process_group(("sd", "writer", None, 0), {}) is None


def test_current_pg_positional_set():
    assert m._current_process_group(("sd", "writer", "PG", 0), {}) == "PG"


def test_current_pg_kwarg():
    assert m._current_process_group(("sd", "writer"), {"process_group": "PG"}) == "PG"
    assert m._current_process_group(("sd", "writer"), {"process_group": None}) is None


def test_current_pg_omitted():
    assert m._current_process_group(("sd", "writer"), {}) is None


def test_inject_positional():
    # The real Megatron shape: process_group is positional arg index 2 (None).
    args, kwargs = m._inject_process_group(
        ("sd", "writer", None, 0), {"planner": "p"}, "GROUP"
    )
    assert args == ("sd", "writer", "GROUP", 0)
    assert kwargs == {"planner": "p"}


def test_inject_kwarg():
    args, kwargs = m._inject_process_group(
        ("sd", "writer"), {"process_group": None, "coordinator_rank": 0}, "GROUP"
    )
    assert args == ("sd", "writer")
    assert kwargs["process_group"] == "GROUP"
    assert kwargs["coordinator_rank"] == 0


def test_inject_omitted():
    # process_group neither positional nor keyword -> add as keyword.
    args, kwargs = m._inject_process_group(("sd", "writer"), {}, "GROUP")
    assert args == ("sd", "writer")
    assert kwargs["process_group"] == "GROUP"


def test_inject_does_not_mutate_caller_dict():
    orig_kwargs = {"process_group": None}
    m._inject_process_group(("sd", "writer"), orig_kwargs, "GROUP")
    assert orig_kwargs["process_group"] is None  # caller's dict untouched


def test_roundtrip_megatron_shape():
    # Detect None at index 2, then inject -> index 2 becomes the override.
    args = ("sd", "writer", None, 0)
    kwargs = {"planner": "p"}
    assert m._current_process_group(args, kwargs) is None
    args, kwargs = m._inject_process_group(args, kwargs, "GROUP")
    assert m._current_process_group(args, kwargs) == "GROUP"


if __name__ == "__main__":
    tests = sorted(
        (name, fn)
        for name, fn in globals().items()
        if name.startswith("test_") and callable(fn)
    )
    for name, fn in tests:
        fn()
    print(f"PASS tests/test_dist_ckpt_patch.py ({len(tests)} tests)")

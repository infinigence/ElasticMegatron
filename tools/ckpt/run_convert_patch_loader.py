#!/usr/bin/env python3
import os
import sys
import argparse
import subprocess
import tempfile
import shutil

"""
Run convert.py with a temporarily patch.

Pass the values of tp pp and ep in the checkpoint as parameters.

"""

THIS_DIR = os.path.abspath(os.path.dirname(__file__))
# suppose ElasticMegatron is the project root, the convert tools is in Megatron-LM/tools/checkpoint/...
PROJECT_ROOT = os.path.dirname(os.path.dirname(THIS_DIR))
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "Megatron-LM", "tools", "checkpoint")
LOADER_CORE = os.path.join(CHECKPOINT_DIR, "loader_core.py")
CONVERT_PY = os.path.join(CHECKPOINT_DIR, "convert.py")

PATCH_BANNER_BEGIN = "# --- BEGIN RUNTIME PATCH: force TP/PP before world_size ---\n"
PATCH_BANNER_END = "# --- END RUNTIME PATCH ---\n"


def build_patched_loader(
    src_text: str, force_tp: int, force_pp: int, force_ep: int = 1
) -> str:
    """Insert two assignment lines just before the first `margs.world_size =` line."""
    lines = src_text.splitlines(keepends=True)

    def indent_of(s: str) -> str:
        return s[: len(s) - len(s.lstrip())]

    def make_patch(indent: str):
        return [
            indent + PATCH_BANNER_BEGIN,
            indent + f"margs.tensor_model_parallel_size = {int(force_tp)}\n",
            indent + f"margs.pipeline_model_parallel_size = {int(force_pp)}\n",
            indent + f"margs.expert_model_parallel_size = {int(force_ep)}\n",
            indent + PATCH_BANNER_END,
        ]

    for i, ln in enumerate(lines):
        if "margs.world_size" in ln and "=" in ln:
            indent = indent_of(ln)
            patch_lines = make_patch(indent)
            return "".join(lines[:i] + patch_lines + lines[i:])

    patch_lines = make_patch("    ")
    return "".join(patch_lines + lines)


def main():
    parser = argparse.ArgumentParser(
        description="Run convert.py using a temporary patched loader.",
        allow_abbrev=False,
        conflict_handler="resolve",
    )
    parser.add_argument(
        "--load-tp",
        type=int,
        required=True,
        help="Value to assign to margs.tensor_model_parallel_size in loader.",
    )
    parser.add_argument(
        "--load-pp",
        type=int,
        required=True,
        help="Value to assign to margs.pipeline_model_parallel_size in loader.",
    )
    parser.add_argument(
        "--load-ep",
        type=int,
        default=1,
        help="Value to assign to margs.expert_model_parallel_size in loader.",
    )
    parser.add_argument(
        "--convert",
        type=str,
        default=CONVERT_PY,
        help="Path to convert.py (default: tools/checkpoint/convert.py).",
    )

    args, forward = parser.parse_known_args()

    forwarded = []
    skip_next = False
    seen_loader = False
    for i, tok in enumerate(forward):
        if skip_next:
            skip_next = False
            continue
        if tok == "--loader":
            seen_loader = True
            skip_next = True
            forwarded += ["--loader", "patched"]
        elif tok.startswith("--loader="):
            seen_loader = True
            forwarded.append("--loader=patched")
        else:
            forwarded.append(tok)
    if not seen_loader:
        forwarded += ["--loader", "patched"]

    forwarded = [tok for tok in forwarded if tok != "--"]

    # Read original loader_core
    with open(LOADER_CORE, "r", encoding="utf-8") as f:
        src = f.read()

    # use the provided loader-side TP/PP
    patched = build_patched_loader(src, args.load_tp, args.load_pp, args.load_ep)

    tmpdir = tempfile.mkdtemp(prefix="loader_patch_")
    try:
        patched_path = os.path.join(tmpdir, "loader_patched.py")
        with open(patched_path, "w", encoding="utf-8") as f:
            f.write(patched)

        env = os.environ.copy()
        py_path = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = (tmpdir + os.pathsep + py_path) if py_path else tmpdir

        cmd = [sys.executable, args.convert] + forwarded
        print("Running:", " ".join(cmd))
        print(f"Using patched loader at: {patched_path}")

        proc = subprocess.run(cmd, env=env)
        sys.exit(proc.returncode)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    main()

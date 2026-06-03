#!/usr/bin/env python3
import argparse
import os
import shutil
import subprocess
import sys
import tempfile

"""
Run convert.py with a temporarily patch.

Pass the values of tp pp and ep in the checkpoint as parameters.

"""

THIS_DIR = os.path.abspath(os.path.dirname(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(THIS_DIR))
_BASE = os.path.dirname(PROJECT_ROOT)

# Resolve the Megatron checkpoint tools directory.
# Prefer the MEGATRON_PATH environment variable, then fall back to sibling
# and sub-directory conventions.
_meg_env = os.environ.get("MEGATRON_PATH")
_MEG_CANDIDATES = (
    [_meg_env] if _meg_env else [
        os.path.join(_BASE, "Megatron-LM"),
        os.path.join(PROJECT_ROOT, "Megatron-LM"),
    ]
)
_checkpoint_dir = next(
    (os.path.join(p, "tools", "checkpoint")
     for p in _MEG_CANDIDATES
     if os.path.isdir(os.path.join(p, "tools", "checkpoint"))),
    None,
)
if _checkpoint_dir is None:
    _searched = "\n  ".join(_MEG_CANDIDATES)
    raise RuntimeError(
        "Could not find Megatron tools/checkpoint directory. "
        "Set MEGATRON_PATH to your Megatron-LM root.\n"
        f"Searched:\n  {_searched}"
    )
CHECKPOINT_DIR = _checkpoint_dir
LOADER_CORE = os.path.join(CHECKPOINT_DIR, "loader_core.py")
# 0.16 把 margs.world_size 的赋值移到了 loader_base.py(loader_core.py 不再含
# margs.world_size 行),patcher 优先尝试 loader_base.py;若不存在则 fall back
# 到 loader_core.py(老版本)。
LOADER_BASE = os.path.join(CHECKPOINT_DIR, "loader_base.py")
LOADER_TO_PATCH = LOADER_BASE if os.path.exists(LOADER_BASE) else LOADER_CORE
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
        # 留一个 stdout 标记:如果 heuristic 把 patch 插到了对的位置,convert 运行时
        # 一定能看到这行;看不到就说明 patch 落空(0.17+ 改了 margs.world_size 的
        # 文件位置或写法),需要更新 build_patched_loader 的搜索锚点。
        return [
            indent + PATCH_BANNER_BEGIN,
            indent + f"margs.tensor_model_parallel_size = {int(force_tp)}\n",
            indent + f"margs.pipeline_model_parallel_size = {int(force_pp)}\n",
            indent + f"margs.expert_model_parallel_size = {int(force_ep)}\n",
            indent + (
                f"print('[loader_patch] forced TP={int(force_tp)} PP={int(force_pp)} "
                f"EP={int(force_ep)}', flush=True)\n"
            ),
            indent + PATCH_BANNER_END,
        ]

    for i, ln in enumerate(lines):
        if "margs.world_size" in ln and "=" in ln:
            indent = indent_of(ln)
            patch_lines = make_patch(indent)
            return "".join(lines[:i] + patch_lines + lines[i:])

    patch_lines = make_patch("    ")
    return "".join(patch_lines + lines)


def patch_loader_core_model_provider(src_text: str) -> str:
    """0.16 loader_core.py 的 import_model_provider 在 GPT 分支里 set 了
    self.model_provider = partial(model_provider, gpt_builder),但 return 的是
    *未包装的* model_provider。loader_base.py 调用 model_provider() 时只传
    pre_process/post_process,缺 model_builder 报 TypeError。

    把 `return model_provider` 改成 `return self.model_provider`,让外层拿到
    带 gpt_builder 的 partial。"""
    needle = "self.model_provider = partial(model_provider, gpt_builder)\n            return model_provider\n"
    fix = "self.model_provider = partial(model_provider, gpt_builder)\n            return self.model_provider\n"
    return src_text.replace(needle, fix)


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

    tmpdir = tempfile.mkdtemp(prefix="loader_patch_")
    try:
        # ---------------------------------------------------------------
        # 0.16 起 margs.world_size 在 loader_base.py 里赋值,而不是
        # loader_core.py。因此真正要 patch 的是 loader_base.py。
        # 我们的策略:
        #   1. 把 patched loader_base.py 写入 tmpdir(同名覆盖);
        #   2. 把 loader_core.py 复制到 tmpdir,重命名为 loader_patched.py
        #      作为 `--loader patched` 插件入口;它在 import 时会从
        #      PYTHONPATH 加载 patched loader_base(因为 tmpdir 在最前)。
        # 旧版本 (≤0.15) 走 fallback:patch loader_core.py 自身。
        # ---------------------------------------------------------------
        if LOADER_TO_PATCH == LOADER_BASE:
            with open(LOADER_BASE, "r", encoding="utf-8") as f:
                base_src = f.read()
            base_patched = build_patched_loader(
                base_src, args.load_tp, args.load_pp, args.load_ep
            )
            with open(os.path.join(tmpdir, "loader_base.py"), "w", encoding="utf-8") as f:
                f.write(base_patched)
            # plugin 入口拷贝 loader_core.py,顺便修一处 0.16 model_provider 签名问题
            with open(LOADER_CORE, "r", encoding="utf-8") as f:
                core_src = f.read()
            core_src = patch_loader_core_model_provider(core_src)
            patched_path = os.path.join(tmpdir, "loader_patched.py")
            with open(patched_path, "w", encoding="utf-8") as f:
                f.write(core_src)
        else:
            with open(LOADER_CORE, "r", encoding="utf-8") as f:
                src = f.read()
            patched = build_patched_loader(
                src, args.load_tp, args.load_pp, args.load_ep
            )
            patched_path = os.path.join(tmpdir, "loader_patched.py")
            with open(patched_path, "w", encoding="utf-8") as f:
                f.write(patched)

        env = os.environ.copy()
        py_path = env.get("PYTHONPATH", "")
        # tmpdir 必须在最前,以便 import loader_base 优先取到 patched 版本;
        # 也要包含原 CHECKPOINT_DIR 让 schema_core 等其他 helper 模块仍能被导入。
        extra = os.pathsep.join([tmpdir, CHECKPOINT_DIR])
        env["PYTHONPATH"] = (extra + os.pathsep + py_path) if py_path else extra

        cmd = [sys.executable, args.convert] + forwarded
        print("Running:", " ".join(cmd))
        print(f"Using patched loader at: {patched_path}")
        if LOADER_TO_PATCH == LOADER_BASE:
            print(f"Patched loader_base at: {os.path.join(tmpdir, 'loader_base.py')}")

        proc = subprocess.run(cmd, env=env, check=False)
        sys.exit(proc.returncode)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    main()

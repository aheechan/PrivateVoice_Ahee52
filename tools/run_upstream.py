"""在打包用的嵌入式 Python 里运行上游 GPT-SoVITS 脚本。

为什么需要它：分发包的 runtime 里有 python312._pth，**只要存在 ._pth，
Python 就会忽略 PYTHONPATH**，于是 `from text.cleaner import ...` 这类
相对 GPT_SoVITS 目录的 import 会失败（ModuleNotFoundError: No module named 'text'）。

用法：
    python -s run_upstream.py <repo> <script.py> [脚本参数...]
"""
from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) < 3:
        print("usage: run_upstream.py <repo> <script.py> [args...]", file=sys.stderr)
        return 2
    repo = Path(sys.argv[1]).resolve()
    script = Path(sys.argv[2]).resolve()
    if not script.exists():
        print(f"script not found: {script}", file=sys.stderr)
        return 2
    for p in (str(repo / "GPT_SoVITS"), str(repo)):
        if p not in sys.path:
            sys.path.insert(0, p)
    # s1（Lightning/DDP）需要的运行时垫片：打包版的 ._pth 会忽略 PYTHONPATH，
    # sitecustomize 不会被自动导入，这里显式加载。
    shim = os.environ.get("VOICELAB_SHIM")
    if shim and Path(shim).is_dir() and shim not in sys.path:
        sys.path.insert(0, shim)
        try:
            import sitecustomize  # noqa: F401
        except Exception as exc:  # noqa: BLE001
            print(f"[run_upstream] shim load failed: {exc}", file=sys.stderr)
    sys.argv = [str(script), *sys.argv[3:]]
    os.chdir(repo)
    runpy.run_path(str(script), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

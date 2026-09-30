"""把目前 Python 環境的套件版本寫成專案的 `requirements.lock`，或檢查環境是否與其相符。

在要鎖定的專案根目錄、以該專案實際使用的直譯器執行：

    python vendor/scripts/freeze_env.py            # 寫出 requirements.lock
    python vendor/scripts/freeze_env.py --check    # 環境與 requirements.lock 不符時結束碼 1

檔頭記錄直譯器版本、平台、torch 與 CUDA 版本及取得清單的工具；內容為目前直譯器的
`pip freeze --all`，直譯器沒有 pip 時（例如 uv 建立的 venv）改用
`uv pip freeze --python <直譯器>`，兩者皆無即中止；去除以路徑或 editable 方式安裝的
本 repo 套件（`immunization-*`）。檢查只比對套件列，不比對檔頭。
"""
from __future__ import annotations

import argparse
import importlib.util
import platform
import shutil
import subprocess
import sys
from pathlib import Path

LOCK = "requirements.lock"


def freeze_command() -> list:
    """目前直譯器可用的 freeze 指令：優先 pip，其次 uv；都沒有即中止。"""
    if importlib.util.find_spec("pip") is not None:
        return [sys.executable, "-m", "pip", "freeze", "--all"]
    uv = shutil.which("uv")
    if uv is not None:
        return [uv, "pip", "freeze", "--python", sys.executable]
    raise SystemExit(f"{sys.executable} 沒有 pip，PATH 上也沒有 uv；無法取得套件清單")


def tool_name() -> str:
    command = freeze_command()
    return "pip freeze --all" if command[1] == "-m" else "uv pip freeze"


def freeze() -> list:
    result = subprocess.run(freeze_command(), capture_output=True, text=True, check=True)
    lines = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("-e "):
            continue
        if line.lower().replace("_", "-").startswith(("immunization-", "immunization ")):
            continue
        lines.append(line)
    return sorted(lines, key=str.lower)


def header() -> list:
    try:
        import torch
        torch_line = f"# torch {torch.__version__}; CUDA {torch.version.cuda}"
    except ImportError:
        torch_line = "# torch 未安裝"
    return [f"# python {platform.python_version()} ({sys.executable})",
            f"# platform {platform.platform()}", torch_line,
            f"# 由 freeze_env.py 產生；套件列取自 {tool_name()}"]


def packages(path: Path) -> list:
    return [line for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.startswith("#")]


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    root = args.project_root.resolve()
    if not (root / "pyproject.toml").is_file():
        raise SystemExit(f"{root} 不是專案根目錄（缺 pyproject.toml）")
    path = root / LOCK
    current = freeze()
    if args.check:
        if not path.is_file():
            raise SystemExit(f"缺少 {path}")
        locked = packages(path)
        missing = sorted(set(locked) - set(current), key=str.lower)
        extra = sorted(set(current) - set(locked), key=str.lower)
        for line in missing:
            print(f"- {line}")
        for line in extra:
            print(f"+ {line}")
        raise SystemExit(1 if missing or extra else 0)
    path.write_text("\n".join(header() + current) + "\n", encoding="utf-8", newline="\n")
    print(f"[DONE] {path}（{len(current)} 個套件）")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""程式碼健康檢查：可達性、測試覆蓋、靜態問題。

    python scripts/code_health.py            # 逐項列出
    python scripts/code_health.py --strict   # 有問題就以非零退出（供 CI）

三項檢查，各自回答一個具體問題：

1. **可達性**：這個模組有沒有任何 script 或 test 到得了？到不了的是死碼候選。
2. **測試覆蓋**：有沒有測試**直接** import 它？只被間接帶到的模組，
   其契約沒有被任何斷言釘住。
3. **靜態問題**：pyflakes。未使用的區域變數尤其要看——在測試裡那常代表
   **漏了斷言**（2026-08-09 就抓到一個：`test_Linf下限在超過tau時才施力`
   算出了低於門檻的擾動卻沒有用它，那一半等於沒測到）。

## 為什麼要自己寫而不是只跑 pyflakes

可達性分析有兩個坑，兩個都踩過：

- **套件的 `__init__` 是節點。** `from src.baselines import REGISTRY` 會執行
  `src/baselines/__init__.py`，而它 import 了五篇 baseline。不把套件當節點，
  那五個檔會被誤判成死碼。
- **`from X import Y` 的邊有兩條。** `n.module` 只給 `X`；真正被載入的是
  `X.Y`。只取前者會讓 `src.experiment.executors` 這種模組整個消失。
"""
from __future__ import annotations

import argparse
import ast
import collections
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent

# 到不了但**刻意保留**的模組，附保留理由。列在這裡的不算問題。
DORMANT = {
    "src.data.pie_bench":
        "PIE-Bench 載入器。DIA 用同一個 benchmark，故留著可直接對照；"
        "但遠端機器連不上 HuggingFace，本輪未採用",
    "src.metrics.battery":
        "保真約束的候選指標組，供 P1/P2 篩選。篩選已完成，"
        "結論（收哪幾項、為什麼）已寫進 src/metrics/suite.py",
}


def build_graph(files):
    mod_of = {}
    for p in files:
        parts = list(p.relative_to(ROOT).with_suffix("").parts)
        if parts[-1] == "__init__":
            parts = parts[:-1]              # 套件本身也是一個節點
        mod_of[p] = ".".join(parts)

    imports = collections.defaultdict(set)
    for p in files:
        try:
            tree = ast.parse(p.read_text(encoding="utf-8", errors="ignore"))
        except SyntaxError:
            continue
        me = mod_of[p]
        for n in ast.walk(tree):
            if isinstance(n, ast.ImportFrom) and n.module:
                imports[me].add(n.module)
                for a in n.names:           # `from X import Y` 也載入 X.Y
                    imports[me].add(f"{n.module}.{a.name}")
            elif isinstance(n, ast.Import):
                for a in n.names:
                    imports[me].add(a.name)
    return mod_of, imports


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--strict", action="store_true",
                    help="有問題就以非零退出")
    args = ap.parse_args()

    files = [p for p in ROOT.rglob("*.py") if "__pycache__" not in p.parts]
    mod_of, imports = build_graph(files)
    nodes = {mod_of[p]: p for p in files
             if p.relative_to(ROOT).parts[0] == "src"}
    leaf = {m: p for m, p in nodes.items() if p.stem != "__init__"}

    entry = [m for m in imports if m.startswith(("scripts.", "tests."))]
    reach, stack = set(), list(entry)
    while stack:
        for tgt in imports.get(stack.pop(), ()):
            if tgt in nodes and tgt not in reach:
                reach.add(tgt)
                stack.append(tgt)

    tested = set()
    for t in (m for m in imports if m.startswith("tests.")):
        for tgt in imports[t]:
            if tgt in nodes:
                tested.add(tgt)

    problems = 0

    dead = sorted(set(leaf) - reach - set(DORMANT))
    print(f"[1] 可達性：{len(leaf)} 個葉模組，到不了的 {len(dead)} 個")
    for m in dead:
        n = len(leaf[m].read_text(encoding="utf-8").splitlines())
        print(f"      {m}（{n} 行）")
        problems += 1
    for m, why in sorted(DORMANT.items()):
        mark = "休眠" if m in leaf and m not in reach else "**已可達，可自 DORMANT 移除**"
        print(f"      {m}：{mark} —— {why}")

    untested = sorted((reach & set(leaf)) - tested)
    print(f"\n[2] 測試覆蓋：可達 {len(reach & set(leaf))} 個，"
          f"沒有測試直接 import 的 {len(untested)} 個")
    for m in untested:
        print(f"      {m}（只被間接帶到，契約沒有斷言釘住）")

    out = subprocess.run([sys.executable, "-m", "pyflakes", "src", "scripts", "tests"],
                         capture_output=True, text=True, encoding="utf-8",
                         cwd=ROOT)
    lines = []
    for ln in out.stdout.splitlines():
        if not ln.strip():
            continue
        # pyflakes 的輸出不含原始碼，故 `# noqa` 要回原檔那一行去看
        try:
            f, no = ln.split(":")[0], int(ln.split(":")[1])
            src_line = (ROOT / f).read_text(encoding="utf-8").splitlines()[no - 1]
            if "noqa" in src_line:
                continue
        except (ValueError, IndexError, OSError):
            pass
        lines.append(ln)
    print(f"\n[3] 靜態問題：{len(lines)} 項")
    for ln in lines:
        print(f"      {ln}")
        problems += 1

    print(f"\n合計需要處理 {problems} 項")
    return 1 if (args.strict and problems) else 0


if __name__ == "__main__":
    sys.exit(main())

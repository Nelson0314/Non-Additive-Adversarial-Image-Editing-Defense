"""本目錄裡所有腳本共用的路徑解析。

`main_table/` 是從主線目錄搬出來的，搬動切斷了兩件事，這個模組把兩件都接回去：

1. **影像原檔**在 `main_table/images/`，不再在主線的 `runs/`。
2. **`src.*` 套件**不在 `main_table/` 底下，而在同一層的主線目錄（含
   `src/metrics`、`src/baselines`、`src/purify`、`src/models`、`src/utils`）。
   資料集（原圖、遮罩、`prompts.yaml`、`data/targets/`）也仍在那一側。

主線目錄的名字改過一次（`non-additive-frequency` → `anti-purification`），
所以這裡不只認一個名字：先照 `IMMUNISATION_SOURCE_HOME` 環境變數，再照
`SOURCE_HOME_NAMES` 逐一找，全部找不到就**直接失敗並列出找過哪些路徑**，
不回退到某個猜測值——猜錯會讓腳本在半途才因為缺檔中止。
"""

from __future__ import annotations

import os
from pathlib import Path

#: `main_table/`，本目錄的上一層。（這個常數的名字沿用，指的就是本目錄。）
BASELINES = Path(__file__).resolve().parent.parent

#: 本目錄的上一層。併進主線之後它就是主線目錄；分開放的時候它是 repo 根。
PROJECT = BASELINES.parent

#: 搬進來的逐格影像。底下是 `defence_portraits/`、`edit_preflight/`、
#: `edit_defended/`、`edit_purified/`、`masks/`。
IMAGES = BASELINES / "images"

#: 讀數 CSV 與腳本自己的輸出。
RESULTS = BASELINES / "results"

#: 主線目錄的候選名字，依序嘗試；改名時把新名字加在最前面。
SOURCE_HOME_NAMES = ("anti-purification", "non-additive-frequency")


def _resolve_source_home() -> Path:
    override = os.environ.get("IMMUNISATION_SOURCE_HOME")
    tried = []
    candidates = [Path(override)] if override else []
    # 併進主線之後，`PROJECT` 就是主線目錄本身；分開放的時候它是 repo 根，
    # 那時這一項找不到 `src/metrics/suite.py`，會自然落到下面的名字清單。
    candidates.append(PROJECT)
    candidates += [PROJECT / name for name in SOURCE_HOME_NAMES]
    for candidate in candidates:
        tried.append(candidate)
        if (candidate / "src" / "metrics" / "suite.py").is_file():
            return candidate.resolve()
    raise SystemExit(
        "找不到主線目錄（要有 src/metrics/suite.py）。找過："
        + "、".join(str(p) for p in tried)
        + "。可用環境變數 IMMUNISATION_SOURCE_HOME 指定。")


#: 主線目錄：`src.*` 套件與資料集的所在。
SOURCE_HOME = _resolve_source_home()

#: 資料集根。原圖在 `man/`、`woman/`，遮罩在 `masks/`，指令在 `prompts.yaml`。
PORTRAITS = SOURCE_HOME / "data" / "portraits"

#: Mist 的目標圖等外部素材。
TARGETS = SOURCE_HOME / "data" / "targets"


def add_source_to_syspath() -> None:
    """讓 `from src...` 能解析到主線目錄，並讓同目錄的腳本能互相 import。"""
    import sys
    for entry in (str(SOURCE_HOME), str(Path(__file__).resolve().parent)):
        if entry not in sys.path:
            sys.path.insert(0, entry)

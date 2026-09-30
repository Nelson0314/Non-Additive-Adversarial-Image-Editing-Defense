"""baseline 專案的預設目錄；只由本套件位置推定專案根，不搜尋其他專案。

各 CLI 以這些位置作為參數預設值，呼叫端可逐一以參數覆寫。`artifacts/` 存放
可由已記錄參數與種子重新產生的影像，不入版控；`results/` 存放 CSV，入版控。
"""
from __future__ import annotations

from pathlib import Path

#: baseline 專案根（含 pyproject.toml）。
PROJECT = Path(__file__).resolve().parents[2]
if not (PROJECT / "pyproject.toml").is_file():
    raise ImportError(f"{PROJECT} 不是 baseline 專案根（缺 pyproject.toml）；"
                      "請自 baseline/src 匯入 immunization_baseline")

DATA = PROJECT / "data"
#: 資料集根：`man/`、`woman/` 原圖，`masks/` 重繪遮罩（白＝重繪），`prompts.yaml` 指令。
PORTRAITS = DATA / "portraits"
#: Mist 的目標影像等外部素材。
TARGETS = DATA / "targets"
CONFIGS = PROJECT / "configs"
RESULTS = PROJECT / "results"
ARTIFACTS = PROJECT / "artifacts"

#: 防禦圖，每條件一個子目錄（`<影像>__<條件>__def.png` 與 `results.csv`）。
DEFENSES = ARTIFACTS / "defenses"
#: 等失真對齊的防禦圖。
ALIGNED_DEFENSES = ARTIFACTS / "aligned" / "defenses"
#: 未防禦的編輯（各 arm 子目錄與 `preflight.csv`）。
UNDEFENDED_EDITS = ARTIFACTS / "undefended_edits"
#: 防禦後的編輯，每條件一個子目錄。
DEFENDED_EDITS = ARTIFACTS / "defended_edits"
#: 淨化後的防禦圖，`<條件>/<淨化>/`。
PURIFIED = ARTIFACTS / "purified"
#: 淨化後重新編輯的結果，`<條件>/<淨化>/<arm>/`。
PURIFIED_EDITS = ARTIFACTS / "purified_edits"
FLUX_EDITS = ARTIFACTS / "flux" / "edits"
ULTRAEDIT_EDITS = ARTIFACTS / "ultraedit" / "edits"
#: 編輯器參數掃描的影像。
SWEEPS = ARTIFACTS / "sweeps"

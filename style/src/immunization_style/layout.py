"""style 專案的預設目錄；只由本套件位置推定專案根，不搜尋其他專案。

`artifacts/` 存放可由已記錄參數與種子重新產生的影像，`runtime/` 存放排程狀態，兩者不入版控；
`results/` 存放 CSV，入版控。
"""
from __future__ import annotations

from pathlib import Path

#: style 專案根（含 pyproject.toml）。
PROJECT = Path(__file__).resolve().parents[2]
if not (PROJECT / "pyproject.toml").is_file():
    raise ImportError(f"{PROJECT} 不是 style 專案根（缺 pyproject.toml）；"
                      "請自 style/src 匯入 immunization_style")

DATA = PROJECT / "data"
#: 資料集根：`man/`、`woman/` 原圖，`masks/` 重繪遮罩（白＝重繪），`prompts.yaml` 指令。
PORTRAITS = DATA / "portraits"
RESULTS = PROJECT / "results"
ARTIFACTS = PROJECT / "artifacts"
RUNTIME = PROJECT / "runtime"

#: 各輪的防禦圖與逐步紀錄，`<輪名>/<工作>/`。
DEFENSES = ARTIFACTS / "defenses"
#: 各輪防禦圖的編輯，`<輪名>/<工作>_<風格>/` 與讀數 `readout_<風格>.csv`。
EDITS = ARTIFACTS / "edits"
#: 未防禦的編輯（讀數的分母 arm `ip2p_si18`）。
UNDEFENDED_EDITS = ARTIFACTS / "undefended_edits"
LOGS = RUNTIME / "logs"

"""color 專案的預設目錄；只由本套件位置推定專案根，不搜尋其他專案。

`artifacts/` 存放可由已記錄參數與種子重新產生的影像，`runtime/` 存放排程狀態，兩者不入版控；
`results/` 存放 CSV，入版控。
"""
from __future__ import annotations

from pathlib import Path

#: color 專案根（含 pyproject.toml）。
PROJECT = Path(__file__).resolve().parents[2]
if not (PROJECT / "pyproject.toml").is_file():
    raise ImportError(f"{PROJECT} 不是 color 專案根（缺 pyproject.toml）；"
                      "請自 color/src 匯入 immunization_color")

DATA = PROJECT / "data"
#: 資料集根：`man/`、`woman/` 原圖，`masks/` 重繪遮罩（白＝重繪），`prompts.yaml` 指令。
PORTRAITS = DATA / "portraits"
RESULTS = PROJECT / "results"
ARTIFACTS = PROJECT / "artifacts"
RUNTIME = PROJECT / "runtime"

#: 防禦圖，每個條件一個子目錄。
DEFENSES = ARTIFACTS / "defenses"
#: 單張防禦圖的分片輸出，`<條件>/<影像>/`。
DEFENSE_SHARDS = ARTIFACTS / "defense_shards"
#: 短步數的單張試跑，`<條件>/<影像>/`。
DEFENSE_PILOTS = ARTIFACTS / "defense_pilots"
#: 未防禦的編輯（對照 arm 與 `preflight.csv`）。
UNDEFENDED_EDITS = ARTIFACTS / "undefended_edits"
DEFENDED_EDITS = ARTIFACTS / "defended_edits"
PURIFIED = ARTIFACTS / "purified"
PURIFIED_EDITS = ARTIFACTS / "purified_edits"

#: 各階段完成標記。
STATE = RUNTIME / "state"
#: 佇列狀態，`<佇列名>/`。
QUEUES = RUNTIME / "queues"
LOGS = RUNTIME / "logs"

"""校準表 — `docs/reference/CODE_CONTRACTS.md` §1.1。

## 存在理由

先驗實驗十次事後診斷都是同一句話：

> 一個為某個對象校準的值，被沿用到另一個對象上，而且沒有症狀。

共通成因是「那個值為誰校準」只寫在註解裡、程式讀不到。實測的代價：
位移場與加性像素殘差的學習率校準值相差 **12.5 倍**，而沿用不會有任何症狀
——優化跑得完、曲線看起來正常、輸出是一張圖，只是量到的不是要量的東西。

結構性修法是把「為誰校準」寫成程式讀得到的表，**未校準的對象直接拋出**。

## 三條不可放寬的規則

1. **沒有回退路徑。** 沒有預設值、沒有內插、沒有「最接近的一組」。
   任何一種回退都會把「這個值不適用」變成一個安靜的近似。
2. **context 必須完全相等**，不是子集比對。若 `calibrated_for` 有而 context
   沒有的鍵可以被忽略，那麼「校準時多記了一個變因」就會變成「比對時少檢查
   一個變因」——恰好是本缺陷的成因。
3. **`gpu` 與 `precision` 是必填**。V100 走 fp16、RTX 5090 走 bf16，
   數值路徑不同。混跑會產生無法歸因的變因（`ARCH` §7.2）。

例外訊息必須同時列出「要求的 context」與「表中記錄的 calibrated_for」。
只說「不符」的話，使用者看到例外仍然不知道差在哪，實務上會直接把檢查關掉。
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional

# 每次查詢都必須提供的 context 欄位。
REQUIRED_CONTEXT = ("model", "resolution", "guidance", "steps", "gpu", "precision")


class CalibrationMismatch(RuntimeError):
    """context 與表中記錄的 calibrated_for 不符，或該鍵根本未校準。"""


class Calibration:
    """校準表。以 JSON 落盤——`.yaml` 會被 `.gitignore` 的 `runs/` 區塊靜默排除。

    該區塊是「全部排除 + 逐副檔名放行」，放行清單為
    `.csv .json .md .log .png .html .txt .pt`，不含 `.yaml`。
    實測 `git check-ignore -v runs/b1/calib/calibration.yaml` 命中
    `.gitignore:16:runs/**`。歷史上已因同一機制靜默漏掉 273 個檔案。
    """

    def __init__(self, entries: Optional[Dict[str, Dict[str, Any]]] = None):
        # entries[key] = {"value": ..., "calibrated_for": {...}, "note": "..."}
        self.entries: Dict[str, Dict[str, Any]] = dict(entries or {})

    # ---- 落盤 ----

    @classmethod
    def load(cls, path: Path) -> "Calibration":
        path = Path(path)
        if not path.exists():
            raise CalibrationMismatch(
                f"校準表不存在：{path}。請先執行段 0（scripts/run_stage.py calib）。"
                "不會回退到預設值——未校準的值沿用正是本專案重複十次的缺陷。"
            )
        return cls(json.loads(path.read_text(encoding="utf-8")))

    def save(self, path: Path) -> None:
        path = Path(path)
        if path.suffix != ".json":
            raise ValueError(
                f"校準表必須是 .json，收到 {path.suffix!r}。"
                "`.gitignore` 的 runs/ 區塊只放行特定副檔名，其他會被靜默排除。"
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.entries, indent=2, ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )

    # ---- 查詢 ----

    @staticmethod
    def _check_context(context: Dict[str, Any]) -> None:
        missing = [k for k in REQUIRED_CONTEXT if k not in context]
        if missing:
            raise CalibrationMismatch(
                f"context 缺少必填欄位 {missing}。"
                f"必填為 {list(REQUIRED_CONTEXT)}，目前提供 {sorted(context)}。"
                "gpu 與 precision 必填的理由：兩張卡的數值路徑不同，"
                "混跑會產生無法歸因的變因。"
            )

    def get(self, key: str, context: Dict[str, Any]) -> Any:
        """回傳該鍵的校準值。未校準或 context 不符一律拋出。"""
        self._check_context(context)
        if key not in self.entries:
            raise CalibrationMismatch(
                f"鍵 {key!r} 未校準。表中已有的鍵：{sorted(self.entries)}。"
                "不會回退到預設值，也不會由鄰近的鍵內插。"
            )
        entry = self.entries[key]
        recorded = entry.get("calibrated_for", {})
        if dict(recorded) != dict(context):
            raise CalibrationMismatch(
                f"鍵 {key!r} 的校準對象不符。\n"
                f"  要求的 context     ：{json.dumps(context, ensure_ascii=False, sort_keys=True)}\n"
                f"  表中 calibrated_for：{json.dumps(recorded, ensure_ascii=False, sort_keys=True)}\n"
                f"  差異：{self._diff(context, recorded)}\n"
                "處置是重跑段 0 校準，不是放寬比對。"
            )
        return entry["value"]

    def put(self, key: str, value: Any, context: Dict[str, Any],
            note: str = "") -> None:
        self._check_context(context)
        self.entries[key] = {
            "value": value,
            "calibrated_for": dict(context),
            "note": note,
        }

    def has(self, key: str, context: Dict[str, Any]) -> bool:
        """供儀表板與續跑判定使用，不供優化流程使用。

        流程一律走 `get()` 讓它拋出——用 `has()` 包一層 if 再走別的路徑，
        就是回退路徑，本模組存在的目的正是消滅它。
        """
        try:
            self.get(key, context)
            return True
        except CalibrationMismatch:
            return False

    @staticmethod
    def _diff(a: Dict[str, Any], b: Dict[str, Any]) -> str:
        keys = sorted(set(a) | set(b))
        parts = [f"{k}: {a.get(k, '<缺>')!r} vs {b.get(k, '<缺>')!r}"
                 for k in keys if a.get(k) != b.get(k)]
        return "；".join(parts) if parts else "（無，應為鍵集合不同）"

"""進度儀表板的寫入端 — `docs/reference/CODE_CONTRACTS.md` §2。

## 真相與快取

| 層級 | 內容 |
|---|---|
| **真相** | `runs/<批次>/_cells/*.json`，每格一份，含 `config_hash` |
| **快取** | `runs/<批次>/progress.json` |

`progress.json` 是快取不是真相。它遺失或損壞時以 `rebuild()` 掃描 `_cells/`
重建。這條規則使「進度檔寫壞導致整批重跑」不可能發生——那會直接損失
不可重跑的實驗（雲端容器會被刪除，`runs/` 是唯一證據來源）。

## 為什麼是 os.replace 而不是直接寫

儀表板是給另一個 session 輪詢用的。若直接對 `progress.json` 寫入，
讀取端有機會讀到寫到一半的檔案，`json.loads` 會拋出。
先寫 `progress.json.tmp` 再 `os.replace()`：該操作在同一檔案系統上是原子的，
讀取端永遠看到完整的舊版或完整的新版。

## Windows 的額外條件（實測）

`os.replace` 的**原子性**在兩個平台都成立，但**可用性**不同：

- POSIX 的 `rename(2)` 換的是目錄項指向的 inode，讀取端持有的舊 handle
  仍指向舊 inode，故 rename 一定成功。
- **Windows 在目標檔案正被另一個行程開啟時會失敗**，回報
  `PermissionError: [WinError 5]`。成因是 Python 的 `open()` 不帶
  `FILE_SHARE_DELETE`，而 rename 需要對目標取得刪除權。

`tests/test_progress.py::test_寫入期間讀取永遠可解析` 在 Windows 上實測到此失敗。
處置是 `_replace_atomic()` 的**有界重試**——這不是掩蓋症狀：根因已確認為
Windows 的檔案共用語意，衝突窗口是讀取端持有 handle 的那幾微秒，
且重試次數用盡仍會拋出，不會靜默吞掉。讀取端「永不看到半份 JSON」這個
不變量由 replace 的原子性保證，不受重試影響。

生產環境（TWCC 容器、RTX 5090 主機）是 Linux，走的是不需重試的路徑；
重試只在本機 Windows 檢視時生效。

## 五態而非四態

`pending / running / done / failed / skipped`。

`skipped` 專供結構上不適用的格——例如 APA 移植在 τ = 0.05、0.10 上因
VAE 重建誤差下限（LPIPS 0.1434）而不可能達成。把不適用算成失敗，
儀表板會永遠是紅的，然後就沒有人看它了。
"""

import json
import os
import socket
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

SCHEMA = 1
STATES = ("pending", "running", "done", "failed", "skipped")
# 進度快取與逐格紀錄的檔名。兩者都是 .json——`.gitignore` 的 runs/ 區塊
# 只放行特定副檔名，換副檔名會被靜默排除。
PROGRESS_NAME = "progress.json"
CELLS_DIR = "_cells"
LOCK_NAME = ".writer.lock"


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _cell_file(batch_dir: Path, cell_id: str) -> Path:
    """`train/N1/pie_0007` → `_cells/train__N1__pie_0007.json`。

    刻意攤平而不建巢狀目錄：逐格紀錄與產物樹是兩件事，
    綁在一起會讓「產物路徑改了」連帶弄丟進度。
    """
    return batch_dir / CELLS_DIR / (cell_id.replace("/", "__") + ".json")


                                    # Windows 上讀取端持有 handle 時 rename 會失敗，
                                    # 衝突窗口是微秒級。20 次 × 5 ms 上限 100 ms，
                                    # 遠長於該窗口；用盡仍失敗代表另有原因，必須拋出。
_REPLACE_RETRIES = 20
_REPLACE_BACKOFF_S = 0.005


def _replace_atomic(tmp: Path, path: Path) -> None:
    """os.replace，並處理 Windows 在讀取端持有 handle 時的失敗（見模組 docstring）。

    只重試 `PermissionError`。其他例外（磁碟滿、路徑錯、跨檔案系統）
    立即拋出——那些重試再多次也不會成功，重試只會延後症狀出現的時間。
    """
    for attempt in range(_REPLACE_RETRIES):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if attempt == _REPLACE_RETRIES - 1:
                raise
            time.sleep(_REPLACE_BACKOFF_S)


def write_atomic(path: Path, text: str) -> None:
    """先寫同目錄的暫存檔再 replace。跨檔案系統不保證原子，故不可改用 /tmp。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    _replace_atomic(tmp, path)


class WriterLockHeld(RuntimeError):
    """已有另一個寫入者。本專案的硬規則是 GPU 工作不可並行。"""


class ProgressWriter:
    """單一寫入者。讀取端一律唯讀，見 `scripts/dashboard.py`。

    單一寫入者不是本模組加上的限制，而是既有事實的落實：實測兩個 GPU 工作
    互搶會把單張 SDEdit 由 222 s 拉長到 30 分鐘以上，故本專案本來就規定
    GPU 工作不可並行。
    """

    def __init__(self, batch_dir, env: Optional[Dict[str, Any]] = None,
                 take_lock: bool = True):
        self.batch_dir = Path(batch_dir)
        self.batch_dir.mkdir(parents=True, exist_ok=True)
        (self.batch_dir / CELLS_DIR).mkdir(exist_ok=True)
        self.path = self.batch_dir / PROGRESS_NAME
        self.lock_path = self.batch_dir / LOCK_NAME
        self._locked = False
        if take_lock:
            self._acquire()
        self.env = dict(env or {})
        self.created = _now()
        self.flush()

    # ---- 鎖 ----

    def _acquire(self) -> None:
        if self.lock_path.exists():
            raise WriterLockHeld(
                f"{self.lock_path} 已存在：{self.lock_path.read_text(encoding='utf-8')}\n"
                "同一批次只允許一個寫入者。若確認前一個行程已結束，手動刪除該檔。"
                "不自動判定行程是否存活——誤判會讓兩個寫入者同時動同一批資料。"
            )
        self.lock_path.write_text(
            json.dumps({"pid": os.getpid(), "host": socket.gethostname(),
                        "since": _now()}, ensure_ascii=False),
            encoding="utf-8",
        )
        self._locked = True

    def release(self) -> None:
        if self._locked and self.lock_path.exists():
            self.lock_path.unlink()
        self._locked = False

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.release()
        return False

    # ---- 逐格 ----

    def _save_cell(self, cell: Dict[str, Any]) -> None:
        write_atomic(_cell_file(self.batch_dir, cell["id"]),
                     json.dumps(cell, indent=2, ensure_ascii=False, sort_keys=True))

    def _load_cell(self, cell_id: str) -> Optional[Dict[str, Any]]:
        f = _cell_file(self.batch_dir, cell_id)
        if not f.exists():
            return None
        return json.loads(f.read_text(encoding="utf-8"))

    def begin(self, cell_id: str, meta: Dict[str, Any]) -> None:
        cell = {
            "id": cell_id,
            "stage": cell_id.split("/")[0],
            "status": "running",
            "started": _now(),
            "finished": None,
            "seconds": None,
            "error": None,
            "artifacts": [],
            **meta,
        }
        self._save_cell(cell)
        self.flush()

    def _close(self, cell_id: str, status: str, **fields) -> None:
        cell = self._load_cell(cell_id)
        if cell is None:
            raise KeyError(f"格 {cell_id!r} 沒有 begin() 過，不能直接收尾")
        cell.update(status=status, finished=_now(), **fields)
        self._save_cell(cell)
        self.flush()

    def finish(self, cell_id: str, seconds: float,
               artifacts: Optional[List[str]] = None) -> None:
        self._close(cell_id, "done", seconds=float(seconds),
                    artifacts=list(artifacts or []), error=None)

    def fail(self, cell_id: str, error: str) -> None:
        self._close(cell_id, "failed", error=str(error))

    def skip(self, cell_id: str, reason: str) -> None:
        """結構上不適用。與 failed 分開，理由見模組 docstring。"""
        self._close(cell_id, "skipped", error=None, skipped_reason=str(reason))

    # ---- 續跑判定 ----

    def is_done(self, cell_id: str, config_hash: str) -> bool:
        """完成的定義是「狀態為 done **且** 雜湊相符 **且** 產物都還在」。

        只看狀態會在改了設定卻沒改路徑時沿用舊結果；
        只看雜湊會在產物被清掉後誤判為完成。三者都要。
        """
        cell = self._load_cell(cell_id)
        if cell is None or cell.get("status") != "done":
            return False
        if cell.get("config_hash") != config_hash:
            return False
        return all((self.batch_dir / a).exists() for a in cell.get("artifacts", []))

    # ---- 快取 ----

    def flush(self) -> None:
        write_atomic(self.path, json.dumps(self.snapshot(), indent=2,
                                           ensure_ascii=False))

    def snapshot(self) -> Dict[str, Any]:
        cells = load_cells(self.batch_dir)
        return build_snapshot(cells, env=self.env, created=self.created,
                              batch=self.batch_dir.name)


# ---- 讀取端共用 ----

def read_progress(batch_dir, retries: int = _REPLACE_RETRIES) -> Dict[str, Any]:
    """讀取 `progress.json`。讀取端一律用本函式，不要裸用 `read_text()`。

    Windows 的限制是雙向的：寫入端 replace 時，讀取端的 `open()` 會拿到
    `PermissionError`（實測，見 `tests/test_progress.py`）。窗口同樣是微秒級。

    這裡重試的是**開檔失敗**，不是解析失敗。`JSONDecodeError` 直接拋出——
    那代表檔案真的壞了，重試只會延後症狀出現的時間，而 `rebuild()` 才是
    對應的處置。
    """
    path = Path(batch_dir) / PROGRESS_NAME
    for attempt in range(retries):
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except PermissionError:
            if attempt == retries - 1:
                raise
            time.sleep(_REPLACE_BACKOFF_S)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"{path} 不存在。若批次已跑過，用 rebuild() 由 _cells/ 重建。"
            ) from None


def load_cells(batch_dir) -> List[Dict[str, Any]]:
    d = Path(batch_dir) / CELLS_DIR
    if not d.exists():
        return []
    return sorted(
        (json.loads(f.read_text(encoding="utf-8")) for f in d.glob("*.json")),
        key=lambda c: c["id"],
    )


def build_snapshot(cells: List[Dict[str, Any]], env: Optional[Dict] = None,
                   created: Optional[str] = None,
                   batch: str = "") -> Dict[str, Any]:
    stages: Dict[str, Dict[str, int]] = {}
    for c in cells:
        st = stages.setdefault(c.get("stage", "?"),
                               {k: 0 for k in STATES} | {"total": 0})
        st["total"] += 1
        st[c.get("status", "pending")] = st.get(c.get("status", "pending"), 0) + 1

    for name, st in stages.items():
        done_secs = [c["seconds"] for c in cells
                     if c.get("stage") == name and c.get("status") == "done"
                     and c.get("seconds")]
        remaining = st["pending"] + st["running"]
        # 中位數而非平均：先驗實驗已見過同一設定在不同影像上耗時差數倍
        # （一張 200 步仍在下降、另一張第 45 步就停住），平均會被離群值帶偏。
        st["median_seconds"] = statistics.median(done_secs) if done_secs else None
        st["eta_seconds"] = (st["median_seconds"] * remaining
                             if done_secs and remaining else None)
        st["status"] = ("done" if st["total"] and remaining == 0
                        else "running" if st["running"] else "pending")

    summary = {k: sum(s.get(k, 0) for s in stages.values()) for k in STATES}
    summary["total"] = sum(s["total"] for s in stages.values())
    etas = [s["eta_seconds"] for s in stages.values() if s["eta_seconds"]]
    summary["eta_seconds"] = sum(etas) if etas else None
    summary["elapsed_seconds"] = sum(
        c["seconds"] for c in cells if c.get("seconds"))

    return {
        "schema": SCHEMA,
        "batch": batch,
        "created": created,
        "updated": _now(),
        "env": dict(env or {}),
        "stages": stages,
        "cells": cells,
        "summary": summary,
    }


def rebuild(batch_dir) -> Dict[str, Any]:
    """忽略 progress.json，由 `_cells/` 重建。回傳快照且**不寫入**。

    不寫入是刻意的：重建由唯讀的儀表板呼叫，而寫入是 ProgressWriter 的職責。
    讓讀取端有寫入能力，就等於允許兩個寫入者。
    """
    batch_dir = Path(batch_dir)
    return build_snapshot(load_cells(batch_dir), batch=batch_dir.name)

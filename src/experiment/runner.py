"""段落執行器 — `docs/ARCH_2026-08-05.md` §5、`docs/RUNBOOK_2026-08-05.md` §3。

## 為什麼骨架與計算分離

本模組只負責**哪些格子要跑、跑過了沒有、進度怎麼記**；實際的訓練與評測由
呼叫端傳入的 `executor` 執行。分離的理由不是美學：

- 骨架的錯誤（漏跑一格、續跑判定失效、進度寫壞）**事後看不出來**，
  產出的表格仍然完整，故它必須有測試；
- 而實際計算需要 SDXL 權重與 GPU，測試不可能涵蓋。

把兩者綁在一起，等於讓可測的部分因為不可測的部分而失去測試。

## 續跑的判定

「已完成」需**三者同時成立**：狀態為 `done`、`config_hash` 相符、產物都還在。

少任何一項都會靜默沿用舊結果。只看狀態，改了設定卻沒改路徑時會沿用；
只看雜湊，產物被清掉後會誤判為完成。這是本專案重複十次的缺陷型態
（`HANDOFF` §4.5），故判定寫在一處、由測試釘住。

## 失敗的處置

單一格失敗**不中止整段**——4449 格跑到一半因為一格拋出而全滅，代價是
數小時的機時。失敗被記錄成 `failed` 並繼續，最後由 `dashboard.py --failed`
一次看完。但**失敗數超過門檻時中止**：那代表問題是系統性的，
繼續跑只是把同一個錯誤重複 4000 次。
"""

import time
import traceback
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence

from src.experiment.grid import Cell
from src.utils.cellid import config_hash
from src.utils.progress import ProgressWriter

# 連續失敗達此數即中止。取 10 而非 1：偶發的單格失敗（例如某張影像的
# 邊界情形）不該讓整批停下；但連續 10 格失敗一定是系統性的。
CONSECUTIVE_FAILURE_LIMIT = 10


@dataclass
class StageResult:
    stage: str
    done: int = 0
    skipped: int = 0
    failed: int = 0
    resumed: int = 0
    aborted: bool = False
    abort_reason: str = ""

    @property
    def attempted(self) -> int:
        return self.done + self.failed


# executor(cell, ctx) -> (artifacts, extra_meta)
#   artifacts：相對批次目錄的路徑清單，供續跑判定檢查存在性
#   extra_meta：要寫進該格 meta 的額外欄位
Executor = Callable[[Cell, Dict[str, Any]], tuple]


def cell_config(cell: Cell, base_config: Dict[str, Any]) -> Dict[str, Any]:
    """該格的完整設定，供雜湊使用。

    `base_config` 是整批共用的部分（模型、解析度、guidance、gpu、precision…），
    格點自己的四個軸疊在上面。**必填鍵一律明寫，即使為 None**——
    `config_hash` 會檢查，缺任何一個就拋出。那是刻意的：忘記把某個變因放進
    雜湊，是「改了設定卻沿用舊結果」的成因。
    """
    purify_kind, purify_strength = (cell.purify or (None, None))
    return dict(
        base_config,
        condition=cell.condition,
        image_id=cell.image_id,
        tau=cell.tau,
        purify=(None if purify_kind is None
                else {"kind": purify_kind, "strength": purify_strength}),
        seed=cell.seed,
    )


def run_stage(
    stage: str,
    cells: Sequence[Cell],
    executor: Executor,
    writer: ProgressWriter,
    base_config: Dict[str, Any],
    ctx: Optional[Dict[str, Any]] = None,
    force: bool = False,
    verbose: bool = True,
) -> StageResult:
    """執行一段。回傳統計；失敗不拋出而是記錄。

    `force=True` 忽略續跑判定重跑全部格子。**預設為 False**：
    重跑不可重跑的實驗（雲端容器會被刪除）是無法回復的損失。
    """
    ctx = dict(ctx or {})
    res = StageResult(stage=stage)
    consecutive = 0

    for i, cell in enumerate(cells, 1):
        cid = cell.cell_id()

        # 結構上不適用的格：標記為 skipped 而非 failed，兩者分開。
        # 把不適用算成失敗，儀表板會永遠是紅的，然後就沒有人看它了。
        if cell.skipped:
            writer.begin(cid, {"config_hash": "", "skipped_reason": cell.skip_reason})
            writer.skip(cid, cell.skip_reason)
            res.skipped += 1
            continue

        cfg = cell_config(cell, base_config)
        chash = config_hash(cfg)

        if not force and writer.is_done(cid, chash):
            res.resumed += 1
            continue

        writer.begin(cid, {"config_hash": chash, "config": cfg,
                           "condition": cell.condition, "image": cell.image_id})
        t0 = time.perf_counter()
        try:
            artifacts, extra = executor(cell, ctx)
            writer.finish(cid, time.perf_counter() - t0, artifacts)
            if extra:
                # 額外欄位在 finish 之後合併，使 finish 的契約保持單純
                _merge_meta(writer, cid, extra)
            res.done += 1
            consecutive = 0
        except Exception:
            # 不吞例外也不中止：完整堆疊寫進該格，供 `dashboard.py --failed`
            # 取用。單格失敗讓整段全滅，代價是數小時的機時。
            writer.fail(cid, traceback.format_exc())
            res.failed += 1
            consecutive += 1
            if verbose:
                print(f"  [fail] {cid}", flush=True)
            if consecutive >= CONSECUTIVE_FAILURE_LIMIT:
                res.aborted = True
                res.abort_reason = (
                    f"連續 {consecutive} 格失敗，中止本段。"
                    "連續失敗代表問題是系統性的，繼續跑只是把同一個錯誤重複數千次。"
                    "用 `dashboard.py <批次> --failed` 看完整堆疊。"
                )
                if verbose:
                    print(f"  [abort] {res.abort_reason}", flush=True)
                break

        if verbose and i % 50 == 0:
            print(f"  [{stage}] {i}/{len(cells)}  done={res.done} "
                  f"failed={res.failed} resumed={res.resumed}", flush=True)

    return res


def _merge_meta(writer: ProgressWriter, cell_id: str, extra: Dict) -> None:
    """把 executor 回傳的額外欄位併入該格的紀錄。"""
    cell = writer._load_cell(cell_id)
    if cell is None:
        return
    cell.update(extra)
    writer._save_cell(cell)
    writer.flush()


def plan_report(cells_by_stage: Dict[str, List[Cell]],
                writer: ProgressWriter,
                base_config: Dict[str, Any]) -> Dict[str, Dict[str, int]]:
    """排程前的乾跑：每段有幾格要跑、幾格可續、幾格不適用。

    **在耗掉任何機時之前**回答「這次會跑多久」。沒有這一步，
    續跑判定是否生效要等跑完才知道。
    """
    out = {}
    for stage, cells in cells_by_stage.items():
        todo = resumable = skipped = 0
        for cell in cells:
            if cell.skipped:
                skipped += 1
            elif writer.is_done(cell.cell_id(),
                                config_hash(cell_config(cell, base_config))):
                resumable += 1
            else:
                todo += 1
        out[stage] = {"todo": todo, "resumable": resumable,
                      "skipped": skipped, "total": len(cells)}
    return out

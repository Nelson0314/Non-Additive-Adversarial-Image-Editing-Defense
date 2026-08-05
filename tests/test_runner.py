"""段落執行器：續跑判定、失敗處置、進度追蹤。

這些是「骨架」的測試。骨架的錯誤事後看不出來——漏跑一格、續跑判定失效、
進度寫壞，產出的表格仍然完整，只是內容不對。實際計算不在此驗（需 GPU）。
"""

import pytest

from src.experiment.grid import Cell
from src.experiment.runner import (CONSECUTIVE_FAILURE_LIMIT, cell_config,
                                   plan_report, run_stage)
from src.utils.cellid import config_hash
from src.utils.progress import ProgressWriter

BASE = {
    "spec_version": 1, "model": "SDXL-1.0-base", "resolution": 1024,
    "guidance": 7.5, "steps": 50, "strength": 0.6,
    "gpu": "V100", "precision": "fp16", "loss_params": {}, "lr": None,
}


@pytest.fixture
def writer(tmp_path):
    w = ProgressWriter(tmp_path / "b1", env={"gpu": "V100"})
    yield w
    w.release()


def _cells(n=3, stage="train"):
    return [Cell(stage, "N1", f"img{i}") for i in range(n)]


def _ok_executor(artifacts=()):
    def run(cell, ctx):
        return list(artifacts), {}
    return run


# ---------------------------------------------------------------------------
# 設定雜湊
# ---------------------------------------------------------------------------

def test_四個軸都進雜湊():
    """漏掉任何一個變因，改了它卻不會重跑。"""
    c = Cell("eval", "N1", "img0", tau=0.20, purify=("jpeg", 30), seed=1)
    h = config_hash(cell_config(c, BASE))
    for other in (
        Cell("eval", "N2", "img0", tau=0.20, purify=("jpeg", 30), seed=1),
        Cell("eval", "N1", "img9", tau=0.20, purify=("jpeg", 30), seed=1),
        Cell("eval", "N1", "img0", tau=0.35, purify=("jpeg", 30), seed=1),
        Cell("eval", "N1", "img0", tau=0.20, purify=("jpeg", 75), seed=1),
        Cell("eval", "N1", "img0", tau=0.20, purify=("blur", 30), seed=1),
        Cell("eval", "N1", "img0", tau=0.20, purify=("jpeg", 30), seed=2),
    ):
        assert config_hash(cell_config(other, BASE)) != h


def test_換卡改變全部格點的雜湊():
    """兩張卡的數值路徑不同（V100 fp16、5090 bf16），混跑會產生
    無法歸因的變因。換卡自動使全部格點視為未完成。"""
    c = Cell("train", "N1", "img0")
    a = config_hash(cell_config(c, BASE))
    b = config_hash(cell_config(c, BASE | {"gpu": "RTX 5090",
                                           "precision": "bf16"}))
    assert a != b


def test_無淨化的格其purify為None而非空字串():
    """`{"purify": None}` 與省略等價，兩者不該產生兩個目錄。"""
    cfg = cell_config(Cell("train", "N1", "img0"), BASE)
    assert cfg["purify"] is None and cfg["tau"] is None and cfg["seed"] is None


# ---------------------------------------------------------------------------
# 續跑
# ---------------------------------------------------------------------------

def test_已完成的格不重跑(writer):
    cells = _cells()
    calls = []

    def counting(cell, ctx):
        calls.append(cell.cell_id())
        return [], {}

    first = run_stage("train", cells, counting, writer, BASE)
    assert first.done == 3 and first.resumed == 0
    second = run_stage("train", cells, counting, writer, BASE)
    assert second.done == 0 and second.resumed == 3
    assert len(calls) == 3, "第二次不該再呼叫 executor"


def test_改了設定就重跑(writer):
    """這正是「改了設定卻沒改路徑」的防線。"""
    cells = _cells(1)
    run_stage("train", cells, _ok_executor(), writer, BASE)
    again = run_stage("train", cells, _ok_executor(), writer,
                      BASE | {"guidance": 1.0})
    assert again.done == 1 and again.resumed == 0


def test_產物被清掉就重跑(writer, tmp_path):
    batch = tmp_path / "b1"
    (batch / "out").mkdir(parents=True)
    (batch / "out" / "phi.pt").write_text("x", encoding="utf-8")

    cells = _cells(1)
    run_stage("train", cells, _ok_executor(["out/phi.pt"]), writer, BASE)
    assert run_stage("train", cells, _ok_executor(["out/phi.pt"]),
                     writer, BASE).resumed == 1

    (batch / "out" / "phi.pt").unlink()
    assert run_stage("train", cells, _ok_executor(["out/phi.pt"]),
                     writer, BASE).done == 1, "產物不在就必須重跑"


def test_force忽略續跑判定(writer):
    cells = _cells(2)
    run_stage("train", cells, _ok_executor(), writer, BASE)
    forced = run_stage("train", cells, _ok_executor(), writer, BASE, force=True)
    assert forced.done == 2 and forced.resumed == 0


def test_force預設為關(writer):
    """重跑不可重跑的實驗是無法回復的損失。"""
    import inspect

    assert inspect.signature(run_stage).parameters["force"].default is False


# ---------------------------------------------------------------------------
# 不適用與失敗
# ---------------------------------------------------------------------------

def test_不適用的格不呼叫executor也不算失敗(writer):
    cells = [Cell("rayscale", "N3", "img0", tau=0.05,
                  skip_reason="低於 VAE 重建下限")]
    called = []
    res = run_stage("rayscale", cells,
                    lambda c, ctx: (called.append(1), ([], {}))[1],
                    writer, BASE)
    assert res.skipped == 1 and res.failed == 0 and res.done == 0
    assert not called


def test_單格失敗不中止整段(writer):
    """4449 格跑到一半因一格拋出而全滅，代價是數小時機時。"""
    cells = _cells(5)

    def flaky(cell, ctx):
        if cell.image_id == "img2":
            raise RuntimeError("模擬失敗")
        return [], {}

    res = run_stage("train", cells, flaky, writer, BASE)
    assert res.done == 4 and res.failed == 1 and not res.aborted


def test_失敗記錄完整堆疊(writer):
    def boom(cell, ctx):
        raise ValueError("這是根本原因")

    run_stage("train", _cells(1), boom, writer, BASE)
    cell = writer.snapshot()["cells"][0]
    assert cell["status"] == "failed"
    assert "這是根本原因" in cell["error"]
    assert "Traceback" in cell["error"], "只有訊息不夠，要能定位"


def test_連續失敗達門檻即中止(writer):
    """連續失敗代表問題是系統性的，繼續跑只是把同一個錯誤重複數千次。"""
    cells = _cells(CONSECUTIVE_FAILURE_LIMIT + 20)
    res = run_stage("train", cells,
                    lambda c, ctx: (_ for _ in ()).throw(RuntimeError("x")),
                    writer, BASE, verbose=False)
    assert res.aborted
    assert res.failed == CONSECUTIVE_FAILURE_LIMIT
    assert "系統性" in res.abort_reason


def test_成功會重置連續失敗計數(writer):
    """偶發的單格失敗不該累積成中止。"""
    cells = _cells(CONSECUTIVE_FAILURE_LIMIT + 5)

    def alternating(cell, ctx):
        if int(cell.image_id[3:]) % 2 == 0:
            raise RuntimeError("x")
        return [], {}

    res = run_stage("train", cells, alternating, writer, BASE, verbose=False)
    assert not res.aborted, "交錯失敗不應觸發中止"


# ---------------------------------------------------------------------------
# 乾跑
# ---------------------------------------------------------------------------

def test_乾跑在耗掉機時之前回答會跑多久(writer):
    cells = {"train": _cells(3), "eval": _cells(2, "eval")}
    before = plan_report(cells, writer, BASE)
    assert before["train"] == {"todo": 3, "resumable": 0, "skipped": 0,
                               "total": 3}

    run_stage("train", cells["train"], _ok_executor(), writer, BASE)
    after = plan_report(cells, writer, BASE)
    assert after["train"]["resumable"] == 3 and after["train"]["todo"] == 0
    assert after["eval"]["todo"] == 2, "另一段不受影響"


def test_乾跑不寫入任何格(writer):
    cells = {"train": _cells(3)}
    plan_report(cells, writer, BASE)
    assert writer.snapshot()["summary"]["total"] == 0


def test_乾跑把不適用的格分開計(writer):
    cells = {"rayscale": [
        Cell("rayscale", "N3", "img0", tau=0.05, skip_reason="下限"),
        Cell("rayscale", "N1", "img0", tau=0.05),
    ]}
    r = plan_report(cells, writer, BASE)["rayscale"]
    assert r == {"todo": 1, "resumable": 0, "skipped": 1, "total": 2}


# ---------------------------------------------------------------------------
# 額外欄位
# ---------------------------------------------------------------------------

def test_executor可回傳額外欄位(writer):
    """例如實際步數、停止原因、耗時分解——報表需要它們。"""
    def with_meta(cell, ctx):
        return [], {"steps_used": 42, "stop_reason": "plateau"}

    run_stage("train", _cells(1), with_meta, writer, BASE)
    cell = writer.snapshot()["cells"][0]
    assert cell["steps_used"] == 42 and cell["stop_reason"] == "plateau"

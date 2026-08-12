"""儀表板：唯讀、輸出穩定、失敗看得見。

儀表板是實驗 agent 唯一的監察入口。它若把 `skipped` 算成失敗，
或在 `progress.json` 遺失時報錯而非重建，agent 就會做出錯誤處置。
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from legacy.scripts import dashboard  # noqa: E402
from src.utils.progress import ProgressWriter  # noqa: E402

META = {"config_hash": "abc123456789"}


@pytest.fixture
def batch(tmp_path):
    d = tmp_path / "b1"
    w = ProgressWriter(d, env={"gpu": "V100", "precision": "fp16",
                               "commit": "8e0ffbc"})
    w.begin("train/N1/img0", META)
    w.finish("train/N1/img0", seconds=120.0, artifacts=[])
    w.begin("train/N2/img0", META)
    w.fail("train/N2/img0", "CalibrationMismatch: lr.N2 不符\n  差異：gpu")
    w.begin("rayscale/N3/img0", META)
    w.skip("rayscale/N3/img0", "低於 VAE 重建下限 LPIPS 0.1434")
    w.begin("eval/N1/img0", META)
    w.release()
    return d


def test_json輸出為單行且可解析(batch):
    snap = dashboard.load(batch, False)
    line = dashboard.render_json(snap)
    assert "\n" not in line
    data = json.loads(line)
    assert data["summary"]["done"] == 1
    assert data["summary"]["failed"] == 1
    assert data["summary"]["skipped"] == 1
    assert data["summary"]["running"] == 1


def test_skipped不算失敗(batch):
    """N3 在低 τ 上不適用是設計預期。算成失敗會讓儀表板永遠是紅的。"""
    data = json.loads(dashboard.render_json(dashboard.load(batch, False)))
    assert data["stages"]["rayscale"]["skipped"] == 1
    assert data["stages"]["rayscale"]["failed"] == 0


def test_終端輸出含失敗提示(batch):
    out = dashboard.render_text(dashboard.load(batch, False))
    assert "FAILED" in out and "train/N2/img0" in out
    assert "不要自行重跑" in out


def test_failed模式給出錯誤全文(batch):
    """只給第一行的話，CalibrationMismatch 的雙方內容就看不到了。"""
    out = dashboard.render_failed(dashboard.load(batch, False))
    assert "CalibrationMismatch" in out and "差異" in out
    assert "abc123456789" in out


def test_沒有失敗時failed模式明講(tmp_path):
    d = tmp_path / "b2"
    ProgressWriter(d, env={}).release()
    assert "沒有失敗" in dashboard.render_failed(dashboard.load(d, False))


def test_progress遺失時自動由cells重建(batch):
    """progress.json 是快取不是真相。遺失不該讓監察中斷。"""
    (batch / "progress.json").unlink()
    snap = dashboard.load(batch, False)
    assert snap["summary"]["done"] == 1


def test_rebuild旗標與快取結果一致(batch):
    a = dashboard.load(batch, False)
    b = dashboard.load(batch, True)
    assert a["cells"] == b["cells"]


def test_段落依流程順序排列(batch):
    out = dashboard.render_text(dashboard.load(batch, False))
    assert out.index("train") < out.index("rayscale") < out.index("eval")


def test_html自足且不含外部資源(batch):
    html = dashboard.render_html(dashboard.load(batch, False))
    assert "http://" not in html and "https://" not in html
    assert "train/N2/img0" in html


def test_有失敗時離開碼非零(batch, capsys):
    assert dashboard.main([str(batch), "--json"]) == 1


def test_全部完成時離開碼為零(tmp_path, capsys):
    d = tmp_path / "b3"
    w = ProgressWriter(d, env={})
    w.begin("train/N1/img0", META)
    w.finish("train/N1/img0", seconds=1.0)
    w.release()
    assert dashboard.main([str(d), "--json"]) == 0


def test_批次目錄不存在時回報而非崩潰(tmp_path, capsys):
    assert dashboard.main([str(tmp_path / "nope"), "--json"]) == 2


def test_儀表板不寫入批次目錄(batch):
    """唯讀是硬性要求：讀取端有寫入能力就等於允許兩個寫入者。
    唯一的例外是 --html 明確要求產生的 dashboard.html。"""
    before = {p.name: p.stat().st_mtime_ns for p in batch.rglob("*")}
    dashboard.main([str(batch), "--json"])
    after = {p.name: p.stat().st_mtime_ns for p in batch.rglob("*")}
    assert before == after


def test_html旗標只新增dashboard_html(batch):
    before = {p.relative_to(batch) for p in batch.rglob("*")}
    dashboard.main([str(batch), "--html", "--json"])
    after = {p.relative_to(batch) for p in batch.rglob("*")}
    assert after - before == {Path("dashboard.html")}

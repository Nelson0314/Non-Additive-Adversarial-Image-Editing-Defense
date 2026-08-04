"""進度儀表板的寫入端。

三件事要釘住：讀取端永遠讀得到完整 JSON、進度檔可由逐格紀錄重建、
以及 `skipped` 不是 `failed`。
"""

import json
import threading
import time

import pytest

from src.utils.progress import (ProgressWriter, WriterLockHeld, rebuild)

META = {"config_hash": "abc123456789", "condition": "N1", "image": "pie_0007"}


def _writer(tmp_path, **kw):
    return ProgressWriter(tmp_path / "b1", env={"gpu": "V100", "precision": "fp16"},
                          **kw)


def test_一格的生命週期(tmp_path):
    with _writer(tmp_path) as w:
        w.begin("train/N1/pie_0007", META)
        assert w.snapshot()["summary"]["running"] == 1
        w.finish("train/N1/pie_0007", seconds=12.5, artifacts=[])
        s = w.snapshot()
    assert s["summary"]["done"] == 1
    assert s["stages"]["train"]["median_seconds"] == 12.5


def test_skipped不計入failed(tmp_path):
    """N3 在低 τ 上因 VAE 重建下限不可能達成，那是不適用不是失敗。
    把不適用算成失敗，儀表板會永遠是紅的，然後就沒有人看它了。"""
    with _writer(tmp_path) as w:
        w.begin("rayscale/N3/pie_0007", META)
        w.skip("rayscale/N3/pie_0007", "低於 VAE 重建下限 LPIPS 0.1434")
        s = w.snapshot()
    assert s["summary"]["skipped"] == 1
    assert s["summary"]["failed"] == 0


def test_失敗記錄錯誤全文(tmp_path):
    with _writer(tmp_path) as w:
        w.begin("train/N2/pie_0031", META)
        w.fail("train/N2/pie_0031", "CalibrationMismatch: lr.N2 不符")
        cell = w.snapshot()["cells"][0]
    assert cell["status"] == "failed"
    assert "CalibrationMismatch" in cell["error"]


def test_沒有begin就收尾會拋出(tmp_path):
    with _writer(tmp_path) as w:
        with pytest.raises(KeyError):
            w.finish("train/N1/nope", seconds=1.0)


def test_續跑判定要同時滿足三件事(tmp_path, monkeypatch):
    """狀態為 done、雜湊相符、產物都還在。缺一不可。"""
    batch = tmp_path / "b1"
    with _writer(tmp_path) as w:
        (batch / "N1").mkdir(parents=True)
        (batch / "N1" / "phi.pt").write_text("x", encoding="utf-8")
        w.begin("train/N1/pie_0007", META)
        w.finish("train/N1/pie_0007", 1.0, artifacts=["N1/phi.pt"])

        assert w.is_done("train/N1/pie_0007", "abc123456789") is True
        # 改了設定卻沒改路徑 —— 這正是本專案重複十次的缺陷型態
        assert w.is_done("train/N1/pie_0007", "different0000") is False
        # 產物被清掉
        (batch / "N1" / "phi.pt").unlink()
        assert w.is_done("train/N1/pie_0007", "abc123456789") is False


def test_eta取中位數而非平均(tmp_path):
    """先驗實驗已見過同一設定在不同影像上耗時差數倍，平均會被離群值帶偏。"""
    with _writer(tmp_path) as w:
        for i, secs in enumerate([10.0, 10.0, 1000.0]):
            cid = f"train/N1/img{i}"
            w.begin(cid, META)
            w.finish(cid, seconds=secs)
        w.begin("train/N1/img9", META)      # 留一格未完成，使 eta 可算
        st = w.snapshot()["stages"]["train"]
    assert st["median_seconds"] == 10.0
    assert st["eta_seconds"] == 10.0


def test_rebuild與快取一致(tmp_path):
    """progress.json 是快取不是真相。刪掉它必須能由 _cells/ 完整重建。"""
    with _writer(tmp_path) as w:
        for i in range(3):
            cid = f"eval/N1/img{i}"
            w.begin(cid, META)
            w.finish(cid, seconds=2.0)
        cached = w.snapshot()
    (tmp_path / "b1" / "progress.json").unlink()
    rebuilt = rebuild(tmp_path / "b1")
    assert rebuilt["cells"] == cached["cells"]
    assert rebuilt["summary"]["done"] == 3


def test_寫入期間讀取永遠可解析(tmp_path):
    """儀表板是另一個 session 在輪詢。直接寫入會讓它讀到寫到一半的檔案。

    要釘住的不變量是**讀取端永遠不會看到半份 JSON**。它由 os.replace 的
    原子性保證，與平台無關。

    `PermissionError` 不算違反：Windows 在 replace 進行中會讓讀取端的
    `open()` 失敗（實測），這是該平台的共用語意而非資料損壞。
    `read_progress()` 已處理它，故此處直接用它——讓測試走的是讀取端
    實際會走的那條路徑。
    """
    from src.utils.progress import read_progress

    w = _writer(tmp_path)
    errors = []
    stop = threading.Event()

    def reader():
        while not stop.is_set():
            try:
                read_progress(tmp_path / "b1")
            except json.JSONDecodeError:
                errors.append("讀到不完整的 JSON —— 原子性假設不成立")
            except FileNotFoundError:
                errors.append("progress.json 在寫入期間消失")
            except PermissionError:
                errors.append("read_progress 的重試次數不足")

    t = threading.Thread(target=reader, daemon=True)
    t.start()
    try:
        for i in range(60):
            cid = f"eval/N1/img{i}"
            w.begin(cid, META)
            w.finish(cid, seconds=0.01)
    finally:
        stop.set()
        t.join(timeout=5)
        w.release()
    assert not errors, errors[:3]


def test_read_progress在檔案損壞時拋出而非重試(tmp_path):
    """重試的對象是開檔失敗，不是解析失敗。檔案真的壞了要立刻知道，
    處置是 rebuild()，不是等它自己好起來。"""
    from src.utils.progress import read_progress

    w = _writer(tmp_path)
    w.release()
    (tmp_path / "b1" / "progress.json").write_text("{壞掉", encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        read_progress(tmp_path / "b1")


def test_read_progress在檔案不存在時給出處置方式(tmp_path):
    from src.utils.progress import read_progress

    (tmp_path / "b1").mkdir(parents=True)
    with pytest.raises(FileNotFoundError) as e:
        read_progress(tmp_path / "b1")
    assert "rebuild" in str(e.value)


def test_第二個寫入者被擋下(tmp_path):
    """本專案的硬規則是 GPU 工作不可並行；實測兩個工作互搶會把單張 SDEdit
    由 222 s 拉長到 30 分鐘以上。"""
    w = _writer(tmp_path)
    try:
        with pytest.raises(WriterLockHeld):
            _writer(tmp_path)
    finally:
        w.release()
    _writer(tmp_path).release()      # 釋放後可再取得


def test_鎖不自動判定行程存活(tmp_path):
    """誤判會讓兩個寫入者同時動同一批不可重跑的資料。訊息要告訴人怎麼手動處理。"""
    w = _writer(tmp_path)
    try:
        with pytest.raises(WriterLockHeld) as e:
            _writer(tmp_path)
        assert "手動刪除" in str(e.value)
    finally:
        w.release()

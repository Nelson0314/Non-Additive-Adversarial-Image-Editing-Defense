"""校準表：確認**沒有任何回退路徑**。

本檔的每一個測試都在證明同一件事——未校準的對象一定拋出。
這是為了堵掉先驗實驗重複十次的缺陷：一個為某個對象校準的值被沿用到另一個
對象上，而且沒有症狀（`PRIOR_FINDINGS` §4.5）。
"""

import json

import pytest

from src.utils.calibration import (REQUIRED_CONTEXT, Calibration,
                                   CalibrationMismatch)

CTX = {
    "model": "SDXL-1.0-base", "resolution": 1024, "guidance": 7.5,
    "steps": 50, "gpu": "Tesla V100-SXM2-32GB", "precision": "fp16",
}


def _table():
    c = Calibration()
    c.put("lr.N1", 0.1, CTX, note="site S, g=32")
    return c


def test_校準過的鍵可取回():
    assert _table().get("lr.N1", CTX) == 0.1


def test_未校準的鍵拋出而非回傳預設值():
    with pytest.raises(CalibrationMismatch) as e:
        _table().get("lr.N2", CTX)
    assert "未校準" in str(e.value)


def test_換卡即不符():
    """gpu 進 context 的理由：V100 走 fp16、RTX 5090 走 bf16，數值路徑不同。"""
    other = CTX | {"gpu": "RTX 5090", "precision": "bf16"}
    with pytest.raises(CalibrationMismatch):
        _table().get("lr.N1", other)


def test_換解析度即不符():
    with pytest.raises(CalibrationMismatch):
        _table().get("lr.N1", CTX | {"resolution": 512})


@pytest.mark.parametrize("missing", REQUIRED_CONTEXT)
def test_context缺任一必填欄位即拋出(missing):
    ctx = {k: v for k, v in CTX.items() if k != missing}
    with pytest.raises(CalibrationMismatch) as e:
        _table().get("lr.N1", ctx)
    assert missing in str(e.value)


def test_例外訊息同時列出雙方內容():
    """只說「不符」的話，使用者看到例外仍不知道差在哪，實務上會直接把檢查關掉。"""
    with pytest.raises(CalibrationMismatch) as e:
        _table().get("lr.N1", CTX | {"guidance": 1.0})
    msg = str(e.value)
    assert "7.5" in msg and "1.0" in msg, "要求值與記錄值都要出現"
    assert "差異" in msg


def test_多出來的context欄位也算不符():
    """子集比對會讓「校準時多記了一個變因」變成「比對時少檢查一個變因」。"""
    with pytest.raises(CalibrationMismatch):
        _table().get("lr.N1", CTX | {"extra_knob": 3})


def test_落盤與讀回等價(tmp_path):
    p = tmp_path / "calibration.json"
    _table().save(p)
    assert Calibration.load(p).get("lr.N1", CTX) == 0.1


def test_拒絕yaml副檔名(tmp_path):
    """`.gitignore` 的 runs/ 區塊只放行特定副檔名，.yaml 會被靜默排除。

    實測 `git check-ignore -v runs/b1/calib/calibration.yaml` 命中 `runs/**`。
    這條測試把該事實變成程式保證。
    """
    with pytest.raises(ValueError) as e:
        _table().save(tmp_path / "calibration.yaml")
    assert "gitignore" in str(e.value) or "排除" in str(e.value)


def test_表不存在時拋出而非產生空表(tmp_path):
    with pytest.raises(CalibrationMismatch) as e:
        Calibration.load(tmp_path / "nope.json")
    assert "段 0" in str(e.value)


def test_has不提供第二條路徑():
    """has() 只給儀表板與續跑判定用。它與 get() 的判定必須完全一致，
    否則就會出現「用 has() 包一層 if 再走別的路徑」這種回退。"""
    t = _table()
    assert t.has("lr.N1", CTX) is True
    assert t.has("lr.N1", CTX | {"precision": "fp32"}) is False
    assert t.has("lr.N9", CTX) is False


def test_落盤內容含calibrated_for(tmp_path):
    """為誰校準必須寫進檔案本身，不能只在註解裡——那正是原缺陷的成因。"""
    p = tmp_path / "calibration.json"
    _table().save(p)
    entry = json.loads(p.read_text(encoding="utf-8"))["lr.N1"]
    assert entry["calibrated_for"] == CTX

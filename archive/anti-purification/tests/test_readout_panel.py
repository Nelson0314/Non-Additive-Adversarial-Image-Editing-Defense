"""`scripts/readout_panel.py` 與 `src/metrics/arcface.py` 的純函式行為。

不測需要 insightface 權重的路徑——那個套件只裝在遠端。這裡測的是兩件在本機
就會出錯、而且錯了會靜默污染 CSV 的事：檔名配對，以及偵測失敗時的回傳值。
"""
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _panel():
    spec = importlib.util.spec_from_file_location(
        "readout_panel", ROOT / "scripts" / "readout_panel.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_配對由編輯圖反推且原圖與防禦圖不被當成格(tmp_path):
    """格是由 `__edit_def.png` 決定的。`__orig.png` 與 `__def.png` 不是格——
    同一個條件的防禦圖被所有指令共用，拿它當格會重複計數。"""
    for n in ["man_00__orig.png",
              "man_00__photoguard_c__def.png",
              "man_00__mist__def.png",
              "man_00__photoguard_c__edit_def.png",
              "man_00__mist__edit_def.png",
              "woman_01__dia_r__edit_def.png"]:
        (tmp_path / n).write_bytes(b"")
    got = sorted(_panel()._pairs(tmp_path))
    assert got == [("man_00", "mist", "man_00__mist"),
                   ("man_00", "photoguard_c", "man_00__photoguard_c"),
                   ("woman_01", "dia_r", "woman_01__dia_r")]


def test_條件名含底線也能配對(tmp_path):
    """`partition` 取第一個 `__`，條件名本身含單底線時不可被切開。"""
    (tmp_path / "img_a__photoguard_linf__edit_def.png").write_bytes(b"")
    assert _panel()._pairs(tmp_path) == [
        ("img_a", "photoguard_linf", "img_a__photoguard_linf")]


def test_指令與種子的檔名切得出條件(tmp_path):
    """`reedit_ip2p.py` 的一格是 (條件 × 指令 × 種子)，但防禦圖只有一張，
    所以 condition 必須從格名裡再切出來，否則找不到 `__def.png`。"""
    (tmp_path / "man_00__photoguard_linf__beach_bg__s17001__edit_def.png").write_bytes(b"")
    assert _panel()._pairs(tmp_path) == [
        ("man_00", "photoguard_linf",
         "man_00__photoguard_linf__beach_bg__s17001")]


def test_偵測不到臉時餘弦是None而不是零():
    """空值與零是兩件事：前者是「沒有臉」，後者是「臉完全不像」。
    填 0 會讓「臉不見了」在表上讀成「防禦把身分推到正交」。"""
    from src.metrics import arcface

    assert arcface.similarity(None, None) is None
    assert arcface.similarity(None, [1.0, 0.0]) is None
    assert arcface.similarity([1.0, 0.0], None) is None


def test_餘弦是內積且已正規化的嵌入落在正負一之間():
    import numpy as np

    from src.metrics import arcface

    a = np.array([1.0, 0.0, 0.0])
    b = np.array([0.0, 1.0, 0.0])
    assert arcface.similarity(a, a) == pytest.approx(1.0)
    assert arcface.similarity(a, b) == pytest.approx(0.0)
    assert arcface.similarity(a, -a) == pytest.approx(-1.0)


def test_無參考指標建立失敗使階段失敗(monkeypatch):
    """必要指標無法建立時不得輸出缺欄的成功結果。"""
    import types

    def fail(name, **kwargs):
        raise RuntimeError(f"無法載入 {name}")

    monkeypatch.setitem(sys.modules, "pyiqa", types.SimpleNamespace(create_metric=fail))
    panel = _panel()
    with pytest.raises(RuntimeError, match="無法載入 niqe"):
        panel._nr_suite("cpu")

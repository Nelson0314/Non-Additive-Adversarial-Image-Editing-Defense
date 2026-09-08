"""身分讀數的三個不變量。

只用 CPU 與合成影像；**不載臉部權重**（那需要下載 107 MB），故測的是
`similarity` 與 `identity_row` 的組合邏輯，偵測與嵌入由假的 embed 注入。
"""

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.metrics import identity  # noqa: E402


def test_任一邊沒有臉時相似度是_None_不是零():
    """填 0 會被讀成「完全不像」，而「沒有臉可比」是另一回事。"""
    v = torch.randn(512)
    assert identity.similarity(None, v) is None
    assert identity.similarity(v, None) is None
    assert identity.similarity(v, v) == pytest.approx(1.0, abs=1e-5)


def test_identity_row_的六個欄位(monkeypatch):
    a, b = torch.randn(512), torch.randn(512)
    seq = [a, a, b]                       # 原圖／編輯原圖 同一人，編輯防禦圖 另一人
    monkeypatch.setattr(identity, "embed", lambda x, d=None: seq.pop(0))
    r = identity.identity_row(*(torch.zeros(1, 3, 8, 8),) * 3)
    assert set(r) == {"face_found_orig", "face_found_edit_orig",
                      "face_found_edit_def", "id_orig", "id_def", "id_drop",
                      "id_embed_weights"}
    assert r["id_orig"] == pytest.approx(1.0, abs=1e-4)
    assert r["id_drop"] == pytest.approx(r["id_orig"] - r["id_def"], abs=1e-4)


def test_編輯輸出沒有臉時_id_def_與_id_drop_都留空(monkeypatch):
    """那是「看不出是誰」最強的形式，不可以被記成 0。"""
    a = torch.randn(512)
    seq = [a, a, None]
    monkeypatch.setattr(identity, "embed", lambda x, d=None: seq.pop(0))
    r = identity.identity_row(*(torch.zeros(1, 3, 8, 8),) * 3)
    assert r["face_found_edit_def"] is False
    assert r["id_def"] == "" and r["id_drop"] == ""
    assert r["id_orig"] != ""              # 這一邊仍然要有值


def test_原圖沒有臉時整列都留空(monkeypatch):
    seq = [None, torch.randn(512), torch.randn(512)]
    monkeypatch.setattr(identity, "embed", lambda x, d=None: seq.pop(0))
    r = identity.identity_row(*(torch.zeros(1, 3, 8, 8),) * 3)
    assert r["id_orig"] == "" and r["id_def"] == "" and r["id_drop"] == ""


def test_門檻只是參照不進算式():
    """`CLAUDE.md`：不得自訂用來判斷「有沒有效果」的判準。"""
    src = (ROOT/"src"/"metrics"/"identity.py").read_text(encoding="utf-8")
    assert "SAME_PERSON_REFERENCE" in src
    body = src.split("SAME_PERSON_REFERENCE = ")[1]
    assert "SAME_PERSON_REFERENCE" not in body    # 定義之後不再被使用

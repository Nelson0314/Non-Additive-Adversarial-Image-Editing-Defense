"""分區 LPIPS 必須與全圖 LPIPS 是同一個度量。

存在理由
────────────────────────────────────────────────────────────────────
分區讀數要與既有的 `edit_lpips` 並列在同一張表上。若兩者不是同一份權重、
同一個聚合方式，兩欄看起來可比而實際上不可比——這正是本專案定義的「靜默
失效」。故第一條測試把「遮罩全為 1 時逐位元退回 `piq.LPIPS()`」釘死。

只用 CPU、隨機小圖，不載任何專案權重。
"""

import sys
from pathlib import Path

import piq
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.metrics.regional import RegionalLPIPS, split_displacement  # noqa: E402


@pytest.fixture(scope="module")
def lp():
    return piq.LPIPS()


@pytest.fixture(scope="module")
def pair():
    g = torch.Generator().manual_seed(20260902)
    x = torch.rand(1, 3, 64, 64, generator=g)
    y = torch.rand(1, 3, 64, 64, generator=g)
    return x, y


def test_遮罩全為一時等於piq的全圖值(lp, pair):
    x, y = pair
    r = RegionalLPIPS(lp)
    ones = torch.ones(1, 1, 64, 64)
    assert r(x, y, ones) == pytest.approx(float(lp(x, y)), abs=1e-6)


def test_不給遮罩時等於piq的全圖值(lp, pair):
    x, y = pair
    r = RegionalLPIPS(lp)
    assert r(x, y, None) == pytest.approx(float(lp(x, y)), abs=1e-6)


def test_遮罩乘上常數不改變結果(lp, pair):
    """加權平均對遮罩的整體尺度不變，否則「面積」會混進讀數。"""
    x, y = pair
    r = RegionalLPIPS(lp)
    m = torch.zeros(1, 1, 64, 64)
    m[..., :32, :] = 1.0
    assert r(x, y, m) == pytest.approx(r(x, y, m * 0.37), abs=1e-6)


def test_只擾動一半時該半的讀數明顯較高(lp):
    """定位能力：改動只落在上半，上半的分區值必須遠高於下半。"""
    g = torch.Generator().manual_seed(7)
    x = torch.rand(1, 3, 64, 64, generator=g)
    y = x.clone()
    y[..., :32, :] = torch.rand(1, 3, 32, 64, generator=g)
    top = torch.zeros(1, 1, 64, 64)
    top[..., :32, :] = 1.0
    r = RegionalLPIPS(lp)
    assert r(x, y, top) > 5.0 * r(x, y, 1.0 - top)


def test_空遮罩拋錯而不是回傳零(lp, pair):
    """回傳 0 會被讀成「這一塊完全沒動」。"""
    x, y = pair
    r = RegionalLPIPS(lp)
    with pytest.raises(ValueError, match="總和為 0"):
        r(x, y, torch.zeros(1, 1, 64, 64))


def test_必須傳piq的LPIPS而不是自建的模組(pair):
    with pytest.raises(TypeError, match="piq.LPIPS"):
        RegionalLPIPS(torch.nn.Identity())


def test_三個位移欄位都由同一份權重算出(lp, pair):
    x, y = pair
    r = RegionalLPIPS(lp)
    m = torch.zeros(1, 1, 64, 64)
    m[..., :, :24] = 1.0
    out = split_displacement(r, x, y, m)
    assert set(out) == {"lpips_full", "lpips_subject", "lpips_background"}
    assert out["lpips_full"] == pytest.approx(float(lp(x, y)), abs=1e-6)
    assert out["lpips_subject"] == pytest.approx(r(x, y, m), abs=1e-9)
    assert out["lpips_background"] == pytest.approx(r(x, y, 1.0 - m), abs=1e-9)

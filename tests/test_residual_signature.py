"""殘差指紋的四個量。**只驗算式本身，不驗任何實驗結論。**

每個量都對應一條構造上的差異（見 `scripts/residual_signature.py` 的
docstring）；這裡用構造得出答案的合成輸入把它們釘住，免得算式改壞了之後
只反映在報表的數字上而沒有症狀。
"""

import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from residual_signature import (  # noqa: E402
    block_energy, gini, hf_share, polar_split,
)


def test_純相位旋轉的幅度那半恰為零():
    torch.manual_seed(0)
    spec = torch.randn(1, 1, 4, 8, 5, dtype=torch.complex128)
    theta = torch.rand(1, 1, 4, 8, 5, dtype=torch.float64) * 2.0 - 1.0
    out = polar_split(spec, spec * torch.exp(1j * theta))
    assert out["energy_mag"] == pytest.approx(0.0, abs=1e-18)
    assert out["phase_share"] == pytest.approx(1.0)


def test_純幅度縮放的相位那半恰為零():
    torch.manual_seed(1)
    spec = torch.randn(1, 1, 4, 8, 5, dtype=torch.complex128)
    out = polar_split(spec, spec * 1.3)
    assert out["energy_phase"] == pytest.approx(0.0, abs=1e-18)
    assert out["phase_share"] == pytest.approx(0.0)


def test_相位差繞過正負pi不被讀成一整圈():
    """`angle(S') − angle(S)` 在 ±π 附近會差 2π，本量必須用 `S'·conj(S)`。"""
    spec = torch.tensor([[[[[1.0 + 0.0j]]]]], dtype=torch.complex128)
    spec = spec * torch.exp(torch.tensor(1j * (math.pi - 0.01), dtype=torch.complex128))
    rotated = spec * torch.exp(torch.tensor(1j * 0.02, dtype=torch.complex128))
    out = polar_split(spec, rotated)
    # 真正的相位改動是 0.02，能量即 (|S|·0.02)^2；讀成一整圈會大三百倍。
    assert out["energy_phase"] == pytest.approx(0.02 ** 2, rel=1e-6)


def test_gini_均勻為零_集中為近一():
    assert gini([1.0] * 16) == pytest.approx(0.0)
    v = [0.0] * 15 + [1.0]
    assert gini(v) == pytest.approx(15 / 16)


def test_gini_拒絕負值():
    with pytest.raises(ValueError):
        gini([1.0, -0.5])


def test_全零殘差的gini定義為零而不是除以零():
    assert gini([0.0, 0.0, 0.0]) == 0.0


def test_逐區塊能量的格數與unfold一致():
    res = torch.zeros(1, 3, 64, 64)
    e = block_energy(res, block=32, hop=8)
    # (64 − 32)/8 + 1 = 5 沿每軸
    assert e.shape == (25,)


def test_逐區塊能量抓得到單一區塊裡的擾動():
    res = torch.zeros(1, 3, 64, 64)
    res[:, :, 0:8, 0:8] = 0.1
    e = block_energy(res, block=32, hop=32)
    assert e[0] > 0
    assert float(e[1:].sum()) == pytest.approx(0.0)


def test_高頻佔比_純高頻殘差接近一():
    n = 64
    k = torch.arange(n, dtype=torch.float32)
    # 逐像素交錯即 Nyquist，歸一化半徑遠大於 0.5
    res = torch.zeros(1, 3, n, n)
    res[:] = ((-1.0) ** k)[None, None, None, :] * 0.01
    assert hf_share(res) == pytest.approx(1.0, abs=1e-6)


def test_高頻佔比_純低頻殘差接近零():
    n = 64
    k = torch.arange(n, dtype=torch.float32)
    res = torch.zeros(1, 3, n, n)
    res[:] = torch.cos(2 * math.pi * k / n)[None, None, None, :] * 0.01
    assert hf_share(res) < 1e-6

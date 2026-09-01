"""殘差—紋理對準探針的守門測試（`scripts/residual_texture_alignment.py`）。

這支探針的結論是「對準這條軸分不開兩個方法」，而那是一個**否定**的結論——
量錯了會長得一模一樣（兩邊都接近某個值），不會有症狀。所以把三件事釘住：
等第相關本身算對、`block_energy` 與 `block_texture` 真的逐方格、以及退化輸入
拋錯而不是回傳一個看似正常的數字。

全部在 CPU 上執行。
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from residual_texture_alignment import (  # noqa: E402
    BLOCK,
    _gini,
    block_energy,
    block_texture,
    permute_block_energy,
    spearman,
)


def test_等第相關_完全同序為一_完全反序為負一():
    a = torch.tensor([1.0, 2.0, 3.0, 4.0])
    assert spearman(a, a) == pytest.approx(1.0, abs=1e-6)
    assert spearman(a, -a) == pytest.approx(-1.0, abs=1e-6)


def test_等第相關只看順序不看尺度():
    """兩個量的分布都極度右偏，用等第正是為了不被少數方格決定。"""
    a = torch.tensor([1.0, 2.0, 3.0, 4.0])
    b = torch.tensor([10.0, 1e3, 1e6, 1e9])
    assert spearman(a, b) == pytest.approx(1.0, abs=1e-6)


def test_常數輸入拋錯而不是回傳零():
    """靜默回傳 0 會被讀成「完全不對準」，那是一個實質結論。"""
    a = torch.ones(16)
    with pytest.raises(ValueError, match="無定義"):
        spearman(a, torch.arange(16.0))


def test_方格能量的形狀與數值():
    x = torch.zeros(1, 3, 2 * BLOCK, 2 * BLOCK)
    x[:, :, :BLOCK, :BLOCK] = 1.0          # 只有左上那一格有能量
    e = block_energy(x)
    assert e.numel() == 4
    assert int(e.argmax()) == 0
    assert float(e[1:].abs().max()) == 0.0


def test_紋理量對平坦區為零_對雜訊區為正():
    g = torch.Generator().manual_seed(0)
    x = torch.zeros(1, 3, 2 * BLOCK, 2 * BLOCK)
    x[:, :, :BLOCK, :] = torch.rand(1, 3, BLOCK, 2 * BLOCK, generator=g)
    t = block_texture(x)
    assert float(t[0]) > 0 and float(t[1]) > 0      # 上半列有紋理
    assert float(t[2]) == pytest.approx(0.0, abs=1e-12)


def test_對準的殘差相關高_均勻的殘差相關低():
    """這一項就是整支探針的主張，用合成輸入釘住它的方向。"""
    g = torch.Generator().manual_seed(1)
    size = 8 * BLOCK
    x = torch.zeros(1, 3, size, size)
    x[:, :, : size // 2, :] = torch.rand(1, 3, size // 2, size, generator=g)
    aligned = torch.zeros(1, 3, size, size)
    aligned[:, :, : size // 2, :] = torch.randn(
        1, 3, size // 2, size, generator=g) * 0.05
    uniform = torch.randn(1, 3, size, size, generator=g) * 0.05
    t = block_texture(x)
    assert spearman(block_energy(aligned), t) > 0.7
    assert abs(spearman(block_energy(uniform), t)) < 0.3


def test_置換把對準打掉():
    g = torch.Generator().manual_seed(2)
    size = 8 * BLOCK
    x = torch.zeros(1, 3, size, size)
    x[:, :, : size // 2, :] = torch.rand(1, 3, size // 2, size, generator=g)
    r = torch.zeros(1, 3, size, size)
    r[:, :, : size // 2, :] = torch.randn(1, 3, size // 2, size, generator=g) * 0.05
    t, e = block_texture(x), block_energy(r)
    assert abs(spearman(permute_block_energy(e, 0), t)) < abs(spearman(e, t))


def test_gini_均勻為零_集中為近一():
    assert _gini(torch.ones(64)) == pytest.approx(0.0, abs=1e-6)
    v = torch.zeros(64)
    v[0] = 1.0
    assert _gini(v) > 0.95


def test_gini_全零拋錯():
    with pytest.raises(ValueError, match="無定義"):
        _gini(torch.zeros(64))


def test_均勻殘差丟掉的能量等於裁掉的面積比():
    """`crop_resize(0.10)` 每邊各裁 10%，保留中央 410/512 的邊長。

    這一項同時釘住幾何：`SURVEY_ARCHITECTURE` H4 引用的「移除 19% 面積」對應的
    是「整條邊裁 10%」，與 `src/purify/ops.py` 的實作不符（實作移除 35.9%）。
    """
    from residual_texture_alignment import crop_ring_fraction

    r = torch.ones(1, 3, 512, 512)
    assert crop_ring_fraction(r) == pytest.approx(1 - (410 / 512) ** 2, abs=1e-6)


def test_能量全在中央時裁切圈為零_全在外圈時為一():
    from residual_texture_alignment import crop_ring_fraction

    centre = torch.zeros(1, 3, 512, 512)
    centre[..., 200:300, 200:300] = 1.0
    assert crop_ring_fraction(centre) == pytest.approx(0.0, abs=1e-9)

    ring = torch.ones(1, 3, 512, 512)
    ring[..., 51:461, 51:461] = 0.0
    assert crop_ring_fraction(ring) == pytest.approx(1.0, abs=1e-9)


def test_殘差全零時拋錯():
    from residual_texture_alignment import crop_ring_fraction

    with pytest.raises(ValueError, match="無定義"):
        crop_ring_fraction(torch.zeros(1, 3, 512, 512))

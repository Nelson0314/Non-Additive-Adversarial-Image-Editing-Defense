"""`src/defense/dct_rotation.py` 的構造性質。

這些測試釘的不是數值大小，是**構造保證**：不成立的話後面所有讀數都要重讀。
"""

from __future__ import annotations

import math

import pytest
import torch

from src.baselines.jpeg_codec import jpeg_encode, jpeg_roundtrip
from src.defense.dct_rotation import (
    DctRotationParam,
    DctRotationRandomParam,
    build_pairs,
    rotate_pairs,
)


def _image(seed: int = 0, n: int = 64) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.rand(1, 3, n, n, generator=g, dtype=torch.float64)


def test_轉置配對不重疊且排除DC():
    pairs = build_pairs("transpose", r_min=0.12)
    assert len(pairs) == 28, "8×8 的非對角上三角是 28 對"
    seen = set()
    for a, b in pairs:
        assert a != (0, 0) and b != (0, 0), "DC 不可入列"
        assert a not in seen and b not in seen, "配對必須不重疊"
        seen.update((a, b))
    # 轉置對的徑向頻率必須相同——這是「保長在感知上有意義」的唯一依據。
    for (u1, v1), (u2, v2) in pairs:
        assert u1 ** 2 + v1 ** 2 == u2 ** 2 + v2 ** 2


def test_zigzag配對不重疊():
    pairs = build_pairs("zigzag", r_min=0.12)
    seen = set()
    for a, b in pairs:
        assert a != (0, 0) and b != (0, 0)
        assert a not in seen and b not in seen
        seen.update((a, b))


def test_未知配對規則要拋出而不是靜默退回預設():
    with pytest.raises(ValueError, match="未知的配對規則"):
        build_pairs("spiral")


def test_旋轉保住每一對的平方和():
    g = torch.Generator().manual_seed(1)
    coef = torch.randn(1, 4, 4, 8, 8, generator=g, dtype=torch.float64)
    pairs = build_pairs("transpose")
    ang = torch.full((1, 4, 4, len(pairs)), 0.7, dtype=torch.float64)
    out = rotate_pairs(coef, pairs, ang)
    for (u1, v1), (u2, v2) in pairs:
        before = coef[..., u1, v1] ** 2 + coef[..., u2, v2] ** 2
        after = out[..., u1, v1] ** 2 + out[..., u2, v2] ** 2
        assert torch.allclose(before, after, atol=1e-12)


def test_旋轉pi就是聯合翻號():
    g = torch.Generator().manual_seed(2)
    coef = torch.randn(1, 2, 2, 8, 8, generator=g, dtype=torch.float64)
    pairs = build_pairs("transpose")
    ang = torch.full((1, 2, 2, len(pairs)), math.pi, dtype=torch.float64)
    out = rotate_pairs(coef, pairs, ang)
    for (u1, v1), (u2, v2) in pairs:
        assert torch.allclose(out[..., u1, v1], -coef[..., u1, v1], atol=1e-12)
        assert torch.allclose(out[..., u2, v2], -coef[..., u2, v2], atol=1e-12)


def test_零角度時輸出等於交付品質的往返而不是原圖():
    """`θ = 0` 是 `R(0) = I` 且 `round(α) = α`，故輸出是 `jpeg_roundtrip`。

    **不是原圖**——交付本身就是壓縮圖，這與 `texture_rephase` 的恆等性質不同。
    """
    x = _image(3)
    p = DctRotationParam(radius=1.0, qd=0.85, gate="band")
    p.reset(x, seed=0)
    out = p.render(x)
    ref = jpeg_roundtrip(x, 0.85)
    assert torch.allclose(out, ref, atol=1e-10)
    assert not torch.allclose(out, x, atol=1e-3), "不該逐位等於原圖"


def test_梯度通得過取整():
    x = _image(4)
    p = DctRotationParam(radius=1.0, qd=0.85, gate="band")
    p.reset(x, seed=0)
    with torch.no_grad():
        for t in p.theta.values():
            t.add_(0.3)
    loss = p.render(x).pow(2).sum()
    grads = torch.autograd.grad(loss, p.params())
    assert all(torch.isfinite(g).all() for g in grads)
    assert max(float(g.abs().max()) for g in grads) > 0, \
        "直通估計沒接上的話梯度會整片為零"


def test_project夾在角度上界內():
    x = _image(5)
    p = DctRotationParam(radius=0.4, qd=0.85, gate="band")
    p.reset(x, seed=0)
    with torch.no_grad():
        for t in p.theta.values():
            t.add_(5.0)
    p.project()
    for t in p.theta.values():
        assert float(t.detach().max()) <= 0.4 + 1e-12


def test_隨機對照不回傳可最佳化的參數():
    x = _image(6)
    p = DctRotationRandomParam(radius=1.0, qd=0.85, gate="band")
    p.reset(x, seed=7)
    assert p.params() == []
    assert max(float(t.detach().abs().max()) for t in p.theta.values()) > 0


def test_紋理閘落在編解碼器自己的區塊格點上():
    """閘的形狀必須與 `jpeg_encode` 的區塊數一致，錯位不會有症狀故要釘住。"""
    x = _image(8, n=64)
    p = DctRotationParam(radius=1.0, qd=0.85, gate="texture")
    p.reset(x, seed=0)
    alpha = jpeg_encode(x, 0.85)
    for name in ("Y", "Cb", "Cr"):
        assert p.gates[name].shape == alpha[name].shape[:3]


def test_delta_stats回報的比例在合理範圍():
    x = _image(9)
    p = DctRotationParam(radius=1.1, qd=0.85, gate="band")
    p.reset(x, seed=0)
    with torch.no_grad():
        for t in p.theta.values():
            t.add_(1.1)
    s = p.delta_stats()
    assert 0.0 <= s["delta_within_1"] <= 1.0
    assert 0.0 <= s["zero_pair_frac"] <= 1.0
    assert s["n_pairs"] == 28

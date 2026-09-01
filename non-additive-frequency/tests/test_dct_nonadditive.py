"""`src/defense/dct_nonadditive.py` 的構造性質。

釘的是構造保證，不是數值大小：不成立的話後面所有讀數都要重讀。
"""

from __future__ import annotations

import math

import pytest
import torch

from src.defense.dct_nonadditive import (
    DctNonAdditiveParam,
    DctNonAdditiveRandomParam,
    band_indices,
    from_planes,
    rotate_in_plane,
    to_planes,
)


def _image(seed: int = 0, n: int = 64) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.rand(1, 3, n, n, generator=g, dtype=torch.float64)


def test_通帶只排除DC():
    idx = band_indices(0.12)
    assert len(idx) == 63, "8×8 上 r_min=0.12 只排除 DC"
    assert (0, 0) not in idx


def test_平面旋轉精確保長():
    g = torch.Generator().manual_seed(1)
    c = torch.randn(4, 5, 63, generator=g, dtype=torch.float64)
    u = torch.randn(4, 5, 63, generator=g, dtype=torch.float64)
    v = torch.randn(4, 5, 63, generator=g, dtype=torch.float64)
    th = torch.full((4, 5, 1), 0.7, dtype=torch.float64)
    out = rotate_in_plane(c, u, v, th)
    assert torch.allclose(c.norm(dim=-1), out.norm(dim=-1), atol=1e-12)


def test_零角度是恆等():
    g = torch.Generator().manual_seed(2)
    c = torch.randn(3, 63, generator=g, dtype=torch.float64)
    u = torch.randn(3, 63, generator=g, dtype=torch.float64)
    v = torch.randn(3, 63, generator=g, dtype=torch.float64)
    out = rotate_in_plane(c, u, v, torch.zeros(3, 1, dtype=torch.float64))
    assert torch.allclose(c, out, atol=1e-14)


def test_平面外的分量完全不動():
    """旋轉只能動 span(u,v) 裡的東西，正交補必須逐位元不變。"""
    k = 8
    c = torch.zeros(1, k, dtype=torch.float64); c[0, 5] = 1.0
    u = torch.zeros(1, k, dtype=torch.float64); u[0, 0] = 1.0
    v = torch.zeros(1, k, dtype=torch.float64); v[0, 1] = 1.0
    out = rotate_in_plane(c, u, v, torch.full((1, 1), 1.3, dtype=torch.float64))
    assert torch.allclose(c, out, atol=1e-14)


def test_退化平面回傳原值而不是NaN():
    c = torch.randn(2, 6, dtype=torch.float64)
    zero = torch.zeros(2, 6, dtype=torch.float64)
    out = rotate_in_plane(c, zero, zero, torch.ones(2, 1, dtype=torch.float64))
    assert torch.isfinite(out).all()
    assert torch.allclose(c, out, atol=1e-14)
    # v 與 u 平行也是退化的
    u = torch.randn(2, 6, dtype=torch.float64)
    out2 = rotate_in_plane(c, u, u * 2.0, torch.ones(2, 1, dtype=torch.float64))
    assert torch.isfinite(out2).all()
    assert torch.allclose(c, out2, atol=1e-12)


def test_平面模式的零角度輸出等於原圖():
    """`theta = 0` 時是恆等映射，而 DCT 往返在浮點下應該回到原圖。"""
    x = _image(3)
    p = DctNonAdditiveParam(radius=1.0, mode="plane", gate="band")
    p.reset(x, seed=0)
    assert torch.allclose(p.render(x), x, atol=1e-9)


def test_平面起點不可為零否則梯度恆為零():
    x = _image(4)
    p = DctNonAdditiveParam(radius=1.0, mode="plane", gate="band")
    p.reset(x, seed=0)
    for name, d in p.params_.items():
        assert float(d["u"].detach().abs().max()) > 0, "u 全零就是退化平面"
        assert float(d["v"].detach().abs().max()) > 0
    assert p.degenerate_fraction() == 0.0


@pytest.mark.parametrize("mode", ["plane", "shared_plane", "gain"])
def test_三個模式都通得過梯度(mode):
    x = _image(5)
    p = DctNonAdditiveParam(radius=0.8, mode=mode, gate="band")
    p.reset(x, seed=1)
    with torch.no_grad():
        key = "g" if mode == "gain" else "theta"
        for d in p.params_.values():
            d[key].add_(0.3)
    loss = p.render(x).pow(2).sum()
    grads = torch.autograd.grad(loss, p.params(), allow_unused=True)
    live = [g for g in grads if g is not None]
    assert live, "沒有任何參數接到梯度"
    assert all(torch.isfinite(g).all() for g in live)
    assert max(float(g.abs().max()) for g in live) > 0


def test_共用平面的參數量不隨影像變大而增加():
    """`shared_plane` 是全域的，`plane` 是逐區塊的——兩者對影像尺寸的相依不同。

    直接寫死倍率會被測試影像的尺寸綁架（64² 只有 64 個亮度區塊），所以改測
    **尺度行為**：影像邊長加倍，逐區塊的參數量要變四倍，共用的要完全不變。
    """
    small, big = _image(6, n=64), _image(6, n=128)
    def n_of(mode, x):
        p = DctNonAdditiveParam(radius=1.0, mode=mode, gate="band")
        p.reset(x, 0)
        return sum(t.numel() for t in p.params())
    assert n_of("shared_plane", small) == n_of("shared_plane", big)
    assert n_of("plane", big) == 4 * n_of("plane", small)
    assert n_of("plane", small) > n_of("shared_plane", small)


def test_project只夾強度不夾平面():
    x = _image(7)
    p = DctNonAdditiveParam(radius=0.4, mode="plane", gate="band")
    p.reset(x, seed=0)
    with torch.no_grad():
        for d in p.params_.values():
            d["theta"].add_(5.0); d["u"].add_(9.0)
    before = {n: d["u"].detach().clone() for n, d in p.params_.items()}
    p.project()
    for n, d in p.params_.items():
        assert float(d["theta"].detach().max()) <= 0.4 + 1e-12
        assert torch.equal(d["u"].detach(), before[n]), "平面是方向，不該被夾"


def test_隨機對照不回傳可最佳化的參數():
    x = _image(8)
    p = DctNonAdditiveRandomParam(radius=0.9, mode="plane", gate="band")
    p.reset(x, seed=5)
    assert p.params() == []
    th = [d["theta"].detach().abs().max() for d in p.params_.values()]
    assert max(float(t) for t in th) > 0


def test_未知模式與未知閘都要拋出():
    with pytest.raises(ValueError, match="未知的 mode"):
        DctNonAdditiveParam(mode="quaternion")
    with pytest.raises(ValueError, match="未知的閘"):
        DctNonAdditiveParam(gate="watson")


def test_不夾取時像素域的保長是精確的():
    """8×8 DCT 正交歸一、區塊不重疊，故保長在像素域也成立——**直到值域夾取**。

    用中間調影像（值域 [0.3, 0.7]）讓旋轉不會把任何像素推出 `[0,1]`，
    這時能量守恆應該精確到浮點誤差。這才是在檢定構造本身。
    """
    from src.baselines.jpeg_codec import block_dct

    g = torch.Generator().manual_seed(9)
    x = 0.3 + 0.4 * torch.rand(1, 3, 64, 64, generator=g, dtype=torch.float64)
    p = DctNonAdditiveParam(radius=1.2, mode="plane", gate="band")
    p.reset(x, seed=2)
    with torch.no_grad():
        for d in p.params_.values():
            d["theta"].add_(1.2)
    y = p.render(x)
    assert p.clip_fraction() == 0.0, "這張圖不該有任何像素被夾"
    a = block_dct(to_planes(x)[0], p._d).pow(2).sum(dim=(-2, -1))
    b = block_dct(to_planes(y)[0], p._d).pow(2).sum(dim=(-2, -1))
    rel = float(((a - b).abs() / a.clamp_min(1e-9)).max().detach())
    assert rel < 1e-9, f"不夾取時能量相對變化應為浮點誤差，得到 {rel:.2e}"


def test_夾取是像素域保長唯一的破壞源():
    """滿值域的影像上，旋轉會把像素推出 `[0,1]`，能量守恆隨之破裂。

    這一格與上一格是一對：構造是精確的，破壞來自「輸出必須是合法影像」。
    報表引用等失真數字時要一併看 `clip_fraction()`。
    """
    from src.baselines.jpeg_codec import block_dct

    x = _image(9, n=64)                       # 滿值域 [0,1) 的隨機圖
    p = DctNonAdditiveParam(radius=1.2, mode="plane", gate="band")
    p.reset(x, seed=2)
    with torch.no_grad():
        for d in p.params_.values():
            d["theta"].add_(1.2)
    y = p.render(x)
    clip = p.clip_fraction()
    a = block_dct(to_planes(x)[0], p._d).pow(2).sum(dim=(-2, -1))
    b = block_dct(to_planes(y)[0], p._d).pow(2).sum(dim=(-2, -1))
    rel = float(((a - b).abs() / a.clamp_min(1e-9)).max().detach())
    assert clip > 0.0, "滿值域影像在這個強度下應該有像素被夾"
    assert rel > 1e-6, "有夾取就不該還是精確守恆"
    assert rel < 0.15, f"破壞幅度異常大（{rel:.2e}），可能不只是夾取"

"""局部銳利度偏差的行為測試。

要釘住的是三條設計性質，任何一條失效這個指標就不該進保真約束：

1. 恆等為 0；模糊使其上升且有號值為負；雜訊使其上升且有號值為正。
2. 不可抵銷——這是它相對 `acutance` 的唯一理由。同一組「半張模糊、
   半張加噪」的構造下，`acutance` 的全域比值可以湊回 1（見
   `tests/test_battery.py`），本指標必須仍然收費。
3. 對次像素位移不敏感——這是它相對 GMSD 的唯一理由。P1b 實測 GMSD 對
   變形收的費比對模糊高，那會使約束因為「site S 是變形」而懲罰它。
"""

import pytest
import torch
import torch.nn.functional as F

from src.metrics.acutance import acutance
from src.metrics.local_acutance import local_acutance, patch_energy


def _img(seed=20260731, size=256):
    """中間調圖樣。飽和的 0/1 二值圖會使夾回 [0,1] 的雜訊反而降低銳利度，
    那是該圖的性質而非指標的性質（見 tests/test_battery.py 的同名說明）。"""
    g = torch.Generator().manual_seed(seed)
    yy, xx = torch.meshgrid(torch.arange(size), torch.arange(size), indexing="ij")
    x = torch.zeros(1, 3, size, size)
    x[0, 0] = torch.where((xx + yy) % 16 < 8, 0.30, 0.70)
    x[0, 1] = torch.where((xx // 8) % 2 == 0, 0.35, 0.65)
    x[0, 2] = 0.35 + 0.30 * torch.rand(size, size, generator=g)
    return x


def _blur(x, k=9, sigma=2.0):
    c = torch.arange(k, dtype=torch.float32) - (k - 1) / 2
    w = torch.exp(-c.pow(2) / (2 * sigma**2))
    w = (w / w.sum()).view(1, 1, 1, k)
    n = x.shape[1]
    x = F.pad(x, (k // 2, k // 2, 0, 0), mode="replicate")
    x = F.conv2d(x, w.expand(n, 1, 1, k), groups=n)
    x = F.pad(x, (0, 0, k // 2, k // 2), mode="replicate")
    return F.conv2d(x, w.transpose(2, 3).expand(n, 1, k, 1), groups=n)


def _shift(x, dx: float):
    """以 grid_sample 做次像素平移，重取樣模式取雙三次以免自身引入模糊。"""
    b, c, h, w = x.shape
    ys = torch.linspace(-1.0, 1.0, h)
    xs = torch.linspace(-1.0, 1.0, w) + dx * 2.0 / (w - 1)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    grid = torch.stack((gx, gy), dim=-1).unsqueeze(0).expand(b, -1, -1, -1)
    return F.grid_sample(x, grid, mode="bicubic", padding_mode="border",
                         align_corners=True).clamp(0, 1)


def test_恆等為零():
    x = _img()
    m = local_acutance(x, x)
    assert m["local_acutance_dev"] == pytest.approx(0.0, abs=1e-6)
    assert m["local_acutance_signed"] == pytest.approx(0.0, abs=1e-6)
    assert m["local_acutance_worst"] == pytest.approx(0.0, abs=1e-6)


def test_模糊使偏差上升且有號值為負():
    x = _img()
    m = local_acutance(x, _blur(x))
    assert m["local_acutance_dev"] > 0.1
    assert m["local_acutance_signed"] < -0.1


def test_雜訊使有號值為正():
    x = _img()
    g = torch.Generator().manual_seed(7)
    y = (x + 0.02 * torch.randn(x.shape, generator=g)).clamp(0, 1)
    assert local_acutance(x, y)["local_acutance_signed"] > 0


def test_區塊能量總和等於全圖梯度能量():
    """切塊不得改變總能量，否則加權平均的權重就不是能量佔比。"""
    from src.metrics.acutance import gradient_energy

    x = _img()
    assert float(patch_energy(x).sum()) == pytest.approx(
        float(gradient_energy(x).sum()), rel=1e-4)


def test_半模糊半加噪無法抵銷():
    """`acutance` 的全域比值可湊回 1，本指標必須仍然收費。

    這是本模組存在的唯一理由；此測試失效等於該理由消失。
    """
    x = _img()
    half = x.shape[-1] // 2
    y = x.clone()
    y[..., :half] = _blur(x)[..., :half]
    n = torch.randn(x.shape, generator=torch.Generator().manual_seed(7))

    def mix(amp):
        z = y.clone()
        z[..., half:] = (y[..., half:] + amp * n[..., half:]).clamp(0, 1)
        return z

    lo, hi = 0.0, 0.5
    for _ in range(40):
        amp = 0.5 * (lo + hi)
        if acutance(x, mix(amp))["acutance_ratio"] < 1.0:
            lo = amp
        else:
            hi = amp
    z = mix(0.5 * (lo + hi))

    # 前提：全域比值確實被湊回 1，即漏洞成立
    assert acutance(x, z)["acutance_ratio"] == pytest.approx(1.0, abs=0.02)
    # 結論：局部偏差不受抵銷影響，仍然收到大筆費用
    assert local_acutance(x, z)["local_acutance_dev"] > 0.1


def test_次像素位移的收費遠低於同等模糊():
    """對位移不敏感是它相對 GMSD 的唯一理由。

    比較的對象取「造成相近 LPIPS 的模糊」不現實（測試不載入 LPIPS 模型），
    故改用一個更強的斷言：0.5 px 的純平移，其收費必須比 σ=2 的模糊低一個
    數量級以上。
    """
    x = _img()
    d_shift = local_acutance(x, _shift(x, 0.5))["local_acutance_dev"]
    d_blur = local_acutance(x, _blur(x))["local_acutance_dev"]
    assert d_shift < 0.1 * d_blur

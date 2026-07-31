"""銳利度保留率的行為測試。

與 tiny-SD 相關的測試不同，這裡的命題全部是演算法層面的確定性行為，
可以直接斷言數值方向：模糊必須降、加雜訊必須升、恆等必須為 1。
"""

import pytest
import torch
import torch.nn.functional as F

from src.metrics.acutance import acutance, gradient_energy


def _img(seed=20260731, size=64):
    g = torch.Generator().manual_seed(seed)
    # 隨機雜訊的梯度太均勻，改用有結構的圖樣：斜條紋 + 方塊，兩者都有明確邊緣
    x = torch.zeros(1, 3, size, size)
    yy, xx = torch.meshgrid(torch.arange(size), torch.arange(size), indexing="ij")
    x[0, 0] = ((xx + yy) % 16 < 8).float()
    x[0, 1] = ((xx // 8) % 2 == 0).float()
    x[0, 2] = torch.rand(size, size, generator=g)
    return x


def _blur(x, k=5, sigma=1.5):
    c = torch.arange(k, dtype=torch.float32) - (k - 1) / 2
    w = torch.exp(-c.pow(2) / (2 * sigma ** 2))
    w = (w / w.sum()).view(1, 1, 1, k)
    n = x.shape[1]
    x = F.pad(x, (k // 2, k // 2, 0, 0), mode="replicate")
    x = F.conv2d(x, w.expand(n, 1, 1, k), groups=n)
    x = F.pad(x, (0, 0, k // 2, k // 2), mode="replicate")
    return F.conv2d(x, w.transpose(2, 3).expand(n, 1, k, 1), groups=n)


def test_恆等時保留率為_1():
    x = _img()
    assert acutance(x, x)["acutance_ratio"] == pytest.approx(1.0, rel=1e-6)


def test_模糊使保留率下降():
    """這是本指標存在的理由：LPIPS 對模糊不敏感，此處必須敏感。"""
    x = _img()
    r = acutance(x, _blur(x))["acutance_ratio"]
    assert r < 0.9, f"高斯模糊後保留率應明顯低於 1，實得 {r}"


def test_模糊越強保留率越低():
    x = _img()
    r1 = acutance(x, _blur(x, sigma=1.0))["acutance_ratio"]
    r2 = acutance(x, _blur(x, sigma=3.0))["acutance_ratio"]
    assert r2 < r1, f"sigma 3.0 應比 1.0 更鈍，實得 {r2} vs {r1}"


def _smooth(size=64):
    """中間調的平滑正弦圖樣。

    不用 `_img()`：它的通道是 0/1 二值，加雜訊後 clamp 會削掉飽和平台，
    邊緣對比下降的幅度大於雜訊帶來的增益，保留率反而小於 1。那是 clamp
    的性質而非本指標的性質，用它測「加雜訊變銳」會測到錯的東西。
    """
    yy, xx = torch.meshgrid(torch.arange(size), torch.arange(size), indexing="ij")
    v = 0.5 + 0.18 * torch.sin(xx.float() / 6.0) * torch.cos(yy.float() / 7.0)
    return v.expand(3, size, size).unsqueeze(0).contiguous()


def test_加雜訊使保留率上升():
    """比原圖更銳不折疊成 1：加性擾動會造成，且那是要看見的資訊。

    以中間調平滑影像為基底，值域遠離 0 與 1，clamp 不會作用。
    """
    x = _smooth()
    g = torch.Generator().manual_seed(7)
    noisy = (x + 0.05 * torch.randn(x.shape, generator=g)).clamp(0, 1)
    assert noisy.min() > 0.0 and noisy.max() < 1.0, "此測試的前提是 clamp 不作用"
    r = acutance(x, noisy)["acutance_ratio"]
    assert r > 1.0, f"加雜訊後保留率應大於 1，實得 {r}"


def test_邊界用_replicate_而非補零():
    """補零會在邊界造出不存在的強梯度。以定值圖檢驗：其梯度能量必須為 0。"""
    flat = torch.full((1, 3, 32, 32), 0.5)
    assert float(gradient_energy(flat)) == pytest.approx(0.0, abs=1e-6)


def test_原圖無梯度時回傳_nan_而非除以零():
    flat = torch.full((1, 3, 32, 32), 0.5)
    other = _img(size=32)
    assert torch.isnan(torch.tensor(acutance(flat, other)["acutance_ratio"]))


def test_亮度加權而非通道相加():
    """純藍與純紅的同樣圖樣，能量應因 Rec.601 權重不同而不同。"""
    size = 32
    yy, xx = torch.meshgrid(torch.arange(size), torch.arange(size), indexing="ij")
    pat = ((xx // 4) % 2 == 0).float()
    red = torch.zeros(1, 3, size, size)
    red[0, 0] = pat
    blue = torch.zeros(1, 3, size, size)
    blue[0, 2] = pat
    er = float(gradient_energy(red))
    eb = float(gradient_energy(blue))
    assert er > eb, f"紅(0.299) 的亮度權重高於藍(0.114)，能量應較大：{er} vs {eb}"

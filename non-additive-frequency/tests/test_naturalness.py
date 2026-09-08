# -*- coding: utf-8 -*-
"""譜斜率／區塊熵／色數：它們要能把「雜訊」與「有結構的圖案」分開。

這一組讀數存在的理由是 NIMA／CNNIQA 分不出那個差別。**新的讀數如果也分不出
來，它同樣不可用**，所以測試釘住的是「在已知答案的合成紋理上排序正確」，
不是「函式有回傳值」。
"""

import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.metrics.naturalness import (block_entropy, colour_count,
                                     naturalness_row, spectral_slope,
                                     support_from_pair)

H = W = 256


def _sup():
    m = torch.zeros(1, 1, H, W)
    m[..., 32:224, 32:224] = 1.0
    return m


def _noise(seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(1, 3, H, W, generator=g)


def _pink(seed=1):
    """1/f 合成紋理：自然影像的功率譜統計。"""
    g = torch.Generator().manual_seed(seed)
    f2 = (torch.fft.rfftfreq(W)[None, :] ** 2
          + torch.fft.fftfreq(H)[:, None] ** 2)
    amp = 1.0 / torch.sqrt(f2.clamp_min(1e-8))
    ph = torch.rand(H, W // 2 + 1, generator=g) * 2 * math.pi
    x = torch.fft.irfft2(amp * torch.exp(1j * ph), s=(H, W)).real
    x = (x - x.min()) / (x.max() - x.min())
    return x[None, None].repeat(1, 3, 1, 1)


def _tiles(period=32):
    x = torch.zeros(1, 3, H, W)
    for i in range(0, H, period):
        for j in range(0, W, period):
            x[..., i:i + period // 2, j:j + period // 2] = 0.8
    return x


def test_譜斜率把白噪聲與自然紋理分開():
    """白噪聲的譜是平的（α≈0），1/f 紋理 α≈2。這是整組讀數的立足點。"""
    s = _sup()
    a = spectral_slope(_noise(), s)
    b = spectral_slope(_pink(), s)
    assert abs(a) < 0.5, f"白噪聲的斜率應該接近 0，得到 {a}"
    assert 1.4 < b < 2.6, f"1/f 紋理的斜率應該接近 2，得到 {b}"
    assert b - a > 1.5


def test_規則重複的圖案斜率明顯高於雜訊():
    """平舖是「高頻加結構」那條路的產物，讀數要能看見它與雜訊的差別。"""
    s = _sup()
    assert spectral_slope(_tiles(), s) > spectral_slope(_noise(), s) + 1.5


def test_色數把雜訊與少色圖案分開():
    s = _sup()
    assert colour_count(_noise(), s) > 500
    assert colour_count(_tiles(), s) <= 4


def test_區塊熵把雜訊與平坦色塊分開():
    s = _sup()
    assert block_entropy(_noise(), s) > block_entropy(_tiles(), s) + 2.0


def test_支撐由兩張影像的差反推():
    """重建載體要再跑一次 ATR，旗標對不上就量到別塊區域且沒有症狀；
    影像之間的差則是交出去那張圖的事實。"""
    x = _noise(2)
    d = x.clone()
    d[..., 10:40, 10:50] = _noise(3)[..., 10:40, 10:50]
    got = support_from_pair(x, d)
    assert abs(float(got.mean()) - 30 * 40 / (H * W)) < 1e-6
    assert float(support_from_pair(x, x).mean()) == 0.0


def test_形狀不同就拋錯():
    with pytest.raises(ValueError, match="形狀不同"):
        support_from_pair(_noise(), torch.zeros(1, 3, 64, 64))


def test_支撐太小時回空值而不是硬算():
    """樣本太少時擬合誤差比要量的差異還大，回 None 才不會被讀成一個數。"""
    tiny = torch.zeros(1, 1, H, W)
    tiny[..., :16, :16] = 1.0
    assert spectral_slope(_noise(), tiny) is None
    assert colour_count(_noise(), tiny) is None
    assert block_entropy(_noise(), tiny) is None


def test_逐列欄位齊全且orig與def都在():
    """`*_orig` 不可省：單張影像的紋理差很多，不扣掉基準就分不出
    「防禦讓它變雜」與「這塊布本來就雜」。"""
    x, d = _pink(), _noise()
    row = naturalness_row(x, d)
    for k in ("support_area", "slope_orig", "slope_def", "slope_drop",
              "entropy_orig", "entropy_def", "entropy_drop",
              "colours_orig", "colours_def", "colours_ratio"):
        assert k in row, k
    # 整張圖都被換掉，故支撐是全圖；雜訊比 1/f 平，drop 必為正。
    assert row["slope_drop"] > 1.0
    assert row["colours_ratio"] > 1.0

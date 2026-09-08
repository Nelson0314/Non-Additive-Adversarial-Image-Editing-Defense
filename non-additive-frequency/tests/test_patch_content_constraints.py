# -*- coding: utf-8 -*-
"""補丁的兩個內容**約束**，以及載體的面積對齊。

這三樣都會靜默失效：約束若沒有真的接上，產物仍然是一張圖、CSV 仍然有值、
報表上看不出來。故每一條都釘住**可觀測的量**而不是「有沒有呼叫到」。
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.defense.carrier_mask import (CARRIER_CLASSES, FACE_CLASS,
                                      carrier_from_seg, erode_mask,
                                      lattice_support, legal_area, match_area)
from src.defense.color_param import luma
from src.defense.patch_param import PatchParam, box_blur

SIGMA = 16


def _scene():
    """一張有低頻結構的假影像、一塊主體遮罩。"""
    g = torch.Generator().manual_seed(7)
    yy, xx = torch.meshgrid(torch.linspace(0, 1, 96), torch.linspace(0, 1, 96),
                            indexing="ij")
    base = torch.stack([yy, xx, 0.3 + 0.4 * yy * xx])[None]
    x = (base + 0.05 * torch.rand(base.shape, generator=g)).clamp(0, 1)
    mask = torch.zeros(1, 1, 96, 96)
    mask[..., 10:30, 10:30] = 1.0
    return x, mask


def _fit(x, mask, **kw):
    p = PatchParam(mask=mask, placement="complement", **kw)
    p.reset(x, 0)
    return p


def _noise_into(p, x):
    g = torch.Generator().manual_seed(3)
    with torch.no_grad():
        p.c.copy_(torch.rand(x.shape, generator=g))
    return p


@pytest.mark.parametrize("kw", [
    {}, {"lowfreq": SIGMA}, {"chroma": True}, {"lowfreq": SIGMA, "chroma": True},
])
def test_兩個約束都保留恆等起點(kw):
    """`init=identity` 的第 0 步必須逐位元等於原圖。

    這是相位族 θ=0、色彩族零半徑的同一條性質：收斂曲線的第一點是恆等。
    約束若寫錯方向（例如減錯一項），起點就會偏掉，而那看起來只像是
    「第一步的損失有點高」。
    """
    x, mask = _scene()
    with torch.no_grad():
        got = _fit(x, mask, **kw).render(x)
    assert torch.allclose(got, x, atol=1e-6)


def test_低頻替換把低頻誤差壓掉一個數量級以上():
    """學到純噪聲時，交出去的圖與原圖的低頻差必須遠小於不加約束的那張。

    量的是**交出去的那張**（夾取之後），不是夾取之前——夾取會把約束破壞一點，
    而使用者看到的是夾取之後的東西。
    """
    x, mask = _scene()
    free = _noise_into(_fit(x, mask), x)
    held = _noise_into(_fit(x, mask, lowfreq=SIGMA), x)
    w = held.support.to(x.dtype)

    def err(p):
        with torch.no_grad():
            d = (box_blur(p.render(x), SIGMA) - box_blur(x, SIGMA)) * w
        return float(d.pow(2).sum())

    assert err(held) < err(free) / 10.0


def test_凍結亮度在夾取之前精確成立且色度不變():
    """BT.601 的三個權重和為 1，故逐通道加同一個純量讓 Y 精確等於原圖的 Y，
    而 `B − Y`、`R − Y` 兩個色度分量逐位元不變。

    這一條是**構造**不是近似，故用嚴格的容差釘住。夾取之後不再精確，
    那由 `content_stats` 逐格量出來。
    """
    x, mask = _scene()
    p = _noise_into(_fit(x, mask, chroma=True), x)
    with torch.no_grad():
        field = p._constrain(p._field(x), x)
    assert torch.allclose(luma(field), luma(x), atol=1e-6)
    free = p._field(x)
    for c in (0, 2):        # R − Y 與 B − Y
        assert torch.allclose(field[:, c:c + 1] - luma(field),
                              free[:, c:c + 1] - luma(free), atol=1e-6)


def test_content_stats_在關閉時留空而不是填零():
    """空字串與 0.0 是不同的事：0.0 會被讀成「約束完全成立」。"""
    x, mask = _scene()
    off = _fit(x, mask).content_stats(x)
    assert off["patch_lowfreq_err"] == "" and off["patch_luma_err"] == ""
    on = _noise_into(_fit(x, mask, lowfreq=SIGMA, chroma=True), x).content_stats(x)
    assert 0.0 < on["patch_lowfreq_err"] < 1.0
    assert 0.0 < on["patch_luma_err"] < 1.0


def test_兩個約束逐列寫進geometry():
    """旗標與實際跑的物件分岔時，只有物件這一邊會說實話。"""
    x, mask = _scene()
    g = _fit(x, mask, lowfreq=SIGMA, chroma=True).geometry()
    assert g["patch_lowfreq"] == SIGMA and g["patch_chroma"] == 1
    g0 = _fit(x, mask).geometry()
    assert g0["patch_lowfreq"] == 0 and g0["patch_chroma"] == 0


def test_負的lowfreq拋錯():
    with pytest.raises(ValueError, match="lowfreq"):
        PatchParam(lowfreq=-1)


# ── 載體的面積對齊 ──────────────────────────────────────────────────

def _carrier_scene():
    mask = torch.zeros(1, 1, 128, 128)
    mask[..., 10:40, 20:60] = 1.0
    carrier = torch.zeros(1, 1, 128, 128)
    carrier[..., 8:120, 6:122] = 1.0
    return carrier, mask


@pytest.mark.parametrize("target", [0.5, 0.32, 0.146, 0.05])
def test_面積對齊精確命中(target):
    """對齊之後的合法面積必須等於目標，不是「接近」。

    只用整數侵蝕會落在離散的階梯上（一級就可能掉好幾個百分點），那時
    「同面積」這個被控制的變因實際上沒有被控制，而報表上看不出來。
    """
    carrier, mask = _carrier_scene()
    got = legal_area(match_area(carrier, mask, target), mask)
    assert abs(got - target) < 1e-5


def test_面積對齊只會讓權重下降():
    """主體那一側恆為 0 的保證由 `reset` 的構造給，前提是載體只會變小。"""
    carrier, mask = _carrier_scene()
    out = match_area(carrier, mask, 0.2)
    assert bool((out <= carrier + 1e-6).all())


def test_目標大於可用面積時拋錯():
    carrier, mask = _carrier_scene()
    with pytest.raises(ValueError, match="達不到"):
        match_area(carrier, mask, 0.99)


def test_方形侵蝕可結合():
    """`match_area` 逐級累進取代大半徑，前提是 `A ⊖ S_{2r+1} = (A ⊖ S_3) ⊖ …`。

    半徑 256 的最小池化核是 513×513，單次就跑不完；逐級累進的代價是線性的，
    但**只有在這個等式成立時兩者才是同一件事**。
    """
    carrier, _ = _carrier_scene()
    step = carrier
    for _ in range(5):
        step = erode_mask(step, 1)
    assert torch.allclose(step, erode_mask(carrier, 5))


def test_背景是登記的載體且只含類別零():
    """類別編號寫錯不會拋錯，只會指到別的東西，故由測試釘住。"""
    assert CARRIER_CLASSES["background"] == (0,)
    assert 0 not in CARRIER_CLASSES["clothes"]


# ── frame 載體與兩種面積縮法 ────────────────────────────────────────

def test_frame是空類別集合且代表整張畫面():
    """空的類別集合是「不由語意限制」的標記，不是「忘了填」。"""
    assert CARRIER_CLASSES["frame"] == ()
    seg = torch.full((64, 64), 4, dtype=torch.long)
    seg[:20, :20] = FACE_CLASS          # 讓有人守門通過
    got = carrier_from_seg(seg, "frame", max_area=1.0)
    assert float(got.mean()) == 1.0


def test_frame仍然擋掉沒有人的影像():
    """整張畫面當載體不代表可以跳過有人守門：那兩件事互相獨立。"""
    seg = torch.full((64, 64), 4, dtype=torch.long)      # 全是衣服、沒有臉
    with pytest.raises(ValueError, match="臉的面積"):
        carrier_from_seg(seg, "frame", max_area=1.0)


@pytest.mark.parametrize("target", [0.4, 0.229, 0.05])
def test_縮放模式精確命中且形狀不變(target):
    """`scale` 只把權重乘上常數：面積精確、而支撐的形狀逐位元不變。

    這是它與 `erode` 的分界——後者改變輪廓，前者不改。兩者的 mean(w) 相同，
    合成一個旋鈕就問不出「同樣的預算攤開還是集中比較好」。
    """
    carrier, mask = _carrier_scene()
    out = match_area(carrier, mask, target, mode="scale")
    assert abs(legal_area(out, mask) - target) < 1e-6
    nz = lambda t: (t > 0).to(torch.int8)
    assert torch.equal(nz(out), nz(carrier))


def test_兩種縮法給出不同的支撐():
    """同一個目標面積、同一個載體，兩種模式的結果必須不同——否則其中一個
    是死程式碼，而報表上兩列看起來像是兩個工作點。"""
    carrier, mask = _carrier_scene()
    a = match_area(carrier, mask, 0.2, mode="erode")
    b = match_area(carrier, mask, 0.2, mode="scale")
    assert not torch.allclose(a, b)
    for t in (a, b):
        assert abs(legal_area(t, mask) - 0.2) < 1e-5


def test_未知的縮法拋錯():
    carrier, mask = _carrier_scene()
    with pytest.raises(ValueError, match="mode"):
        match_area(carrier, mask, 0.2, mode="shrink")


# ── 點陣支撐 ────────────────────────────────────────────────────────

def test_點陣的token覆蓋率是滿的而像素面積不是():
    """這是點陣存在的**唯一**理由：實測效果由「碰到多少 latent token」決定，
    而 SD 的 VAE 降採樣 8 倍。pitch 8 讓每個 latent 格恰好被碰到一次，
    故 token 覆蓋率 100% 而像素面積只有 π·r²/64。

    這一條若壞了（例如格點偏移算錯），支撐看起來仍然像點陣、面積也對，
    只有覆蓋率悄悄掉下去——那正是這個方法的全部價值所在。
    """
    import torch.nn.functional as F
    full = torch.ones(1, 1, 128, 128)
    m = lattice_support(full, 8, 2.0)
    assert 0.15 < float(m.mean()) < 0.22
    tok = F.max_pool2d(m, 8, 8)
    assert float(tok.min()) == 1.0, "有 latent 格完全沒被碰到"


def test_點陣只落在載體內():
    """臉的排除與「主體那一側恆為 0」的保證，靠的是與載體相乘。"""
    carrier = torch.zeros(1, 1, 128, 128)
    carrier[..., 32:96, 32:96] = 1.0
    m = lattice_support(carrier, 8, 2.0)
    assert float(m[..., :32, :].max()) == 0.0
    assert float(m[..., 96:, :].max()) == 0.0


def test_圓斑相連時拋錯():
    """2r ≥ pitch 時支撐退化成一整片，那就不是點陣了——要滿版有別的旗標。"""
    full = torch.ones(1, 1, 64, 64)
    with pytest.raises(ValueError, match="太大"):
        lattice_support(full, 8, 4.0)
    with pytest.raises(ValueError, match="pitch"):
        lattice_support(full, 1, 0.4)


def test_半徑越大面積越大而覆蓋率不變():
    import torch.nn.functional as F
    full = torch.ones(1, 1, 128, 128)
    areas = [float(lattice_support(full, 8, r).mean()) for r in (1.0, 1.5, 2.0, 2.5)]
    assert areas == sorted(areas)
    for r in (1.0, 2.5):
        assert float(F.max_pool2d(lattice_support(full, 8, r), 8, 8).min()) == 1.0

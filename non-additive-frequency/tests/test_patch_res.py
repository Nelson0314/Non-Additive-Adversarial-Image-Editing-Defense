"""帶限參數化（`--patch-res`）的六個不變量。

存在理由
────────────────────────────────────────────────────────────────────
`docs/DIRECTION.md` §6.1b 的處境是「補丁載體有效但撐不住低通」：模糊 1%、
重取樣 1%、JPEG 75 只剩 12%。`res` 是針對這一條的旋鈕——把可學張量限制在
`1/res` 解析度上，於是內容**依構造**沒有細於 `res` 像素的分量。

這裡要釘住的是「依構造」那三個字。四個會靜默失效的地方：

1. **`res = 1` 必須逐位元等於加入這個旗標之前。** 否則所有既有的補丁批次
   都不再可比，而 CSV 上看不出來。
2. **高頻真的被砍掉了。** 只檢查張量形狀不夠——形狀對而升取樣寫錯（例如用
   `nearest`）的話，輸出會有階梯狀的高頻，而那正是要避免的東西。故用頻譜
   能量直接量。
3. **`res` 與 `tile` 的順序。** 先平舖再升取樣會跨過磚縫內插，週期性被抹掉
   一條；兩者都開時輸出必須仍然是嚴格週期的。
4. **隨機臂也要吃到這兩個旗標。** 它的可學張量是在 `reset` 裡另外抽的，
   形狀寫成 `x01.shape` 的話兩個旗標對它完全沒有作用，而地板因此量錯——
   詳見檔尾那一組。

只用 CPU 與小張合成影像，不載任何權重。
"""

import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.defense.patch_param import PatchParam  # noqa: E402


def img(side=128, seed=1):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(1, 3, side, side, generator=g)


def mask_block(h=128, w=128, top=40, left=40, side=40):
    m = torch.zeros(1, 1, h, w)
    m[..., top:top + side, left:left + side] = 1.0
    return m


def high_band_ratio(field: torch.Tensor, cutoff: float) -> float:
    """歸一化頻率高於 `cutoff` 的能量佔比。`cutoff` 以 Nyquist 為 1。

    **必須先扣掉直流。** 不扣的話自然影像的 DC 佔掉幾乎全部能量，白噪聲量
    出來的高頻佔比會是 0.098 而不是 0.989——那個數字看起來完全合理，而且會
    讓「帶限有沒有生效」的判斷整個反過來。
    """
    g = field.mean(1).detach()
    g = g - g.mean()
    f = torch.fft.fftshift(torch.fft.fft2(g), dim=(-2, -1))
    h, w = f.shape[-2:]
    fy = torch.fft.fftshift(torch.fft.fftfreq(h))[:, None] * 2.0
    fx = torch.fft.fftshift(torch.fft.fftfreq(w))[None, :] * 2.0
    r = (fy ** 2 + fx ** 2).sqrt()
    e = (f.abs() ** 2)[0]
    return float(e[r > cutoff].sum() / e.sum())


def test_res為1時逐位元等於沒有這個旗標():
    """關閉值必須是恆等，否則既有的補丁批次全部不可比而報表上看不出來。"""
    x, m = img(), mask_block()
    a = PatchParam(placement="complement", mask=m, init="random")
    b = PatchParam(placement="complement", mask=m, init="random", res=1)
    a.reset(x, seed=3)
    b.reset(x, seed=3)
    assert torch.equal(a.c, b.c)
    assert torch.equal(a.render(x), b.render(x))


def test_可學張量縮小到1除以res():
    x, m = img(), mask_block()
    p = PatchParam(placement="complement", mask=m, init="random", res=8)
    p.reset(x, seed=0)
    assert tuple(p.c.shape) == (1, 3, 16, 16)
    # 內容場仍是全解析度——縮的是參數，不是輸出。
    assert tuple(p._field(x).shape[-2:]) == (128, 128)


def test_遠高於截止的能量掉兩個數量級():
    """`res` 存在的理由。形狀對而升取樣寫錯（例如 nearest）時這裡會抓到。

    量的是 `f > 1/2`——**一次 2× 降取樣會抹掉的那一整段**，也就是模糊與重
    取樣真正吃掉的東西。實測 res=1 是 0.806、res=4 是 0.007（115 倍）。

    刻意**不**量 `f > 1/res`：雙線性核的 sinc² 旁瓣讓那一段在每個 `res` 上
    都留著 12–22%，拿它當判準會得到「帶限沒有生效」這個錯誤的結論。
    """
    x, m = img(), mask_block()
    ratios = {}
    for r in (1, 4, 16):
        p = PatchParam(placement="complement", mask=m, init="random", res=r)
        p.reset(x, seed=5)
        ratios[r] = high_band_ratio(p._field(x), cutoff=0.5)
    assert ratios[1] > 0.5, ratios
    assert ratios[4] < ratios[1] / 50, ratios
    assert ratios[16] < ratios[4], ratios


def test_平舖與帶限同時開時仍是嚴格週期():
    """順序若對調（先平舖再升取樣），磚縫會被內插掉，這裡會抓到。"""
    x, m = img(), mask_block()
    p = PatchParam(placement="complement", mask=m, init="random",
                   tile=32, res=4)
    p.reset(x, seed=7)
    assert tuple(p.c.shape) == (1, 3, 8, 8)
    f = p._field(x)
    assert torch.equal(f[..., :32, :32], f[..., 32:64, 32:64])


def test_res的恆等起點是原圖的低通版本():
    """子空間裡沒有原圖，但有一個離它最近的點；不是平舖那個常數場。"""
    x, m = img(), mask_block()
    p = PatchParam(placement="complement", mask=m, init="identity", res=4)
    p.reset(x, seed=0)
    ref = F.interpolate(x, size=(32, 32), mode="bilinear",
                        align_corners=False, antialias=True)
    assert torch.allclose(p.c, ref)
    # 不是常數場——平舖的恆等起點是畫面均值，標準差恰為零。
    tiled = PatchParam(placement="complement", mask=m, init="identity", tile=8)
    tiled.reset(x, seed=0)
    assert float(tiled.c.std()) == 0.0
    assert float(p.c.detach().std()) > 0.02


def test_res的值域檢查():
    with pytest.raises(ValueError, match="res"):
        PatchParam(res=0)
    with pytest.raises(ValueError, match="res"):
        PatchParam(tile=8, res=16)


def test_res逐列寫進geometry():
    """CSV 要有 `patch_res` 欄：同一個參數量在頻譜上可以是兩件不同的事。"""
    x, m = img(), mask_block()
    p = PatchParam(placement="complement", mask=m, init="random", res=4)
    p.reset(x, seed=0)
    g = p.geometry()
    assert g["patch_res"] == 4
    assert g["patch_tile"] == 0


# ------------------------------------------------ 隨機臂也必須吃到這兩個旗標
#
# 第六個不變量，比前五個更容易靜默失效：**隨機臂的可學張量是在 `reset` 裡
# 另外抽的**，若那一行用 `x01.shape` 而不是 `_content_shape`，`res` 與 `tile`
# 對隨機臂就完全沒有作用——`_field` 的 `interpolate` 變成同尺寸的恆等、
# `repeat` 之後的裁切又切回原張量，兩步都不拋錯。
#
# 後果不是「隨機臂稍微不準」，而是**地板量錯了**：帶限批的 `s16_rand` 會與
# `free` 的隨機臂逐位元相同，於是「最佳化在帶限之下買到了什麼」這個問題
# 被拿一個錯的分母去答，而 CSV 的 `patch_res` 欄仍然寫著 16。


def _random_arm(**kw):
    from src.defense.patch_param import PatchRandomParam
    return PatchRandomParam(placement="complement", mask=mask_block(), **kw)


def test_隨機臂的可學張量也縮小到1除以res():
    x = img()
    p = _random_arm(res=8)
    p.reset(x, seed=0)
    assert tuple(p.c.shape) == (1, 3, 16, 16)
    assert tuple(p._field(x).shape[-2:]) == (128, 128)


def test_隨機臂不同res必須給出不同的圖():
    """形狀對而渲染路徑沒吃到，這一條才會抓到。"""
    x = img()
    outs = {}
    for res in (1, 4, 8):
        p = _random_arm(res=res)
        p.reset(x, seed=0)
        outs[res] = p.render(x)
    assert not torch.equal(outs[1], outs[4])
    assert not torch.equal(outs[4], outs[8])


def test_隨機臂的帶限也真的砍掉高頻():
    """與 `test_遠高於截止的能量掉兩個數量級` 同一個量，換成隨機臂。"""
    x = img()
    e = {}
    for res in (1, 8):
        p = _random_arm(res=res)
        p.reset(x, seed=0)
        f = p._field(x)
        F_ = torch.fft.fftshift(torch.fft.fft2(f - f.mean()), dim=(-2, -1))
        pw = (F_.abs() ** 2).sum(1)[0]
        h, w = pw.shape
        yy, xx = torch.meshgrid(torch.arange(h) - h // 2,
                                torch.arange(w) - w // 2, indexing="ij")
        r = ((yy / (h / 2)) ** 2 + (xx / (w / 2)) ** 2).sqrt()
        e[res] = float(pw[r > 0.5].sum() / pw.sum())
    assert e[8] < e[1] / 20, f"res=8 的高頻佔比 {e[8]:.4f} 沒有明顯低於 res=1 的 {e[1]:.4f}"


def test_隨機臂的平舖也真的是週期的():
    x = img()
    p = _random_arm(tile=32)
    p.reset(x, seed=0)
    f = p._field(x)
    assert torch.equal(f[..., :32, :32], f[..., 32:64, :32])
    assert torch.equal(f[..., :32, :32], f[..., :32, 32:64])


def test_隨機臂在res為1時逐位元不變():
    """既有的 `patch_rand` 批次必須仍然可重現——`_content_shape` 在
    `res=1, tile=0` 時回傳的形狀與 `x01.shape` 相同、抽樣順序也相同。"""
    x = img()
    p = _random_arm(res=1)
    p.reset(x, seed=7)
    g = torch.Generator(device="cpu").manual_seed(7)
    assert torch.equal(p.c, torch.rand(x.shape, generator=g))

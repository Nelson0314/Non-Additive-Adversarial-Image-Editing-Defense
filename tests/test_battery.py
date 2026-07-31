"""候選指標組的行為測試。

篩選實驗的結論全部建立在「這些指標對模糊收費」之上，故此處斷言的不是
數值本身，而是**方向與可鑽性**：

1. 恆等時每個指標都取到最佳值；
2. 模糊必須被判為比恆等差；
3. **`acutance` 的抵銷漏洞真的存在，而 GMSD 不受其影響**——這是把 GMSD
   而非 `acutance` 放進約束的直接理由，必須有測試釘住，否則哪天有人把
   約束換回 `acutance` 不會有任何東西擋。

影像取 256²：MS-SSIM 需要五層金字塔（下限約 176），MUSIQ 需要 ≥224。
"""

import pytest
import torch
import torch.nn.functional as F

from src.metrics.battery import HIGHER_IS_BETTER, MetricBattery


@pytest.fixture(scope="module")
def bat():
    return MetricBattery()


def _img(seed=20260731, size=256):
    """有結構的圖樣。純隨機雜訊的梯度過於均勻，模糊與加噪的差別不明顯。"""
    g = torch.Generator().manual_seed(seed)
    yy, xx = torch.meshgrid(torch.arange(size), torch.arange(size), indexing="ij")
    x = torch.zeros(1, 3, size, size)
    x[0, 0] = ((xx + yy) % 16 < 8).float()
    x[0, 1] = ((xx // 8) % 2 == 0).float()
    x[0, 2] = torch.rand(size, size, generator=g)
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


def test_方向表涵蓋所有有參考指標(bat):
    """新增指標卻忘了登記方向，報告端會靜默把它當成低者佳。"""
    got = set(bat.full_reference(_img(), _img()).keys())
    assert got - set(HIGHER_IS_BETTER) == {"acutance_ratio"}


def test_恆等時取到最佳值(bat):
    x = _img()
    m = bat.full_reference(x, x)
    assert m["gmsd"] == pytest.approx(0.0, abs=1e-5)
    assert m["nlpd"] == pytest.approx(0.0, abs=1e-4)
    assert m["lpips"] == pytest.approx(0.0, abs=1e-4)
    assert m["stlpips"] == pytest.approx(0.0, abs=1e-4)
    assert m["ms_ssim"] == pytest.approx(1.0, abs=1e-4)
    assert m["haarpsi"] == pytest.approx(1.0, abs=1e-4)
    assert m["acutance_ratio"] == pytest.approx(1.0, rel=1e-6)


@pytest.mark.parametrize("key", ["gmsd", "nlpd", "lpips", "dists", "stlpips"])
def test_模糊使低者佳的指標變差(bat, key):
    x = _img()
    assert bat.full_reference(x, _blur(x))[key] > bat.full_reference(x, x)[key]


@pytest.mark.parametrize("key", ["ms_ssim", "haarpsi", "vif_p", "ssim"])
def test_模糊使高者佳的指標變差(bat, key):
    x = _img()
    assert bat.full_reference(x, _blur(x))[key] < bat.full_reference(x, x)[key]


def _img_midtone(seed=20260731, size=256):
    """中間調的圖樣，像素值落在 0.3–0.7。

    此測試不能用 `_img()`：後者的通道 0/1 是飽和的 0/1 二值，加性雜訊夾回
    [0,1] 後只能單向壓縮邊緣對比，實測會使銳利度比**下降**（半張模糊 0.632，
    加噪到 amp=0.4 反而降到 0.530），抵銷根本不可能發生。那是該測試圖的
    性質，不是 `acutance` 的性質。自然影像多半不飽和，故此處用中間調圖樣。
    """
    g = torch.Generator().manual_seed(seed)
    yy, xx = torch.meshgrid(torch.arange(size), torch.arange(size), indexing="ij")
    x = torch.zeros(1, 3, size, size)
    x[0, 0] = torch.where((xx + yy) % 16 < 8, 0.30, 0.70)
    x[0, 1] = torch.where((xx // 8) % 2 == 0, 0.35, 0.65)
    x[0, 2] = 0.35 + 0.30 * torch.rand(size, size, generator=g)
    return x


def test_acutance_的抵銷漏洞存在而_GMSD_不受影響(bat):
    """一半模糊、一半加噪：全域梯度能量比可以湊回 1，但影像明顯失真。

    這是把 GMSD 而非 `acutance` 放進保真約束的理由。`acutance` 是整張圖的
    能量比，兩種相反的失真會互相抵銷；GMSD 取逐點梯度相似度的**空間標準差**，
    局部差異無法透過他處的反向差異補回。
    """
    x = _img_midtone()
    half = x.shape[-1] // 2
    y = x.clone()
    y[..., :half] = _blur(x)[..., :half]
    assert bat.full_reference(x, y)["acutance_ratio"] < 0.7, "半張模糊應大幅降低銳利度"

    # 對另一半加噪，二分搜尋雜訊強度使全域銳利度比回到 1
    n = torch.randn(x.shape, generator=torch.Generator().manual_seed(7))

    def mix(amp):
        z = y.clone()
        z[..., half:] = (y[..., half:] + amp * n[..., half:]).clamp(0, 1)
        return z

    lo, hi = 0.0, 0.5
    assert bat.full_reference(x, mix(hi))["acutance_ratio"] > 1.0, "搜尋上限不足"
    for _ in range(40):
        amp = 0.5 * (lo + hi)
        if bat.full_reference(x, mix(amp))["acutance_ratio"] < 1.0:
            lo = amp
        else:
            hi = amp

    m = bat.full_reference(x, mix(0.5 * (lo + hi)))
    # 漏洞成立：銳利度比湊回 1 附近，該指標對這張圖幾乎不收費
    assert m["acutance_ratio"] == pytest.approx(1.0, abs=0.02)
    # 但影像確實失真，且 GMSD 與 LPIPS 都收得到費
    assert m["gmsd"] > 0.05
    assert m["lpips"] > 0.05


def test_無參考指標對模糊變差(bat):
    x = _img()
    a, b = bat.no_reference(x), bat.no_reference(_blur(x))
    assert b["musiq"] < a["musiq"]


def test_過小影像的無參考指標回傳_nan(bat):
    """定義域限制不該升級成故障：回傳 NaN 讓它在 CSV 裡看得見缺席。"""
    m = bat.no_reference(torch.rand(1, 3, 64, 64))
    assert m["niqe"] != m["niqe"]
    assert m["musiq"] != m["musiq"]

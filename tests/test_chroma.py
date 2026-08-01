"""色度度量的性質。

多數測試釘的是由定義直接推導的性質（白點、灰階、對稱性、零點）。CIEDE2000
另有一項對 `scikit-image` 的交叉驗證，因為它的絕對尺度不是由簡單性質
就能確認的——公式含分項加權與藍區旋轉項，寫錯其中一項仍可能通過所有性質
測試。該測試在缺 `scikit-image` 的環境自動跳過（遠端容器就沒有）。
"""

import pytest
import torch

from src.metrics.chroma import (
    dchroma_map, de2000_map, de76_map, local_dchroma_dev, rgb_to_lab,
)

SIZE = 64
DEV = torch.device("cpu")


def _const(rgb) -> torch.Tensor:
    x = torch.tensor(rgb, dtype=torch.float32).view(1, 3, 1, 1)
    return x.expand(1, 3, SIZE, SIZE).contiguous()


# ---- CIELAB 的錨點：由定義可直接推導 ----

def test_白點的Lab為100與零色度():
    lab = rgb_to_lab(_const([1.0, 1.0, 1.0]))
    assert float(lab[0, 0].mean()) == pytest.approx(100.0, abs=1e-3)
    assert float(lab[0, 1].abs().max()) < 1e-3
    assert float(lab[0, 2].abs().max()) < 1e-3


def test_黑點的L為零():
    lab = rgb_to_lab(_const([0.0, 0.0, 0.0]))
    assert float(lab[0, 0].mean()) == pytest.approx(0.0, abs=1e-4)


@pytest.mark.parametrize("v", [0.2, 0.5, 0.8])
def test_任何灰階的色度都是零(v):
    """R=G=B 必落在中性軸上。site C 的『無彩區域沒有容量』與此對偶：
    無彩處色度為零，故色度約束在該處也不施力。"""
    lab = rgb_to_lab(_const([v, v, v]))
    assert float(lab[0, 1].abs().max()) < 1e-4
    assert float(lab[0, 2].abs().max()) < 1e-4


# ---- 色差的基本性質 ----

def _pair():
    g = torch.Generator().manual_seed(20260728)
    a = torch.rand(1, 3, SIZE, SIZE, generator=g)
    b = (a + 0.05 * torch.randn(a.shape, generator=g)).clamp(0, 1)
    return a, b


@pytest.mark.parametrize("fn", [de76_map, de2000_map, dchroma_map])
def test_同一張圖的色差為零(fn):
    a, _ = _pair()
    assert float(fn(a, a).abs().max()) < 1e-4


@pytest.mark.parametrize("fn", [de76_map, de2000_map, dchroma_map])
def test_色差對調兩張圖不變(fn):
    """ΔE 是距離，必須對稱。CIEDE2000 的分項加權以兩點的平均為準，
    故對稱性成立——這是實作是否照定義寫的一個直接檢查。"""
    a, b = _pair()
    assert torch.allclose(fn(a, b), fn(b, a), atol=1e-4)


def test_中性區的de00與de76接近():
    """CIEDE2000 的分項加權在 L*≈50、C*≈0 附近趨近 1，故兩者應相近。
    這是 de00 的實作沒有寫錯量級的一個檢查（不是絕對尺度的驗證）。"""
    a = _const([0.5, 0.5, 0.5])
    b = _const([0.52, 0.52, 0.52])
    d76 = float(de76_map(a, b).mean())
    d00 = float(de2000_map(a, b).mean())
    assert d76 > 0.5, "此對的色差不該小到無法比較"
    assert d00 == pytest.approx(d76, rel=0.25)


def test_純亮度變化不計入dchroma():
    """`dchroma` 丟掉 ΔL*，故灰階變亮不算色度失真——亮度那一軸由 LPIPS 與
    局部銳利度各自把關，重複懲罰同一件事會變成另一種循環論證。"""
    a, b = _const([0.4, 0.4, 0.4]), _const([0.6, 0.6, 0.6])
    assert float(dchroma_map(a, b).mean()) < 1e-3
    assert float(de76_map(a, b).mean()) > 1.0, "同一對在 ΔE76 上必須有明顯色差"


# ---- 局部版本的存在理由 ----

def test_一處偏紅他處偏綠無法互相抵銷():
    """本函式存在的理由。色度位移是有號的二維向量，全域平均可以
    正負相消；逐區塊先取量值再平均就不行。與 `local_acutance_dev` 的
    `test_半模糊半加噪無法抵銷` 是同一種構造。
    """
    # 構造要真的相消，兩半就必須沿同一條 CIELAB 軸反向移動。加紅與加綠
    # 不行：兩者在 b* 上同號，實測有號量值仍達 8.87（a* −2.52、b* +8.51），
    # 只有 a* 相消。改以同一通道的加減構造：R 增／R 減使 a* 與 b* 都反向。
    a = _const([0.5, 0.5, 0.5])
    b = a.clone()
    b[:, 0, :, : SIZE // 2] += 0.12          # 左半 R 增
    b[:, 0, :, SIZE // 2:] -= 0.12           # 右半 R 減
    b = b.clamp(0, 1)

    lab_a, lab_b = rgb_to_lab(a), rgb_to_lab(b)
    signed = (lab_b[:, 1:] - lab_a[:, 1:]).flatten(2).mean(-1).norm()
    assert float(signed) < 1.0, (
        f"此構造的有號全域平均應該大幅相消，實得 {float(signed):.3f}")
    assert float(local_dchroma_dev(a, b)) > 8.0, "逐區塊版本必須照樣收費"


def test_可微且梯度抵達輸入():
    a, b = _pair()
    b = b.requires_grad_(True)
    local_dchroma_dev(a, b).backward()
    assert b.grad is not None and float(b.grad.abs().sum()) > 0


@pytest.mark.parametrize("fn", [de76_map, de2000_map, dchroma_map])
def test_極端值不產生NaN(fn):
    """飽和與純黑處的分母、`atan2(0,0)` 與 `t^(1/3)` 在 0 的導數都是風險點。
    產生 NaN 會讓損失整個變成 NaN 並靜默毀掉一次訓練。"""
    a = torch.zeros(1, 3, 8, 8)
    b = torch.ones(1, 3, 8, 8)
    for x, y in ((a, b), (a, a), (b, b)):
        v = fn(x, y)
        assert torch.isfinite(v).all(), f"{fn.__name__} 在極端輸入上產生非有限值"

    z = torch.zeros(1, 3, 8, 8, requires_grad=True)
    out = fn(z, torch.full((1, 3, 8, 8), 0.5))
    out.mean().backward()
    assert torch.isfinite(z.grad).all(), f"{fn.__name__} 的梯度在純黑處非有限"


# ---- local_chroma_bias：P9 之後唯一合格的候選 ----

def test_逐像素取量值再池化等於沒有池化():
    """釘住一個已證明無效的構造。`local_dchroma_dev` 先逐像素取量值、
    再做區塊平均與全域平均，兩者合起來就是全域平均——區塊結構完全沒有作用。
    P9 的實測顯示它與 `dchroma` 在每一臂上數值完全相同。

    保留這個測試是為了讓「取量值與取平均的順序寫反」這件事有一個具名的位置：修法不是調參數，
    而是改成先在區塊內取有號平均、再取量值（`local_chroma_bias`）。
    """
    from src.metrics.chroma import dchroma_map, local_dchroma_dev

    a, b = _pair()
    assert float(local_dchroma_dev(a, b)) == pytest.approx(
        float(dchroma_map(a, b).mean()), rel=1e-5)


def test_隨機色度雜訊在區塊內相消而連貫色偏不會():
    """`local_chroma_bias` 存在的理由。

    兩種失真的逐像素色度誤差量值相同，但一個是隨機的、一個是連貫的。
    ΔE 那一族分不出來（P9 實測等 LPIPS 下雜訊 2.44 對色偏 2.79），
    而人眼分得出來。
    """
    from src.metrics.chroma import dchroma_map, local_chroma_bias

    g = torch.Generator().manual_seed(20260728)
    base = _const([0.5, 0.5, 0.5])

    # 兩臂的振幅是量出來的，不是猜的：雜訊 0.06 與 R+0.12 使逐像素色度誤差
    # 幾乎相等（12.82 vs 12.58），此時兩者的差別只剩空間結構。
    noisy = (base + 0.06 * torch.randn(base.shape, generator=g)).clamp(0, 1)
    shifted = base.clone()
    shifted[:, 0] += 0.12                      # 整張連貫偏紅
    shifted = shifted.clamp(0, 1)

    dn = float(dchroma_map(base, noisy).mean())
    ds = float(dchroma_map(base, shifted).mean())
    bn = float(local_chroma_bias(base, noisy))
    bs = float(local_chroma_bias(base, shifted))

    # 兩者的逐像素量值刻意取到相近；若構造偏離，下面的比較就沒有意義
    assert 0.5 < dn / ds < 2.0, (
        f"兩臂的逐像素色度誤差應相近才有比較意義，實得 {dn:.3f} vs {ds:.3f}")
    assert bs / max(bn, 1e-9) > 5.0, (
        f"連貫色偏的偏壓應遠高於隨機雜訊，實得 {bs:.3f} vs {bn:.3f}")


def test_local_chroma_bias可微():
    from src.metrics.chroma import local_chroma_bias

    a, b = _pair()
    b = b.requires_grad_(True)
    local_chroma_bias(a, b).backward()
    assert b.grad is not None and float(b.grad.abs().sum()) > 0


def test_de00與skimage一致():
    """對第三方實作的交叉驗證。CIEDE2000 的公式含亮度／彩度／色相的
    分項加權與藍區的旋轉項；寫錯其中任一項仍可能通過上面所有的性質測試
    （零點、對稱性、中性區與 ΔE76 相近都不觸及那些項）。故絕對尺度必須另外
    對照，否則「τ 由實測值決定上下界」夾的可能是一把刻度錯的尺。

    實測差距屬 float32 對 float64 的精度層級：CIELAB 最大絕對差 0.0048、
    ΔE00 最大絕對差 0.0071（相對 0.016%）。
    """
    skimage_color = pytest.importorskip("skimage.color")
    import numpy as np

    from src.metrics.chroma import de2000_map, rgb_to_lab

    g = np.random.default_rng(0)
    a = g.random((64, 64, 3)).astype(np.float32)
    b = g.random((64, 64, 3)).astype(np.float32)
    ta = torch.from_numpy(a).permute(2, 0, 1)[None]
    tb = torch.from_numpy(b).permute(2, 0, 1)[None]

    lab_mine = rgb_to_lab(ta)[0].permute(1, 2, 0).numpy()
    lab_ref = skimage_color.rgb2lab(a)
    assert np.abs(lab_mine - lab_ref).max() < 0.05

    mine = de2000_map(ta, tb)[0, 0].numpy()
    ref = skimage_color.deltaE_ciede2000(skimage_color.rgb2lab(a),
                                         skimage_color.rgb2lab(b))
    assert np.abs(mine - ref).max() < 0.05
    assert mine.mean() == pytest.approx(float(ref.mean()), rel=1e-4)

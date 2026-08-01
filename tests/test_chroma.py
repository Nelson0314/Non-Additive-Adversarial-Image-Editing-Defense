"""色度度量的性質。

**這些是由定義直接推導的性質，不是與第三方實作的交叉驗證。** 本機沒有
`skimage` / `colour`，Sharma 等人的 34 組 CIEDE2000 參考值也不在手邊，故
`de00` 的**絕對尺度未經第三方驗證**（見 `src/metrics/chroma.py` 的模組
docstring）。本專案的 τ 一律由探針的實測值夾出，不依賴 ΔE 的文獻常數，
故此限制不影響用途；但若日後要引用「JND ≈ 1–2 ΔE 單位」就必須先補驗證。
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
    """**本函式存在的全部理由。** 色度位移是有號的二維向量，全域平均可以
    正負相消；逐區塊先取量值再平均就不行。與 `local_acutance_dev` 的
    `test_半模糊半加噪無法抵銷` 是同一種構造。
    """
    # 構造要真的相消，兩半就必須沿**同一條** CIELAB 軸反向移動。加紅與加綠
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

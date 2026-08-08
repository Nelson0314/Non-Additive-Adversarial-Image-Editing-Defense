"""`scripts/seam_discontinuity.py` 的接縫量 —— 邊界帶與比值的定義。

不驗真實影像上的數值（那要跑完的批次），驗的是**構造**：合成一張「接得起來」
與一張「接不起來」的圖，比值必須分得開；否則這個量對它要偵測的現象沒有
鑑別力，而它照樣會產出一個看起來合理的數字。
"""

import importlib.util
from pathlib import Path

import pytest
import torch

SPEC = importlib.util.spec_from_file_location(
    "seam_discontinuity",
    Path(__file__).resolve().parent.parent / "scripts"
    / "seam_discontinuity.py")
sd = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(sd)

SIZE = 64
BOX = (16, 48, 16, 48)


@pytest.fixture
def mask():
    m = torch.zeros(1, 1, SIZE, SIZE, dtype=torch.bool)
    m[..., BOX[0]:BOX[1], BOX[2]:BOX[3]] = True
    return m


def test_邊界帶是膨脹減侵蝕(mask):
    ring = sd.boundary_ring(mask, k=2)
    assert ring.shape == mask.shape
    # 帶寬 2k=4，故中心與遠處都不在帶上
    assert not bool(ring[0, 0, 32, 32]), "遮罩正中心不該在邊界帶上"
    assert not bool(ring[0, 0, 2, 2]), "遠離邊界處不該在邊界帶上"
    assert bool(ring[0, 0, BOX[0], 32]), "遮罩上緣應在邊界帶上"


def test_接得起來時比值接近一(mask):
    """整張同一個平滑梯度：邊界上沒有任何特殊之處。"""
    ramp = torch.linspace(0, 1, SIZE).view(1, 1, 1, SIZE).expand(1, 3, SIZE, SIZE)
    r = sd.seam_ratio(ramp.contiguous(), sd.boundary_ring(mask))
    assert r["seam"] == pytest.approx(1.0, abs=0.15), r


def test_接不起來時比值明顯大於一(mask):
    """遮罩內整塊抬高一個亮度：邊界上出現一圈階梯。

    底圖用有紋理的雜訊而非純色——純色的話遮罩外梯度為零、比值無定義，
    那是下一條測試在管的事。
    """
    g = torch.Generator().manual_seed(0)
    x = torch.rand(1, 3, SIZE, SIZE, generator=g) * 0.1 + 0.45
    x[..., BOX[0]:BOX[1], BOX[2]:BOX[3]] += 0.4
    plain = sd.seam_ratio(x - 0, sd.boundary_ring(mask))
    # 對照：沒有階梯的同一張底圖
    base = torch.rand(1, 3, SIZE, SIZE, generator=torch.Generator()
                      .manual_seed(0)) * 0.1 + 0.45
    ref = sd.seam_ratio(base, sd.boundary_ring(mask))
    assert plain["seam"] > 2.0 * ref["seam"], (
        f"階梯沒有被量到：有階梯 {plain['seam']:.3f} vs 無階梯 "
        f"{ref['seam']:.3f}")


def test_分母為零時不回傳一個看似合理的數(mask):
    """全平的圖沒有梯度，比值無定義。回 0 會被讀成「接得非常好」。"""
    r = sd.seam_ratio(torch.full((1, 3, SIZE, SIZE), 0.3),
                      sd.boundary_ring(mask))
    assert r["seam"] != r["seam"], "無定義時必須是 NaN"

"""SkinLocusCurveParam：保護是色彩空間裡的，不是座標上的。

最重要的一條是 `test_is_a_pure_colour_map`——把像素順序打亂之後輸出必須
跟著打亂，也就是輸出只由該像素的 RGB 決定。這條硬約束（純顏色全域濾鏡）
如果被破壞，抗淨化的理由就不成立，而看圖看不出來。
"""

import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.defense.ncf_param import lab_to_rgb  # noqa: E402
from src.defense.skin_locus_curve import (  # noqa: E402
    HUE_CENTRE, SkinLocusCurveParam, protection)

K = 64


def image(seed: int = 0, size: int = 64) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    coarse = torch.rand((1, 3, 8, 8), generator=generator)
    return torch.nn.functional.interpolate(
        coarse, size=(size, size), mode="bicubic", align_corners=False).clamp(0, 1)


def lab_patch(lightness, chroma, hue_degrees, n=4):
    h = math.radians(hue_degrees)
    lab = torch.tensor([lightness, chroma * math.cos(h), chroma * math.sin(h)])
    return lab_to_rgb(lab.view(1, 3, 1, 1).expand(1, 3, n, n).contiguous()).clamp(0, 1)


def carrier(protect=1.0, seed=0, jitter=1.0):
    c = SkinLocusCurveParam(radius=5.0, pieces=K, protect_scale=protect,
                            init_jitter=jitter)
    c.reset(image(seed=0), seed=seed)
    return c


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_is_a_pure_colour_map(seed):
    """打亂像素順序，輸出必須跟著打亂——輸出只能由該像素的 RGB 決定。"""
    c = carrier(seed=seed)
    x = image(seed=seed + 90)
    y = c.render(x)
    flat = x.reshape(1, 3, -1)
    order = torch.randperm(flat.shape[-1], generator=torch.Generator().manual_seed(seed))
    shuffled = flat[:, :, order].reshape(x.shape)
    y_shuffled = c.render(shuffled)
    want = y.reshape(1, 3, -1)[:, :, order].reshape(x.shape)
    assert float((y_shuffled - want).abs().max()) < 1e-6


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_output_stays_in_range_without_clamping(seed):
    c = carrier(seed=seed)
    y = c.render(image(seed=seed + 95))
    assert float(y.min()) >= -1e-6 and float(y.max()) <= 1.0 + 1e-6


def test_skin_band_is_held_still():
    """帶心、彩度夠高的顏色逐位元不動。"""
    c = carrier(protect=1.0, seed=4)
    skin = lab_patch(60.0, 25.0, HUE_CENTRE)
    assert float((c.render(skin) - skin).abs().max()) < 1e-5


def test_colours_outside_the_band_do_move():
    c = carrier(protect=1.0, seed=4)
    far = lab_patch(60.0, 25.0, HUE_CENTRE + 180.0)
    moved = float((c.render(far) - far).abs().max())
    assert moved > 1e-3, f"帶外的顏色沒有動（{moved:.2e}），保護太寬"


def test_near_neutral_colours_are_not_protected():
    """彩度趨近 0 時色相沒有意義，保護度必須趨近 0，atan2 的奇點才壓得住。"""
    grey = torch.full((1, 3, 4, 4), 0.5)
    assert float(protection(grey).max()) < 1e-3


def test_protection_is_finite_and_bounded():
    x = image(seed=17)
    p = protection(x)
    assert torch.isfinite(p).all()
    assert float(p.min()) >= 0.0 and float(p.max()) <= 1.0


def test_protect_scale_zero_is_plain_advcf():
    from src.defense.color_param import ColorCurveParam
    x = image(seed=23)
    a = carrier(protect=0.0, seed=6)
    b = ColorCurveParam(radius=5.0, pieces=K, bound_mode="advcf", init_jitter=1.0)
    b.reset(image(seed=0), seed=6)
    assert float((a.render(x) - b.render(x)).abs().max()) < 1e-6


def test_gradient_reaches_theta_and_is_finite():
    c = carrier(seed=8)
    x = image(seed=29)
    c.render(x).square().mean().backward()
    assert c.theta.grad is not None and torch.isfinite(c.theta.grad).all()
    assert float(c.theta.grad.abs().max()) > 0


def test_gradient_is_finite_on_exactly_neutral_pixels():
    """純灰的像素會讓 atan2 落在奇點上，梯度仍必須是有限值。"""
    c = carrier(seed=9)
    x = torch.full((1, 3, 8, 8), 0.5, requires_grad=False)
    x[:, :, :4] = 0.0
    c.render(x).square().mean().backward()
    assert torch.isfinite(c.theta.grad).all()


def test_construction_rejects_bad_bands():
    with pytest.raises(ValueError):
        SkinLocusCurveParam(hue_inner=40.0, hue_outer=20.0).weight(image())
    with pytest.raises(ValueError):
        SkinLocusCurveParam(protect_scale=1.5)

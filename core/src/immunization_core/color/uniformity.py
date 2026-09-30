"""Lab 位移場的均勻性與飽和讀數；色差大小見 `color.difference`。

`tv_offset` 為位移場的加權全變差，對應細碎色斑與位移邊界；`u16_offset` 為以
σ=16 平滑後的空間標準差，對應大片區域之間的色偏差異，兩者互不取代。
`new_endpoint_fraction` 只計入原圖沒有的端點像素；`raw_excursion` 量測各段在
gamut 壓縮之前的越界幅度，因 `soft_gamut` 在深度越界處的導數接近零。
"""
from __future__ import annotations

from typing import Dict, List

import torch

from immunization_core.color.space import rgb_to_lab
from immunization_core.purifiers.operators import gaussian_blur

ENDPOINT_LO = 1.0 / 255.0
ENDPOINT_HI = 254.0 / 255.0
RAMP_LO = 0.02
RAMP_HI = 0.98


def lab_offset(x01: torch.Tensor, y01: torch.Tensor) -> torch.Tensor:
    """(B,3,H,W) 的 Lab 位移場。約束下在最終交付圖上，不是參數場上。"""
    return rgb_to_lab(y01.clamp(0, 1)) - rgb_to_lab(x01.clamp(0, 1))


def _weights(off: torch.Tensor, support) -> torch.Tensor:
    if support is None:
        return torch.ones_like(off[:, :1])
    return support.to(device=off.device, dtype=off.dtype)


def tv_offset(off: torch.Tensor, support=None) -> torch.Tensor:
    """位移場的全變差，三通道平均。

    權重加在**完整位移場的相鄰差分**上，不先把支撐外設成零——那會在遮罩邊界
    憑空製造一圈巨大的差分。
    """
    w = _weights(off, support)
    dx = (off[..., :, 1:] - off[..., :, :-1]).abs()
    dy = (off[..., 1:, :] - off[..., :-1, :]).abs()
    wx = 0.5 * (w[..., :, 1:] + w[..., :, :-1])
    wy = 0.5 * (w[..., 1:, :] + w[..., :-1, :])
    tx = (dx * wx).sum() / wx.sum().clamp_min(1e-8) / off.shape[1]
    ty = (dy * wy).sum() / wy.sum().clamp_min(1e-8) / off.shape[1]
    return 0.5 * (tx + ty)


def u16_offset(off: torch.Tensor, support=None, sigma: float = 16.0) -> torch.Tensor:
    """位移場平滑之後的空間標準差，三通道平均。量的是大片區域之間的色偏差異。"""
    w = _weights(off, support)
    smooth = gaussian_blur(off, sigma)
    total = w.sum().clamp_min(1e-8)
    mean = (smooth * w).sum(dim=(2, 3), keepdim=True) / total
    var = (((smooth - mean) ** 2) * w).sum(dim=(2, 3)) / total
    return var.clamp_min(1e-12).sqrt().mean()


def endpoint_mask(x01: torch.Tensor) -> torch.Tensor:
    """任一通道落在端點上的像素。"""
    z = x01.clamp(0, 1)
    return ((z <= ENDPOINT_LO) | (z >= ENDPOINT_HI)).any(1, keepdim=True)


def new_endpoint_fraction(x01: torch.Tensor, y01: torch.Tensor,
                          support=None) -> float:
    """**新增**的端點像素比例。原圖本來就有的黑白端點不算。量測用，不可微。"""
    with torch.no_grad():
        fresh = endpoint_mask(y01) & (~endpoint_mask(x01))
        w = _weights(y01, support)
        return float((fresh.to(w.dtype) * w).sum() / w.sum().clamp_min(1e-8))


def endpoint_ramp(x01: torch.Tensor, y01: torch.Tensor, support=None) -> torch.Tensor:
    """端點的可微代理：越接近端點罰越多，原圖本身的貢獻為零。

    布林比例不能反傳，所以軟端用斜坡、硬端用 Q8 上的布林比例，兩者分開回報。
    """
    def ramp(z):
        z = z.clamp(0, 1)
        return ((z - RAMP_HI).clamp_min(0) + (RAMP_LO - z).clamp_min(0)).amax(
            1, keepdim=True) / RAMP_LO
    w = _weights(y01, support)
    fresh = (ramp(y01) - ramp(x01)).clamp_min(0)
    return (fresh * w).sum() / w.sum().clamp_min(1e-8)


def raw_excursion(carrier, x01: torch.Tensor) -> List[torch.Tensor]:
    """逐段在 gamut 壓縮之前的越界幅度，支撐內以像素平均，保持可微。

    深度越界時 `soft_gamut` 幾乎沒有梯度，所以這一條要下在壓縮**之前**的值上。
    """
    out, current = [], x01
    for stage in carrier.stages:
        raw = stage.raw_rgb(current)
        w = stage.support.to(device=raw.device, dtype=raw.dtype)
        over = torch.maximum((-raw).clamp_min(0), (raw - 1).clamp_min(0))
        over = over.amax(1, keepdim=True)
        out.append((over * w).sum() / w.sum().clamp_min(1e-8))
        current = stage.render(current)
    return out


def uniformity_row(x01: torch.Tensor, y01: torch.Tensor,
                   supports: Dict[str, torch.Tensor]) -> Dict[str, float]:
    """一整列的均勻性與飽和讀數，給 CSV 用。"""
    with torch.no_grad():
        off = lab_offset(x01, y01)
        row = {}
        for name, w in supports.items():
            row[f'tv_{name}'] = round(float(tv_offset(off, w)), 5)
            row[f'u16_{name}'] = round(float(u16_offset(off, w)), 5)
            row[f'endpoint_new_{name}'] = round(
                new_endpoint_fraction(x01, y01, w), 5)
        for c, ch in enumerate('Lab'):
            row[f'tv_{ch}'] = round(float(tv_offset(off[:, c:c + 1])), 5)
        row['offset_std'] = round(
            float(off.flatten(2).std(dim=2).mean()), 5)
        return row

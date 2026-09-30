"""逐像素 Lab 通道位移的分位數與最大值。"""
from __future__ import annotations

import torch

from immunization_core.color.space import rgb_to_lab


def _shift(x01, y01, channel, sign):
    d = rgb_to_lab(y01.clamp(0, 1).float()) - rgb_to_lab(x01.clamp(0, 1).float())
    v = d[:, channel].reshape(-1)
    return v.abs() if sign == 0 else torch.relu(sign * v)


def channel_shift_p95(x01, y01, channel, sign, q=0.95):
    """逐像素 Lab 位移在某通道、某方向的分位數；沒有往該方向動的像素記 0。"""
    return torch.quantile(_shift(x01, y01, channel, sign), q)


def channel_shift_max(x01, y01, channel, sign):
    """同上的精確最大值：p95 管不到佔比 1–3% 的小區域（如嘴唇）。"""
    return _shift(x01, y01, channel, sign).max()

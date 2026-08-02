"""E31 Task 6：`MetricSuite.pairwise` 的兩個新欄位。

與文獻的預算對比需要 RMS 這一軸。本專案在 τ=0.10 的實測是 LPIPS 0.0856、
RMS 0.0319、L∞ 0.373——L∞ 是文獻 ε=16/255 的六倍而 LPIPS 是其三分之一，
只報單一軸的對比會失真。
"""

import torch

from src.metrics.suite import HIGHER_IS_BETTER, MetricSuite


def test_pairwise含RMS與尖峰比例():
    s = MetricSuite()
    a = torch.zeros(1, 3, 64, 64)
    b = torch.zeros(1, 3, 64, 64)
    b[..., :8, :] = 0.1          # 1/8 的列偏離 0.1
    m = s.pairwise(a, b)
    assert abs(m["rms"] - 0.1 * (1 / 8) ** 0.5) < 1e-6
    # 0.1 > 16/255 ≈ 0.0627，故該 1/8 全部計入
    assert abs(m["frac_gt_16_255"] - 1 / 8) < 1e-6


def test_完全相同時兩欄皆為零():
    s = MetricSuite()
    a = torch.rand(1, 3, 64, 64)
    m = s.pairwise(a, a.clone())
    assert m["rms"] == 0.0
    assert m["frac_gt_16_255"] == 0.0


def test_兩個新鍵都登記了方向():
    # 報告與繪圖依 HIGHER_IS_BETTER 決定「較好」的方向；漏登記會讓新欄位
    # 在報告端被當成「越高越好」而畫反。
    assert HIGHER_IS_BETTER["rms"] is False
    assert HIGHER_IS_BETTER["frac_gt_16_255"] is False

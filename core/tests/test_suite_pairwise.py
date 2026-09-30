"""E31 Task 6：`MetricSuite.pairwise` 的兩個新欄位。"""

import pytest
import torch

from immunization_core.metrics.suite import HIGHER_IS_BETTER, MetricSuite


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
    assert HIGHER_IS_BETTER["rms"] is False
    assert HIGHER_IS_BETTER["frac_gt_16_255"] is False


# ---------------------------------------------------------------------------
# 計算精度與指標的邊界
# ---------------------------------------------------------------------------


def test_成對指標接受計算精度的張量():
    """指標量的是影像，不是產生它的計算精度。"""
    s = MetricSuite()
    a = torch.rand(1, 3, 64, 64)
    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        b = (a * 0.9).to(dtype)
        m = s.pairwise(a, b)          # 混精度：這正是實際的呼叫形式
        assert all(v == v for v in m.values()), f"{dtype} 產生了 NaN"
        assert s.pairwise(a.to(dtype), b) is not None


def test_niqe接受計算精度的張量():
    """段 2 的 `rayscale_executor` 對 N3 傳的 `x_tau` 直接來自 `gen.generate`。

    NIQE 需要短邊 ≥ 96，故此處用 128²；小於該值會回傳 NaN 而測不到 dtype。
    """
    s = MetricSuite()
    x = torch.rand(1, 3, 128, 128)
    ref = s.niqe(x)
    for dtype in (torch.float16, torch.bfloat16):
        got = s.niqe(x.to(dtype))
        assert got == got, f"{dtype} 下 NIQE 為 NaN"
        # 半精度的輸入本身已量化，故只要求同量級，不要求逐位相同
        assert abs(got - ref) < 1.0, f"{dtype} 下 NIQE 偏離 fp32 過多：{got} vs {ref}"

pytestmark = pytest.mark.weights

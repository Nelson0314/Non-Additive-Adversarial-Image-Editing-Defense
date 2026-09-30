"""以解析特徵圖驗證分區 LPIPS 的遮罩權重，不下載 piq 權重。"""
import sys
from types import SimpleNamespace

import pytest
import torch

from immunization_core.metrics.regional import RegionalLPIPS, split_displacement
from immunization_core.metrics.standard import blocked_by_siglip, standard_row


class FeatureMetric:
    def __init__(self):
        self.model = torch.nn.Identity()
        self.weights = [torch.ones(1, 1, 1, 1)]

    def get_features(self, image):
        return [image[:, :1]]

    def compute_distance(self, a, b):
        return [(a[0] - b[0]).square()]


def test_subject_and_complement_use_the_same_metric_weights(monkeypatch):
    monkeypatch.setitem(sys.modules, "piq", SimpleNamespace(LPIPS=FeatureMetric))
    regional = RegionalLPIPS(FeatureMetric())
    a = torch.zeros(1, 3, 2, 2)
    b = a.clone()
    b[..., 0, :] = 1
    mask = torch.zeros(1, 1, 2, 2)
    mask[..., 0, :] = 1
    assert split_displacement(regional, a, b, mask) == {
        "lpips_full": 0.5, "lpips_subject": 1.0, "lpips_background": 0.0,
    }
    assert regional(a, b, torch.ones_like(mask)) == regional(a, b)
    with pytest.raises(ValueError, match="總和為 0"):
        regional(a, b, torch.zeros_like(mask))


def test_standard_schema_keeps_required_columns_and_strict_threshold():
    row = standard_row("disp_", dict(lpips=0.123456, ssim=0.9, psnr=24.12345, vif_p=0.6, dists=0.2))
    assert row["disp_lpips"] == 0.12346
    assert row["disp_psnr"] == 24.123
    assert not blocked_by_siglip(0.837)
    assert blocked_by_siglip(0.8369)
    with pytest.raises(KeyError, match="缺欄位"):
        standard_row("disp_", {"lpips": 0.1})

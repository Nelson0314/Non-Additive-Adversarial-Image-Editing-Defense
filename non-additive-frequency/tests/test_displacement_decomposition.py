"""位移的模糊對照。

存在理由：本專案的主讀數是位移 `LPIPS(編輯(原圖), 編輯(防禦圖))`，而**防禦
成功的定義是攻擊方拿不到他要的東西**，兩者不是同一件事。人眼在比對頁上直接
看到這個落差——位移 0.137 的那一格，指令要求的顏色改變仍然成功、場景完整。

這裡釘住那個判別實驗的兩個性質：模糊確實單調地降低銳利度（二分搜尋才有
意義），以及純模糊本身會產生可觀的 LPIPS（故它是位移的下界解釋，不是零）。
"""

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src.metrics.acutance import acutance  # noqa: E402
from src.purify.ops import gaussian_blur  # noqa: E402

from displacement_decomposition import sigma_matching_acutance  # noqa: E402


def _image(size: int = 96, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.rand(1, 3, size, size, generator=g)


def test_acutance_decreases_monotonically_with_sigma():
    """二分搜尋的前提。高斯是低通，sigma 越大通帶越窄，銳利度必須單調下降。"""
    x = _image()
    ratios = [acutance(x, gaussian_blur(x, s))["acutance_ratio"]
              for s in (0.25, 0.5, 1.0, 2.0, 4.0)]
    assert ratios == sorted(ratios, reverse=True), ratios
    assert ratios[0] < 1.0


def test_matched_sigma_reproduces_the_target_acutance():
    x = _image(seed=3)
    for target_sigma in (0.5, 1.5, 3.0):
        target = acutance(x, gaussian_blur(x, target_sigma))["acutance_ratio"]
        found = sigma_matching_acutance(x, target)
        got = acutance(x, gaussian_blur(x, found))["acutance_ratio"]
        assert abs(got - target) < 1e-3, (target_sigma, found, target, got)


def test_unreachable_target_returns_nan_rather_than_extrapolating():
    """目標比最大模糊還鈍時不外插。協定禁止外插，這裡同樣不容許。"""
    x = _image(seed=5)
    assert sigma_matching_acutance(x, 1e-9) != sigma_matching_acutance(x, 1e-9)


def test_target_sharper_than_original_needs_no_blur():
    """防禦後編輯比未防禦編輯更銳時，模糊對照的 sigma 是 0，不是負數。"""
    x = _image(seed=7)
    assert sigma_matching_acutance(x, 1.5) == 0.0


def test_blur_alone_produces_substantial_lpips():
    """模糊本身就會產生可觀的 LPIPS——這正是「位移不等於防禦」的原因。

    若這條失敗（模糊的 LPIPS 趨近 0），模糊對照就沒有鑑別力，整個判別實驗
    的前提要重新檢視。
    """
    piq = pytest.importorskip("piq")
    x = _image(size=128, seed=11)
    blurred = gaussian_blur(x, 2.0)
    d = float(piq.LPIPS()(x.clamp(0, 1), blurred.clamp(0, 1)))
    assert d > 0.05, d

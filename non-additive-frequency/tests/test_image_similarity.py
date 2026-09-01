"""影像—影像的語意相似度。

存在理由：`semantic` 量的是影像對一句**文字**的對齊，而 OmniEdit 給的是
**指令**，那條路徑在服從率驗收上近乎隨機。要回答「防禦後的編輯是不是變成
另一個場景」，需要的是兩張影像之間的語意距離，不需要 caption。
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.metrics.suite import MetricSuite  # noqa: E402


def _img(seed: int, size: int = 224) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.rand(1, 3, size, size, generator=g)


@pytest.fixture(scope="module")
def suite():
    pytest.importorskip("transformers")
    return MetricSuite(device=torch.device("cpu"))


def test_identical_images_score_one(suite):
    x = _img(0)
    s = suite.image_similarity(x, x)
    for k in ("clip", "siglip"):
        assert s[k] == pytest.approx(1.0, abs=1e-3), (k, s[k])


def test_symmetric(suite):
    a, b = _img(1), _img(2)
    s1, s2 = suite.image_similarity(a, b), suite.image_similarity(b, a)
    for k in ("clip", "siglip"):
        assert s1[k] == pytest.approx(s2[k], abs=1e-5)


def test_a_shifted_copy_scores_higher_than_unrelated_noise(suite):
    """判別力的下限：同一張圖的小幅改動必須比無關的另一張圖更相似。

    這一條若失敗，這個量就分不出「內容被換掉」與「內容只是變髒」，
    也就無法用來判定防禦有沒有擋下攻擊。
    """
    a = _img(3)
    near = (a * 0.9 + 0.05).clamp(0, 1)
    far = _img(99)
    s_near = suite.image_similarity(a, near)
    s_far = suite.image_similarity(a, far)
    for k in ("clip", "siglip"):
        assert s_near[k] > s_far[k], (k, s_near[k], s_far[k])

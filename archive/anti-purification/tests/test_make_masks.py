"""`scripts/make_masks.py` 的極性與二值性。

這兩件事寫錯都不會拋錯，只會讓後續整批 inpainting 重繪到錯的地方：極性反了
會把主體重畫掉（`c_a` 落在遮罩內），非二值會讓「白到底算不算重繪」取決於下游
怎麼取整。故由測試釘住，不靠看圖。
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from make_masks import inpaint_mask  # noqa: E402


def test_重繪區是主體的補集():
    subject = torch.zeros(1, 1, 8, 8)
    subject[..., 2:6, 2:6] = 1.0
    m = inpaint_mask(subject)
    assert m[..., 2:6, 2:6].sum() == 0.0          # 主體不重繪
    assert m.sum() == 64 - 16                      # 其餘全部重繪


def test_非二值的主體遮罩拋錯():
    subject = torch.zeros(1, 1, 8, 8)
    subject[..., 2:6, 2:6] = 0.5
    with pytest.raises(ValueError, match="二值"):
        inpaint_mask(subject)


def test_feather_0_的_subject_mask_是二值的():
    """`inpaint_mask` 的前提來自 `subject_mask(..., feather=0)`。"""
    from src.defense.subject_mask import _feather

    hard = (torch.rand(1, 1, 16, 16) > 0.5).float()
    assert torch.equal(_feather(hard, 0), hard)

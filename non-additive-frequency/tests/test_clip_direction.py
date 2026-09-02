"""指令 → 敘述的機械推導，以及方向相似度的邊界。

這一份釘的是**解析**：敘述若推導錯了，CLIP 的數字仍然算得出來、看起來也正常，
但量到的是別的東西。本專案已實測到 OmniEdit 給的是**指令**不是描述，
把指令直接當 caption 做對齊近乎隨機（25 張裡 15 張為正）。
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from clip_direction import captions, direction  # noqa: E402


@pytest.mark.parametrize("instruction,src,tgt", [
    ("turn the color of potted plant to pink", "a potted plant",
     "a pink potted plant"),
    ("turn the color of toolbox to red", "a toolbox", "a red toolbox"),
    # `to be` 的寫法在 OmniEdit 裡同樣存在。
    ("turn the color of LCD screen to be green", "a LCD screen",
     "a green LCD screen"),
    # 冠詞由模板吃掉，不重複。
    ("turn the color of a stone fountain to red", "a stone fountain",
     "a red stone fountain"),
    # 英式拼法與句點。
    ("Turn the colour of the cushion to black.", "a cushion",
     "a black cushion"),
])
def test_colour_instructions_map_to_captions(instruction, src, tgt):
    assert captions(instruction) == (src, tgt)


@pytest.mark.parametrize("instruction", [
    "add a red umbrella in the background",
    "remove the dog",
    "make it snowy",
    "replace the cat with a dog",
])
def test_other_task_types_are_refused_not_guessed(instruction):
    """對不上模板就回傳 None，呼叫端跳過並印出來。

    **不套用別的模板、不猜**：五類任務裡只有 `attribute_modification` 的指令
    能機械地變成一對敘述，其餘四類沒有這個結構。
    """
    assert captions(instruction) is None


def test_direction_is_one_when_the_move_matches_the_text_axis():
    a = torch.tensor([1.0, 0.0, 0.0])
    b = torch.tensor([1.0, 0.0, 0.0])
    assert direction(torch.zeros(3), a, torch.zeros(3), b) == pytest.approx(1.0)


def test_direction_is_minus_one_when_the_move_opposes_it():
    a = torch.tensor([-1.0, 0.0, 0.0])
    b = torch.tensor([1.0, 0.0, 0.0])
    assert direction(torch.zeros(3), a, torch.zeros(3), b) == pytest.approx(-1.0)


def test_direction_is_nan_when_the_edit_did_nothing():
    """編輯完全沒動時方向無定義，回報 nan 而不是補零。

    補零會讓「沒動」與「動了但與指令正交」在表上長得一樣。
    """
    v = torch.tensor([1.0, 0.0, 0.0])
    out = direction(v, v, torch.zeros(3), torch.tensor([1.0, 0.0, 0.0]))
    assert out != out


# ── 主體遮罩 ─────────────────────────────────────────────────────────


def _mask(box, size=64, feather=8):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from decoy_masked import subject_mask

    return subject_mask(box, size, feather, torch.device("cpu"), torch.float32)


def test_subject_box_is_exactly_one_inside():
    """羽化只往外：框內恆為 1，主體因此逐位元保留。"""
    m = _mask([0.25, 0.25, 0.75, 0.75])
    inner = m[..., 20:44, 20:44]
    assert float(inner.min()) == 1.0


def test_composite_leaves_the_subject_bitwise_untouched():
    torch.manual_seed(0)
    x = torch.rand(1, 3, 64, 64)
    decoy = torch.rand(1, 3, 64, 64)
    m = _mask([0.25, 0.25, 0.75, 0.75])
    out = m * x + (1.0 - m) * decoy
    inside = m >= 1.0
    assert float(((out - x) * inside).abs().max()) == 0.0


def test_far_outside_the_box_is_entirely_decoy():
    m = _mask([0.4, 0.4, 0.6, 0.6], feather=4)
    assert float(m[0, 0, 0, 0]) == 0.0


def test_zero_feather_gives_a_hard_edge():
    m = _mask([0.25, 0.25, 0.75, 0.75], feather=0)
    assert set(m.unique().tolist()) <= {0.0, 1.0}

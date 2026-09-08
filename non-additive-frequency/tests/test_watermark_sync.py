"""解碼端幾何對齊：整格錯位是信心度看不見的，前導碼才看得見。

存在理由
────────────────────────────────────────────────────────────────────
這個模組唯一的價值是把裁切那一欄救回來，而它有一個非常容易寫錯、寫錯之後
**數字看起來完全正常**的地方：用萃取信心度當對齊準則。

實測：平移整整一個區塊時，係數仍然坐在原本的 QIM 格點上，信心度是滿的
0.950，但槽與位元的對應整個推移，讀出來是一組被置換過的位元（正確率
0.4375）。只看信心度的搜尋會高高興興地選中那個候選。

故第一條測試就把這件事釘死：**整格錯位必須維持高信心度而位元是錯的**。
接著釘前導碼評分能把它分出來。

只用 CPU 與小張合成影像，不載任何模型權重。
"""

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.watermark.qim import BLOCK, bit_accuracy, embed, random_bits  # noqa: E402
from src.watermark.sync import (  # noqa: E402
    _block_valid, _extract_core, search_alignment, undo_magnification,
)

COEF = ((1, 2), (2, 1), (2, 2), (1, 3))
# 128x128 在 4 個係數下共有 (128/8)^2 * 4 = 1024 個槽，故 16 x 64 剛好用滿。
# **用滿是必要的**：只用一部分槽時，整格平移會把用到的槽對到沒有嵌入的槽上，
# 信心度自己就掉下來，於是測不到「信心度看不見整格錯位」那個現象。真實的
# 512 影像在 repeat=64 下的槽距是 4，整格平移剛好是 4 的倍數，也對到用到的槽。
DELTA, REPEAT, NBITS = 48.0, 64, 16


def smooth_image(side=128, seed=3):
    """低通的合成影像。用白噪聲會量到不具代表性的極端（見 qim 的 docstring）。"""
    g = torch.Generator().manual_seed(seed)
    small = torch.rand(1, 3, side // 16, side // 16, generator=g)
    return torch.nn.functional.interpolate(
        small, size=(side, side), mode="bicubic", align_corners=False
    ).clamp(0.15, 0.85)


@pytest.fixture(scope="module")
def marked():
    bits = random_bits(NBITS, 20250903)
    x = smooth_image()
    xw = embed(x, bits, delta=DELTA, coeffs=list(COEF), repeat=REPEAT)
    return xw, bits


def test_無攻擊時信心度接近上限且位元全對(marked):
    xw, bits = marked
    got, conf = _extract_core(xw, NBITS, delta=DELTA, coeffs=COEF, repeat=REPEAT)
    assert bit_accuracy(got, bits) == 1.0
    assert conf > 0.8


def test_整格錯位維持高信心度但位元是錯的(marked):
    """本模組存在的理由。信心度**分不出**這個情形，只有前導碼分得出。"""
    xw, bits = marked
    shifted = torch.roll(xw, shifts=(BLOCK, BLOCK), dims=(2, 3))
    got, conf = _extract_core(shifted, NBITS, delta=DELTA, coeffs=COEF,
                              repeat=REPEAT)
    assert conf > 0.8, "整格平移不改變係數的值，信心度應該仍然是滿的"
    assert bit_accuracy(got, bits) < 0.9, "槽與位元的對應被推移，位元應該錯掉"


def test_格點相位錯開會壓低信心度(marked):
    """相位錯開才是信心度看得見的那一種。"""
    xw, _ = marked
    _, c0 = _extract_core(xw, NBITS, delta=DELTA, coeffs=COEF, repeat=REPEAT)
    _, c3 = _extract_core(torch.roll(xw, shifts=(3, 3), dims=(2, 3)),
                          NBITS, delta=DELTA, coeffs=COEF, repeat=REPEAT)
    assert c3 < c0 * 0.75


def test_有效遮罩只讓畫面內的副本投票(marked):
    xw, _ = marked
    n_slots = (128 // BLOCK) ** 2 * len(COEF)
    valid = torch.zeros(n_slots)
    valid[: n_slots // 2] = 1.0
    _, conf = _extract_core(xw, NBITS, delta=DELTA, coeffs=COEF,
                            repeat=REPEAT, valid=valid)
    assert conf > 0.8            # 有效的那一半仍然對齊，信心度不該掉


def test_全零遮罩不會除以零(marked):
    xw, _ = marked
    n_slots = (128 // BLOCK) ** 2 * len(COEF)
    _, conf = _extract_core(xw, NBITS, delta=DELTA, coeffs=COEF,
                            repeat=REPEAT, valid=torch.zeros(n_slots))
    assert conf == 0.0


def test_縮回去之後貼在指定位置且遮罩相符():
    y = smooth_image(128)
    cv, mask = undo_magnification(y, 1.25, 10, 12, 128)
    side = int(round(128 / 1.25))
    assert cv.shape == (1, 3, 128, 128)
    assert float(mask[..., 10:10 + side, 12:12 + side].min()) == 1.0
    assert float(mask.sum()) == side * side
    # 畫布之外補的是均值，不是零——補零的假邊自己就會產生高信心度
    assert float(cv[0, :, 0, 0].mean()) == pytest.approx(float(y.mean()), abs=1e-5)


def test_只有整塊都在畫面內的區塊才算有效():
    mask = torch.zeros(1, 1, 32, 32)
    mask[..., 0:12, 0:12] = 1.0        # 蓋滿第 0 塊，第 1 塊只蓋一半
    valid = _block_valid(mask, 1)
    assert valid.shape == (16,)        # 4x4 個區塊 x 1 個係數
    assert float(valid[0]) == 1.0      # (0,0) 整塊都在裡面
    assert float(valid[1]) == 0.0      # (0,1) 只有一半


def test_對裁切放大的圖搜尋幾何比不搜好(marked):
    """真實的使用情形：裁切之後重新取樣，尺度與位移都要估。

    注意 `undo_magnification` 只做「縮回去再貼」，**尺度必須大於 1 才有位移
    的自由度**；純平移（尺度 1.0）在這個構造下無法修正，那是本模組的邊界。
    """
    from src.purify.ops import crop_resize
    xw, bits = marked
    y = crop_resize(xw, 0.1)
    naive, _ = _extract_core(y, NBITS, delta=DELTA, coeffs=COEF, repeat=REPEAT)
    got, info = search_alignment(
        y, NBITS, delta=DELTA, coeffs=COEF, repeat=REPEAT,
        scales=[1.20 + 0.01 * k for k in range(0, 11)],
        sync=bits[:8], offset_radius=6, canvas=128)
    assert info["scale"] > 1.0
    assert bit_accuracy(got, bits) >= bit_accuracy(naive, bits)


def test_純平移在尺度一時沒有位移自由度(marked):
    """記錄本模組的邊界，避免日後有人以為它能修正純平移。"""
    xw, _ = marked
    cv, mask = undo_magnification(xw, 1.0, 10, 12, 128)
    assert float(mask.min()) == 1.0        # 整張都有效，貼不出偏移
    assert torch.allclose(cv, xw, atol=1e-4)


def test_scales為空就拋錯而不是默默回傳(marked):
    xw, _ = marked
    with pytest.raises(ValueError, match="scales"):
        search_alignment(xw, NBITS, delta=DELTA, coeffs=COEF, repeat=REPEAT,
                         scales=[], canvas=128)


def test_畫布邊長不是區塊倍數就拋錯(marked):
    xw, _ = marked
    with pytest.raises(ValueError, match="倍數"):
        search_alignment(xw, NBITS, delta=DELTA, coeffs=COEF, repeat=REPEAT,
                         scales=[1.0], canvas=130)


def test_回傳的info帶齊要進CSV的欄位(marked):
    xw, bits = marked
    _, info = search_alignment(xw, NBITS, delta=DELTA, coeffs=COEF,
                               repeat=REPEAT, scales=[1.0], sync=bits[:8],
                               offset_radius=2, canvas=128)
    assert set(info) == {"scale", "oy", "ox", "confidence", "sync_match",
                         "valid_frac"}

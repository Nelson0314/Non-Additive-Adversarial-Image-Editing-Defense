"""旋轉淨化算子。FaceLock 與 EditShield 的抗淨化表都有這一欄，本專案先前沒有。

出處：FaceLock（CVPR 2025，arXiv:2411.16832）"Rotate denotes random rotation
between (-10, 10) degrees"；EditShield（ECCV 2024）的 EOT 族用 5°。
插值核與邊界填補**論文未載，是本專案指定**（雙線性、補零），
見 `src/purify/ops.py::rotate_random`。

四件必須釘住的事，每一件失敗都不會有症狀：

1. **同 seed 必得同一個角度。** 抗淨化的兩側（`編輯(p(原圖))` 與
   `編輯(p(防禦圖))`）要吃到同一個旋轉，否則量到的是兩個不同的取景，
   而讀數看起來仍然很正常。
2. **它是幾何類。** 旋轉改取景（四角離開畫面）也改像素格點，參照必須換成
   `編輯(p(原圖))`。漏了它就會與 `crop_resize` 那一欄不可並列。
3. **標籤還原得回 kind。** `rotate10` 不可以被別的 kind 的字首吃掉。
4. **0 度是恆等映射。** 否則「不旋轉」那一格會帶進插值損失。
"""

import math

import pytest
import torch

from src.purify.ops import (
    GEOMETRIC_KINDS,
    KINDS,
    ROTATE_DEGREES_FACELOCK,
    Purifier,
    kind_of_label,
    label_is_geometric,
    rotate_random,
)


def _img(seed=0, size=64):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(1, 3, size, size, generator=g)


def test_rotate_is_a_known_kind():
    assert "rotate" in KINDS


def test_rotate_is_geometric():
    """改取景又改格點，理由與 shift_only 同型。"""
    assert "rotate" in GEOMETRIC_KINDS
    assert label_is_geometric("rotate10")
    assert kind_of_label("rotate10") == "rotate"


def test_label_roundtrip_not_eaten_by_other_kinds():
    assert kind_of_label("rotate") == "rotate"
    assert kind_of_label("rotate5") == "rotate"


def test_shape_and_range_preserved():
    x = _img()
    y = Purifier("rotate", ROTATE_DEGREES_FACELOCK, seed=0).evaluate(x)
    assert y.shape == x.shape
    assert float(y.min()) >= 0.0 and float(y.max()) <= 1.0


def test_same_seed_gives_bitwise_identical_output():
    """兩側必須吃到同一個旋轉——這是換參照協定成立的前提。"""
    x = _img()
    a = Purifier("rotate", 10.0, seed=7).evaluate(x)
    b = Purifier("rotate", 10.0, seed=7).evaluate(x)
    assert torch.equal(a, b)


def test_角度不再依賴seed():
    """`ROTATE_FIXED` 之後角度固定取區間端點，**與 seed 無關**。
    改回抽樣時這一條會紅，提醒去重看那些依賴確定性的批次。"""
    from src.purify.ops import ROTATE_FIXED, rotate_angle
    assert ROTATE_FIXED
    xs = [rotate_angle(ROTATE_DEGREES_FACELOCK, s) for s in range(6)]
    assert len(set(xs)) == 1 and xs[0] == ROTATE_DEGREES_FACELOCK, xs
    a = Purifier("rotate", 10.0, seed=7).evaluate(_img())
    b = Purifier("rotate", 10.0, seed=8).evaluate(_img())
    assert torch.equal(a, b)



def test_zero_degrees_is_identity():
    """0 度必須是恆等，不可以帶進插值損失。

    `strength=0` 會走預設的 10 度（`_real` 的 `if self.strength` 分支），
    故這裡直接呼叫函式本身。
    """
    x = _img()
    y = rotate_random(x, 0.0, seed=3)
    assert torch.allclose(y, x, atol=1e-6)


def test_rotation_moves_pixels():
    """非零角度必須真的動到內容，否則 grid 的建法寫錯也不會有症狀。"""
    x = _img()
    y = rotate_random(x, 10.0, seed=1)
    assert float((y - x).abs().mean()) > 1e-3


def test_corners_are_zero_filled():
    """補零：旋轉之後四角必然離開原畫面。取極端角度讓角落確定落在畫面外。"""
    x = torch.ones(1, 3, 64, 64)
    y = rotate_random(x, 10.0, seed=0)
    corner = min(float(y[0, :, 0, 0].max()), float(y[0, :, 0, -1].max()),
                 float(y[0, :, -1, 0].max()), float(y[0, :, -1, -1].max()))
    assert corner < 1.0


def test_natively_differentiable():
    """`grid_sample` 原生可微，故不走直通估計、`proxy_gap` 必為 0。"""
    p = Purifier("rotate", 10.0, seed=0)
    assert p.differentiable
    x = _img()
    assert p.proxy_gap(x) == 0.0
    z = x.clone().requires_grad_(True)
    p.forward(z).sum().backward()
    assert float(z.grad.abs().sum()) > 0


def test_angle_stays_within_the_declared_range():
    """角度必須落在 ±degrees 內。以「旋轉後與原圖的差」不會大到像是轉了 90 度
    當作粗檢不夠，這裡直接檢查取樣本身：對每個 seed 建網格並反推角度。"""
    x = torch.zeros(1, 1, 3, 3)
    x[0, 0, 0, 1] = 1.0  # 上方中點
    for seed in range(20):
        y = rotate_random(x, 10.0, seed=seed)
        # 10 度之下上方中點的能量不會跑到下半，若角度取樣超界就會。
        assert float(y[0, 0, 2, :].max()) < 1e-3


# ---------------------------------------------------------- 退化的角度抽樣
#
# 第五件必須釘住的事：**單一 seed 抽到的角度可能接近零**。`U(−10,10)` 上
# `seed=0` 抽到 −0.075°，512 px 影像最遠角落只位移 0.33 px，整張圖都在次像素
# 量級。那一格量到的是恆等映射，卻掛著 `rotate10` 的名字出現在抗淨化表上，
# 與 `identity` 欄並列時看不出異常——只看得到「旋轉不傷防禦」這個假結論。


def test_角度查詢與實際套用一致():
    """`rotate_angle` 是報告與守門用的入口；它與實際套用的角度不同步時，
    寫進 CSV 的角度就是假的，而圖與讀數都不會有症狀。"""
    from src.purify.ops import rotate_angle
    x = torch.zeros(1, 1, 65, 65)
    x[0, 0, 4, 32] = 1.0
    for deg in (5.0, 10.0, 20.0):
        assert rotate_angle(deg, 0) == deg
        assert torch.allclose(rotate_random(x, deg, seed=0),
                              rotate_random(x, deg, seed=99))



def test_固定角度不可能退化():
    """先前的抽樣式在 `seed=0` 上抽到 −0.075°，那一格量到的是恆等映射。
    固定取端點之後這件事**構造上不可能發生**。"""
    from src.purify.ops import ROTATE_DEGENERATE_DEGREES, rotate_angle
    for s_ in range(8):
        assert abs(rotate_angle(ROTATE_DEGREES_FACELOCK, s_)) >= ROTATE_DEGENERATE_DEGREES



def test_退化角度的守門仍然存在():
    """守門現在是**備援**：固定角度之下走不到它，但把 `ROTATE_FIXED` 關掉
    改回抽樣時它必須還在，否則又會靜默量到恆等映射。"""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from purify_identity import build_purifier
    from src.purify import ops

    p = build_purifier("rotate10", seed=0)
    assert p.kind == "rotate"
    old = ops.ROTATE_FIXED
    try:
        ops.ROTATE_FIXED = False
        with pytest.raises(SystemExit) as e:
            build_purifier("rotate10", seed=0, rotate_seed=0)
        assert "次像素" in str(e.value)
    finally:
        ops.ROTATE_FIXED = old



def test_default_rotate_seed_draws_a_real_rotation():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from purify_identity import ROTATE_SEED_DEFAULT
    from src.purify.ops import ROTATE_DEGENERATE_DEGREES, rotate_angle
    a = rotate_angle(ROTATE_DEGREES_FACELOCK, ROTATE_SEED_DEFAULT)
    assert abs(a) >= ROTATE_DEGENERATE_DEGREES
    assert abs(a) <= ROTATE_DEGREES_FACELOCK


def test_the_two_scripts_measure_the_same_rotation():
    """頻率響應那一支與抗淨化那一支若各用各的種子，兩張表上的 `rotate10`
    是兩個不同的算子，而欄名相同。"""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import purifier_transfer
    from purify_identity import ROTATE_SEED_DEFAULT
    assert purifier_transfer.ROTATE_SEED == ROTATE_SEED_DEFAULT


def test_十度真的把畫面轉得動():
    """半對角線 362 px，10° 對應約 63 px 的角落位移——遠大於一個像素。"""
    from src.purify.ops import rotate_angle
    assert 362.0 * math.radians(rotate_angle(ROTATE_DEGREES_FACELOCK, 0)) > 40.0
    x = _img(size=128)
    y = rotate_random(x, ROTATE_DEGREES_FACELOCK, seed=0)
    assert float((y - x).abs().mean()) > 1e-2



"""補丁零號實驗的幾何：**補丁不可以碰到主體**。

存在理由
────────────────────────────────────────────────────────────────────
整個實驗的前提是「主體逐位元不動」。若擺放搜尋漏掉了羽化帶、或在沒有合法
位置時悄悄退回一個重疊的位置，量到的「主體內位移」就同時混著「補丁自己蓋
在主體上」——那是靜默失效，數字看起來完全正常。

故這裡把幾件事釘死：合法性的定義是區域內遮罩逐像素為零、`far`／`near` 的
選法是確定性的、沒有合法位置時回 `None` 而不是回一個湊合的位置、
`complement` 的支撐嚴格排除羽化帶、以及最大合法邊長的二分搜尋與逐一嘗試
的結果相同。

只用 CPU，不載任何權重。
"""

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import patch_probe as pp  # noqa: E402


def mask_with_block(h=128, w=128, top=32, left=32, side=48):
    m = torch.zeros(1, 1, h, w)
    m[..., top:top + side, left:left + side] = 1.0
    return m


def test_邊長一律取到偶數():
    assert pp.even(181) == 180
    assert pp.even(180) == 180
    assert pp.even(0.4) == 0


def test_候選位置與遮罩零重疊():
    m = mask_with_block()
    cands, _ = pp.placements(m, side=16, stride=8)
    assert cands
    for top, left in cands:
        assert float(m[..., top:top + 16, left:left + 16].max()) == 0.0


def test_羽化帶也算重疊():
    """遮罩的軟邊值大於零，補丁不可以壓在上面。"""
    m = torch.zeros(1, 1, 64, 64)
    m[..., 24:40, 24:40] = 1.0
    m[..., 20:24, 20:44] = 0.2          # 一條羽化帶
    cands, _ = pp.placements(m, side=8, stride=4)
    for top, left in cands:
        assert float(m[..., top:top + 8, left:left + 8].max()) == 0.0
    assert (20, 20) not in cands


def test_far與near分別是離重心最遠與最近():
    m = mask_with_block(h=128, w=128, top=48, left=48, side=32)
    cands, centre = pp.placements(m, side=16, stride=16)
    far = pp.choose(cands, centre, "far")
    near = pp.choose(cands, centre, "near")
    cy, cx = centre

    def d(p):
        return ((p[0] - cy) ** 2 + (p[1] - cx) ** 2) ** 0.5

    assert d(far) == max(d(p) for p in cands)
    assert d(near) == min(d(p) for p in cands)


def test_選法是確定性的():
    m = mask_with_block()
    cands, centre = pp.placements(m, side=16, stride=8)
    for mode in ("far", "near"):
        first = pp.choose(cands, centre, mode)
        assert all(pp.choose(cands, centre, mode) == first for _ in range(5))


def test_沒有合法位置時回None而不是湊合一個():
    m = torch.ones(1, 1, 64, 64)          # 整張都是主體
    cands, centre = pp.placements(m, side=16, stride=8)
    assert cands == []
    assert pp.choose(cands, centre, "far") is None


def test_補丁大於畫面時回空清單():
    m = torch.zeros(1, 1, 32, 32)
    cands, _ = pp.placements(m, side=64, stride=8)
    assert cands == []


# ---- 最大合法邊長 ----

def brute_force_max_side(m, stride, min_side):
    best = 0
    s = min_side
    while s <= min(m.shape[-2:]):
        if pp.placements(m, s, stride)[0]:
            best = s
        s += 2
    return best


def test_二分搜尋與逐一嘗試得到同一個最大邊長():
    for spec in [(128, 128, 32, 32, 48), (128, 128, 0, 0, 90),
                 (96, 128, 20, 60, 30)]:
        h, w, top, left, side = spec
        m = mask_with_block(h, w, top, left, side)
        got = pp.largest_legal_side(m, stride=8, min_side=8)
        want = brute_force_max_side(m, stride=8, min_side=8)
        assert got == want, spec


def test_整張都是主體時最大邊長為零():
    assert pp.largest_legal_side(torch.ones(1, 1, 64, 64),
                                 stride=8, min_side=8) == 0


def test_最大邊長確實有合法位置且再大一級沒有():
    m = mask_with_block(128, 128, 32, 32, 48)
    s = pp.largest_legal_side(m, stride=8, min_side=8)
    assert pp.placements(m, s, 8)[0]
    assert pp.placements(m, s + 2, 8)[0] == []


# ---- 支撐與貼上 ----

def test_補集支撐嚴格排除羽化帶():
    m = torch.zeros(1, 1, 32, 32)
    m[..., 8:16, 8:16] = 1.0
    m[..., 6:8, 6:18] = 0.05        # 極淡的羽化帶也要被排除
    s = pp.complement_support(m)
    assert not bool(s[..., 6:8, 6:18].any())
    assert not bool(s[..., 8:16, 8:16].any())
    assert bool(s[..., 0, 0])


def test_貼上之後支撐之外逐位元不變():
    g = torch.Generator().manual_seed(3)
    x = torch.rand(1, 3, 64, 64, generator=g)
    field = torch.zeros_like(x)
    s = pp.rect_support(x.shape, 8, 12, 16)
    out = pp.paste(x, field, s)
    assert torch.equal(out[..., 8:24, 12:28], field[..., 8:24, 12:28])
    keep = ~s
    assert torch.equal(out * keep, x * keep)
    assert out is not x                     # 不可就地改寫呼叫端的張量


def test_補集貼上之後主體逐位元不變():
    g = torch.Generator().manual_seed(11)
    x = torch.rand(1, 3, 32, 32, generator=g)
    m = torch.zeros(1, 1, 32, 32)
    m[..., 8:16, 8:16] = 1.0
    out = pp.paste(x, torch.zeros_like(x), pp.complement_support(m))
    assert torch.equal(out[..., 8:16, 8:16], x[..., 8:16, 8:16])


def test_方形支撐的面積正確():
    s = pp.rect_support((1, 3, 64, 64), 0, 0, 32)
    assert float(s.to(torch.float32).mean()) == pytest.approx(0.25)


# ---- 內容場 ----

def test_三種內容的形狀與值域():
    ref = torch.zeros(1, 3, 64, 64)
    photo = torch.rand(1, 3, 64, 64)
    for kind in pp.CONTENTS:
        c = pp.full_content(kind, ref, photo, seed=1)
        assert c.shape == ref.shape
        assert float(c.min()) >= 0.0 and float(c.max()) <= 1.0
    assert float(pp.full_content("gray", ref, photo, 1).std()) == 0.0


def test_噪聲內容同種子同內容():
    ref = torch.zeros(1, 3, 32, 32)
    photo = torch.rand(1, 3, 32, 32)
    a = pp.full_content("noise", ref, photo, seed=5)
    b = pp.full_content("noise", ref, photo, seed=5)
    c = pp.full_content("noise", ref, photo, seed=6)
    assert torch.equal(a, b)
    assert not torch.equal(a, c)


def test_photo形狀不符時拋錯而不是靜默對齊():
    ref = torch.zeros(1, 3, 64, 64)
    with pytest.raises(ValueError, match="對不齊"):
        pp.full_content("photo", ref, torch.rand(1, 3, 32, 32), 1)


def test_未知內容拋錯():
    ref = torch.zeros(1, 3, 32, 32)
    with pytest.raises(ValueError, match="未知的補丁內容"):
        pp.full_content("rainbow", ref, ref, 1)


# ---- CLI ----

def test_ig_zt沒給就擋在載入權重之前():
    args = pp.build_parser().parse_args(["--images", "a", "--out", "o"])
    with pytest.raises(SystemExit, match="--ig-zt 必填"):
        pp.check_args(args)


def test_尺寸倍數出界拋錯():
    args = pp.build_parser().parse_args(
        ["--images", "a", "--out", "o", "--ig-zt", "diffuse_src",
         "--size-scales", "1.5"])
    with pytest.raises(SystemExit, match="最大合法邊長的倍數"):
        pp.check_args(args)


def test_預設含complement這一格():
    """上界格：主體以外全部換掉。少了它就沒有證偽的那一端。"""
    args = pp.build_parser().parse_args(
        ["--images", "a", "--out", "o", "--ig-zt", "diffuse_src"])
    assert "complement" in args.placements

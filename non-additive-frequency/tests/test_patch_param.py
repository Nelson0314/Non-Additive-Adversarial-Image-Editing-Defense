"""可學補丁的三個不變量：主體逐位元不動、起點即恆等、放不下就拋錯。

存在理由
────────────────────────────────────────────────────────────────────
這個參數化的整個前提是「主體不動」。若支撐算錯了一個像素，量到的「主體內
位移」就同時混著「補丁自己蓋在主體上」——那是靜默失效，數字看起來完全正常，
而且會讓結論反過來。

第二個容易錯的是**起點**：`reset` 之後第 0 步的輸出必須與原圖逐位元相同，
否則收斂曲線的第一點不是恆等，跨族比較的起點就不一樣了。

第三個是**放不下時的行為**。主體遮罩太大時某個面積沒有合法位置，這時安靜地
縮小補丁會讓 `radius` 欄與實際跑的面積對不上，報表上看不出來。

只用 CPU 與小張合成影像，不載任何權重。
"""

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.defense.patch_param import (  # noqa: E402
    PatchParam, PatchRandomParam, choose, largest_legal_side, placements,
    rect_support,
)


def img(side=128, seed=1):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(1, 3, side, side, generator=g)


def mask_block(h=128, w=128, top=40, left=40, side=40):
    m = torch.zeros(1, 1, h, w)
    m[..., top:top + side, left:left + side] = 1.0
    return m


def test_起點即恆等():
    """第 0 步的輸出必須與原圖逐位元相同。"""
    x, m = img(), mask_block()
    p = PatchParam(radius=0.05, mask=m, min_side=8)
    p.reset(x, seed=0)
    assert torch.equal(p.render(x), x)


def test_主體核心逐位元不動():
    x, m = img(), mask_block()
    p = PatchParam(radius=0.05, mask=m, min_side=8)
    p.reset(x, seed=0)
    with torch.no_grad():
        p.c.uniform_(0, 1)               # 把補丁內容整個換掉
    out = p.render(x)
    core = m >= 1.0
    assert torch.equal(out * core, x * core)


def test_支撐與遮罩零重疊且含羽化帶():
    m = mask_block()
    m[..., 36:40, 36:84] = 0.05          # 極淡的羽化帶也要被排除
    p = PatchParam(radius=0.02, mask=m, min_side=8)
    p.reset(img(), seed=0)
    assert float((p.support.float() * m).max()) == 0.0


def test_面積比例換算成邊長():
    p = PatchParam(radius=0.25)
    assert p.side_for(128, 128) == 64     # sqrt(0.25*128*128) = 64
    p2 = PatchParam(radius=0.0625)
    assert p2.side_for(512, 512) == 128


def test_實際幾何逐列可讀():
    x, m = img(), mask_block()
    p = PatchParam(radius=0.05, mask=m, placement="far", min_side=8)
    p.reset(x, seed=0)
    g = p.geometry()
    assert set(g) == {"patch_side", "patch_top", "patch_left",
                      "patch_placement", "patch_area", "patch_tile",
                      "patch_res",
                      "patch_alpha", "patch_carrier", "patch_count",
                      "patch_crop_keep", "patch_rects",
                      # 兩個內容約束。加旋鈕而不登記時就是這一條會紅——
                      # 新欄位同時要進 `distortion_axis_analysis.SETTING_DEFAULTS`，
                      # 否則跨批分析會對未登記的欄位拋錯。
                      "patch_lowfreq", "patch_chroma"}
    assert g["patch_side"] == p.side_for(128, 128)
    assert g["patch_placement"] == "far"
    assert g["patch_carrier"] == "none"
    assert g["patch_count"] == 1
    assert g["patch_rects"] == f"{p.top},{p.left}"
    assert g["patch_lowfreq"] == 0 and g["patch_chroma"] == 0
    assert g["patch_res"] == 1


def test_complement_支撐是整個補集且主體不動():
    """`complement` 是「主體逐位元不動」底下可用面積的最大值。"""
    x, m = img(), mask_block()
    p = PatchParam(placement="complement", mask=m)
    p.reset(x, seed=0)
    assert torch.equal(p.render(x), x)               # 起點仍是恆等
    with torch.no_grad():
        p.c.uniform_(0, 1)
    core = m >= 1.0
    assert torch.equal(p.render(x) * core, x * core)
    assert float((p.support.float() * m).max()) == 0.0
    assert p.geometry()["patch_area"] == pytest.approx(float((m <= 0).float().mean()), abs=1e-4)


def test_random起點在支撐內是噪聲而主體仍不動():
    """`edit_divergence` 在恆等起點是駐點，故需要 random 起點。"""
    x, m = img(), mask_block()
    p = PatchParam(radius=0.05, mask=m, min_side=8, init="random")
    p.reset(x, seed=5)
    out = p.render(x)
    assert not torch.equal(out, x)                    # 起點不再是恆等
    core = m >= 1.0
    assert torch.equal(out * core, x * core)          # 主體仍逐位元不動
    q = PatchParam(radius=0.05, mask=m, min_side=8, init="random")
    q.reset(x, seed=5)
    assert torch.equal(q.render(x), out)              # 同種子同內容


def test_恆等起點在散度型損失上是駐點():
    """釘住這個陷阱本身：x_def = x 時平方差的梯度恰為零。"""
    x, m = img(), mask_block()
    p = PatchParam(radius=0.05, mask=m, min_side=8, init="identity")
    p.reset(x, seed=0)
    div = (p.render(x) - x).pow(2).mean()
    div.backward()
    assert float(p.c.grad.abs().sum()) == 0.0


def test_平舖只學一塊磚且鋪滿支撐():
    """規則重複是「刻意放上去的標記」的視覺訊號，也把參數量降兩個數量級。"""
    x, m = img(), mask_block()
    p = PatchParam(placement="complement", mask=m, init="random", tile=16)
    p.reset(x, seed=2)
    assert tuple(p.c.shape) == (1, 3, 16, 16)
    out = p.render(x)
    core = m >= 1.0
    assert torch.equal(out * core, x * core)
    # 支撐內相隔一個磚距的兩點內容相同（皆在支撐內時）
    f = p._field(x)
    assert torch.allclose(f[..., 0, 0], f[..., 16, 16])


def test_半透明讓原圖透出來():
    """alpha 界定支撐內與原圖的最大偏離，故它就是「可見度」的旋鈕。"""
    x, m = img(), mask_block()
    outs = {}
    for a in (0.25, 1.0):
        p = PatchParam(placement="complement", mask=m, init="random", alpha=a)
        p.reset(x, seed=4)
        outs[a] = float((p.render(x) - x).abs().max())
    assert outs[0.25] < outs[1.0]
    assert outs[0.25] <= 0.25 + 1e-6


def test_alpha與tile的值域檢查():
    with pytest.raises(ValueError, match="alpha"):
        PatchParam(alpha=0.0)
    with pytest.raises(ValueError, match="alpha"):
        PatchParam(alpha=1.5)
    with pytest.raises(ValueError, match="tile"):
        PatchParam(tile=-1)


def test_平舖的恆等起點是常數場():
    """一塊磚蓋不出原圖，故恆等起點取畫面均值——最接近「不改變觀感」的常數。"""
    x, m = img(), mask_block()
    p = PatchParam(placement="complement", mask=m, init="identity", tile=8)
    p.reset(x, seed=0)
    assert float(p.c.std()) == 0.0
    assert float(p.c.mean()) == pytest.approx(float(x.mean()), abs=1e-6)


def test_未知的init拋錯():
    with pytest.raises(ValueError, match="init"):
        PatchParam(init="亂填")


def test_未知的placement拋錯():
    with pytest.raises(ValueError, match="placement"):
        PatchParam(placement="隨便放")


def test_沒有遮罩就拋錯而不是放中央():
    p = PatchParam(radius=0.05, min_side=8)
    with pytest.raises(ValueError, match="主體遮罩"):
        p.reset(img(), seed=0)


def test_放不下時拋錯而不是自動縮小():
    """遮罩佔滿畫面時沒有合法位置。安靜縮小會讓 radius 欄對不上實際面積。"""
    x = img()
    m = torch.ones(1, 1, 128, 128)
    p = PatchParam(radius=0.05, mask=m, min_side=8)
    with pytest.raises(ValueError, match="沒有與主體遮罩"):
        p.reset(x, seed=0)


def test_太小的補丁拋錯():
    p = PatchParam(radius=0.0001, mask=mask_block(), min_side=32)
    with pytest.raises(ValueError, match="min_side"):
        p.reset(img(), seed=0)


def test_far與near挑到不同位置():
    x, m = img(), mask_block(top=48, left=48, side=32)
    out = {}
    for mode in ("far", "near"):
        p = PatchParam(radius=0.02, mask=m, placement=mode, min_side=8)
        p.reset(x, seed=0)
        out[mode] = (p.top, p.left)
    assert out["far"] != out["near"]


def test_投影把內容夾回值域():
    x, m = img(), mask_block()
    p = PatchParam(radius=0.05, mask=m, min_side=8)
    p.reset(x, seed=0)
    with torch.no_grad():
        p.c.add_(5.0)
    p.project()
    assert float(p.c.max()) <= 1.0 and float(p.c.min()) >= 0.0


def test_隨機版不最佳化且同種子同內容():
    x, m = img(), mask_block()
    a, b = (PatchRandomParam(radius=0.05, mask=m, min_side=8) for _ in range(2))
    a.reset(x, 7); b.reset(x, 7)
    assert a.params() == []                       # 沒有可學參數
    assert torch.equal(a.render(x), b.render(x))
    c = PatchRandomParam(radius=0.05, mask=m, min_side=8); c.reset(x, 8)
    assert not torch.equal(a.render(x), c.render(x))
    core = m >= 1.0
    assert torch.equal(a.render(x) * core, x * core)   # 主體仍然不動


def test_幾何helper只有一份實作():
    """`scripts/patch_probe.py` 必須用同一份，兩份會在改動時悄悄分岔。"""
    src = (ROOT / "scripts" / "patch_probe.py").read_text(encoding="utf-8")
    assert "from src.defense.patch_param import" in src
    for name in ("def placements(", "def choose(", "def largest_legal_side(",
                 "def rect_support("):
        assert name not in src, f"{name} 在腳本裡有第二份實作"


def test_最大合法邊長與逐一嘗試相同():
    m = mask_block()
    brute = 0
    s = 8
    while s <= 128:
        if placements(m, s, 8)[0]:
            brute = s
        s += 2
    assert largest_legal_side(m, stride=8, min_side=8) == brute


def test_支撐面積正確():
    s = rect_support((1, 3, 64, 64), 0, 0, 32)
    assert int(s.sum()) == 32 * 32
    assert choose([], (0, 0), "far") is None

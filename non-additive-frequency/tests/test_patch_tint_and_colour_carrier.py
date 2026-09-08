"""兩個讓產物不再像雜訊的旋鈕。

B · `--patch-tint`：粗尺度上像原本那件衣服
────────────────────────────────────────────────────────────────────
`--patch-tv` 罰掉支撐內**所有**局部變化，連花紋本身都被壓平——實測平舖配
tv 0.3 只剩基準的 32%，而單獨用時分別是 86% 與 90%。tint 只罰**低頻**：

    λ · mean_over_support( ( blur_σ(c) − blur_σ(x) )² )

於是局部色調被拉向原本的衣服，而花紋的高頻結構完全不受懲罰。兩者不可
互相取代，故 `--patch-tint 0` 時呼叫路徑必須逐位元不變。

C · 色彩重映射施加在衣服上
────────────────────────────────────────────────────────────────────
`ColorCurveParam` 與 `ColorGridParam` 的 `apply_where` 是軟權重，輸出是
`w·F(x) + (1−w)·x`。把它設成 `載體 ∧ 主體補集`，色彩重映射就只作用在衣服
上，而 `w = 0` 的地方**逐位元恆等**——主體不動的保證與補丁族同一個構造。

只用 CPU 與小張合成影像，不載任何權重。
"""

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import ip2p_run  # noqa: E402
from src.defense.color_param import ColorGridParam  # noqa: E402
from src.defense.patch_param import PatchParam  # noqa: E402


def img(side=64, seed=1):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(1, 3, side, side, generator=g)


def mask_block(side=64, h=16):
    m = torch.zeros(1, 1, side, side)
    m[..., :h, :] = 1.0
    return m


def _args(**over):
    ns = ip2p_run.build_parser().parse_args(["--out", "x"])
    for k, v in over.items():
        setattr(ns, k, v)
    return ns


# ── B · tint ────────────────────────────────────────────────────────

def test_tint_為零時損失函式逐位元不變():
    x = img()
    p = PatchParam(radius=0.05, mask=mask_block(), placement="complement",
                   min_side=8)
    p.reset(x, seed=0)
    base = lambda x_def: torch.zeros(())     # noqa: E731
    got = ip2p_run._with_patch_tint(p, x, _args(patch_tint=0.0), base)
    assert got is base


def test_tint_在恆等起點為零而在偏離時為正():
    """起點 c = x，粗尺度差為零；把內容整個換掉之後必須為正。"""
    x = img()
    p = PatchParam(radius=0.05, mask=mask_block(), placement="complement",
                   min_side=8)
    p.reset(x, seed=0)
    fn = ip2p_run._with_patch_tint(
        p, x, _args(patch_tint=1.0, patch_tint_sigma=8),
        lambda x_def: torch.zeros(()))
    assert float(fn(x)) == pytest.approx(0.0, abs=1e-7)
    with torch.no_grad():
        p.c.uniform_(0, 1)
    assert float(fn(x)) > 0.0


def test_tint_只罰低頻():
    """加一個零均值的高頻擾動，tint 幾乎不動；加一個常數色偏，tint 明顯上升。

    這一條把 tint 與 tv 的差別釘住——tv 對前者反應最大，tint 對它幾乎不反應。
    """
    x = img()
    p = PatchParam(radius=0.05, mask=mask_block(), placement="complement",
                   min_side=8)
    p.reset(x, seed=0)
    fn = ip2p_run._with_patch_tint(
        p, x, _args(patch_tint=1.0, patch_tint_sigma=8),
        lambda x_def: torch.zeros(()))
    hi = torch.zeros_like(p.c)
    hi[..., ::2, :] = 0.2
    hi[..., 1::2, :] = -0.2                 # 棋盤式，局部均值幾乎為零
    with torch.no_grad():
        p.c.add_(hi)
    high_freq = float(fn(x))
    with torch.no_grad():
        p.c.sub_(hi).add_(0.2)              # 同幅度的常數色偏
    dc = float(fn(x))
    assert dc > 20 * high_freq, (dc, high_freq)


def test_tint_配非補丁條件時拋錯():
    with pytest.raises(SystemExit, match="patch-tint"):
        ip2p_run._with_patch_tint(
            object(), img(), _args(patch_tint=1.0, conditions=["color_grid"]),
            lambda x_def: torch.zeros(()))


def test_tint_兩個欄位以字面字串寫進列():
    src = (ROOT / "scripts" / "ip2p_run.py").read_text(encoding="utf-8")
    for col in ("patch_tint", "patch_tint_sigma"):
        assert f'"{col}":' in src, col


def test_tint_登記進_SETTING_DEFAULTS():
    import distortion_axis_analysis as daa
    assert daa.SETTING_DEFAULTS["patch_tint"] == "0.0"
    assert daa.SETTING_DEFAULTS["patch_tint_sigma"] == "16"


# ── C · 色彩重映射施加在衣服上 ──────────────────────────────────────

def test_apply_where_為零處逐位元恆等():
    """主體不動的保證：`w = 0` 的地方輸出必須與原圖逐位元相同。"""
    x = img()
    m = mask_block()
    carrier = torch.zeros(1, 1, 64, 64)
    carrier[..., 8:40, :] = 1.0
    where = ((carrier > 0.5) & (m <= 0.0)).to(x.dtype)
    p = ColorGridParam(radius=0.60, grid=4, luma_bins=4, apply_where=where)
    p.reset(x, seed=0)
    with torch.no_grad():
        for t in p.params():
            t.uniform_(-1, 1)               # 把色彩場整個打亂
    out = p.render(x)
    off = where <= 0.0
    assert torch.equal(out * off, x * off)
    assert not torch.equal(out, x)          # 有作用的地方確實變了


def test_色彩族的載體交集由驅動腳本組出來():
    """`apply_where = 載體 ∧ 主體補集` 這條算式必須只有一份實作。"""
    src = (ROOT / "scripts" / "ip2p_run.py").read_text(encoding="utf-8")
    assert src.count("carrier_apply_where(") >= 2   # 定義一次、呼叫至少一次


def test_carrier_apply_where_的交集正確():
    m = mask_block()
    carrier = torch.zeros(1, 1, 64, 64)
    carrier[..., 8:40, :] = 1.0
    w = ip2p_run.carrier_apply_where(carrier, m)
    assert torch.equal(w, ((carrier > 0.5) & (m <= 0.0)).to(m.dtype))
    assert float(w[..., :16, :].max()) == 0.0       # 主體那一段全為零
    assert float(w[..., 16:40, :].min()) == 1.0     # 載體減去主體那一段全為一


def test_色彩族配載體時不需要_complement():
    """載體用在色彩族上是 `apply_where`，不是支撐，故那道守門不該擋它。"""
    ip2p_run.validate_patch_args(
        _args(patch_carrier="clothes", conditions=["color_grid"],
              patch_placement="far",
              subject_mask=Path("data/carrier_catalogue.yaml")))

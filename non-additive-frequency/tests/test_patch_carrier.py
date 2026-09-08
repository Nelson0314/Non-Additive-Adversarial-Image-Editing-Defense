"""補丁的**載體**：把支撐限制在語意上自然的區域（衣服）。

存在理由
────────────────────────────────────────────────────────────────────
補丁族原本的支撐是「主體遮罩的補集」或補集裡的一塊方塊，於是花紋落在牆面
與地板上，讀起來是一塊壞掉的區域而不是影像的一部分。載體把支撐再交集一層
語意區域（ATR 人體解析的衣物類別），花紋止於衣物輪廓，讀起來是布料印花。

三個會靜默失效的地方，本檔逐一釘住：

1. **交集算錯一個像素**，量到的「主體內位移」就混著「補丁蓋在主體上」。
   數字看起來完全正常而結論會反過來。
2. **載體給了卻沒有生效**（例如配上矩形擺放模式），報表上寫著 `clothes`
   而實際跑的是方塊。
3. **交集之後支撐為空**（該影像上沒有合法衣物）時安靜地退回補集，那會讓
   `patch_carrier` 欄與實際跑的東西對不上。

只用 CPU 與合成的分割圖，不載任何權重。
"""

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.defense.carrier_mask import (  # noqa: E402
    CARRIER_CLASSES, CARRIER_REPO, carrier_from_seg, carrier_stats,
)
from src.defense.patch_param import PatchParam  # noqa: E402


def img(side=64, seed=1):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(1, 3, side, side, generator=g)


def seg_map(side=64, face=True):
    """合成的 ATR 分割：上排 Upper-clothes(4)，中排 Pants(6)，其餘 Background(0)。

    `face=True` 時最上面四列是 Face(11)——`carrier_from_seg` 要求有人的證據，
    沒有臉的分割圖代表「這張圖裡沒有人」。臉那四列落在衣物之外，故三個載體
    的面積不受影響。
    """
    s = torch.zeros(side, side, dtype=torch.long)
    if face:
        s[:4, :] = 11
    s[8:24, :] = 4
    s[24:40, :] = 6
    return s


def mask_block(side=64, top=0, left=0, h=16, w=64):
    """主體遮罩（1 = 主體）。預設蓋住最上面 16 列，與上衣那一段重疊一半。"""
    m = torch.zeros(1, 1, side, side)
    m[..., top:top + h, left:left + w] = 1.0
    return m


# ── 載體本身 ────────────────────────────────────────────────────────

def test_類別集合對應到正確的像素():
    seg = seg_map()
    up = carrier_from_seg(seg, "upper")
    pa = carrier_from_seg(seg, "pants")
    cl = carrier_from_seg(seg, "clothes")
    assert up.shape == (1, 1, 64, 64)
    assert torch.equal(up[0, 0], (seg == 4).float())
    assert torch.equal(pa[0, 0], (seg == 6).float())
    assert torch.equal(cl, torch.maximum(up, pa))


def test_未登記的類別名稱拋錯():
    with pytest.raises(ValueError, match="kind"):
        carrier_from_seg(seg_map(), "hat")


def test_面積出界拋錯而不是靜默接受():
    """空載體會讓支撐退化，滿載體表示解析器把整張圖都算成衣服。"""
    empty = torch.zeros(64, 64, dtype=torch.long)
    empty[:4, :] = 11                       # 有人但沒有衣物
    with pytest.raises(ValueError, match="載體"):
        carrier_from_seg(empty, "clothes")
    full = torch.full((64, 64), 6, dtype=torch.long)
    full[:4, :] = 11                        # 有人但整張都是衣物
    with pytest.raises(ValueError, match="載體"):
        carrier_from_seg(full, "clothes")


def test_載體的類別表與模型的標註體系一致():
    """ATR 的 18 類裡，衣物是 4/5/6/7。寫錯編號不會拋錯，只會指到別的東西。"""
    assert CARRIER_CLASSES["clothes"] == (4, 5, 6, 7)
    assert CARRIER_CLASSES["upper"] == (4, 7)
    assert CARRIER_CLASSES["pants"] == (5, 6)
    assert CARRIER_REPO == "mattmdjaga/segformer_b2_clothes"


def test_carrier_stats_回報實際面積():
    st = carrier_stats(carrier_from_seg(seg_map(), "clothes"))
    assert st["carrier_area"] == pytest.approx(32 / 64, abs=1e-4)
    assert st["carrier_source"] == CARRIER_REPO


# ── 與 PatchParam 的交集 ────────────────────────────────────────────

def _built(carrier, kind="clothes", mask=None, x=None):
    x = img() if x is None else x
    p = PatchParam(radius=0.05, mask=mask_block() if mask is None else mask,
                   placement="complement", min_side=8)
    p.set_carrier(carrier, kind)
    p.reset(x, seed=0)
    return p, x


def test_支撐等於載體與補集的交集():
    carrier = carrier_from_seg(seg_map(), "clothes")
    m = mask_block()
    p, x = _built(carrier, mask=m)
    expect = (m <= 0.0) & (carrier > 0.5)
    assert torch.equal(p.support, expect)
    # 上衣的 8..16 那八列被主體吃掉，剩 16..24 與褲子 24..40 共 24 列
    assert float(p.support.float().mean()) == pytest.approx(24 / 64, abs=1e-4)


def test_主體核心在載體模式下仍逐位元不動():
    carrier = carrier_from_seg(seg_map(), "clothes")
    m = mask_block()
    p, x = _built(carrier, mask=m)
    with torch.no_grad():
        p.c.uniform_(0, 1)
    out = p.render(x)
    core = m >= 1.0
    assert torch.equal(out * core, x * core)


def test_起點即恆等():
    p, x = _built(carrier_from_seg(seg_map(), "clothes"))
    assert torch.equal(p.render(x), x)


def test_沒給載體時與現行補集模式逐位元相同():
    """載體是**加上去**的自由度，不給它時呼叫路徑必須完全不變。"""
    x, m = img(), mask_block()
    a = PatchParam(radius=0.05, mask=m, placement="complement", min_side=8)
    a.reset(x, seed=0)
    b = PatchParam(radius=0.05, mask=m, placement="complement", min_side=8)
    b.set_carrier(None, "none")
    b.reset(x, seed=0)
    assert torch.equal(a.support, b.support)
    assert torch.equal(a.render(x), b.render(x))


def test_交集後支撐為空時拋錯():
    """該影像上沒有落在主體之外的衣物——盆栽人那張就是這個情形。"""
    carrier = carrier_from_seg(seg_map(), "clothes")
    m = mask_block(h=64)                      # 主體蓋住整張圖
    p = PatchParam(radius=0.05, mask=m, placement="complement", min_side=8)
    p.set_carrier(carrier, "clothes")
    with pytest.raises(ValueError, match="支撐"):
        p.reset(img(), seed=0)


def test_載體配矩形擺放模式時拋錯():
    """靜默忽略的症狀是「報表寫著 clothes，實際跑的是方塊」。"""
    p = PatchParam(radius=0.05, mask=mask_block(), placement="far", min_side=8)
    p.set_carrier(carrier_from_seg(seg_map(), "clothes"), "clothes")
    with pytest.raises(ValueError, match="complement"):
        p.reset(img(), seed=0)


def test_載體張量與名稱必須同時給():
    """只給其中一個會讓 CSV 的 patch_carrier 欄與實際支撐對不上。"""
    p = PatchParam(radius=0.05, mask=mask_block(), placement="complement",
                   min_side=8)
    with pytest.raises(ValueError, match="carrier_kind"):
        p.set_carrier(carrier_from_seg(seg_map(), "clothes"), "none")
    with pytest.raises(ValueError, match="carrier_kind"):
        p.set_carrier(None, "clothes")


def test_geometry_帶出載體名稱():
    p, _ = _built(carrier_from_seg(seg_map(), "pants"), kind="pants")
    assert p.geometry()["patch_carrier"] == "pants"
    q = PatchParam(radius=0.05, mask=mask_block(), placement="complement",
                   min_side=8)
    q.reset(img(), seed=0)
    assert q.geometry()["patch_carrier"] == "none"


# ── 驅動腳本的欄位與守門 ────────────────────────────────────────────

sys.path.insert(0, str(ROOT / "scripts"))

import ip2p_run  # noqa: E402


def _args(**over):
    ns = ip2p_run.build_parser().parse_args(["--out", "x"])
    for k, v in over.items():
        setattr(ns, k, v)
    return ns


def test_三個載體欄位以字面字串寫進列():
    """只活在 CLI 預設值裡的設定，合併分片之後在報表上分不出來。"""
    src = (ROOT / "scripts" / "ip2p_run.py").read_text(encoding="utf-8")
    for col in ("patch_carrier", "carrier_area", "carrier_source"):
        assert f'"{col}":' in src, col


def test_三個載體欄位登記進_SETTING_DEFAULTS():
    """沒登記時跨批分析會對未知欄位拋錯，整份報表跑不出來。"""
    import distortion_axis_analysis as daa
    assert daa.SETTING_DEFAULTS["patch_carrier"] == "none"
    assert daa.SETTING_DEFAULTS["carrier_area"] == ""
    assert daa.SETTING_DEFAULTS["carrier_source"] == ""


def test_補丁守門不受_loss_的提前_return_影響():
    """`validate_loss_args` 在別的損失底下提前 return，補丁的守門不可以住在
    那裡——住在那裡的症狀是「換一個 --loss 就整段跳過」而且沒有任何症狀。"""
    with pytest.raises(SystemExit, match="complement"):
        ip2p_run.validate_patch_args(
            _args(patch_carrier="clothes", conditions=["patch"],
                  patch_placement="far", loss="encoder_target"))


def test_載體配色彩族時不需要_complement_但需要主體遮罩():
    """載體有兩個用途：補丁族拿它當**支撐**，色彩族拿它當 `apply_where`。

    後者不需要 complement——那個旗標在色彩族上沒有意義；但它需要主體遮罩，
    因為 `apply_where = 載體 ∧ 主體補集` 的交集要有遮罩才算得出來。不給遮罩
    時載體會被靜默忽略，而 CSV 的 patch_carrier 欄仍然寫著它。
    """
    from pathlib import Path as _P
    ip2p_run.validate_patch_args(
        _args(patch_carrier="clothes", conditions=["color_grid"],
              patch_placement="far",
              subject_mask=_P("data/carrier_catalogue.yaml")))
    with pytest.raises(SystemExit, match="subject-mask"):
        ip2p_run.validate_patch_args(
            _args(patch_carrier="clothes", conditions=["color_grid"],
                  patch_placement="far", subject_mask=None))


def test_載體配矩形擺放模式時派工前就拋錯():
    with pytest.raises(SystemExit, match="complement"):
        ip2p_run.validate_patch_args(
            _args(patch_carrier="clothes", conditions=["patch"],
                  patch_placement="far"))


# ── 固定形狀浮水印：crop_safe 擺放與多塊 ──────────────────────────

from src.defense.patch_param import crop_safe_box, place_many  # noqa: E402


def test_crop_safe_box_對上_crop_resize_的裁切量():
    """`crop_resize0.1` 每邊裁 10%，故留存框是中央 80%。"""
    assert crop_safe_box(512, 512, 0.8) == (51, 51, 461, 461)
    assert crop_safe_box(100, 200, 1.0) == (0, 0, 100, 200)


def test_crop_safe_只挑落在留存框內的位置():
    """落在框外的候選會被裁掉一部分，那正是要避免的。"""
    m = mask_block(side=64, top=0, left=0, h=8, w=64)
    got = place_many(m, side=8, count=1, stride=4, mode="crop_safe",
                     crop_keep=0.5)
    (top, left), = got
    t0, l0, t1, l1 = crop_safe_box(64, 64, 0.5)
    assert t0 <= top and top + 8 <= t1
    assert l0 <= left and left + 8 <= l1


def test_兩塊互不重疊():
    m = mask_block(side=64, top=0, left=0, h=8, w=64)
    got = place_many(m, side=12, count=2, stride=4, mode="crop_safe",
                     crop_keep=1.0)
    assert len(got) == 2
    (t1, l1), (t2, l2) = got
    assert not (abs(t1 - t2) < 12 and abs(l1 - l2) < 12), got


def test_放不下第二塊時拋錯而不是只放一塊():
    """安靜地少放一塊會讓 patch_count 欄與實際面積對不上。"""
    m = torch.ones(1, 1, 64, 64)
    m[..., 56:, 56:] = 0.0                # 合法區只有右下角一塊 8×8
    assert place_many(m, side=8, count=1, stride=4, mode="near") == [(56, 56)]
    with pytest.raises(ValueError, match="count"):
        place_many(m, side=8, count=2, stride=4, mode="near")


def test_多塊的支撐是聯集且面積正確():
    x = img()
    p = PatchParam(radius=0.08, mask=mask_block(side=64, h=8, w=64),
                   placement="crop_safe", min_side=8, count=2, crop_keep=1.0)
    p.reset(x, seed=0)
    assert p.count == 2
    assert float(p.support.float().sum()) == 2 * p.side * p.side
    assert torch.equal(p.render(x), x)          # 起點即恆等


def test_geometry_帶出_count_與每一塊的座標():
    x = img()
    p = PatchParam(radius=0.08, mask=mask_block(side=64, h=8, w=64),
                   placement="crop_safe", min_side=8, count=2, crop_keep=1.0)
    p.reset(x, seed=0)
    g = p.geometry()
    assert g["patch_count"] == 2
    assert g["patch_crop_keep"] == 1.0
    assert g["patch_rects"].count(";") == 1     # 兩塊，一個分號
    assert g["patch_rects"].split(";")[0] == f"{p.top},{p.left}"


def test_載體配多塊時拋錯():
    """載體是不規則區域，方塊擺放與它是兩種互斥的支撐構造。"""
    p = PatchParam(radius=0.08, mask=mask_block(), placement="complement",
                   min_side=8, count=2)
    p.set_carrier(carrier_from_seg(seg_map(), "clothes"), "clothes")
    with pytest.raises(ValueError, match="count"):
        p.reset(img(), seed=0)


def test_crop_safe_的守門欄位以字面字串寫進列():
    src = (ROOT / "scripts" / "ip2p_run.py").read_text(encoding="utf-8")
    for col in ("patch_count", "patch_crop_keep", "patch_rects"):
        assert f'"{col}":' in src, col


def test_多塊配補集模式時派工前就拋錯():
    with pytest.raises(SystemExit, match="patch-count"):
        ip2p_run.validate_patch_args(
            _args(conditions=["patch"], patch_placement="complement",
                  patch_count=2, radius=0.05))


# ── 「這張圖裡有沒有人」的守門 ──────────────────────────────────────

from src.defense.carrier_mask import FACE_CLASS, MIN_FACE  # noqa: E402


def test_沒有臉時拋錯():
    """ATR 沒有「沒有人」這個輸出：貓的毛皮會被標成 Upper-clothes。"""
    with pytest.raises(ValueError, match="沒有人"):
        carrier_from_seg(seg_map(face=False), "clothes")


def test_臉那幾列不影響三個載體的面積():
    """守門用的證據與載體本身必須分得開，否則門檻會連帶改變支撐。"""
    a = carrier_from_seg(seg_map(face=True), "clothes")
    assert float(a.mean()) == pytest.approx(32 / 64, abs=1e-4)
    assert FACE_CLASS not in CARRIER_CLASSES["clothes"]


def test_臉的門檻落在實測的兩群之間():
    """貓 0.0002／蘑菇 0.0000／手掌 0.0000 對真人 0.0225–0.1800。"""
    assert 0.0002 < MIN_FACE < 0.0225

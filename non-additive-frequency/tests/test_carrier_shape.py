"""載體的貼合與融入：四個旋鈕的不變量。

產物的不自然有很大一部分來自**載體邊界本身**，不是花紋內容：ATR 的分割是
粗的，硬 0/1 邊界會讓花紋溢出到皮膚、或在衣物中途硬切。四個正交的旋鈕：

    refine    導引濾波把邊界吸附到影像真實的衣物邊緣
    erode     邊界往內縮，花紋永不溢出
    feather   往內羽化，花紋在邊緣淡入衣服而不是硬切
    scatter   減小面積並分散成 N 個斑塊

**羽化方向與 `subject_mask` 相反。** 那一支往外羽化是為了「寧可多保留一點
主體」；載體往外會讓花紋溢出衣物輪廓，故一律往內，且**永遠不增加**任何一
個像素的權重——主體那一側恆為 0 的保證因此不受影響。

只用 CPU 與合成遮罩，不載任何權重。
"""

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.defense.carrier_mask import (  # noqa: E402
    erode_mask, feather_inward, guided_refine, scatter_support,
)
from src.defense.patch_param import PatchParam  # noqa: E402


def disc(side=64, cx=32, cy=32, r=20):
    y, x = torch.meshgrid(torch.arange(side).float(),
                          torch.arange(side).float(), indexing="ij")
    return (((x - cx) ** 2 + (y - cy) ** 2) <= r * r).float()[None, None]


# ── erode / feather：永不增加 ─────────────────────────────────────

def test_內縮永遠不增加權重():
    m = disc()
    for n in (1, 3, 8):
        out = erode_mask(m, n)
        assert float((out - m).max()) <= 0.0, n
        assert float(out.sum()) < float(m.sum()), n


def test_內縮零時是恆等():
    m = disc()
    assert torch.equal(erode_mask(m, 0), m)


def test_往內羽化_內部仍為一_邊界介於零與一_且不外擴():
    m = disc()
    out = feather_inward(m, 6)
    assert float((out - m).max()) <= 0.0          # 永不增加
    assert float(out[0, 0, 32, 32]) == pytest.approx(1.0, abs=1e-6)  # 圓心仍是 1
    band = out[(m > 0) & (out < 1.0)]
    assert band.numel() > 0 and float(band.min()) >= 0.0
    assert float(out[m <= 0].max()) == 0.0        # 遮罩之外恆為零


def test_往內羽化零時是恆等():
    m = disc()
    assert torch.equal(feather_inward(m, 0), m)


def test_羽化方向與_subject_mask_相反():
    """`subject_mask._feather` 往外、面積變大；載體往內、面積變小。"""
    from src.defense.subject_mask import _feather
    m = disc()
    assert float(_feather(m, 6).sum()) > float(m.sum())
    assert float(feather_inward(m, 6).sum()) < float(m.sum())


# ── refine：吸附到影像邊緣 ────────────────────────────────────────

def test_導引濾波輸出落在零一之間且不遠離原遮罩():
    """吸附會讓邊界移動，但不該把遮罩搬到別的地方去。"""
    g = torch.zeros(1, 3, 64, 64)
    g[..., :, 32:] = 1.0                          # 影像的真實邊界在 x=32
    m = disc(cx=32, r=20)
    out = guided_refine(m, g, radius=4, eps=1e-3)
    assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0
    iou = float(((out > 0.5) & (m > 0.5)).sum()) / float(((out > 0.5) | (m > 0.5)).sum())
    assert iou > 0.7, iou


def test_導引濾波半徑零時是恆等():
    m = disc()
    assert torch.equal(guided_refine(m, torch.zeros(1, 3, 64, 64), 0, 1e-3), m)


# ── scatter：減小面積並分散 ──────────────────────────────────────

def test_分散斑塊全部落在載體內且面積接近要求():
    m = disc(r=24)
    out = scatter_support(m, count=5, area=0.04, seed=0)
    assert float((out * (m <= 0)).max()) == 0.0           # 不溢出載體
    got = float(out.mean())
    assert got == pytest.approx(0.04, rel=0.12), got      # 實現面積要對得上要求

def test_分散斑塊互不相連():
    m = disc(r=24)
    out = scatter_support(m, count=4, seed=0, area=0.03)[0, 0].numpy()
    from scipy import ndimage
    _, n = ndimage.label(out > 0)
    assert n == 4, n


def test_塊數越多每一塊越小():
    m = disc(r=24)
    a = scatter_support(m, count=2, area=0.04, seed=0)
    b = scatter_support(m, count=8, area=0.04, seed=0)
    assert float(a.mean()) == pytest.approx(float(b.mean()), rel=0.35)


def test_載體放不下要求的面積時拋錯():
    m = disc(r=4)                                          # 很小的載體
    with pytest.raises(ValueError, match="面積"):
        scatter_support(m, count=3, area=0.5, seed=0)


def test_實現面積由二分搜尋對齊要求而不是由公式直接算():
    """圓斑靠近載體邊界會被裁掉；直接用 sqrt(area·HW/count/pi) 會少一大截。

    實測在真實載體上要求 4% 只拿到 1.7%，而那是靜默的——面積旗標在 scatter
    模式下的意義會與其他模式不同。改成二分搜尋半徑之後對得上。

    容差取 20%：小面積時每個圓斑只有幾十個像素，半徑加一格就跨過目標，
    離散化必然讓實現面積略高於要求。**只會高不會低**，由下一條釘住。
    """
    m = disc(r=20)
    for area in (0.02, 0.05, 0.08):
        got = float(scatter_support(m, count=4, area=area, seed=0).mean())
        assert got == pytest.approx(area, rel=0.20), (area, got)


def test_實現面積永遠不低於要求():
    """寧可多一點也不能少——少了就是「跑的比登記的小」而報表看不出來。"""
    m = disc(r=20)
    for area in (0.02, 0.05, 0.08):
        got = float(scatter_support(m, count=4, area=area, seed=0).mean())
        assert got >= area, (area, got)


# ── PatchParam：軟權重支撐 ───────────────────────────────────────

def img(side=64, seed=1):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(1, 3, side, side, generator=g)


def mask_block(side=64, h=16):
    m = torch.zeros(1, 1, side, side)
    m[..., :h, :] = 1.0
    return m


def _built(carrier, side=64):
    x, m = img(side), mask_block(side)
    p = PatchParam(radius=0.05, mask=m, placement="complement", min_side=8)
    p.set_carrier(carrier, "clothes")
    p.reset(x, seed=0)
    return p, x, m


def test_軟權重支撐_權重為零處逐位元不動():
    carrier = torch.zeros(1, 1, 64, 64)
    carrier[..., 20:50, :] = 1.0
    p, x, m = _built(feather_inward(carrier, 5))
    with torch.no_grad():
        p.c.uniform_(0, 1)
    out = p.render(x)
    off = p.support <= 0.0
    assert torch.equal(out * off, x * off)
    core = m >= 1.0
    assert torch.equal(out * core, x * core)          # 主體仍逐位元不動


def test_軟權重支撐真的在邊界混色():
    """羽化被門檻掉的話這一條會紅——那正是先前的行為。"""
    carrier = torch.zeros(1, 1, 64, 64)
    carrier[..., 20:50, :] = 1.0
    p, x, m = _built(feather_inward(carrier, 5))
    assert p.support.dtype != torch.bool
    band = (p.support > 0) & (p.support < 1)
    assert int(band.sum()) > 0                        # 確實存在中間權重
    with torch.no_grad():
        p.c.fill_(1.0)                                # 內容全白
    out = p.render(x)
    v = out[band.expand_as(out)]
    assert float(v.min()) > float(x[band.expand_as(x)].min())   # 被推向白
    assert float(v.max()) < 1.0                                  # 但沒有到全白


def test_零一權重時與布林支撐逐位元相同():
    """軟權重是**加上去**的自由度，硬遮罩下輸出必須完全不變。"""
    carrier = torch.zeros(1, 1, 64, 64)
    carrier[..., 20:50, :] = 1.0
    a, x, _ = _built(carrier)
    with torch.no_grad():
        a.c.uniform_(0, 1)
    b, _, _ = _built(carrier)
    with torch.no_grad():
        b.c.copy_(a.c)
    assert torch.equal(a.render(x), b.render(x))
    assert float(a.support.max()) == 1.0 and float(a.support.min()) == 0.0


def test_面積欄用權重平均而不是非零像素數():
    carrier = torch.zeros(1, 1, 64, 64)
    carrier[..., 20:50, :] = 1.0
    hard, _, _ = _built(carrier)
    soft, _, _ = _built(feather_inward(carrier, 5))
    assert soft.geometry()["patch_area"] < hard.geometry()["patch_area"]


# ── 臉框守門：ATR 會把臉的皮膚判成衣服 ────────────────────────────

from src.defense.carrier_mask import box_conflict, exclude_boxes  # noqa: E402


def test_臉框被挖掉且外擴有生效():
    m = torch.ones(1, 1, 64, 64)
    out = exclude_boxes(m, [(20, 20, 40, 40)], margin=4)
    assert float(out[..., 20:41, 20:41].max()) == 0.0      # 框內全空
    assert float(out[..., 16:45, 16:45].max()) == 0.0      # 外擴也空
    assert float(out[..., :12, :12].min()) == 1.0          # 框外不動


def test_沒有框時是恆等():
    m = torch.rand(1, 1, 32, 32)
    assert torch.equal(exclude_boxes(m, []), m)


def test_衝突量在臉被判成衣服時很高_在正常情形時很低():
    """這一條釘住的是失效的簽名，不是某個門檻。"""
    bad = torch.ones(1, 1, 64, 64)                          # 整張都被判成衣物
    good = torch.zeros(1, 1, 64, 64)
    good[..., 45:, :] = 1.0                                 # 衣物只在臉框之下
    box = [(16, 8, 48, 40)]
    assert box_conflict(bad, box) == pytest.approx(1.0, abs=1e-6)
    assert box_conflict(good, box) == pytest.approx(0.0, abs=1e-6)
    assert box_conflict(bad, []) == 0.0


def test_random_起點在軟支撐上也能建出來():
    """`init="random"` 的內容初始化當初漏改，軟支撐會拋「condition 不是布林」。

    症狀是 `RuntimeError`，不是靜默失效——但它擋住整條 scatter 的路徑。
    """
    carrier = torch.zeros(1, 1, 64, 64)
    carrier[..., 20:50, :] = 1.0
    x, m = img(), mask_block()
    p = PatchParam(radius=0.05, mask=m, placement="complement", min_side=8,
                   init="random")
    p.set_carrier(feather_inward(carrier, 5), "clothes")
    p.reset(x, seed=3)
    out = p.render(x)
    off = p.support <= 0.0
    assert torch.equal(out * off, x * off)
    assert not torch.equal(out, x)                 # 支撐內確實是噪聲


def test_carrier_of_只引用實際存在的旗標():
    """`_carrier_of` 曾引用不存在的 `args.defense_seed`，整批死於 AttributeError。

    用真的 parser 產出的 Namespace 跑一次，缺任何旗標都會在這裡爆而不是在
    GPU 上跑了幾分鐘之後。
    """
    import sys as _s
    _s.path.insert(0, str(ROOT / "scripts"))
    import ip2p_run

    ns = ip2p_run.build_parser().parse_args(
        ["--out", "x", "--patch-carrier", "clothes", "--carrier-refine", "2",
         "--carrier-erode", "1", "--carrier-feather", "2",
         "--carrier-scatter", "3", "--radius", "0.05"])
    import inspect
    src = inspect.getsource(ip2p_run._carrier_of)
    import re
    for attr in set(re.findall(r"args\.(\w+)", src)):
        assert hasattr(ns, attr), f"_carrier_of 引用了不存在的旗標 args.{attr}"


def test_四個載體函式的輸出裝置跟著輸入走():
    """本機測試只跑 CPU，故「在 CPU 上建張量」的錯誤要到 GPU 才會炸。

    這裡用 `meta` 裝置代替 GPU：它不需要顯示卡，但同樣會在跨裝置運算時
    拋錯，於是這一類缺陷在本機就攔得下來。
    """
    m = disc()
    for fn, args in ((erode_mask, (2,)), (feather_inward, (3,))):
        assert fn(m.to("meta"), *args).device.type == "meta", fn.__name__
    g = torch.zeros(1, 3, 64, 64)
    assert guided_refine(m.to("meta"), g.to("meta"), 2, 1e-3).device.type == "meta"


def test_scatter_的輸出裝置跟著輸入走():
    """`scatter_support` 用 nonzero 取索引，不能在 meta 上跑，故單獨檢查
    它建出來的張量與輸入同裝置（CPU 上恆真，但釘住那行程式碼的意圖）。"""
    m = disc(r=20)
    out = scatter_support(m, count=3, area=0.03, seed=0)
    assert out.device == m.device
    src = (ROOT/"src"/"defense"/"carrier_mask.py").read_text(encoding="utf-8")
    body = src[src.index("def scatter_support("):]
    body = body[:body.index("\ndef ")] if "\ndef " in body else body
    assert "device=dev" in body, "scatter_support 沒有把裝置傳給新建的張量"


def test_取卡預設不超過五張():
    """使用者定的規則：機器是多人共用的，一次最多五張卡。

    **上限做在 `free_cards.sh` 本身**，不是各派工腳本裡——寫在文件或個別
    腳本上的規則遲早會有一支漏掉，而漏掉不會報錯。這裡比對字面字串：那支
    是 shell，測試不執行它。
    """
    src = (ROOT / "scripts" / "free_cards.sh").read_text(encoding="utf-8")
    assert "MAX_CARDS=5" in src
    assert "--max-cards" in src
    # `--assert` 是檢查指定的卡空不空，與取幾張無關，不可被上限影響
    i = src.index("MAX_CARDS\" -gt 0 ]")
    assert '-z "$ASSERT"' in src[i - 60:i]

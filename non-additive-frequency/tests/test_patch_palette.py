"""有限調色盤參數化的六個不變量。

存在理由
────────────────────────────────────────────────────────────────────
`docs/DIRECTION.md` 的證據清單第 5 項還沒達成：四個既有參數化都產不出
「像標記」的東西。既有的兩個美化旋鈕（`lowfreq` 低頻替換、`chroma` 凍結
亮度）砍的是**頻帶與通道**，而 §3.4b 的階梯正好說效果來自逐像素的高頻
自由度——所以那兩個旋鈕是拿效果換好看。

調色盤砍的是**色彩的基數**，逐像素的空間自由度完全保留。這一族要釘住的是
「真的只用得到 K 個顏色」與「那個限制不是靠 clamp 事後補的」。

只用 CPU 與小張合成影像，不載任何權重。
"""

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.defense.patch_param import PatchParam, PatchPaletteParam  # noqa: E402


def img(side=64, seed=1):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(1, 3, side, side, generator=g)


def mask_block(h=64, w=64, top=20, left=20, side=20):
    m = torch.zeros(1, 1, h, w)
    m[..., top:top + side, left:left + side] = 1.0
    return m


def support_colours(p, x, atol=2e-2):
    """支撐內出現過幾個「與某個調色盤顏色相符」的獨立顏色。"""
    out = p.render(x).detach()
    sup = p.support.to(torch.bool).expand_as(out)
    px = out.permute(0, 2, 3, 1).reshape(-1, 3)[sup[:, 0].reshape(-1)]
    pal = p.p.detach()[0]
    d = (px[:, None] - pal[None]).abs().amax(-1)
    return d.amin(-1), px


def test_主體核心逐位元不動():
    """整族共同的前提。調色盤換的是支撐內長什麼樣子，不是支撐本身。"""
    x, m = img(), mask_block()
    p = PatchPaletteParam(placement="complement", mask=m, palette=5)
    p.reset(x, seed=0)
    core = m >= 1.0
    assert torch.equal(p.render(x) * core, x * core)


def test_支撐內的顏色全部落在調色盤上():
    """**這一條是整個參數化的定義。** 落不上去就表示 softmax 沒有硬到
    足以當成指派，那時「K 個顏色」只是一個說法而不是一個約束。"""
    x, m = img(), mask_block()
    p = PatchPaletteParam(placement="complement", mask=m, palette=4, temp=0.05)
    p.reset(x, seed=0)
    dist, px = support_colours(p, x)
    assert float(dist.max()) < 2e-2, float(dist.max())
    # 而且真的用到不只一個顏色——全部塌到同一個顏色也會通過上一行。
    assert len(torch.unique((px * 255).round().to(torch.int32), dim=0)) >= 2


def test_可學張量的通道數是K不是3():
    x, m = img(), mask_block()
    p = PatchPaletteParam(placement="complement", mask=m, palette=7)
    p.reset(x, seed=0)
    assert tuple(p.c.shape) == (1, 7, 64, 64)
    assert tuple(p.p.shape) == (1, 7, 3)
    assert [t.shape for t in p.params()] == [p.c.shape, p.p.shape]


def test_恆等起點是原圖的K色海報化():
    """子空間裡沒有原圖，但有一個離它最近的點。與 `res` 的低通起點同性質。"""
    x, m = img(), mask_block()
    p = PatchPaletteParam(placement="complement", mask=m, palette=6,
                          init="identity")
    p.reset(x, seed=0)
    out = p.render(x)
    sup = p.support.to(torch.bool)
    # 不是逐位元恆等（那個子空間裡沒有原圖）……
    assert float(((out - x) * sup).abs().max()) > 1e-3
    # ……但比隨機起點近得多。
    q = PatchPaletteParam(placement="complement", mask=m, palette=6,
                          init="random")
    q.reset(x, seed=0)
    d_id = float((((out - x) * sup) ** 2).mean())
    d_rand = float((((q.render(x) - x) * sup) ** 2).mean())
    assert d_id < d_rand, (d_id, d_rand)


def test_梯度同時通到logits與調色盤():
    """兩個張量都必須可學：只學 logits 就退化成「從固定色票裡挑」，
    只學調色盤就退化成「整塊同色」。任一邊斷掉都不會報錯。"""
    x, m = img(), mask_block()
    p = PatchPaletteParam(placement="complement", mask=m, palette=5,
                          init="random")
    p.reset(x, seed=0)
    loss = (p.render(x) ** 2).sum()
    gc, gp = torch.autograd.grad(loss, p.params(), allow_unused=True)
    assert gc is not None and float(gc.abs().max()) > 0, "logits 沒有梯度"
    assert gp is not None and float(gp.abs().max()) > 0, "調色盤沒有梯度"


def test_與tile和res疊得起來():
    """三個旋鈕砍的是不同的東西（週期／頻帶／色彩基數），必須能各自獨立掃。"""
    x, m = img(), mask_block()
    p = PatchPaletteParam(placement="complement", mask=m, palette=4,
                          tile=16, res=4, init="random")
    p.reset(x, seed=0)
    assert tuple(p.c.shape) == (1, 4, 4, 4)
    f = p._field(x)
    assert tuple(f.shape[-2:]) == (64, 64) and f.shape[1] == 3
    # 平舖仍是嚴格週期的（先升取樣、後平舖的順序沒有被 softmax 打斷）
    assert torch.allclose(f[..., :16, :16], f[..., 16:32, 16:32])


def test_值域檢查():
    with pytest.raises(ValueError, match="palette"):
        PatchPaletteParam(palette=1)
    with pytest.raises(ValueError, match="temp"):
        PatchPaletteParam(temp=0.0)


def test_實際用到幾個顏色逐列寫出():
    """只記 K 分不出「要求 8 個而只用到 3 個」與「要求 3 個」。"""
    x, m = img(), mask_block()
    p = PatchPaletteParam(placement="complement", mask=m, palette=6)
    p.reset(x, seed=0)
    g = p.geometry()
    assert g["patch_palette"] == 6 and g["patch_palette_temp"] == 0.05
    assert 1 <= g["patch_palette_used"] <= 6
    # 父類的欄位一個都不能掉
    assert {"patch_side", "patch_tile", "patch_res", "patch_carrier"} <= set(g)


def test_隨機臂與最佳化臂共用同一組色票():
    """`docs/DIRECTION.md` §3.5 記過：色彩網格的隨機解與最佳化解**看起來
    一樣**——外觀由參數化決定不由最佳化決定。調色盤同樣會大幅改變外觀，
    沒有這個對照就分不出「像印花」是學出來的還是參數化本來就長那樣。

    色票**不重抽**（兩臂相同），隨機的只有「每個像素挑哪一個顏色」。
    色票若也重抽，兩臂的差別就混著色票的差別。
    """
    from src.defense.patch_param import PatchPaletteRandomParam

    x, m = img(), mask_block()
    a = PatchPaletteParam(placement="complement", mask=m, palette=5)
    b = PatchPaletteRandomParam(placement="complement", mask=m, palette=5)
    a.reset(x, seed=3)
    b.reset(x, seed=3)
    assert torch.equal(a.p.detach(), b.p.detach()), "兩臂的色票必須相同"
    assert not torch.equal(a.c.detach(), b.c.detach()), "圖樣必須不同"
    assert b.params() == [], "隨機對照不最佳化，params() 必須是空的"
    # 支撐幾何逐位元相同，差別只剩梯度
    assert torch.equal(a.support, b.support)


def test_旗標關閉時建出來的是原本的類別():
    """`--patch-palette 0` 必須逐位元退回 `PatchParam`，否則既有的補丁批次
    全部不再可比，而 CSV 上看不出來。"""
    sys.path.insert(0, str(ROOT / "scripts"))
    import phase_ablation

    from src.defense.patch_param import PatchPaletteParam as PP
    from src.defense.patch_param import PatchParam as P

    off, _, _ = phase_ablation.build("patch", seed=0)
    on, _, _ = phase_ablation.build("patch", seed=0, patch_palette=4)
    assert type(off) is P and type(on) is PP
    rnd_off, _, _ = phase_ablation.build("patch_rand", seed=0)
    rnd_on, _, _ = phase_ablation.build("patch_rand", seed=0, patch_palette=4)
    assert type(rnd_off).__name__ == "PatchRandomParam"
    assert type(rnd_on).__name__ == "PatchPaletteRandomParam"


# ── Voronoi（arXiv:2606.17711）─────────────────────────────────────


def test_voronoi的可學張量是座標不是場():
    """這是它與調色盤唯一的結構差別，也是參數量少三個數量級的來源。"""
    from src.defense.patch_param import PatchVoronoiParam

    x, m = img(), mask_block()
    p = PatchVoronoiParam(placement="complement", mask=m, seeds=16)
    p.reset(x, seed=0)
    assert tuple(p.c.shape) == (1, 16, 2), "種子是座標 (1,K,2)"
    assert tuple(p.p.shape) == (1, 16, 3)
    assert sum(t.numel() for t in p.params()) == 16 * 5
    # 自由補丁在同一張圖上是 3·H·W
    assert 16 * 5 < 3 * 64 * 64 / 100


def test_voronoi的梯度同時通到種子與顏色():
    """硬 argmin 對座標的導數處處為零——那正是本專案記過的零梯度陷阱，
    所以指派必須是 softmax。任一邊斷掉都不會報錯。"""
    from src.defense.patch_param import PatchVoronoiParam

    x, m = img(), mask_block()
    p = PatchVoronoiParam(placement="complement", mask=m, seeds=12)
    p.reset(x, seed=0)
    gs, gp = torch.autograd.grad((p.render(x) ** 2).sum(), p.params())
    assert float(gs.abs().max()) > 0, "種子點沒有梯度"
    assert float(gp.abs().max()) > 0, "顏色沒有梯度"


def test_voronoi的支撐內顏色落在色票上():
    """胞邊界要夠硬，否則「一胞一色」只是說法而不是約束。"""
    from src.defense.patch_param import PatchVoronoiParam

    x, m = img(), mask_block()
    p = PatchVoronoiParam(placement="complement", mask=m, seeds=8)
    p.reset(x, seed=0)
    out = p.render(x).detach()
    sup = p.support.to(torch.bool).expand_as(out)
    px = out.permute(0, 2, 3, 1).reshape(-1, 3)[sup[:, 0].reshape(-1)]
    # `amax(-1)` 是「這個像素與該色票色的通道最大差」，**還要對 K 取 amin**
    # 才是「離最近的色票色有多遠」。漏掉 amin 量到的是對最遠那一色的距離。
    d = (px[:, None] - p.p.detach()[0][None]).abs().amax(-1).amin(-1)
    # 邊界上的像素會落在兩色之間，故用分位數而不是最大值。
    assert float(d.quantile(0.95)) < 5e-2, float(d.quantile(0.95))


def test_voronoi主體逐位元不動():
    from src.defense.patch_param import PatchVoronoiParam

    x, m = img(), mask_block()
    p = PatchVoronoiParam(placement="complement", mask=m, seeds=16)
    p.reset(x, seed=0)
    core = m >= 1.0
    assert torch.equal(p.render(x) * core, x * core)


def test_voronoi不與tile或res併用():
    """兩者作用在可學張量的空間解析度上，而這裡的可學張量是座標。
    靜默忽略的症狀是 CSV 寫著 tile 64 而實際沒有平舖。"""
    from src.defense.patch_param import PatchVoronoiParam

    with pytest.raises(ValueError, match="tile"):
        PatchVoronoiParam(seeds=8, tile=32)
    with pytest.raises(ValueError, match="tile"):
        PatchVoronoiParam(seeds=8, res=4)
    with pytest.raises(ValueError, match="seeds"):
        PatchVoronoiParam(seeds=1)


def test_voronoi隨機臂共用色票只換種子():
    from src.defense.patch_param import PatchVoronoiParam, PatchVoronoiRandomParam

    x, m = img(), mask_block()
    a = PatchVoronoiParam(placement="complement", mask=m, seeds=16)
    b = PatchVoronoiRandomParam(placement="complement", mask=m, seeds=16)
    a.reset(x, seed=2)
    b.reset(x, seed=2)
    assert torch.equal(a.p.detach(), b.p.detach()), "色票必須相同"
    assert not torch.equal(a.c.detach(), b.c.detach()), "種子必須不同"
    assert b.params() == []


def test_voronoi的geometry不炸且記實際胞數():
    """父類算 `patch_palette_used` 時假設 c 是 (1,K,H,W)，形狀對不上會拋
    IndexError——這一條釘住那個覆寫。"""
    from src.defense.patch_param import PatchVoronoiParam

    x, m = img(), mask_block()
    p = PatchVoronoiParam(placement="complement", mask=m, seeds=16)
    p.reset(x, seed=0)
    g = p.geometry()
    assert g["patch_seeds"] == 16
    assert 1 <= g["patch_palette_used"] <= 16
    assert {"patch_side", "patch_tile", "patch_res", "patch_carrier"} <= set(g)

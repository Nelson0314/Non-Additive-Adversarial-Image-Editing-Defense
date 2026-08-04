"""複合模塊：APA 移植需要同時掛 LoRA 與 latent 注入，
那是 `generator.py` 的分派從未走過的組合（`ARCH` §3.2）。

四項不變量：φ=0 逐位元等同、停用逐位元等同、參數群可分開取用、`remove()` 廣播。
"""

import pytest
import torch
import torch.nn as nn

from src.residual.base import ResidualModule
from src.residual.composite import CompositeResidual
from src.residual.site_latent import LatentResidual
from src.residual.site_warp import WarpResidual


class _FakeAttn2(nn.Module):
    """最小的 attn2 結構，讓 WeightResidual 找得到 to_q/to_k/to_v/to_out.0。

    用假的而非真的 SD：本檔要驗的是複合邏輯，載入 SD 會把單元測試變成
    需要 GPU 與權重的整合測試。
    """

    def __init__(self, dim=8):
        super().__init__()
        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)
        self.to_out = nn.ModuleList([nn.Linear(dim, dim)])

    def forward(self, x):
        return self.to_out[0](self.to_q(x) + self.to_k(x) + self.to_v(x))


class _FakeUNet(nn.Module):
    def __init__(self, dim=8):
        super().__init__()
        self.blocks = nn.ModuleDict({"0": nn.ModuleDict({"attn2": _FakeAttn2(dim)})})

    def forward(self, x):
        return self.blocks["0"]["attn2"](x)


def _weight_module(unet, **kw):
    from src.residual.site_weight import WeightResidual
    return WeightResidual(unet, rank=2, init_std=0.02, seed=0, **kw)


def _apa_like():
    unet = _FakeUNet()
    w = _weight_module(unet)
    lat = LatentResidual(steps=4, channels=4, size=8, max_rank=2, const_rank=2,
                         init_std=0.0, seed=0)
    return unet, CompositeResidual([w, lat], ["stage1", "stage2"])


# ---- 建構 ----

def test_名稱與成員數不符即拋出():
    with pytest.raises(ValueError) as e:
        CompositeResidual([WarpResidual(size=8, grid_size=4)], ["a", "b"])
    assert "一一對應" in str(e.value)


def test_組名重複即拋出():
    ms = [WarpResidual(size=8, grid_size=4), WarpResidual(size=8, grid_size=4)]
    with pytest.raises(ValueError):
        CompositeResidual(ms, ["s", "s"])


def test_空成員即拋出():
    with pytest.raises(ValueError):
        CompositeResidual([], [])


# ---- 不變量 1：φ=0 逐位元等同 ----

def test_φ為零時eps_hook不改變預測噪聲():
    _, comp = _apa_like()
    ts = torch.tensor([0, 250, 500, 750, 999])
    hook = comp.eps_hook(ts, 4)
    eps = torch.randn(1, 4, 8, 8)
    out = hook(eps.clone(), 0, ts[-1])
    assert torch.equal(out, eps), "初值為零時必須逐位元等同"


def test_φ為零時LoRA不改變UNet輸出():
    unet, comp = _apa_like()
    x = torch.randn(2, 8)
    with torch.no_grad():
        got = unet(x)
    comp.remove()
    with torch.no_grad():
        want = unet(x)
    assert torch.equal(got, want), "B 初值為零，修正量應恆為零"


# ---- 不變量 2：停用逐位元等同 ----

def test_停用廣播到全部成員():
    _, comp = _apa_like()
    comp.disable()
    assert not comp.enabled
    assert all(not m.enabled for m in comp.members)
    comp.enable()
    assert all(m.enabled for m in comp.members)


def test_停用時eps_hook為None():
    _, comp = _apa_like()
    comp.disable()
    assert comp.eps_hook(torch.tensor([0, 500, 999]), 2) is None


def test_停用時LoRA輸出逐位元等同未掛載():
    """x_base = G(x; φ=0) 是全部保真度量測的基準。它與「模塊不存在」
    若有任何差異，該差異會被算進防禦的失真預算裡。"""
    unet = _FakeUNet()
    x = torch.randn(2, 8)
    with torch.no_grad():
        bare = unet(x)

    w = _weight_module(unet)
    # 讓 B 非零，確保測的是「停用」而不是「初值恰好為零」
    with torch.no_grad():
        for h in w.hooks.values():
            h.B.normal_(0, 0.5)
    comp = CompositeResidual([w], ["stage1"])

    with torch.no_grad():
        active = unet(x)
    assert not torch.equal(active, bare), "前提：啟用時必須真的有作用"

    comp.disable()
    with torch.no_grad():
        off = unet(x)
    assert torch.equal(off, bare), "停用必須逐位元等同模塊不存在"


# ---- 不變量 3：參數群 ----

def test_參數群依成員分開且不相交():
    _, comp = _apa_like()
    g = comp.param_groups()
    assert set(g) == {"stage1", "stage2"}
    ids1 = {id(p) for p in g["stage1"]}
    ids2 = {id(p) for p in g["stage2"]}
    assert ids1 and ids2
    assert ids1.isdisjoint(ids2)


def test_參數群的聯集等於全部參數():
    """少了任何一個參數，該階段就有東西訓練不到而且不會有症狀。"""
    _, comp = _apa_like()
    union = {id(p) for ps in comp.param_groups().values() for p in ps}
    assert union == {id(p) for p in comp.parameters()}


def test_成員以ModuleList持有使parameters可見():
    """用純 list 持有的話 parameters() 會是空的，優化器拿不到參數而靜默不更新。"""
    _, comp = _apa_like()
    assert comp.num_trainable() > 0


def test_巢狀複合的組名以斜線展開():
    inner = CompositeResidual(
        [WarpResidual(size=8, grid_size=4), WarpResidual(size=8, grid_size=4)],
        ["a", "b"])
    outer = CompositeResidual([inner, WarpResidual(size=8, grid_size=4)],
                              ["nest", "solo"])
    assert set(outer.param_groups()) == {"nest/a", "nest/b", "solo"}


def test_基底類別預設為單一組():
    assert list(WarpResidual(size=8, grid_size=4).param_groups()) == ["default"]


# ---- 不變量 4：remove 廣播 ----

def test_remove廣播且unet不殘留hook():
    """hook 註冊在 SD 的模組上，模塊被垃圾回收不會移除它們，
    殘留的 hook 會污染共用同一個 SDWrapper 的後續實驗。"""
    unet, comp = _apa_like()
    lin = unet.blocks["0"]["attn2"].to_q
    assert len(lin._forward_hooks) == 1
    comp.remove()
    assert len(lin._forward_hooks) == 0


def test_基底類別的remove不拋出():
    """呼叫端應可無條件呼叫，不需要先判斷模塊是哪一種。"""
    WarpResidual(size=8, grid_size=4).remove()


# ---- 能力聚合 ----

def test_patches_model任一為真即為真():
    _, comp = _apa_like()
    assert comp.patches_model() is True


def test_像素側成員串接():
    a = WarpResidual(size=8, grid_size=4, max_disp=1.0)
    b = WarpResidual(size=8, grid_size=4, max_disp=1.0)
    with torch.no_grad():
        a.flow.fill_(0.5)
        b.flow.fill_(0.5)
    comp = CompositeResidual([a, b], ["a", "b"])
    x = torch.rand(1, 3, 8, 8)
    once = a.pixel_residual(x)
    assert torch.allclose(comp.pixel_residual(x), b.pixel_residual(once))


def test_能力判定不隨開關狀態改變():
    """generator.prepare() 用「有沒有 pixel_residual」決定走哪條路徑。
    若該判定隨 disable() 改變，φ=0 的對照就會走到另一條路徑上去。"""
    comp = CompositeResidual([WarpResidual(size=8, grid_size=4)], ["s"])
    x = torch.rand(1, 3, 8, 8)
    assert comp.pixel_residual(x) is not None
    comp.disable()
    assert comp.pixel_residual(x) is not None, "停用不等於不提供該能力"


def test_非像素側成員的複合不提供pixel_residual():
    _, comp = _apa_like()
    assert comp.pixel_residual(torch.rand(1, 3, 8, 8)) is None


def test_eps_hook串接順序即members順序():
    """串接不可交換，自動排序會讓「換一個順序」變成看不見的變因。"""

    class _Add(ResidualModule):
        def __init__(self, v):
            super().__init__()
            self.v = v
            self.p = nn.Parameter(torch.zeros(1))

        def eps_hook(self, ts, steps):
            return lambda eps, i, t: eps * 2 if self.v == 0 else eps + 1

    comp = CompositeResidual([_Add(0), _Add(1)], ["first", "second"])
    hook = comp.eps_hook(torch.tensor([0, 999]), 1)
    # 先 ×2 再 +1
    assert hook(torch.tensor([3.0]), 0, torch.tensor(999)).item() == 7.0


def test_嵌入殘差相加():
    class _Emb(ResidualModule):
        def __init__(self, v):
            super().__init__()
            self.p = nn.Parameter(torch.full((1,), float(v)))

        def emb_residual(self, emb):
            return torch.ones_like(emb) * self.p

    comp = CompositeResidual([_Emb(1), _Emb(2)], ["a", "b"])
    out = comp.emb_residual(torch.zeros(1, 4))
    assert torch.allclose(out, torch.full((1, 4), 3.0))

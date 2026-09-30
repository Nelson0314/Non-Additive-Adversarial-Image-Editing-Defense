"""`src/baselines/sifm.py` 的驗收 —— 用替身 SD，不載入真模型，全程 CPU。

SIFM 沒有官方程式碼，整個實作是依 arXiv:2512.14320v1 的式 (3)–(7) 與
Algorithm 1 重建的，而重建錯掉的地方**全部沒有症狀**：損失照樣下降、
輸出照樣是一張合理的防禦圖。故這裡釘四件事：

1. 預算換算（`eps` / `eps_pixel01` / 值域）；
2. 合成損失的正負號方向（式 (6)、Algorithm 1 第 13 行）；
3. 兩個目標項**各自**的作用方向（壓低 L1 範數、推高特徵距離）；
4. `BaselineSpec` 的一致性（改寫註記、步數、起點、更新規則）。
"""

import pytest
import torch
import torch.nn as nn

from src.baselines.pgd import run_pgd
from src.baselines.sifm import (
    DEFAULT_LAYERS, DEFAULT_STEP_SIZE, DEFAULT_TIMESTEPS, PAPER_EPS,
    PAPER_LAMBDA, PAPER_STEPS, SPEC_PAPER, FeatureRecorder, loss_fn, prepare,
    sifm_terms,
)


# ---------------------------------------------------------------------------
# 替身
# ---------------------------------------------------------------------------


class _TinyUNet(nn.Module):
    """三個具名子模組：`mid_block`（與真 UNet 同名）、形狀不同的 `alt`、
    以及永遠不會被呼叫的 `never`。後兩者供錯誤路徑的測試使用。"""

    def __init__(self, dtype=torch.float64):
        super().__init__()
        self.mid_block = nn.Conv2d(4, 6, 1, dtype=dtype)
        self.alt = nn.Conv2d(4, 6, 2, stride=2, dtype=dtype)
        self.never = nn.Conv2d(4, 6, 1, dtype=dtype)
        self.out = nn.Conv2d(6, 4, 1, dtype=dtype)
        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, z, t, emb):
        h = self.mid_block(z)
        self.alt(z)                      # 形狀 (1,6,4,4)，與 mid_block 不同
        return self.out(h)


class _StubSD:
    """VAE 換成平均池化、UNet 換成 `_TinyUNet`。兩者都可微且可手算。"""

    num_train_timesteps = 1000

    def __init__(self, device="cpu", dtype=torch.float64):
        self.device = torch.device(device)
        self.dtype = dtype
        self.unet = _TinyUNet(dtype=dtype)
        self.is_inpainting = False
        self.calls = []

    def alphas_cumprod(self, device=None):
        t = torch.arange(self.num_train_timesteps, dtype=self.dtype)
        return (1.0 - t / self.num_train_timesteps).clamp(min=1e-6).to(
            device or self.device)

    def encode_text(self, prompt):
        self.calls.append(("encode_text", prompt))
        return torch.zeros(1, 77, 8, dtype=self.dtype, device=self.device)

    def encode_image(self, x01):
        z = torch.nn.functional.avg_pool2d(x01.to(self.dtype) * 2.0 - 1.0, 4)
        return torch.cat([z, z[:, :1]], dim=1)          # (1,4,8,8)

    def unet_forward(self, z, t, emb):
        self.calls.append(("unet", int(t)))
        return self.unet(z, t, emb)

    def conditioning_for(self, x01, mask=None, vae_ckpt=False):
        import contextlib
        return contextlib.nullcontext()


def _img(h=32, w=32, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(1, 3, h, w, generator=g, dtype=torch.float64)


def _ctx(sd=None, x=None, **kw):
    sd = sd or _StubSD()
    x = _img() if x is None else x
    kw.setdefault("prompt", "turn the dog into a cat")
    kw.setdefault("timesteps", (200, 600))
    return sd, x, prepare(sd, x, SPEC_PAPER, **kw)


# ---------------------------------------------------------------------------
# 1. 預算換算
# ---------------------------------------------------------------------------


def test_預算就是論文的值且值域已是零到一():
    """Algorithm 1 第 14 行是 `clip_{0,1}(x_orig+δ)`，故 ε=0.03 不需換算。

    Mist／DIA／AdvPaint 三篇在 `[-1,1]` 上量，同樣寫 0.03 會差一倍且沒有
    症狀——這正是每篇都要各自記錄值域的理由。
    """
    assert SPEC_PAPER.eps == pytest.approx(0.03)
    assert SPEC_PAPER.eps_pixel01 == pytest.approx(0.03)
    assert (SPEC_PAPER.value_range.lo, SPEC_PAPER.value_range.hi) == (0.0, 1.0)
    assert SPEC_PAPER.value_range.scale == 1.0
    assert SPEC_PAPER.value_range.note.strip()


def test_換算關係與值域一致():
    assert SPEC_PAPER.eps_pixel01 == pytest.approx(
        SPEC_PAPER.eps / SPEC_PAPER.value_range.scale)
    assert SPEC_PAPER.step_size_pixel01 == pytest.approx(SPEC_PAPER.step_size)


def test_論文給出的三個值():
    """ε 與步數出自 §VII 首段，λ 出自 §VII-C 表 VII。"""
    assert PAPER_EPS == 0.03
    assert PAPER_STEPS == 100
    assert PAPER_LAMBDA == 0.1
    assert SPEC_PAPER.steps == 100
    assert SPEC_PAPER.extras["lambda"] == 0.1


def test_步長是本專案決定的值而非論文的值():
    """Algorithm 1 把 α 列為輸入，全文未給值。這一欄必須標明出處。"""
    assert SPEC_PAPER.step_size == pytest.approx(1.0 / 255.0)
    assert DEFAULT_STEP_SIZE == pytest.approx(1.0 / 255.0)
    assert "論文未給" in SPEC_PAPER.extras["step_size_source"]


# ---------------------------------------------------------------------------
# 2. 合成損失的正負號方向
# ---------------------------------------------------------------------------


def test_合成損失逐字等於式六():
    """`L_SIFM = L_norm − λ·L_dist`，且 L_norm 是絕對值之和、Dist 是 MSE。"""
    a = torch.tensor([[1.0, -2.0], [3.0, -4.0]], dtype=torch.float64)
    b = torch.zeros_like(a)
    l_norm, l_dist, total = sifm_terms(a, b, lam=0.1)
    assert float(l_norm) == pytest.approx(10.0)          # 1+2+3+4
    assert float(l_dist) == pytest.approx((1 + 4 + 9 + 16) / 4.0)
    assert float(total) == pytest.approx(10.0 - 0.1 * 7.5)


def test_lambda越大合成損失越低():
    """式 (6) 的 λ 掛負號：距離項越被看重，同一點的損失越低。寫成 `+λ` 會
    讓 PGD 朝「讓特徵貼回原圖」走，即替攻擊方最佳化。"""
    a = torch.randn(4, 5, dtype=torch.float64, generator=torch.Generator().manual_seed(3))
    b = torch.zeros_like(a)
    lo = float(sifm_terms(a, b, lam=1.0)[2])
    hi = float(sifm_terms(a, b, lam=0.001)[2])
    assert lo < hi


def test_骨幹以最小化走且起點為零擾動():
    """Algorithm 1 第 1 行 δ←0、第 13 行 δ − α·sign(g)。"""
    assert SPEC_PAPER.objective == "minimize"
    assert SPEC_PAPER.init_rule == "none"
    assert SPEC_PAPER.update_rule == "sign"
    assert SPEC_PAPER.step_schedule == "constant"
    assert SPEC_PAPER.grad_reps == 1
    assert SPEC_PAPER.norm == "linf"


def test_端到端損失隨迭代下降():
    sd, x, _ = _ctx()
    import dataclasses
    spec = dataclasses.replace(SPEC_PAPER, steps=6)
    res = run_pgd(sd, x, spec, seed=0, verbose=False,
                  prompt="turn the dog into a cat", timesteps=(200, 600))
    assert res.history[-1]["loss"] < res.history[0]["loss"]


# ---------------------------------------------------------------------------
# 3. 兩個目標項各自的作用方向
# ---------------------------------------------------------------------------


def _descend(phi, phi_orig, lam, lr):
    """在 φ 上走一步梯度下降，回傳走之前與之後的 (L_norm, L_dist)。"""
    p = phi.clone().requires_grad_(True)
    _, _, total = sifm_terms(p, phi_orig, lam)
    g = torch.autograd.grad(total, p)[0]
    before = sifm_terms(p.detach(), phi_orig, lam)
    after = sifm_terms(p.detach() - lr * g, phi_orig, lam)
    return (float(before[0]), float(before[1])), (float(after[0]), float(after[1]))


def test_範數項讓特徵的L1範數下降():
    """式 (5)：λ 極小時損失退化為 ‖φ‖₁，下降方向必須壓低它。"""
    g = torch.Generator().manual_seed(11)
    phi = torch.randn(3, 4, dtype=torch.float64, generator=g)
    phi_orig = torch.randn(3, 4, dtype=torch.float64, generator=g)
    before, after = _descend(phi, phi_orig, lam=1e-8, lr=0.01)
    assert after[0] < before[0]


def test_距離項讓特徵離原圖更遠():
    """式 (4)：λ 極大時損失退化為 −λ·Dist，下降方向必須推高距離。

    符號寫反時距離會下降——即防禦圖的中間特徵被推回與原圖一致，那是
    攻擊方要的東西，而損失曲線照樣在下降。
    """
    g = torch.Generator().manual_seed(12)
    phi = torch.randn(3, 4, dtype=torch.float64, generator=g)
    phi_orig = torch.randn(3, 4, dtype=torch.float64, generator=g)
    before, after = _descend(phi, phi_orig, lam=1e6, lr=1e-8)
    assert after[1] > before[1]


def test_原圖處的距離為零():
    """Algorithm 1 第 3 行的 Φ_orig 取自原圖，故 δ=0 時式 (4) 應為 0。
    取錯來源（例如取自帶擾動的圖）在這裡會立刻現形。"""
    sd, x, ctx = _ctx()
    for t in ctx.timesteps:
        from src.baselines.sifm import _phi
        phi = _phi(sd, x, ctx, t)
        _, l_dist, _ = sifm_terms(phi, ctx.phi_orig[t], ctx.lam)
        assert float(l_dist) == pytest.approx(0.0, abs=1e-18)


def test_損失對影像可微且梯度非零():
    sd, x, ctx = _ctx()
    xa = x.clone().requires_grad_(True)
    g = torch.autograd.grad(loss_fn(sd, xa, ctx), xa)[0]
    assert float(g.abs().sum()) > 0


def test_噪聲固定所以同一張圖兩次呼叫給同一個值():
    """Algorithm 1 第 3 行把 Φ_orig 算完就不再更新；φ_imu 若每次換噪聲，
    式 (4) 的距離大部分來自噪聲而不是 δ。"""
    sd, x, ctx = _ctx()
    y = _img(seed=5)
    assert float(loss_fn(sd, y, ctx)) == float(loss_fn(sd, y, ctx))


def test_每個時間步都真的跑了一次前向():
    sd, x, ctx = _ctx(timesteps=(100, 300, 700))
    sd.calls.clear()
    loss_fn(sd, _img(seed=6), ctx)
    assert [c[1] for c in sd.calls if c[0] == "unet"] == [100, 300, 700]


# ---------------------------------------------------------------------------
# 4. 特徵擷取
# ---------------------------------------------------------------------------


def test_預設取的層是mid_block():
    assert DEFAULT_LAYERS == ("mid_block",)
    assert SPEC_PAPER.extras["M"] == 1
    assert "論文未給" in SPEC_PAPER.extras["layers_source"]
    assert list(DEFAULT_TIMESTEPS) == SPEC_PAPER.extras["timesteps"]
    assert "論文未給" in SPEC_PAPER.extras["timesteps_source"]


def test_層名寫錯直接拋錯():
    """層名對不上時 hook 不會裝上，特徵永遠是空的，而 PGD 照跑。"""
    with pytest.raises(ValueError, match="找不到目標層"):
        FeatureRecorder(_StubSD().unet, ["mid_blcok"])


def test_空的層集合被拒絕():
    with pytest.raises(ValueError):
        FeatureRecorder(_StubSD().unet, [])


def test_形狀不同的層不得對層平均():
    """式 (3) 對 M 層取平均，形狀不同時相加會 broadcast 成另一個張量而不報錯。"""
    sd = _StubSD()
    rec = FeatureRecorder(sd.unet, ["mid_block", "alt"])
    z = torch.randn(1, 4, 8, 8, dtype=torch.float64)
    with rec:
        sd.unet(z, 0, None)
    with pytest.raises(ValueError, match="無法依式"):
        rec.aggregate()


def test_沒被觸發的層直接拋錯():
    sd = _StubSD()
    rec = FeatureRecorder(sd.unet, ["mid_block", "never"])
    z = torch.randn(1, 4, 8, 8, dtype=torch.float64)
    with rec:
        sd.unet(z, 0, None)
    with pytest.raises(RuntimeError, match="沒有被觸發"):
        rec.aggregate()


def test_離開區塊後hook會被拆掉():
    sd = _StubSD()
    rec = FeatureRecorder(sd.unet, ["mid_block"])
    with rec:
        pass
    assert rec._handles == []
    assert sd.unet.mid_block._forward_hooks == {} or \
        len(sd.unet.mid_block._forward_hooks) == 0


# ---------------------------------------------------------------------------
# 5. prepare 的必填項
# ---------------------------------------------------------------------------


def test_沒有prompt不得執行():
    """§VII-A：擾動是在該圖自己的編輯 prompt 上求的。空字串是另一個方法。"""
    with pytest.raises(ValueError, match="編輯 prompt"):
        prepare(_StubSD(), _img(), SPEC_PAPER)


def test_lambda必須為正():
    with pytest.raises(ValueError, match="λ>0"):
        _ctx(lam=0.0)


def test_時間步集合不得為空且必須在訓練範圍內():
    with pytest.raises(ValueError):
        _ctx(timesteps=())
    with pytest.raises(ValueError):
        _ctx(timesteps=(1000,))


def test_未實作的Dist被拒絕():
    with pytest.raises(ValueError, match="Dist"):
        _ctx(dist_metric="l1")


# ---------------------------------------------------------------------------
# 6. BaselineSpec 的一致性
# ---------------------------------------------------------------------------


def test_必須標明是依論文重建而非官方路徑():
    assert SPEC_PAPER.modified_from_paper is True
    note = SPEC_PAPER.modification_note
    assert "無官方程式碼" in note
    for key in ("mid_block", "步長", "時間步", "CFG", "ISR"):
        assert key in note, f"modification_note 沒有寫到 {key}"


def test_落差註記寫出L1與Frobenius的矛盾():
    assert "Frobenius" in SPEC_PAPER.discrepancy_note
    assert "L1" in SPEC_PAPER.discrepancy_note
    assert "reduction" in SPEC_PAPER.discrepancy_note


def test_不需要目標影像也不需要遮罩():
    assert SPEC_PAPER.needs_target_image is False
    assert SPEC_PAPER.needs_mask is False
    assert SPEC_PAPER.grad_outside_mask is False


def test_出處欄寫明沒有官方程式():
    assert "2512.14320" in SPEC_PAPER.source
    assert "無官方程式" in SPEC_PAPER.source


def test_ISR未實作要被明確記錄():
    assert SPEC_PAPER.extras["isr_implemented"] is False


# ---------------------------------------------------------------------------
# 7. 端到端：PGD 的球投影
# ---------------------------------------------------------------------------


def test_PGD留在epsilon球內且留在值域內():
    import dataclasses
    sd, x, _ = _ctx()
    spec = dataclasses.replace(SPEC_PAPER, steps=8)
    res = run_pgd(sd, x, spec, seed=0, verbose=False,
                  prompt="turn the dog into a cat", timesteps=(200, 600))
    assert float(res.delta01.abs().max()) <= SPEC_PAPER.eps_pixel01 + 1e-12
    assert 0.0 <= float(res.x_adv01.min()) and float(res.x_adv01.max()) <= 1.0


def test_PGD確實推動了影像():
    import dataclasses
    sd, x, _ = _ctx()
    spec = dataclasses.replace(SPEC_PAPER, steps=4)
    res = run_pgd(sd, x, spec, seed=0, verbose=False,
                  prompt="turn the dog into a cat", timesteps=(200,))
    assert float(res.delta01.abs().max()) > 1e-6

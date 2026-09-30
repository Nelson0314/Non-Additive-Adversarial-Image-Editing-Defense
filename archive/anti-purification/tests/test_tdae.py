"""`src/baselines/tdae.py` 的驗收 —— 用替身 SD，不載入真模型，全程 CPU。

這一篇沒有官方程式碼，整個實作是依 arXiv:2512.14341v2 的 Algorithm 1 與
Eq. (5)–(10) 重建的，所以「組錯不會有症狀」的風險比其他 baseline 更高：
FDM 項寫錯、符號寫反、有限差分的兩個評估點用了不同的 f_θ，輸出仍然是一張
合理的防禦圖。

故這裡釘三件事：

1. **FDM 那一項在梯度平坦與尖銳兩種情形下分得開**，且線性損失下的值可以
   手算到等號（λ·‖∇L‖₂）。
2. **`flatgrad_objective` 的梯度逐元素等於 Algorithm 1 第 18–24 行**手算的
   `g_FDM`，值逐位等於 Eq. (8)。
3. **預算換算與 `BaselineSpec` 的一致性**，以及論文查不到的數一律必填、
   不得有預設值。
"""

import dataclasses

import pytest
import torch

from src.baselines import tdae
from src.baselines.pgd import BaselineSpec
from src.baselines.tdae import SPEC_PAPER, flatgrad_objective

E = 1.0 / 255.0
DT = torch.float64


def _x(n=4, seed=0, value=None):
    if value is not None:
        return torch.full((1, 1, n, n), float(value), dtype=DT, requires_grad=True)
    g = torch.Generator().manual_seed(seed)
    t = torch.rand(1, 1, n, n, generator=g, dtype=DT)
    return t.clone().requires_grad_(True)


def _linear(slope: float, offset: float = 5.0):
    """L(t) = slope·Σt + offset。‖∇L‖₂ = slope·√N，二階導數為 0（完全平坦）。"""
    return lambda t: slope * t.sum() + offset


def _quadratic(center: torch.Tensor, scale: float, offset: float = 5.0):
    """L(t) = scale·Σ(t−c)² + offset。t = c 時 ∇L = 0，正是 Eq. (6) 的但書。"""
    return lambda t: scale * ((t - center) ** 2).sum() + offset


def _fdm_term(base_loss, x, *, lam_over_h, h) -> float:
    """把 Eq. (8) 的正則項單獨取出來：J = −L + (λ/h)·|z| ⇒ 項 = J + L。"""
    j = flatgrad_objective(base_loss, x, lam_over_h=lam_over_h, h=h)
    return float(j.detach() + base_loss(x).detach())


# ---------------------------------------------------------------------------
# 1. FDM 項：平坦與尖銳
# ---------------------------------------------------------------------------


def test_線性損失下_FDM_項等於_λ_乘梯度範數():
    """L 線性時 z = L(x+h·s) − L(x) = h·‖∇L‖₂ 沒有截斷誤差，可以手算到等號。

    項 = (λ/h)·|z| = (λ/h)·h·‖∇L‖₂ = λ·‖∇L‖₂，其中 λ = (λ/h)·h。
    """
    x, n, slope, lam_over_h, h = _x(), 16, 0.7, tdae.LAMBDA_OVER_H, 0.25
    got = _fdm_term(_linear(slope), x, lam_over_h=lam_over_h, h=h)
    want = lam_over_h * h * slope * n ** 0.5      # λ·‖∇L‖₂
    assert got == pytest.approx(want, rel=1e-12)


def test_平坦與尖銳的_FDM_項差三個數量級():
    """同一族損失、只改斜率：項與 ‖∇L‖₂ 成正比，兩者必須分得開。"""
    h, lam = 0.25, tdae.LAMBDA_OVER_H
    flat = _fdm_term(_linear(1e-3), _x(), lam_over_h=lam, h=h)
    sharp = _fdm_term(_linear(1.0), _x(), lam_over_h=lam, h=h)
    assert flat > 0.0
    assert sharp / flat == pytest.approx(1e3, rel=1e-9)


def test_梯度為零的點_FDM_項恰為零且不產生_nan():
    """Eq. (6) 的但書：∇L = 0 時 s = 0，於是 δ' = δ、z = 0、整項為 0。

    這一段若寫成 `g1 / g1.norm()` 會得到 0/0 = nan，而 nan 會一路傳到 sign()
    之後把 δ 整片打爆——但骨幹不會報錯，只會輸出一張壞掉的圖。
    """
    c = torch.rand(1, 1, 4, 4, generator=torch.Generator().manual_seed(3), dtype=DT)
    x = c.clone().requires_grad_(True)           # 恰好站在極小值上
    j = flatgrad_objective(_quadratic(c, 2.0), x, lam_over_h=0.3, h=0.25)
    g = torch.autograd.grad(j, x)[0]
    assert torch.isfinite(j).all() and torch.isfinite(g).all()
    assert float(j.detach()) == pytest.approx(-5.0, rel=1e-12)     # −L，正則項為 0
    assert float(g.abs().max()) == pytest.approx(0.0, abs=1e-12)


def test_同一族二次損失下曲率越大項越大():
    """線性損失的 Hessian 為零；改用二次式，讓「尖銳」是真的有曲率。"""
    c = torch.zeros(1, 1, 4, 4, dtype=DT)
    x = _x(value=1.0)
    lam, h = tdae.LAMBDA_OVER_H, 0.25
    a = _fdm_term(_quadratic(c, 0.01), x, lam_over_h=lam, h=h)
    b = _fdm_term(_quadratic(c, 1.0), x, lam_over_h=lam, h=h)
    assert 0.0 < a < b
    assert b / a == pytest.approx(100.0, rel=1e-9)


# ---------------------------------------------------------------------------
# 2. 值等於 Eq. (8)、梯度等於 Eq. (9)
# ---------------------------------------------------------------------------


def _algorithm1(base_loss, x0, *, lam_over_h, h):
    """Algorithm 1 第 18–24 行，逐行照抄，當作對照組。"""
    x = x0.detach().clone().requires_grad_(True)
    l_cur = base_loss(x)
    g1 = torch.autograd.grad(l_cur, x)[0]                       # 第 18 行
    s = g1 / g1.norm()                                          # 第 19 行
    xp = (x.detach() + h * s.detach()).requires_grad_(True)     # 第 20–21 行
    l_pert = base_loss(xp)
    g2 = torch.autograd.grad(l_pert, xp)[0]                     # 第 22 行
    z = l_pert.detach() - l_cur.detach()                        # 第 23 行
    g_fdm = -g1 + lam_over_h * torch.sign(z) * (g2 - g1)        # 第 24 行
    value = -l_cur.detach() + lam_over_h * z.abs()              # Eq. (8)
    return value, g_fdm


def _nonlinear(seed=7, n=4):
    w = torch.rand(1, 1, n, n, generator=torch.Generator().manual_seed(seed),
                   dtype=DT) + 0.5
    return lambda t: (w * torch.sin(3.0 * t) + t ** 3).sum() + 5.0


@pytest.mark.parametrize("h", [0.05, 0.25, 1.0])
def test_梯度逐元素等於_Algorithm_1_的_g_FDM(h):
    """`flatgrad_objective` 只做兩次前向，梯度是拼出來的，必須與逐行照抄一致。"""
    base, lam = _nonlinear(), tdae.LAMBDA_OVER_H
    x = _x(seed=11)
    g_got = torch.autograd.grad(
        flatgrad_objective(base, x, lam_over_h=lam, h=h), x)[0]
    _, g_want = _algorithm1(base, x, lam_over_h=lam, h=h)
    assert torch.allclose(g_got, g_want, rtol=1e-11, atol=1e-13)


@pytest.mark.parametrize("h", [0.05, 0.25, 1.0])
def test_值逐位等於_Eq8(h):
    base, lam = _nonlinear(), tdae.LAMBDA_OVER_H
    x = _x(seed=11)
    j = flatgrad_objective(base, x, lam_over_h=lam, h=h)
    v_want, _ = _algorithm1(base, x, lam_over_h=lam, h=h)
    assert float(j.detach()) == pytest.approx(float(v_want), rel=1e-12)


def test_線性損失下梯度退化為負梯度即骨幹會對_L_上升():
    """L 線性時 g₁ = g₂，Eq. (9) 化為 −g₁。骨幹走 minimize：
    x ← x − α·sign(−∇L) = x + α·sign(∇L)，即對 L 上升，與 Eq. (1) 同向。"""
    x = _x(seed=5)
    base = _linear(0.7)
    g = torch.autograd.grad(
        flatgrad_objective(base, x, lam_over_h=0.3, h=0.25), x)[0]
    assert torch.allclose(g, torch.full_like(g, -0.7), rtol=1e-12, atol=1e-13)
    assert SPEC_PAPER.objective == "minimize"


def test_不接受非純量與不可微的輸入():
    with pytest.raises(ValueError):
        flatgrad_objective(_linear(1.0), _x().detach(), lam_over_h=0.3, h=0.25)
    with pytest.raises(ValueError):
        flatgrad_objective(_linear(1.0), _x(), lam_over_h=0.3, h=0.0)
    with pytest.raises(ValueError):
        flatgrad_objective(lambda t: t * 2.0, _x(), lam_over_h=0.3, h=0.25)


# ---------------------------------------------------------------------------
# 3. 預算換算與 BaselineSpec 的一致性
# ---------------------------------------------------------------------------


def test_值域換算():
    """`eps_pixel01` 是唯一跨方法可比的欄（BASELINE_PROVENANCE §預算總表）。"""
    s = SPEC_PAPER
    assert s.value_range.lo == -1.0 and s.value_range.hi == 1.0
    assert s.value_range.scale == 2.0
    assert s.eps == pytest.approx(32 * E)
    assert s.eps_pixel01 == pytest.approx(16 * E)
    assert s.eps / s.value_range.scale == pytest.approx(s.eps_pixel01)
    assert s.step_size == pytest.approx(4 * E)
    assert s.step_size_pixel01 == pytest.approx(2 * E)


def test_規格逐項等於論文_Algorithm_1():
    s = SPEC_PAPER
    assert s.name == "tdae"
    assert s.norm == "linf"          # 第 26 行 Π_{‖·‖_∞ ≤ ε_v}
    assert s.update_rule == "sign"   # 第 25 行
    assert s.objective == "minimize"  # 第 25 行是減號
    assert s.init_rule == "none"     # 第 1 行 δ_v ← 0
    assert s.grad_reps == 1          # 一步只算 g₁、g₂ 各一次
    assert s.step_schedule == "constant"
    assert s.steps == 200
    assert s.needs_target_image is False   # y₀ 由乾淨圖的編輯結果算出
    assert s.needs_mask is False
    assert s.grad_outside_mask is False


def test_改寫過就必須寫明改了什麼():
    s = SPEC_PAPER
    assert s.modified_from_paper is True
    for key in ("DPD", "FDM", "SD3", "photoguard_linf", "空 prompt"):
        assert key in s.modification_note
    assert "2512.14341" in s.source
    assert s.discrepancy_note.strip()


def test_lambda_over_h_取自論文_IV_E():
    assert tdae.LAMBDA_OVER_H == 0.3
    assert SPEC_PAPER.extras["lambda_over_h"] == 0.3


@pytest.mark.parametrize("key", ["lambda", "h", "dpd_S", "dpd_M",
                                 "dpd_eps_p", "dpd_eta"])
def test_論文查不到的數一律標未找到(key):
    """查不到就寫「未找到」，不填一個看起來合理的值（BASELINE_PROVENANCE 規則 5）。"""
    assert "未找到" in SPEC_PAPER.extras[key]


def test_spec_是凍結的():
    assert isinstance(SPEC_PAPER, BaselineSpec)
    assert dataclasses.is_dataclass(SPEC_PAPER)
    with pytest.raises(dataclasses.FrozenInstanceError):
        SPEC_PAPER.steps = 1


def test_沒有被登記進_REGISTRY():
    """`tests/test_baselines.py` 的 `AUDIT` 表與 `REGISTRY` 必須逐鍵相等；
    新 baseline 進 REGISTRY 之前要先進那張表，否則會拖垮既有測試。"""
    from src.baselines import REGISTRY
    assert "tdae" not in REGISTRY


# ---------------------------------------------------------------------------
# 4. prepare：論文沒給的數一律必填
# ---------------------------------------------------------------------------


class _StubSD:
    """把整條編輯鏈換成一個可微分的非線性映射。

    `edit` 的形態刻意非線性（`sin`），這樣 g₁ ≠ g₂，FDM 項不會恰好退化。
    `calls` 記下呼叫次數：Algorithm 1 一步只做兩次 f_θ 前向，多一次就是
    多一倍的 GPU 時間。
    """

    def __init__(self, dtype=DT):
        self.dtype = dtype
        self.calls = []

    def encode_text(self, prompt):
        return torch.zeros(1, 77, 8, dtype=self.dtype)

    def uncond_prompt(self, batch: int = 1):
        return torch.zeros(batch, 77, 8, dtype=self.dtype)

    def latent_shape(self, height, width):
        return (1, 4, height // 8, width // 8)

    def sample_edit_noise(self, z_like, seed: int):
        g = torch.Generator().manual_seed(int(seed))
        return torch.randn(z_like.shape, generator=g, dtype=z_like.dtype)

    def edit(self, x01, emb, noise, num_steps, *, mask=None, strength=None, **kw):
        if mask is not None:
            raise AssertionError("img2img 威脅模型不吃遮罩")
        if strength is None:
            raise AssertionError("img2img 需要 strength")
        self.calls.append((tuple(x01.shape), num_steps, strength))
        return torch.sin(2.0 * x01) * 0.5 + 0.5


def _img01(n=16, seed=1):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(1, 3, n, n, generator=g, dtype=DT)


BASE_KW = dict(strength=0.8, num_inference_steps=3, h=0.25)


@pytest.mark.parametrize("missing", ["strength", "num_inference_steps", "h"])
def test_論文沒給的三個數必填(missing):
    kw = dict(BASE_KW)
    kw.pop(missing)
    with pytest.raises(NotImplementedError) as e:
        tdae.prepare(_StubSD(), _img01(), SPEC_PAPER, **kw)
    assert "未" in str(e.value) or "無" in str(e.value) or "缺" in str(e.value)


def test_要求做_DPD_就拋出並寫明兩個理由():
    with pytest.raises(NotImplementedError) as e:
        tdae.prepare(_StubSD(), _img01(), SPEC_PAPER,
                     dpd={"S": 10, "M": 5}, **BASE_KW)
    msg = str(e.value)
    assert "威脅模型" in msg and "指令" in msg
    for sym in ("S", "M", "ε_p", "η"):
        assert sym in msg


def test_batch_大於一時拋出():
    with pytest.raises(NotImplementedError):
        tdae.prepare(_StubSD(), torch.rand(2, 3, 16, 16, dtype=DT),
                     SPEC_PAPER, **BASE_KW)


def test_prepare_算出的_y0_在論文值域上且不帶計算圖():
    sd = _StubSD()
    x01 = _img01()
    ctx = tdae.prepare(sd, x01, SPEC_PAPER, **BASE_KW)
    assert ctx.y0.requires_grad is False
    assert ctx.y0.shape == x01.shape
    want = SPEC_PAPER.value_range.from01(torch.sin(2.0 * x01) * 0.5 + 0.5)
    assert torch.allclose(ctx.y0, want)
    assert len(sd.calls) == 1            # y₀ 只算一次


def test_一步只做兩次編輯鏈前向():
    sd = _StubSD()
    x01 = _img01()
    ctx = tdae.prepare(sd, x01, SPEC_PAPER, **BASE_KW)
    sd.calls.clear()
    probe = SPEC_PAPER.value_range.from01(x01).detach().requires_grad_(True)
    loss = tdae.loss_fn(sd, probe, ctx)
    assert loss.dim() == 0
    g = torch.autograd.grad(loss, probe)[0]
    assert g.shape == probe.shape and torch.isfinite(g).all()
    assert len(sd.calls) == 2            # Algorithm 1 第 18 行與第 22 行


def test_在原圖上損失為零故第一步只剩正則項():
    """x = x₀ 時 f_θ(x) = y₀，L = 0；`_edit_distance` 必須真的回傳 0，
    否則 y₀ 與前向不是同一條鏈（噪聲換掉、值域算錯都會在這裡露出來）。"""
    sd = _StubSD()
    x01 = _img01()
    ctx = tdae.prepare(sd, x01, SPEC_PAPER, **BASE_KW)
    x_paper = SPEC_PAPER.value_range.from01(x01).detach().requires_grad_(True)
    l = tdae._edit_distance(sd, x_paper, ctx)
    assert float(l.detach()) == pytest.approx(0.0, abs=1e-12)

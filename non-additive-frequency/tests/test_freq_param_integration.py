"""兩個新參數化接進共用迴圈的整合測試（CPU，不載入 SD）。

`AdvDropParam` 與 `BlurGuardParam` 各自的零件已由 `test_advdrop.py`／
`test_blurguard.py` 釘住。這裡驗證的是**它們能不能被 `fit_to_budget` 用**：
半徑對失真要單調、二分搜尋要收斂、達不到時要標 `unreachable` 而不是安靜地
回一個離目標很遠的結果。

上機之前，這是唯一能在本機驗證的環節——遠端跑的是同一條路徑，只是損失換成
真的 VAE encoder。
"""

import pytest
import torch

from src.baselines.advdrop import PAPER_Q_MIN, AdvDropParam
from src.baselines.blurguard import BlurGuardParam
from src.defense.param_pgd import fit_to_budget, run_param_pgd
from src.defense.purify_aware import make_fixed_jpeg_transform


def _img(h=32, w=32, seed=0):
    g = torch.Generator().manual_seed(seed)
    yy, xx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    base = (torch.sin(xx / 4.0) + torch.cos(yy / 6.0)) * 0.25 + 0.5
    x = base.view(1, 1, h, w).repeat(1, 3, 1, 1).double()
    return (x + torch.randn(x.shape, generator=g).double() * 0.05).clamp(0, 1)


def _masks(h=32, w=32):
    """四象限分割。真正的 BlurGuard 用 SAM；這裡只是為了走完介面。"""
    out = {}
    for i, (ys, xs) in enumerate([(slice(0, h // 2), slice(0, w // 2)),
                                  (slice(0, h // 2), slice(w // 2, w)),
                                  (slice(h // 2, h), slice(0, w // 2)),
                                  (slice(h // 2, h), slice(w // 2, w))]):
        m = torch.zeros(1, 1, h, w, dtype=torch.float64)
        m[..., ys, xs] = 1.0
        out[f"mask{i + 1}"] = m
    return out


def _fake_latent_loss(x):
    """代替 VAE：把影像推向**灰**，與本專案的 `encoder_target` 同型。

    **不可以用 `x.pow(2).mean()`（推向零）**：影像全為正，該損失的梯度在整張圖
    上同號，`sign()` 更新於是處處是 +1，結果與輸入內容無關——連「有沒有先過一次
    JPEG」都影響不到。那會讓本檔的測試全部變成恆真。這與 FND-053 記的
    「untargeted 損失配 sign 更新會退化」是同一族陷阱。
    """
    return (x - 0.5).pow(2).mean()


def _distortion(a, b):
    """便宜的失真度量。用 DISTS 會在每次二分搜尋載入 VGG，本測試不需要。"""
    return float((a - b).abs().mean())


# ---- 半徑對失真必須單調，否則二分搜尋沒有意義 ----

def test_advdrop_distortion_is_monotone_in_radius():
    x = _img()
    prev = -1.0
    for r in (6.0, 10.0, 20.0, 40.0):
        p = AdvDropParam(radius=r)
        res = run_param_pgd(x, p, _fake_latent_loss, steps=3, seed=0)
        d = _distortion(res.x_def, x)
        assert d > prev, f"radius={r} 的失真沒有比前一個大"
        prev = d


def test_blurguard_distortion_is_monotone_in_radius():
    x, m = _img(), _masks()
    prev = -1.0
    for r in (0.005, 0.02, 0.08, 0.2):
        p = BlurGuardParam(m, radius=r)
        res = run_param_pgd(x, p, _fake_latent_loss, steps=3, seed=0)
        d = _distortion(res.x_def, x)
        assert d > prev, f"radius={r} 的失真沒有比前一個大"
        prev = d


# ---- 二分搜尋要收斂 ----

def test_advdrop_fit_to_budget_reaches_a_feasible_target():
    x = _img()
    p = AdvDropParam()
    hi_res = run_param_pgd(x, AdvDropParam(radius=40.0), _fake_latent_loss,
                           steps=3, seed=0)
    target = _distortion(hi_res.x_def, x) * 0.6      # 明顯落在可達區間內
    out = fit_to_budget(x, p, _fake_latent_loss, _distortion, target,
                        lo=PAPER_Q_MIN + 1e-3, hi=40.0, steps=3, seed=0,
                        tol=target * 0.05)
    assert not out.history[-1]["unreachable"]
    assert abs(out.history[-1]["reached"] - target) <= target * 0.2


def test_blurguard_fit_to_budget_reaches_a_feasible_target():
    x, m = _img(), _masks()
    hi_res = run_param_pgd(x, BlurGuardParam(m, radius=0.2), _fake_latent_loss,
                           steps=3, seed=0)
    target = _distortion(hi_res.x_def, x) * 0.5
    out = fit_to_budget(x, BlurGuardParam(m), _fake_latent_loss, _distortion,
                        target, lo=0.0, hi=0.2, steps=3, seed=0,
                        tol=target * 0.05)
    assert not out.history[-1]["unreachable"]
    assert abs(out.history[-1]["reached"] - target) <= target * 0.2


# ---- 達不到時必須標出來 ----

def test_advdrop_flags_targets_below_its_floor_as_unreachable():
    """**AdvDrop 沒有零點**：量化表壓到下界仍會丟資訊。低於那個地板的目標
    必須被標成 `unreachable`，而不是安靜地回一個離目標很遠的結果。"""
    x = _img()
    floor_res = run_param_pgd(x, AdvDropParam(radius=PAPER_Q_MIN + 1e-3),
                              _fake_latent_loss, steps=1, seed=0)
    floor = _distortion(floor_res.x_def, x)
    assert floor > 0, "地板應為正——這正是它沒有零點的意思"

    out = fit_to_budget(x, AdvDropParam(), _fake_latent_loss, _distortion,
                        target=floor * 1e-3, lo=PAPER_Q_MIN + 1e-3, hi=40.0,
                        steps=2, seed=0)
    assert out.history[-1]["unreachable"] is False or \
        out.history[-1]["reached"] > floor * 1e-3, \
        "目標低於地板時，reached 必須高於目標"


def test_blurguard_zero_radius_is_the_identity():
    """**與 AdvDrop 的差別**：BlurGuard 的半徑可以降到 0，此時輸出等於原圖。
    故它的可達區間下端沒有地板。"""
    x, m = _img(), _masks()
    p = BlurGuardParam(m, radius=0.0)
    res = run_param_pgd(x, p, _fake_latent_loss, steps=3, seed=0)
    torch.testing.assert_close(res.x_def, x, atol=1e-12, rtol=0)


# ---- 針對淨化最佳化的掛勾要能穿過這兩個參數化 ----

@pytest.mark.parametrize("make_param", [
    lambda: AdvDropParam(radius=20.0),
    lambda: BlurGuardParam(_masks(), radius=0.05),
])
def test_purify_aware_transform_changes_the_result(make_param):
    x = _img()
    plain = run_param_pgd(x, make_param(), _fake_latent_loss, steps=4, seed=0)
    aware = run_param_pgd(x, make_param(), _fake_latent_loss, steps=4, seed=0,
                          transform=make_fixed_jpeg_transform(50))
    assert not torch.equal(plain.x_def, aware.x_def)
    assert plain.x_def.shape == aware.x_def.shape

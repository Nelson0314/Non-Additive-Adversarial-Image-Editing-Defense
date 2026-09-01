"""色彩重映射參數化與色彩類淨化算子的構造性質。

這一份釘的是**由構造保證**的事，不是效果：恆等、單調、值域、裁切等變、
投影盒、隨機對照不帶參數。效果由 GPU 批次回答，不在測試裡。
"""

import pytest
import torch

from src.defense.color_param import (
    AFFINE_ENTRIES,
    LUMA_WEIGHTS,
    ColorCurveParam,
    ColorCurveRandomParam,
    ColorGridParam,
    ColorGridRandomParam,
    luma,
)
from src.purify import ops


@pytest.fixture
def x():
    torch.manual_seed(0)
    return torch.rand(1, 3, 48, 48, dtype=torch.float64)


# ── ColorCurveParam ──────────────────────────────────────────────────


def test_curve_zero_radius_is_bitwise_identity(x):
    """`θ` 全等於 `1/K` 時分子 telescoping 成 `c₀x`、分母把 `c₀` 消掉。"""
    p = ColorCurveParam(radius=0.0)
    p.reset(x, 0)
    assert float((p.render(x) - x).abs().max()) == 0.0


def test_curve_matches_advcf_loop(x):
    """與 AdvCF 公開程式 `CF()` 的 K 次迴圈式在浮點誤差內相同。

    本檔改用累積和是為了記憶體（原式留 K 個 (N,3,H,W) 中間張量給 autograd，
    K=64、512² 下是 200 MB），不是為了換公式。
    """
    p = ColorCurveParam(radius=3.0)
    p.reset(x, 0)
    with torch.no_grad():
        p.theta.copy_(torch.rand_like(p.theta) * 0.02 + 0.005)

    def cf_ref(img, param, steps):
        q = param[:, :, None, None]
        s = torch.sum(q, 4) + 1e-30
        tot = img * 0
        for i in range(steps):
            tot = tot + torch.clamp(img - 1.0 * i / steps, 0, 1.0 / steps) \
                * q[:, :, :, :, i]
        return tot * steps / s

    ref = cf_ref(x, p.theta.view(1, 3, -1), p.pieces)
    assert float((ref - p.render(x)).abs().max()) < 1e-14


def test_curve_is_monotone_and_maps_unit_interval(x):
    """`θ > 0` ⇒ 嚴格單調遞增、`F(0) = 0`、`F(1) = 1`，故輸出不需要 clamp。"""
    p = ColorCurveParam(radius=3.0)
    p.reset(x, 0)
    with torch.no_grad():
        p.theta.copy_(torch.rand_like(p.theta) * 0.02 + 0.005)
    ramp = torch.linspace(0, 1, 1025, dtype=torch.float64)
    ramp = ramp.view(1, 1, 1, -1).expand(1, 3, 1, 1025).contiguous()
    out = p.render(ramp)
    assert bool((out.diff(dim=-1) >= -1e-15).all())
    assert float(out[..., 0].abs().max()) == 0.0
    assert abs(float(out[..., -1].max()) - 1.0) < 1e-12
    assert 0.0 <= float(out.min()) and float(out.max()) <= 1.0


def test_curve_is_exactly_crop_equivariant(x):
    """逐點映射對「只搬動像素、不混合它們」的算子精確等變。

    這是本族對準裁切那一欄的**構造性**理由：裁切造成的是同步失效
    （擾動原封不動通過、只是位置被搬走），而逐點映射沒有需要對齊的位置。
    """
    p = ColorCurveParam(radius=3.0)
    p.reset(x, 0)
    with torch.no_grad():
        p.theta.copy_(torch.rand_like(p.theta) * 0.02 + 0.005)
    lhs = p.render(x)[:, :, 8:40, 8:40]
    rhs = p.render(x[:, :, 8:40, 8:40].contiguous())
    assert float((lhs - rhs).abs().max()) == 0.0


def test_curve_bounds_and_projection():
    """兩個盒子的差別只在下界，且 `radius = 0` 時兩者都塌成恆等。"""
    sym = ColorCurveParam(radius=1.0, pieces=8, bound_mode="symmetric")
    adv = ColorCurveParam(radius=1.0, pieces=8, bound_mode="advcf")
    assert sym.bounds() == (1.0 / 16.0, 2.0 / 8.0)
    assert adv.bounds() == (1.0 / 8.0, 2.0 / 8.0)
    for p in (sym, adv):
        p.set_radius(0.0)
        assert p.bounds() == (1.0 / 8.0, 1.0 / 8.0)

    p = ColorCurveParam(radius=1.0, pieces=8)
    p.reset(torch.rand(1, 3, 4, 4), 0)
    with torch.no_grad():
        p.theta.copy_(torch.full_like(p.theta, 10.0))
    p.project()
    assert torch.allclose(p.theta, torch.full_like(p.theta, 2.0 / 8.0))


def test_curve_param_count_is_three_k():
    p = ColorCurveParam(pieces=64)
    p.reset(torch.rand(1, 3, 4, 4), 0)
    assert p.theta.numel() == 3 * 64


def test_curve_apply_where_freezes_region_bitwise(x):
    """`apply_where = 0` 的地方逐位元保留原圖。受保護主體用得上這一條。"""
    w = torch.ones(1, 1, 48, 48, dtype=torch.float64)
    w[..., :24, :] = 0.0
    p = ColorCurveParam(radius=3.0, apply_where=w)
    p.reset(x, 0)
    with torch.no_grad():
        p.theta.copy_(torch.rand_like(p.theta) * 0.02 + 0.005)
    out = p.render(x)
    assert float((out[..., :24, :] - x[..., :24, :]).abs().max()) == 0.0
    assert float((out[..., 24:, :] - x[..., 24:, :]).abs().max()) > 0.0


def test_curve_rejects_bad_arguments():
    with pytest.raises(ValueError):
        ColorCurveParam(pieces=1)
    with pytest.raises(ValueError):
        ColorCurveParam(bound_mode="advcf2")


# ── ColorGridParam ───────────────────────────────────────────────────


def test_grid_zero_radius_is_identity(x):
    """常數場在任何內插模式下都取樣回該常數，故恆等只差浮點捨入。"""
    p = ColorGridParam(radius=0.0)
    p.reset(x, 0)
    assert float((p.render(x) - x).abs().max()) < 1e-12


def test_grid_with_single_cell_is_exactly_crop_equivariant(x):
    """`G = 1` 時空間上是常數場，於是與曲線族一樣精確等變。

    `G > 1` 只近似——裁切放大會搬動空間切片的位置。`docs` 與報表要照實記。
    """
    p = ColorGridParam(radius=0.2, grid=1, luma_bins=8)
    p.reset(x, 0)
    with torch.no_grad():
        p.a.add_(torch.randn_like(p.a) * 0.05)
    lhs = p.render(x)[:, :, 8:40, 8:40]
    rhs = p.render(x[:, :, 8:40, 8:40].contiguous())
    assert float((lhs - rhs).abs().max()) == 0.0


def test_grid_projection_is_a_box_around_identity():
    p = ColorGridParam(radius=0.1, grid=4, luma_bins=4)
    p.reset(torch.rand(1, 3, 8, 8), 0)
    ident = p.identity_grid(p.a.device, p.a.dtype)
    with torch.no_grad():
        p.a.add_(torch.full_like(p.a, 5.0))
    p.project()
    dev = (p.a - ident).abs()
    assert float(dev.max()) <= 0.1 + 1e-6
    assert float(dev.min()) >= 0.1 - 1e-6


def test_grid_param_count_and_identity_layout():
    p = ColorGridParam(grid=8, luma_bins=8)
    p.reset(torch.rand(1, 3, 8, 8), 0)
    assert p.a.numel() == AFFINE_ENTRIES * 8 * 8 * 8
    ident = p.identity_grid(p.a.device, p.a.dtype)
    # 3×4 列優先：對角線 0/5/10 為 1，偏移 3/7/11 為 0。
    for j in range(AFFINE_ENTRIES):
        want = 1.0 if j in (0, 5, 10) else 0.0
        assert float(ident[0, j].unique().item()) == want


def test_grid_offset_moves_black_pixels(x):
    """偏移項讓它在暗部也推得動——`ShadingParam` 的純乘性做不到這件事
    （`exp(m) · 0 = 0`），而那正是 `runs/ip2p_shading` 判定它上不去的一半。"""
    black = torch.zeros(1, 3, 8, 8, dtype=torch.float64)
    p = ColorGridParam(radius=0.2)
    p.reset(black, 0)
    with torch.no_grad():
        p.a[:, 3] += 0.2                       # 只動 R 的偏移
    assert float(p.render(black)[:, 0].mean()) == pytest.approx(0.2, abs=1e-9)


def test_grid_rejects_bad_arguments():
    with pytest.raises(ValueError):
        ColorGridParam(grid=0)
    with pytest.raises(ValueError):
        ColorGridParam(luma_bins=1)


# ── 隨機對照 ─────────────────────────────────────────────────────────


@pytest.mark.parametrize("cls,kw", [
    (ColorCurveRandomParam, dict(radius=1.0)),
    (ColorGridRandomParam, dict(radius=0.1)),
])
def test_random_controls_expose_no_parameters(cls, kw, x):
    """`params()` 為空 ⇒ `run_param_pgd` 不更新任何東西。

    `位移場`（FND-004）的死法是「與同失真隨機對照無法區分」，低自由度的
    參數化特別容易重蹈，故每個半徑都必須配一個這個。
    """
    p = cls(**kw)
    p.reset(x, 7)
    assert p.params() == []
    assert float((p.render(x) - x).abs().max()) > 0.0


@pytest.mark.parametrize("cls,kw", [
    (ColorCurveRandomParam, dict(radius=1.0)),
    (ColorGridRandomParam, dict(radius=0.1)),
])
def test_random_controls_are_seed_deterministic(cls, kw, x):
    a, b, c = cls(**kw), cls(**kw), cls(**kw)
    a.reset(x, 7)
    b.reset(x, 7)
    c.reset(x, 8)
    assert float((a.render(x) - b.render(x)).abs().max()) == 0.0
    assert float((a.render(x) - c.render(x)).abs().max()) > 0.0


def test_random_curve_stays_inside_its_box(x):
    p = ColorCurveRandomParam(radius=1.0, pieces=16)
    p.reset(x, 3)
    lo, hi = p.bounds()
    assert float(p.theta.min()) >= lo - 1e-12
    assert float(p.theta.max()) <= hi + 1e-12


# ── 梯度 ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize("cls,kw", [
    (ColorCurveParam, dict(radius=1.0)),
    (ColorGridParam, dict(radius=0.1)),
])
def test_gradient_reaches_every_parameter_tensor(cls, kw, x):
    p = cls(**kw)
    p.reset(x, 0)
    p.render(x).sum().backward()
    for t in p.params():
        assert t.grad is not None
        assert float(t.grad.abs().sum()) > 0.0


# ── 亮度係數的一致性 ─────────────────────────────────────────────────


def test_luma_weights_match_the_other_two_definitions(x):
    """三處的亮度係數必須相同：本模組的導引、銳度量、灰階淨化算子。

    不一致的話「防禦把能量放在哪個亮度上」與「淨化拿走哪個亮度」講的
    就不是同一件事，而兩者都不會報錯。
    """
    from src.metrics.acutance import _luma

    assert LUMA_WEIGHTS == (0.299, 0.587, 0.114)
    assert float((luma(x) - _luma(x)).abs().max()) < 1e-12
    gray = ops.Purifier("grayscale").evaluate(x.float())
    assert float((gray[:, 0:1] - luma(x).float()).abs().max()) < 1e-6


# ── 色彩類淨化算子 ───────────────────────────────────────────────────


def test_colour_purifiers_are_registered_and_not_geometric():
    for kind in ("grayscale", "gray_world", "auto_levels", "clahe"):
        assert kind in ops.KINDS
        assert kind not in ops.GEOMETRIC_KINDS
        assert ops.kind_of_label(kind) == kind


def test_grayscale_collapses_the_three_channels(x):
    y = ops.Purifier("grayscale").evaluate(x.float())
    assert float((y[:, 0] - y[:, 1]).abs().max()) == 0.0
    assert float((y[:, 0] - y[:, 2]).abs().max()) == 0.0


def test_gray_world_equalises_channel_means():
    torch.manual_seed(1)
    x = torch.rand(1, 3, 32, 32) * torch.tensor([1.0, 0.6, 0.3]).view(1, 3, 1, 1)
    y = ops.Purifier("gray_world").evaluate(x)
    m = y.mean(dim=(0, 2, 3))
    assert float((m - m.mean()).abs().max()) < 1e-3


def test_auto_levels_stretches_to_the_unit_interval():
    torch.manual_seed(2)
    x = torch.rand(1, 3, 64, 64) * 0.4 + 0.3
    y = ops.Purifier("auto_levels", 0.01).evaluate(x)
    assert float(y.min()) == 0.0
    assert float(y.max()) == 1.0


def test_clahe_runs_and_uses_straight_through():
    torch.manual_seed(3)
    x = torch.rand(1, 3, 64, 64)
    p = ops.Purifier("clahe", 2.0)
    assert p.differentiable is False
    assert float((p.evaluate(x) - x).abs().max()) > 0.0
    # 直通估計：前向等於真實實作，故 proxy_gap 為零。
    assert p.proxy_gap(x) == 0.0


def test_delta_e00_is_zero_for_identical_images_and_positive_otherwise():
    from src.metrics.suite import _delta_e00

    torch.manual_seed(4)
    a = torch.rand(1, 3, 32, 32)
    assert _delta_e00(a, a) == 0.0
    assert _delta_e00(a, (a * 0.9 + 0.05).clamp(0, 1)) > 0.0


# ── 共防禦參照的重建 ─────────────────────────────────────────────────


def _write_weights(tmp_path, image, cond, tensor):
    import torch as _t

    _t.save([tensor], tmp_path / f"{image}__{cond}__w.pt")


def test_codefense_identity_for_the_blank_floor(tmp_path, x):
    from src.defense.codefense import STATUS_IDENTITY, build_codefense

    fn, status = build_codefense({"condition": "none"}, tmp_path, "img",
                                 x.device, x.dtype)
    assert status == STATUS_IDENTITY
    assert float((fn(x) - x).abs().max()) == 0.0


def test_codefense_is_not_applicable_to_the_phase_family(tmp_path, x):
    """相位族的兩個閘由原圖自己的頻譜算出，套到另一張影像上不是同一個算子。

    這不是實作缺口，是那一族沒有這個物件；呼叫端要照實記，不得當成恆等。
    """
    from src.defense.codefense import STATUS_NOT_APPLICABLE, build_codefense

    fn, status = build_codefense({"condition": "phase_gain"}, tmp_path, "img",
                                 x.device, x.dtype)
    assert fn is None
    assert status == STATUS_NOT_APPLICABLE


def test_codefense_rebuilds_the_curve_exactly(tmp_path, x):
    """重建出來的 D 套在**另一張影像**上，與原參數逐位元相同。"""
    from src.defense.codefense import STATUS_EXACT, build_codefense

    p = ColorCurveParam(radius=2.0, pieces=32)
    p.reset(x, 0)
    with torch.no_grad():
        p.theta.copy_(torch.rand_like(p.theta) * 0.05 + 0.01)
    _write_weights(tmp_path, "img", "color_curve", p.theta.detach().cpu())

    row = {"condition": "color_curve", "radius": "2.0", "color_pieces": "32",
           "color_bound_mode": "symmetric", "defense_seed": "0"}
    fn, status = build_codefense(row, tmp_path, "img", x.device, x.dtype)
    assert status == STATUS_EXACT
    other = torch.rand_like(x)
    assert float((fn(other) - p.render(other)).abs().max()) == 0.0


def test_codefense_rebuilds_the_grid_exactly(tmp_path, x):
    from src.defense.codefense import build_codefense

    p = ColorGridParam(radius=0.15, grid=4, luma_bins=4)
    p.reset(x, 0)
    with torch.no_grad():
        p.a.add_(torch.randn_like(p.a) * 0.05)
    _write_weights(tmp_path, "img", "color_grid", p.a.detach().cpu())

    row = {"condition": "color_grid", "radius": "0.15", "color_grid": "4",
           "color_luma_bins": "4", "defense_seed": "0"}
    fn, _ = build_codefense(row, tmp_path, "img", x.device, x.dtype)
    other = torch.rand_like(x)
    assert float((fn(other) - p.render(other)).abs().max()) < 1e-12


def test_codefense_rebuilds_random_controls_from_the_seed(tmp_path, x):
    """隨機對照的 `params()` 是空的，存不到權重，只能由種子重抽。

    抽出來的張量形狀與影像尺寸無關，故拿小探針重建與拿原尺寸重建相同。
    """
    from src.defense.codefense import build_codefense

    p = ColorGridRandomParam(radius=0.1, grid=4, luma_bins=4)
    p.reset(x, 4242)
    row = {"condition": "color_grid_rand", "radius": "0.1", "color_grid": "4",
           "color_luma_bins": "4", "defense_seed": "4242"}
    fn, _ = build_codefense(row, tmp_path, "img", x.device, x.dtype)
    other = torch.rand_like(x)
    assert float((fn(other) - p.render(other)).abs().max()) < 1e-12


def test_codefense_refuses_a_random_row_without_the_seed(tmp_path, x):
    from src.defense.codefense import build_codefense

    row = {"condition": "color_grid_rand", "radius": "0.1"}
    with pytest.raises(ValueError, match="defense_seed"):
        build_codefense(row, tmp_path, "img", x.device, x.dtype)


def test_codefense_refuses_missing_weights(tmp_path, x):
    from src.defense.codefense import build_codefense

    row = {"condition": "color_curve", "radius": "1.0", "color_pieces": "64",
           "defense_seed": "0"}
    with pytest.raises(FileNotFoundError):
        build_codefense(row, tmp_path, "img", x.device, x.dtype)


def test_codefense_refuses_a_shape_mismatch(tmp_path, x):
    """構造設定與存檔時不同 ⇒ 載進去的是別的東西，寧可拋錯。"""
    from src.defense.codefense import build_codefense

    p = ColorCurveParam(radius=1.0, pieces=16)
    p.reset(x, 0)
    _write_weights(tmp_path, "img", "color_curve", p.theta.detach().cpu())
    row = {"condition": "color_curve", "radius": "1.0", "color_pieces": "64",
           "defense_seed": "0"}
    with pytest.raises(ValueError, match="形狀"):
        build_codefense(row, tmp_path, "img", x.device, x.dtype)


def test_codefense_reference_is_zero_when_nothing_is_applied(tmp_path):
    """`D = identity`、`p = identity` 時共防禦參照恰為 0。

    這是它與幾何類換參照同性質的地方：地板由構造為 0，而不是量出來的。
    影像取 fp32：`Purifier._run` 會先轉 fp32 再轉回，fp64 進來時那一趟不是
    恆等，而那與共防禦參照無關。
    """
    from src.defense.codefense import build_codefense

    fn, _ = build_codefense({"condition": "none"}, tmp_path, "img",
                            torch.device("cpu"), torch.float32)
    edit_orig = torch.rand(1, 3, 32, 32)
    ref = ops.Purifier("identity").evaluate(fn(edit_orig))
    assert float((ref - edit_orig).abs().max()) == 0.0


# ── 步長的尺度 ───────────────────────────────────────────────────────


def test_step_scale_defaults_to_radius_for_every_other_family():
    """沒有定義 `step_scale()` 的參數化照舊用 `radius`，軌跡逐位元不變。"""
    from src.defense.param_pgd import AdditiveParam, ShadingParam, step_scale_of

    assert step_scale_of(AdditiveParam(radius=0.05)) == 0.05
    assert step_scale_of(ShadingParam(radius=0.2)) == 0.2
    assert step_scale_of(ColorGridParam(radius=0.1)) == 0.1


def test_curve_step_scale_is_the_box_width_not_the_radius():
    """曲線族的半徑是斜率剖面的動態範圍，與 `θ` 的尺度差兩個數量級。

    K=64、r=3：盒是 [1/256, 4/64]，寬 0.0586；而 radius 是 3.0。用 radius
    當步長尺度的話 sign 更新五步就撞到邊界，之後只在兩個角落之間彈跳。
    """
    from src.defense.param_pgd import step_scale_of

    p = ColorCurveParam(radius=3.0, pieces=64)
    lo, hi = p.bounds()
    assert step_scale_of(p) == pytest.approx(hi - lo)
    assert step_scale_of(p) == pytest.approx(4.0 / 64 - 1.0 / 256)
    assert step_scale_of(p) < p.radius / 50

    p.set_radius(0.0)
    assert step_scale_of(p) == 0.0        # 恆等時步長為零

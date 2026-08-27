"""整併版：學出來的旋轉平面直接作用在量化後的整數係數上，交付即參數。

四件事釘住：`theta = 0` 的恆等對象是**壓縮圖**不是原圖、梯度通得過取整、
保長在取整前精確成立、以及診斷欄位真的量到它宣稱的東西。
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch

from src.baselines.jpeg_codec import jpeg_encode, jpeg_roundtrip
from src.defense.dct_unified import DctUnifiedParam, DctUnifiedRandomParam

ROOT = Path(__file__).resolve().parents[1]


def _img(seed: int = 0, size: int = 64):
    torch.manual_seed(seed)
    return torch.rand(1, 3, size, size)


def test_theta_zero_is_the_compressed_image_not_the_original():
    """**這一條是整份設計的分界線。** 交付本身就是壓縮圖，所以恆等的對象是
    `jpeg_roundtrip(x, qd)`；`texture_rephase` 的 `theta = 0` 才等於原圖。"""
    x = _img()
    p = DctUnifiedParam(radius=2.2, qd=0.85)
    p.reset(x, 0)
    out = p.render(x)
    assert torch.equal(out, jpeg_roundtrip(x, 0.85))
    assert not torch.equal(out, x)


def test_rotation_changes_the_image():
    x = _img(1)
    p = DctUnifiedParam(radius=2.2, qd=0.85)
    p.reset(x, 0)
    ref = p.render(x)
    with torch.no_grad():
        for q in p.params_.values():
            q["theta"].fill_(1.5)
    assert not torch.equal(p.render(x), ref)


def test_gradient_flows_through_the_rounding():
    """取整的導數幾乎處處為零；直通估計是這個參數化能被最佳化的前提。"""
    x = _img(2)
    p = DctUnifiedParam(radius=2.2, qd=0.85)
    p.reset(x, 0)
    with torch.no_grad():
        for q in p.params_.values():
            q["theta"].fill_(0.5)
    loss = p.render(x).pow(2).mean()
    grads = torch.autograd.grad(loss, p.params())
    assert all(torch.isfinite(g).all() for g in grads)
    assert any(float(g.norm()) > 0 for g in grads)


def test_rotation_preserves_length_before_rounding():
    """8×8 DCT 正交歸一、區塊不重疊，故保長在像素域也成立——**但那是取整
    之前的性質**，取整之後有誤差。這一條量的是取整前。"""
    from src.defense.dct_nonadditive import rotate_in_plane

    torch.manual_seed(3)
    c = torch.randn(2, 4, 4, 63)
    u = torch.randn(2, 4, 4, 63)
    v = torch.randn(2, 4, 4, 63)
    th = torch.full((2, 4, 4, 1), 1.3)
    out = rotate_in_plane(c, u, v, th)
    assert torch.allclose(out.norm(dim=-1), c.norm(dim=-1), atol=1e-4)


def test_delta_within_one_is_measured_not_assumed():
    """`delta_within_1` 是新穎性主張的邊界讀數，必須真的量到位移的大小。"""
    x = _img(4)
    p = DctUnifiedParam(radius=3.14, qd=0.85)
    p.reset(x, 0)
    with torch.no_grad():
        for q in p.params_.values():
            q["theta"].fill_(0.0)
    p.render(x)
    small = p.delta_stats()
    assert small["delta_within_1"] == 1.0        # theta=0：位移恆為 0
    assert small["delta_nonzero"] == 0.0
    with torch.no_grad():
        for q in p.params_.values():
            q["theta"].fill_(3.14)
    p.render(x)
    big = p.delta_stats()
    assert big["delta_nonzero"] > small["delta_nonzero"]
    assert big["delta_within_1"] < 1.0


def test_zero_coefficient_fraction_matches_the_encoder():
    """`zero_coef_frac` 要與 `jpeg_encode` 的實際輸出一致，不是估的。"""
    x = _img(5)
    p = DctUnifiedParam(radius=2.0, qd=0.85)
    p.reset(x, 0)
    p.render(x)
    stats = p.delta_stats()
    rows = [u for u, _ in p.idx]
    cols = [v for _, v in p.idx]
    a = jpeg_encode(x, 0.85)
    zeros = tot = 0
    for name in ("Y", "Cb", "Cr"):
        sub = a[name][..., rows, cols]
        zeros += int((sub == 0).sum()); tot += sub.numel()
    assert stats["zero_coef_frac"] == pytest.approx(zeros / tot, abs=1e-5)


def test_project_clamps_the_angle_only():
    x = _img(6)
    p = DctUnifiedParam(radius=1.0, qd=0.85)
    p.reset(x, 0)
    with torch.no_grad():
        for q in p.params_.values():
            q["theta"].fill_(5.0)
            q["u"].fill_(9.0)
    p.project()
    for q in p.params_.values():
        assert float(q["theta"].detach().max()) == pytest.approx(1.0)
        assert float(q["u"].detach().max()) == pytest.approx(9.0)   # 平面不是預算


def test_random_variant_has_no_trainable_parameters():
    x = _img(7)
    r = DctUnifiedRandomParam(radius=2.2, qd=0.85)
    r.reset(x, 0)
    assert r.params() == []
    assert not torch.equal(r.render(x), jpeg_roundtrip(x, 0.85))


def test_unknown_channel_and_gate_are_rejected():
    with pytest.raises(ValueError, match="未知通道"):
        DctUnifiedParam(channels=("Y", "Alpha"))
    with pytest.raises(ValueError, match="未知的閘"):
        DctUnifiedParam(gate="sharpen")


def test_builder_returns_the_unified_param():
    spec = importlib.util.spec_from_file_location(
        "phase_ablation_unified", ROOT / "scripts" / "phase_ablation.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    param, lo, hi = mod.build("dct_unified", 0, dct_qd=0.85)
    assert isinstance(param, DctUnifiedParam)
    assert 0 < lo < hi
    rnd, _, _ = mod.build("dct_unified_rand", 0, dct_qd=0.85)
    assert isinstance(rnd, DctUnifiedRandomParam)


def test_deliver_jpeg_is_refused_on_the_unified_condition():
    """它自己就交付壓縮圖；再疊一次 `--deliver-jpeg` 是壓兩次。"""
    spec = importlib.util.spec_from_file_location(
        "ip2p_run_unified", ROOT / "scripts" / "ip2p_run.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    args = mod.build_parser().parse_args(
        ["--out", "o", "--data", "d", "--radius", "2.2", "--deliver-jpeg", "0.85"])
    with pytest.raises(SystemExit, match="交付品質請用 --dct-qd"):
        mod.defend(None, None, "dct_unified", _img(), args, lambda z: z.mean())

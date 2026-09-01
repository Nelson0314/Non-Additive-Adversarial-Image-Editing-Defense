"""頻譜加性下限：在乘法之外多一個加法項。

存在理由
────────────────────────────────────────────────────────────────────
相位旋轉乘的是模長 1 的複數，幅度增益乘的是 `exp(g)`。**兩者都是乘法**，
而平坦區的 `|spec| ≈ 0`——乘任何東西還是接近零，強度開再大也動不了。
13 張裡失敗的那幾張全是大面積平滑主體（瓶子在純黑底上只花得出 DISTS 0.085）。

加性下限在頻譜上加一項，值由 JPEG 亮度量化表定價。**它只乘徑向帶通，不乘
紋理閘**——紋理閘在平坦區就是零，乘了它這一項等於沒加。

代價：方法不再是純粹的非加性重參數化。兩個設定都是主線，分開報。
"""

import math

import pytest
import torch

from src.residual.perceptual_weight import freq_weight
from src.residual.texture_rephase import PhaseResidual, radial_gate

BLOCK = 32
DT = torch.float64
DEV = torch.device("cpu")


def _module(**kw):
    m = PhaseResidual(size=128, block=BLOCK, hop=8, r_min=0.12, theta_max=1.0,
                      gain_max=1.0, energy_quantile=0.0, **kw).to(DT)
    torch.manual_seed(0)
    m.prepare_gates(torch.rand(1, 3, 128, 128, dtype=DT))
    return m


def test_default_is_off_and_bit_identical():
    """預設 0：不建參數、前向逐位元等於加這個選項之前。"""
    torch.manual_seed(1)
    x = torch.rand(1, 3, 128, 128, dtype=DT)
    a, b = _module(), _module(spectral_floor=0.0)
    assert a.spectral_floor == 0.0
    for m in (a, b):
        m.prepare_gates(x)
        with torch.no_grad():
            m.theta.fill_(0.4)
            m.gain.fill_(0.2)
    with torch.no_grad():
        assert torch.equal(a.pixel_residual(x), b.pixel_residual(x))


def test_identity_still_holds_when_every_parameter_is_zero():
    """theta = gain = floor = 0 時輸出仍逐位元等於原圖。這是本模組唯一的
    構造保證，加法項不得破壞它。"""
    torch.manual_seed(2)
    x = torch.rand(1, 3, 128, 128, dtype=DT)
    m = _module(spectral_floor=0.05)
    m.prepare_gates(x)
    with torch.no_grad():
        m.theta.zero_(); m.gain.zero_(); m.floor.zero_()
        out = m.pixel_residual(x)
    assert float((out - x).abs().max()) < 1e-12


def test_floor_price_skips_the_texture_gate():
    """加法項只乘徑向帶通與知覺權重。乘了紋理閘就進不去平坦區，
    這一項也就沒有存在的理由。"""
    m = _module(spectral_floor=0.05)
    want = (radial_gate(BLOCK, 0.12, DEV, DT)
            * freq_weight("jpeg_luma", BLOCK, DEV, DT))
    assert torch.allclose(m.floor_price(), want)
    # 紋理閘確實不是全 1，否則這條測試沒有鑑別力
    assert float(m.tex_gate.min()) < 0.99


def _flat_with_sensor_noise():
    """幾乎平坦但不是精確的零——真實照片的暗部長這樣。"""
    torch.manual_seed(7)
    return (torch.full((1, 3, 128, 128), 0.5, dtype=DT)
            + torch.randn(1, 3, 128, 128, dtype=DT) * 1e-3)


def test_floor_moves_a_flat_region_that_multiplication_cannot():
    """幾乎平坦的區域：乘法幾乎動不了，加法動得了。這是這一項存在的理由。"""
    x = _flat_with_sensor_noise()
    out = {}
    for fl in (0.0, 0.05):
        m = PhaseResidual(size=128, block=BLOCK, hop=8, r_min=0.12,
                          theta_max=1.0, gain_max=1.0, energy_quantile=0.0,
                          spectral_floor=fl).to(DT)
        m.prepare_gates(x)
        with torch.no_grad():
            m.theta.fill_(1.0)
            m.gain.fill_(1.0)
            if fl > 0:
                m.floor.fill_(1.0)
            out[fl] = float((m.pixel_residual(x) - x).abs().max())
    assert out[0.05] > 10 * out[0.0]


def test_a_uniform_image_is_the_construction_boundary():
    """精確的純色塊上加法項幾乎動不了：殘差是 1e-6 的量級。

    原因是它加上去的量沿用原係數的相位並正比於 `|rot|` 的方向，而純色塊
    加窗之後的能量幾乎全落在 `r_min` 以下、被徑向帶通擋掉。**這是構造上的
    邊界，不是缺陷**——真實照片的暗部有感測器雜訊，通帶內的係數小但非零，
    上一條測試量的就是那個情形。釘住它是為了不讓人以為這一項無所不能。
    """
    x = torch.full((1, 3, 128, 128), 0.5, dtype=DT)
    m = PhaseResidual(size=128, block=BLOCK, hop=8, r_min=0.12, theta_max=1.0,
                      gain_max=1.0, energy_quantile=0.0,
                      spectral_floor=0.05).to(DT)
    m.prepare_gates(x)
    with torch.no_grad():
        m.theta.fill_(1.0); m.gain.fill_(1.0); m.floor.fill_(1.0)
        moved = float((m.pixel_residual(x) - x).abs().max())
    assert moved < 1e-4


def test_output_stays_real():
    """rfft2 的共軛對稱由 `radial_gate` 歸零的兩行保證，加法項乘了它
    所以不會破壞。輸出必須是實數張量而非複數。"""
    torch.manual_seed(3)
    x = torch.rand(1, 3, 128, 128, dtype=DT)
    m = _module(spectral_floor=0.05)
    m.prepare_gates(x)
    with torch.no_grad():
        m.theta.uniform_(-1, 1); m.gain.uniform_(-1, 1); m.floor.uniform_(-1, 1)
        out = m.pixel_residual(x)
    assert not out.is_complex()
    assert torch.isfinite(out).all()


def test_negative_floor_raises():
    with pytest.raises(ValueError, match="spectral_floor"):
        PhaseResidual(size=128, block=BLOCK, spectral_floor=-0.1)


def test_param_and_cli_carry_it_through():
    import sys
    from pathlib import Path

    from src.defense.param_pgd import PhaseParam

    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "scripts"))
    import ip2p_run

    src = (root / "scripts" / "ip2p_run.py").read_text(encoding="utf-8")
    assert '"spectral_floor":' in src
    ns = ip2p_run.build_parser().parse_args(["--out", "x"])
    assert ns.spectral_floor == 0.0

    x = torch.rand(1, 3, 128, 128, dtype=DT)
    p = PhaseParam(size=128, block=BLOCK, hop=8, r_min=0.12, radius=1.0,
                   gain_ratio=1.0, spectral_floor=0.04)
    p.reset(x, seed=0)
    assert p.module.spectral_floor == 0.04
    # 三組參數都要進最佳化，漏掉任何一組都不會報錯，只會安靜地不學
    assert len(p.params()) == 3
    with torch.no_grad():
        p.module.floor.fill_(5.0)
    p.project()
    assert float(p.module.floor.max()) == 1.0

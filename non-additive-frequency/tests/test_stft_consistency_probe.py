"""探針重建的 `rot` 必須與 `_rephase` 走同一條路。

這一支存在的唯一理由：`requested_spectrum` 是把 `PhaseResidual._rephase` 的
前半段抄出來的。抄出來的東西會在模組改動時**悄悄分岔**——不會拋錯，只會讓
量到的偏差是另一個東西的偏差。所以釘住「`synthesize(requested_spectrum(x))`
與 `render(x)` 逐位相同」。

`gl_iters` 與 `pixel_mask` 是 `render` 在 `synthesize` **之後**才做的事，兩者
都關著時才有這個等式；打開時本探針量的是它們之前的那一步，那是刻意的——
投影誤差就發生在 `synthesize` 那一步。
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from stft_consistency_probe import consistency_deviation
from src.residual.texture_rephase import PhaseResidual


def _module(**kw):
    m = PhaseResidual(size=64, block=16, hop=4, **kw)
    return m


def _fill(m, x, scale=0.3):
    m.prepare_gates(x)
    with torch.no_grad():
        m.theta.normal_(0.0, scale)
        if m.gain_max > 0:
            m.gain.normal_(0.0, 0.1)
    return m


def test_合成要求的頻譜等於_render():
    torch.manual_seed(0)
    x = torch.rand(1, 3, 64, 64)
    m = _fill(_module(), x)
    got = m.synthesize(m.requested_spectrum(x))
    assert torch.allclose(got, m._rephase(x), atol=1e-6)


def test_增益開著時仍然等於_render():
    """`--gain-ratio > 0` 是本專案批次的實際設定，這一格必須成立。"""
    torch.manual_seed(1)
    x = torch.rand(1, 3, 64, 64)
    m = _fill(_module(gain_max=0.2), x)
    got = m.synthesize(m.requested_spectrum(x))
    assert torch.allclose(got, m._rephase(x), atol=1e-6)


def test_theta_為零時偏差為零():
    """沒有要求任何旋轉時，要求的頻譜就是原圖的，投影誤差必須是 0。"""
    torch.manual_seed(2)
    x = torch.rand(1, 3, 64, 64)
    m = _module()
    m.prepare_gates(x)
    with torch.no_grad():
        m.theta.zero_()
    d = consistency_deviation(m, x)
    assert d["amp_dev"] < 1e-4
    assert d["phase_rho"] > 0.9999


def test_旋轉越大偏差越大():
    """偏差要真的隨「要求得多離譜」單調增加，否則它量的不是投影誤差。"""
    torch.manual_seed(3)
    x = torch.rand(1, 3, 64, 64)
    devs = []
    for s in (0.05, 0.3, 1.0):
        m = _fill(_module(), x, scale=s)
        devs.append(consistency_deviation(m, x)["amp_dev"])
    assert devs[0] < devs[1] < devs[2]


def test_不與模組自己的_amplitude_deviation_混用():
    """增益開著時模組那一支量的是別的東西，兩者不可互相取代。"""
    torch.manual_seed(4)
    x = torch.rand(1, 3, 64, 64)
    m = _fill(_module(gain_max=0.4), x)
    mine = consistency_deviation(m, x)["amp_dev"]
    theirs = m.amplitude_deviation(x)
    assert abs(mine - theirs) > 1e-3

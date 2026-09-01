"""增益的閘可以與相位的閘分開。

存在理由
────────────────────────────────────────────────────────────────────
`PhaseResidual` 至今對相位與增益套**同一個閘**，理由寫在前向那一段：讓兩者
被允許出現的位置一致，比較才是乾淨的「動什麼」而不是「動哪裡」。歸因做完
之後那個約束就沒有必要了，而 `runs/encoder_frequency_response` 指出分開是
有理由的。

該診斷逐帶量到（theta = 1 探針、latent_norm 損失、6 張）：

    半徑帶        DISTS 代價   latent 位移   位移／DISTS
    0.00-0.18      0.2451        80.8            330
    0.18-0.35      0.0278        32.9           1181
    0.35-0.53      0.0059        18.1           3072
    0.53-0.71      0.00044       10.8          24772
    0.71-0.88      0.00005        6.7         134544

高頻便宜到近乎免費，但**位移也近乎為零**——自然影像的功率譜按 1/f² 掉，
那裡沒有能量可以旋轉（`METHOD.md` 的構造限制第一條）。相位在那裡無事可做，
**增益不然**：`exp(g)·|spec|` 在 |spec| 微小但非零處造得出容量，而代價落在
人眼看不見的頻帶。

於是：相位保留完整的二值帶通（用中頻那些真的有能量的格），增益改用知覺
權重（把振幅的創造推到看不見的地方）。這不是加性項——|spec| 為零處乘任何
東西仍是零。
"""

import math

import pytest
import torch

from src.residual.perceptual_weight import freq_weight
from src.residual.texture_rephase import PhaseResidual, radial_gate

BLOCK = 32
DEV, DT = torch.device("cpu"), torch.float64


def _module(**kw):
    m = PhaseResidual(size=128, block=BLOCK, r_min=0.12, theta_max=1.0,
                      gain_max=1.0, **kw).to(DT)
    torch.manual_seed(0)
    m.prepare_gates(torch.rand(1, 3, 128, 128, dtype=DT))
    return m


def test_shared_is_the_default_and_is_bit_identical():
    """預設值必須逐位元等於加這個選項之前：相位與增益同一個閘。"""
    m = _module()
    assert m.gain_weight == "shared"
    assert torch.equal(m.gain_gate(), m.gate())


def test_jnd_gain_gate_is_the_phase_gate_times_the_perceptual_weight():
    m = _module(gain_weight="jnd")
    w = freq_weight("jpeg_luma", BLOCK, DEV, DT)
    assert torch.allclose(m.gain_gate(), m.gate() * w)
    # 相位那一側不受影響
    assert torch.allclose(m.freq_gate,
                          radial_gate(BLOCK, 0.12, DEV, DT))


def test_jnd_moves_the_gain_budget_towards_high_frequency():
    """增益的預算重心必須往高頻移，否則這個改動沒有做到它要做的事。"""
    fy = torch.fft.fftfreq(BLOCK, dtype=DT) * 2.0
    fx = torch.fft.rfftfreq(BLOCK, dtype=DT) * 2.0
    r = torch.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)

    def centroid(gate):
        g = gate.sum(dim=1).squeeze(0) if gate.dim() > 2 else gate
        return float((g * r).sum() / g.sum())

    shared = _module()
    jnd = _module(gain_weight="jnd")
    assert centroid(jnd.gain_gate()) > centroid(shared.gain_gate())


def test_unknown_gain_weight_raises():
    with pytest.raises(ValueError, match="gain_weight"):
        _module(gain_weight="csf")


def test_identity_at_zero_parameters_survives_the_split_gate():
    """theta = 0 且 gain = 0 時輸出仍逐位元等於原圖。"""
    m = _module(gain_weight="jnd")
    torch.manual_seed(1)
    x = torch.rand(1, 3, 128, 128, dtype=DT)
    m.prepare_gates(x)
    with torch.no_grad():
        m.theta.zero_(); m.gain.zero_()
        out = m.pixel_residual(x)
    assert float((out - x).abs().max()) < 1e-12


def test_gain_weight_reaches_the_cli_and_the_csv():
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "scripts"))
    import ip2p_run

    src = (root / "scripts" / "ip2p_run.py").read_text(encoding="utf-8")
    assert '"gain_weight":' in src
    ns = ip2p_run.build_parser().parse_args(["--out", "x"])
    assert ns.gain_weight == "shared"
    ns2 = ip2p_run.build_parser().parse_args(
        ["--out", "x", "--gain-weight", "jnd"])
    assert ns2.gain_weight == "jnd"

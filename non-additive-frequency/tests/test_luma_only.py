"""只在亮度通道上重相位。

存在理由
────────────────────────────────────────────────────────────────────
本算子至今對 R、G、B 三個通道各自做同一件事，於是增益在色度上累積成**全域
色偏**——13 張的目視稽核裡，帶內最好的那個工作點（gamma 0.25、DISTS 0.142）
輸出整片洋紅，原圖仍認得出來。使用者的判準把「單純劣化」排除在成功之外，
而色偏正屬於那一類；`DECISIONS.md` 也記載位移在裁切縮放那一格會把均勻色偏
算成防禦。

`RESULTS.md` 的 DCT-Shield 重現記著「真正把失真砍半的是**只動 Y 通道**」
（LPIPS 0.4578 → 0.3918），而該篇的 Y-only 變體正是它在本專案失真帶內最好
的一格（11/13、PSNR 24.26）。本方法從未有 Y-only 版本。

**色彩轉換用精確反矩陣，不用 JFIF 的公布常數。** 後者正逆各自四捨五入到
小數第六位，往返只到 1.2e-6，那會破壞「theta = 0 時輸出逐位元等於原圖」這條
構造保證。本模組不是在模擬 libjpeg，沒有理由繼承那個誤差。
"""

import pytest
import torch

from src.residual.texture_rephase import PhaseResidual, luma_join, luma_split

DT = torch.float64


def test_luma_round_trip_is_exact():
    """精確互逆。JFIF 的公布常數只到 1.2e-6，那個誤差會吃掉恆等保證。"""
    torch.manual_seed(0)
    x = torch.rand(2, 3, 16, 16, dtype=DT)
    y, chroma = luma_split(x)
    assert y.shape == (2, 1, 16, 16) and chroma.shape == (2, 2, 16, 16)
    assert float((luma_join(y, chroma) - x).abs().max()) < 1e-13


def test_luma_is_the_bt601_weighted_sum():
    x = torch.zeros(1, 3, 2, 2, dtype=DT)
    x[:, 0] = 1.0
    y, _ = luma_split(x)
    assert float(y.mean()) == pytest.approx(0.299, abs=1e-9)


def test_channels_y_leaves_chroma_untouched():
    """只動 Y：防禦圖的 Cb、Cr 必須與原圖逐位元相同，否則色偏又回來了。"""
    torch.manual_seed(0)
    x = torch.rand(1, 3, 128, 128, dtype=DT)
    m = PhaseResidual(size=128, block=32, r_min=0.12, theta_max=1.0,
                      gain_max=1.0, channels="y").to(DT)
    m.prepare_gates(x)
    with torch.no_grad():
        m.theta.uniform_(-1.0, 1.0)
        m.gain.uniform_(-1.0, 1.0)
        out = m.pixel_residual(x)
    _, c_in = luma_split(x)
    _, c_out = luma_split(out)
    assert float((c_out - c_in).abs().max()) < 1e-12
    # 亮度確實被動到了，否則這個測試會在算子壞掉時仍然通過
    y_in, _ = luma_split(x)
    y_out, _ = luma_split(out)
    assert float((y_out - y_in).abs().max()) > 1e-3


def test_identity_at_zero_survives_the_luma_path():
    torch.manual_seed(0)
    x = torch.rand(1, 3, 128, 128, dtype=DT)
    m = PhaseResidual(size=128, block=32, r_min=0.12, theta_max=1.0,
                      channels="y").to(DT)
    m.prepare_gates(x)
    with torch.no_grad():
        m.theta.zero_()
        out = m.pixel_residual(x)
    assert float((out - x).abs().max()) < 1e-12


def test_rgb_remains_the_default_and_unchanged():
    torch.manual_seed(0)
    x = torch.rand(1, 3, 128, 128, dtype=DT)
    a = PhaseResidual(size=128, block=32, r_min=0.12, theta_max=1.0).to(DT)
    b = PhaseResidual(size=128, block=32, r_min=0.12, theta_max=1.0,
                      channels="rgb").to(DT)
    assert a.channels == "rgb"
    for m in (a, b):
        m.prepare_gates(x)
        with torch.no_grad():
            m.theta.fill_(0.5)
    with torch.no_grad():
        assert torch.equal(a.pixel_residual(x), b.pixel_residual(x))


def test_unknown_channels_raises():
    with pytest.raises(ValueError, match="channels"):
        PhaseResidual(size=128, block=32, channels="yuv")


def test_channels_reaches_the_cli_and_the_csv():
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "scripts"))
    import ip2p_run

    src = (root / "scripts" / "ip2p_run.py").read_text(encoding="utf-8")
    assert '"phase_channels":' in src
    ns = ip2p_run.build_parser().parse_args(["--out", "x"])
    assert ns.phase_channels == "rgb"
    ns2 = ip2p_run.build_parser().parse_args(
        ["--out", "x", "--phase-channels", "y"])
    assert ns2.phase_channels == "y"

"""重疊步長 `hop`：擾動的紋理有多粗由它決定。

存在理由
────────────────────────────────────────────────────────────────────
`hop` 自始固定在 `block // 2`，從未被掃過——`PhaseResidual` 收這個參數，
但 `PhaseParam` 不轉交，CLI 也沒有旗標，於是它只活在預設值裡。

使用者看防禦圖時指出擾動的紋理偏粗。相鄰區塊的相位各自獨立旋轉，重疊越少
接縫越明顯（`METHOD.md` 構造限制第三條「重疊相加會部分抵銷」的同一個成因）。
`hop = block // 4` 讓每個像素被四個區塊覆蓋而不是兩個，平均之後應該更平滑。

NOLA 條件（`OLA(w²) > 0`）在 hop 更小時只會更寬鬆，故 `theta = 0` 的逐位元
恆等不受影響——這一條由測試釘住，因為它是本模組唯一的構造保證。
"""

import math

import pytest
import torch

from src.residual.texture_rephase import PhaseResidual

DT = torch.float64


def _module(hop):
    return PhaseResidual(size=128, block=32, hop=hop, r_min=0.12,
                         theta_max=1.0).to(DT)


@pytest.mark.parametrize("hop", [16, 8, 4])
def test_identity_at_zero_theta_holds_for_every_hop(hop):
    torch.manual_seed(0)
    x = torch.rand(1, 3, 128, 128, dtype=DT)
    m = _module(hop)
    m.prepare_gates(x)
    with torch.no_grad():
        m.theta.zero_()
        out = m.pixel_residual(x)
    assert float((out - x).abs().max()) < 1e-12


def test_smaller_hop_gives_more_blocks():
    """覆蓋每個像素的區塊變多，是「平滑」這個效果的直接來源。"""
    assert _module(8).n_blocks > _module(16).n_blocks


def test_smaller_hop_smooths_the_residual():
    """同一組 theta 下，hop 越小殘差的區塊尺度結構越弱。

    量的是殘差在 block 網格上的**分段常數程度**：把殘差按 32 像素切格取
    平均，再看該平均能解釋多少總變異。接縫明顯時這個比例高。
    """
    torch.manual_seed(0)
    x = torch.rand(1, 3, 128, 128, dtype=DT)
    frac = {}
    for hop in (16, 4):
        m = _module(hop)
        m.prepare_gates(x)
        with torch.no_grad():
            g = torch.Generator().manual_seed(0)
            m.theta.copy_(torch.randn(m.theta.shape, generator=g)
                          .clamp(-1, 1).to(DT))
            r = m.pixel_residual(x) - x
        cell = torch.nn.functional.avg_pool2d(r, 32)
        blocky = float(cell.var(unbiased=False) * (32 * 32))
        frac[hop] = blocky / float(r.var(unbiased=False))
    assert frac[4] < frac[16]


def test_hop_reaches_the_cli_and_the_csv():
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "scripts"))
    import ip2p_run

    src = (root / "scripts" / "ip2p_run.py").read_text(encoding="utf-8")
    # hop 此前是由 block 推導寫進 CSV 的常數，改成逐列記下實際用的值
    assert 'args.block // 2 if args.hop is None else args.hop' in src
    ns = ip2p_run.build_parser().parse_args(["--out", "x"])
    assert ns.hop is None
    ns2 = ip2p_run.build_parser().parse_args(["--out", "x", "--hop", "8"])
    assert ns2.hop == 8


def test_hop_larger_than_block_still_raises():
    with pytest.raises(ValueError, match="hop"):
        PhaseResidual(size=128, block=32, hop=64)

"""知覺加權的頻率閘：把「通／不通」換成「這一格值多少錢」。

存在理由
────────────────────────────────────────────────────────────────────
`radial_gate` 是**二值**的——歸一化半徑 0.15 的頻格與 0.9 的頻格拿到同一個
`theta_max`，但人眼對前者的對比敏感度高一個數量級。實測「每單位 DISTS 換到
多少位移」：DCT-Shield 13.2、本方法 3.7–6.6，而 RESULTS 已把那 2 倍歸因給
JPEG 量化階的約束，並註明「那不是加性本身帶來的」。量化表就是一張對比敏感度
的價目表，故此處把同一個約束搬進本方法的閘。

**這不改參數化**：仍是頻譜重參數化、非加性，`theta = 0` 的恆等仍成立。改的
只有「擾動被允許出現在哪裡」，`DECISIONS.md` 的「頻譜加性項不做」明文寫
「閘的開度不受此限」。
"""

import math

import pytest
import torch

from src.residual.perceptual_weight import (
    FREQ_WEIGHTS,
    JPEG_LUMA_TABLE,
    freq_weight,
)
from src.residual.texture_rephase import PhaseResidual, radial_gate


BLOCK = 32


def test_jpeg_table_is_the_standard_annex_k_luminance_table():
    """ITU-T T.81 Annex K 表 K.1。抄錯一個數不會有任何症狀，故釘住。"""
    assert JPEG_LUMA_TABLE.shape == (8, 8)
    assert JPEG_LUMA_TABLE[0].tolist() == [16, 11, 10, 16, 24, 40, 51, 61]
    assert JPEG_LUMA_TABLE[7].tolist() == [72, 92, 95, 98, 112, 100, 103, 99]
    assert int(JPEG_LUMA_TABLE.max()) == 121


def test_binary_weight_is_bit_identical_to_all_ones():
    """預設值必須逐位元等於加這個選項之前的行為。"""
    w = freq_weight("binary", BLOCK, torch.device("cpu"), torch.float64)
    assert w.shape == (BLOCK, BLOCK // 2 + 1)
    assert torch.equal(w, torch.ones_like(w))


def test_jpeg_luma_weight_is_normalised_to_unit_maximum():
    w = freq_weight("jpeg_luma", BLOCK, torch.device("cpu"), torch.float64)
    assert w.shape == (BLOCK, BLOCK // 2 + 1)
    assert float(w.max()) == pytest.approx(1.0)
    assert float(w.min()) > 0.0


def test_jpeg_luma_prices_low_frequency_higher_than_high_frequency():
    """人眼對低頻敏感 ⇒ 量化階小 ⇒ 允許的擾動小。

    比的是 `fy = 0` 那一列上的兩格：`fx` 索引 2（歸一化半徑 0.125）對
    索引 12（0.75）。方向搞反的話這整個改動會把預算推到最看得見的地方。
    """
    w = freq_weight("jpeg_luma", BLOCK, torch.device("cpu"), torch.float64)
    assert float(w[0, 2]) < float(w[0, 12])


def test_unknown_weight_name_raises_instead_of_falling_back():
    with pytest.raises(ValueError, match="freq_weight"):
        freq_weight("csf_wilson", BLOCK, torch.device("cpu"), torch.float64)
    assert "binary" in FREQ_WEIGHTS and "jpeg_luma" in FREQ_WEIGHTS


def test_gate_is_the_band_pass_times_the_weight():
    """閘 = 帶通遮罩 × 知覺權重。權重不得讓通帶外的格復活。"""
    x = torch.rand(1, 3, 128, 128, dtype=torch.float64)
    m = PhaseResidual(size=128, block=BLOCK, r_min=0.25,
                      freq_weight="jpeg_luma").to(torch.float64)
    m.prepare_gates(x)
    band = radial_gate(BLOCK, 0.25, torch.device("cpu"), torch.float64)
    w = freq_weight("jpeg_luma", BLOCK, torch.device("cpu"), torch.float64)
    assert torch.allclose(m.freq_gate, band * w)
    assert float(m.freq_gate[band == 0].abs().max()) == 0.0


def test_identity_at_zero_theta_survives_the_weight():
    """`theta = 0` 時輸出逐位元等於原圖，這是構造保證的性質，權重不得破壞。"""
    x = torch.rand(1, 3, 128, 128, dtype=torch.float64)
    m = PhaseResidual(size=128, block=BLOCK, r_min=0.12,
                      freq_weight="jpeg_luma").to(torch.float64)
    m.prepare_gates(x)
    with torch.no_grad():
        m.theta.zero_()
        out = m.pixel_residual(x)
    assert float((out - x).abs().max()) < 1e-12


def test_weight_shrinks_the_reachable_distortion_at_equal_theta():
    """同一個 theta 下，加權之後動到的能量必然不高於二值閘。

    這條釘住的是「加權是收緊而不是放寬」——若它反而放大了失真，後續所有
    等失真比較的解讀都會反過來。
    """
    torch.manual_seed(0)
    x = torch.rand(1, 3, 128, 128, dtype=torch.float64)
    out = {}
    for name in ("binary", "jpeg_luma"):
        m = PhaseResidual(size=128, block=BLOCK, r_min=0.12, theta_max=1.0,
                          freq_weight=name).to(torch.float64)
        m.prepare_gates(x)
        with torch.no_grad():
            m.theta.fill_(1.0)
            out[name] = float((m.pixel_residual(x) - x).pow(2).mean())
    assert out["jpeg_luma"] < out["binary"]


def test_phase_param_passes_the_weight_through_to_the_module():
    """`PhaseParam` 是最佳化迴圈看到的介面，選項在那裡斷掉不會有症狀——
    整批會安靜地跑成基準的重複，而 CSV 的 `freq_weight` 欄仍寫著新名字。"""
    from src.defense.param_pgd import PhaseParam

    x = torch.rand(1, 3, 128, 128, dtype=torch.float64)
    p = PhaseParam(size=128, block=BLOCK, r_min=0.12, radius=1.0,
                   freq_weight="jpeg_luma")
    p.reset(x, seed=0)
    assert p.module.freq_weight == "jpeg_luma"
    band = radial_gate(BLOCK, 0.12, torch.device("cpu"), torch.float64)
    w = freq_weight("jpeg_luma", BLOCK, torch.device("cpu"), torch.float64)
    assert torch.allclose(p.module.freq_gate.double(), band * w)


def test_ablation_builder_accepts_the_weight_for_every_phase_condition():
    import sys
    from pathlib import Path as _Path

    sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "scripts"))
    from phase_ablation import build

    for name in ("phase", "phase_rand", "phase_gain", "gain_only"):
        param, _lo, _hi = build(name, seed=0, gain_ratio=1.0,
                                freq_weight="jpeg_luma")
        assert param.freq_weight == "jpeg_luma", name


def test_freq_weight_is_a_csv_column_and_a_cli_flag():
    import sys
    from pathlib import Path as _Path

    root = _Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "scripts"))
    import ip2p_run

    src = (root / "scripts" / "ip2p_run.py").read_text(encoding="utf-8")
    assert '"freq_weight":' in src
    ns = ip2p_run.build_parser().parse_args(["--out", "x"])
    assert ns.freq_weight == "binary"
    ns2 = ip2p_run.build_parser().parse_args(
        ["--out", "x", "--freq-weight", "jpeg_luma"])
    assert ns2.freq_weight == "jpeg_luma"


def test_weight_power_interpolates_between_binary_and_full_weighting():
    """`w ** gamma`：gamma = 0 逐位元等於二值閘，gamma = 1 是完整的量化表定價。

    存在理由：兩端都不是操作點。二值閘的位移／DISTS 只有 3.3–4.3；完整加權
    把它拉到 8–14.5，但通帶有效容量掉到 0.544，要摸到會擋下的強度就得把半徑
    推到 theta 封頂（pi）之外，那之後只有**增益**在長，PSNR 直接被打掉
    （radius 8 時只剩 18.97）。中間值讓兩者可以取捨。
    """
    dev, dt = torch.device("cpu"), torch.float64
    full = freq_weight("jpeg_luma", BLOCK, dev, dt)
    assert torch.equal(freq_weight("jpeg_luma", BLOCK, dev, dt, power=0.0),
                       torch.ones_like(full))
    assert torch.equal(freq_weight("jpeg_luma", BLOCK, dev, dt, power=1.0), full)
    half = freq_weight("jpeg_luma", BLOCK, dev, dt, power=0.5)
    assert torch.allclose(half, full ** 0.5)
    # 單調：gamma 越大，定價越低（放行越少）
    assert float(full.mean()) < float(half.mean()) < 1.0


def test_negative_weight_power_raises():
    with pytest.raises(ValueError, match="power"):
        freq_weight("jpeg_luma", BLOCK, torch.device("cpu"), torch.float64,
                    power=-1.0)


def test_weight_power_reaches_the_module_and_the_cli():
    import sys
    from pathlib import Path as _Path

    root = _Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "scripts"))
    import ip2p_run

    src = (root / "scripts" / "ip2p_run.py").read_text(encoding="utf-8")
    assert '"freq_weight_power":' in src
    ns = ip2p_run.build_parser().parse_args(["--out", "x"])
    assert ns.freq_weight_power == 1.0

    x = torch.rand(1, 3, 128, 128, dtype=torch.float64)
    m = PhaseResidual(size=128, block=BLOCK, r_min=0.12,
                      freq_weight="jpeg_luma",
                      freq_weight_power=0.0).to(torch.float64)
    m.prepare_gates(x)
    band = radial_gate(BLOCK, 0.12, torch.device("cpu"), torch.float64)
    assert torch.allclose(m.freq_gate, band)

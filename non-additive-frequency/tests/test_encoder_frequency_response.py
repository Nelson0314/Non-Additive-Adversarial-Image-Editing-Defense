"""`scripts/encoder_frequency_response.py` 的分帶邏輯與欄位。

存在理由
────────────────────────────────────────────────────────────────────
知覺加權閘（`freq_weight=jpeg_luma`）把預算推到人眼看不見的高頻，效率因此
從 3.3–4.3 拉到 18.1，但**位移封頂在 0.50**：radius 由 1.5 推到 12 只讓位移
從 0.22 走到 0.50，效率同時掉回 3.4。

假說：VAE 編碼器把影像降採樣 8 倍，接近 Nyquist 的內容在進 latent 之前就被
丟掉了——「人眼看不見」的地方**模型也看不見**。若成立，正確的權重是
`模型敏感度 / 人眼代價`，而不是人眼代價的倒數本身。

本模組只測**分帶與欄位**，不載 IP2P 權重（那需要 GPU）。
"""

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import encoder_frequency_response as efr  # noqa: E402


def test_bands_tile_the_radius_range_without_gaps_or_overlap():
    """相鄰帶必須首尾相接：有縫會讓某些頻格從未被量到，重疊會重複計數，
    兩者都不會有症狀，只會讓導出的權重是錯的。"""
    edges = efr.band_edges(n=6)
    assert edges[0] == pytest.approx(0.0)
    assert edges[-1] > 1.4  # 角落的半徑是 sqrt(2)
    assert all(b > a for a, b in zip(edges[:-1], edges[1:]))
    assert len(efr.bands(n=6)) == 6
    for (lo, hi), (lo2, _) in zip(efr.bands(n=6)[:-1], efr.bands(n=6)[1:]):
        assert hi == lo2


def test_band_energy_sums_to_the_whole_spectrum():
    """逐帶能量加起來要等於整體能量，否則有格子掉了。"""
    torch.manual_seed(0)
    spec = torch.randn(2, 5, 32, 17, dtype=torch.float64)
    total = float(spec.pow(2).sum())
    parts = efr.band_energy(spec, block=32)
    assert sum(parts) == pytest.approx(total, rel=1e-9)


def test_required_columns_are_written_literally():
    src = (ROOT / "scripts" / "encoder_frequency_response.py").read_text(
        encoding="utf-8")
    for col in ("r_lo", "r_hi", "grad_energy", "dists_cost",
                "latent_move", "move_per_dists"):
        assert f'"{col}":' in src, col


def test_cli_defaults_are_the_operator_geometry():
    ns = efr.build_parser().parse_args(["--out", "x"])
    assert ns.block == 32 and ns.bands == 8
    assert ns.loss == "latent_norm"

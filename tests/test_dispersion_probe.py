"""色散度探針的接線與內插規則。

沒有權重的機器上測得到的是：條件表齊全、K=1 這個對照組不能被漏掉、以及
**等失真內插拒絕外插**——最後這一條是 `DECISIONS.md` 的規定，而它的失效
方式（悄悄外插出一個數字）沒有症狀。
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _mod():
    spec = importlib.util.spec_from_file_location(
        "dispersion_probe", ROOT / "scripts" / "dispersion_probe.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


P = _mod()


def test_condition_families_are_assigned_correctly():
    assert P.condition_family("disp_k1") == "disp"
    assert P.condition_family("disp_k8") == "disp"
    assert P.condition_family("disp_kfull") == "phase"
    assert P.condition_family("warp_smooth") == "warp"
    assert P.condition_family("warp_fold") == "warp"


def test_each_family_has_its_own_amplitude_ladder():
    """三族的強度單位不同（像素／弧度／像素），梯子不可共用。"""
    assert set(P.DEFAULT_AMPS) == {"disp", "phase", "warp"}
    for ladder in P.DEFAULT_AMPS.values():
        assert list(ladder) == sorted(ladder)
        assert ladder[0] > 0


def test_k1_is_in_the_default_sweep():
    """K=1 就是古典位移場，是整條軸的對照組。"""
    assert 1 in P.build_parser().parse_args([]).bands


def test_default_hop_matches_the_mainline_decision():
    assert P.build_parser().parse_args([]).hop == 8


def test_two_warp_grids_are_the_two_measured_ones():
    """16 是失真對照表所在的粗網格，64 是走到失真帶的那一格。"""
    assert P.WARP_GRIDS == {"warp_smooth": 16, "warp_fold": 64}


# ---- 內插：範圍外一律拒絕 ----

CURVE = [(0.05, 10.0), (0.10, 30.0), (0.20, 50.0)]


def test_interpolation_inside_the_range():
    assert P.interpolate(CURVE, 0.05) == pytest.approx(10.0)
    assert P.interpolate(CURVE, 0.075) == pytest.approx(20.0)
    assert P.interpolate(CURVE, 0.15) == pytest.approx(40.0)


@pytest.mark.parametrize("anchor", [0.049, 0.201, 0.0, 1.0])
def test_interpolation_refuses_to_extrapolate(anchor):
    assert P.interpolate(CURVE, anchor) is None


def test_summary_marks_out_of_range_instead_of_inventing_a_number():
    rows = [
        {"condition": "disp_k1", "amp": 1.0, "dists": 0.05, "latent_move": 10.0},
        {"condition": "disp_k1", "amp": 2.0, "dists": 0.10, "latent_move": 30.0},
    ]
    out = P.build_summary(rows, [0.075, 0.30])
    inside = [r for r in out if r["anchor_dists"] == 0.075][0]
    outside = [r for r in out if r["anchor_dists"] == 0.30][0]
    assert inside["out_of_range"] == 0
    assert inside["latent_move"] == pytest.approx(20.0)
    assert outside["out_of_range"] == 1
    assert outside["latent_move"] == ""
    assert outside["move_per_dists"] == ""


def test_summary_averages_over_images_before_interpolating():
    """先對影像取平均再內插。逐圖內插再平均會被構不到錨點的那幾張靜默丟掉。"""
    rows = [
        {"condition": "c", "amp": 1.0, "dists": 0.04, "latent_move": 10.0},
        {"condition": "c", "amp": 1.0, "dists": 0.06, "latent_move": 20.0},
        {"condition": "c", "amp": 2.0, "dists": 0.10, "latent_move": 30.0},
        {"condition": "c", "amp": 2.0, "dists": 0.10, "latent_move": 50.0},
    ]
    out = P.build_summary(rows, [0.05])
    assert out[0]["latent_move"] == pytest.approx(15.0)
    assert out[0]["dists_lo"] == pytest.approx(0.05)
    assert out[0]["n_amps"] == 2

"""`scripts/tradeoff_curve.py` 的曲線與兩個錨點（DEC-029）。

這支不跑 GPU，全部邏輯都是純函式，故可以完整測。要釘的是三件事：

1. 同一強度的多張影像摺成一點、依 x 遞增排序；
2. 內插取得的錨點值正確，且**範圍外一律回 None 不外插**；
3. 錨點回傳所依據的兩個端點——沒有端點的內插值無法判斷可靠度。

第 2 條是 FND-062 的教訓：單點對齊看不出附近的曲線長什麼樣，外插出來的
數字看起來一樣像個數字。
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from tradeoff_curve import anchors, curve_points, interp_at  # noqa: E402


def _rows():
    # 兩個條件 × 兩個強度 × 兩張影像。
    out = []
    for cond, base in (("phase", 0.0), ("dct_shield", 0.1)):
        for rad, x, y in ((1.0, 0.02, 0.30), (2.0, 0.06, 0.50)):
            for img, jitter in (("a", -0.002), ("b", +0.002)):
                out.append({"image": img, "condition": cond, "radius": str(rad),
                            "fid_dists": str(x + base + jitter),
                            "edit_lpips": str(y + jitter)})
    return out


def test_同強度的多張影像摺成一點且依x排序():
    pts = curve_points(_rows(), "phase", "fid_dists", "edit_lpips")
    assert len(pts) == 2
    assert [p[2] for p in pts] == [1.0, 2.0]        # 依 x 遞增即依 radius 遞增
    assert pts[0][3] == 2                            # n_images
    assert pts[0][0] == pytest.approx(0.02, abs=1e-9)   # jitter 相消


def test_內插取中點():
    pts = [(0.0, 0.0, 1.0, 1), (1.0, 10.0, 2.0, 1)]
    hit = interp_at(pts, 0.5, axis=0)
    assert hit["value"] == pytest.approx(5.0)
    assert hit["lo"][2] == 1.0 and hit["hi"][2] == 2.0


def test_依y內插():
    pts = [(0.0, 0.0, 1.0, 1), (1.0, 10.0, 2.0, 1)]
    assert interp_at(pts, 2.5, axis=1)["value"] == pytest.approx(0.25)


def test_範圍外不外插():
    pts = [(0.0, 0.0, 1.0, 1), (1.0, 10.0, 2.0, 1)]
    assert interp_at(pts, 1.5, axis=0) is None
    assert interp_at(pts, -0.1, axis=0) is None


def test_單點曲線無法內插():
    assert interp_at([(0.0, 0.0, 1.0, 1)], 0.0, axis=0) is None


def test_錨點回報端點與範圍外旗標():
    rows = _rows()
    ref = curve_points(rows, "phase", "fid_dists", "edit_lpips")
    other = curve_points(rows, "dct_shield", "fid_dists", "edit_lpips")
    got = anchors(ref, other, ref_radius=2.0)
    by_name = {a["anchor"]: a for a in got}

    # 等失真：phase 在 r=2 的失真是 0.06，而 dct_shield 的失真範圍是
    # 0.12–0.16（整條偏移 0.1），故落在範圍外，必須拒絕。
    assert by_name["等失真"]["out_of_range"] is True
    assert by_name["等失真"]["other_value"] == ""

    # 等效果：兩條曲線的 y 範圍相同（0.30–0.50），故 0.50 在端點上、可內插。
    eq = by_name["等效果"]
    assert eq["out_of_range"] is False
    assert eq["seg_lo_radius"] == 1.0 and eq["seg_hi_radius"] == 2.0


def test_參照強度不存在時拋錯():
    rows = _rows()
    ref = curve_points(rows, "phase", "fid_dists", "edit_lpips")
    other = curve_points(rows, "dct_shield", "fid_dists", "edit_lpips")
    with pytest.raises(ValueError, match="沒有 radius"):
        anchors(ref, other, ref_radius=1.3)

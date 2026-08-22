"""等失真表的分組與拒絕外插。**只驗協定，不驗任何實驗結論。**

兩件事必須釘住：

1. `--group-by` 沒把「同一個 condition、不同旗標」分開的話，兩條曲線會被摺
   成一條而**不會有任何症狀**——那正是本支存在的理由。
2. 錨點落在掃描範圍外一律標 `out_of_range`，不外插（DEC-029）。
3. 各組的張數不同時直接拋錯：張數不同的平均並排讀會把「跑到哪張」讀成
   方法差異。
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from matched_distortion_table import (  # noqa: E402
    build_curves, common_images, group_key,
)


def _row(image, cond, floor, radius, dists, disp, psnr=30.0, clip=0.9):
    return {"image": image, "condition": cond, "spectral_floor": str(floor),
            "radius": str(radius), "fid_dists": str(dists),
            "edit_lpips": str(disp), "fid_psnr": str(psnr),
            "fid_lpips": "0.3", "fid_ssim": "0.9",
            "edit_clip_sim": str(clip)}


def _rows():
    """兩個設定，**半徑互不重疊**——兩批掃描各自取自己的強度，實務上就是
    這樣，於是不分組時它們會被摺成一條曲線而不會有任何症狀。"""
    out = []
    for img in ("a", "b"):
        for floor, radii in ((0.0, ((1.0, 0.05, 0.3), (2.0, 0.15, 0.6))),
                             (0.04, ((1.5, 0.07, 0.5), (2.5, 0.17, 0.8)))):
            for rad, d, y in radii:
                out.append(_row(img, "phase_gain", floor, rad, d, y))
    return out


def test_不分組會把兩個設定摺成一條曲線():
    rows = _rows()
    one = build_curves(rows, ["condition"], "fid_dists", "edit_lpips",
                       ["a", "b"])
    assert len(one) == 1
    # 四個強度全部落在同一條曲線上，看不出其中兩個是另一個設定
    assert len(one[next(iter(one))]) == 4
    two = build_curves(rows, ["condition", "spectral_floor"], "fid_dists",
                       "edit_lpips", ["a", "b"])
    assert len(two) == 2


def test_分組鍵含每一個指定的欄():
    k = group_key({"condition": "phase_gain", "spectral_floor": "0.04"},
                  ["condition", "spectral_floor"])
    assert "condition=phase_gain" in k and "spectral_floor=0.04" in k


def test_張數不同時拋錯而不是靜默平均():
    rows = _rows()
    rows = [r for r in rows
            if not (r["image"] == "b" and r["radius"] == "2.0")]
    with pytest.raises(SystemExit, match="沒跑完"):
        build_curves(rows, ["condition", "spectral_floor"], "fid_dists",
                     "edit_lpips", ["a", "b"])


def test_共同影像取交集():
    rows = _rows()
    rows = [r for r in rows if not (r["image"] == "b" and r["radius"] == "2.0")]
    assert common_images(rows, ["condition", "spectral_floor"]) == ["a"]


def test_沒有共同影像時拋錯():
    rows = [_row("a", "phase", 0.0, 1.0, 0.05, 0.3),
            _row("b", "phase", 0.0, 2.0, 0.15, 0.6)]
    with pytest.raises(SystemExit, match="共同的影像"):
        common_images(rows, ["condition"])


def test_錨點落在範圍外不外插():
    from tradeoff_curve import interp_at
    pts = [(0.05, 0.3, 1.0, 2), (0.15, 0.6, 2.0, 2)]
    assert interp_at(pts, 0.30, axis=0) is None
    assert interp_at(pts, 0.01, axis=0) is None
    hit = interp_at(pts, 0.10, axis=0)
    assert hit is not None
    assert hit["value"] == pytest.approx(0.45)

"""兩支「只讀已存的圖」的探針，把設定錯誤擋在載入權重之前。

存在理由
────────────────────────────────────────────────────────────────────
兩支探針的共同失效方式是**量到另一條軸而不報錯**：`--ig-zt` 填錯、固定抽樣
的組數或種子與訓練端不同、或是 `--entry` 指到一個沒有 `results.csv` 的目錄
（於是整批靜默跳過、輸出一張只有 `original` 的表）。這些都要在載入 IP2P
之前就拋出來，否則一格的代價是把權重載進顯存之後才發現。

`defense_images` 另外釘一件事：**缺圖要印出來而不是靜默略過**，並且回傳的
條件名取自 `results.csv` 而不是由檔名反推。

只用 CPU，不載任何權重。
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import ig_probe  # noqa: E402
import regional_displacement as rd  # noqa: E402

HEADER = "image,condition,fid_dists,edit_lpips,best_eval,loss\n"


def make_run_dir(tmp_path: Path, name: str, conds, files=("def",)):
    d = tmp_path / "run"
    d.mkdir(exist_ok=True)
    rows = "".join(f"{name},{c},0.1,0.2,0.003,image_guidance\n" for c in conds)
    (d / "results.csv").write_text(HEADER + rows, encoding="utf-8")
    for c in conds:
        for sub in files:
            (d / f"{name}__{c}__{sub}.png").write_bytes(b"")
    return d


# ---- ig_probe ----

def test_ig_zt必填且訊息說明理由():
    args = ig_probe.build_parser().parse_args(
        ["--entry", "a=x", "--images", "i", "--out", "o.csv"])
    with pytest.raises(SystemExit, match="--ig-zt 必填"):
        ig_probe.check_args(args)


def test_固定抽樣的預設值與訓練端相同():
    """`scripts/ip2p_run.py` 的 `--eval-draws` 8、`--eval-seed` 99991。

    不同就不是同一條軸，而且不會有症狀。
    """
    args = ig_probe.build_parser().parse_args(
        ["--entry", "a=x", "--images", "i", "--out", "o.csv",
         "--ig-zt", "diffuse_src"])
    assert (args.eval_draws, args.eval_seed) == (8, 99991)
    assert (args.ig_t_min, args.ig_t_max) == (1, 1000)


def test_entry指到沒有results的目錄就拋錯(tmp_path):
    args = ig_probe.build_parser().parse_args(
        ["--entry", f"a={tmp_path}", "--images", "i", "--out", "o.csv",
         "--ig-zt", "diffuse_src"])
    with pytest.raises(SystemExit, match="results.csv"):
        ig_probe.check_args(args)


def test_entry格式錯誤就拋錯(tmp_path):
    args = ig_probe.build_parser().parse_args(
        ["--entry", "沒有等號", "--images", "i", "--out", "o.csv",
         "--ig-zt", "diffuse_src"])
    with pytest.raises(SystemExit, match="LABEL=DIR"):
        ig_probe.check_args(args)


def test_t範圍顛倒就拋錯():
    args = ig_probe.build_parser().parse_args(
        ["--entry", "a=x", "--images", "i", "--out", "o.csv",
         "--ig-zt", "diffuse_src", "--ig-t-min", "800", "--ig-t-max", "300"])
    with pytest.raises(SystemExit, match="ig-t-min"):
        ig_probe.check_args(args)


def test_條件名取自results而不是由檔名反推(tmp_path):
    d = make_run_dir(tmp_path, "img_a", ["color_grid", "color_grid_rand"])
    out = ig_probe.defense_images(d, "img_a")
    assert [c for c, _, _ in out] == ["color_grid", "color_grid_rand"]


def test_缺圖的條件被跳過而不是拋錯(tmp_path, capsys):
    d = make_run_dir(tmp_path, "img_a", ["color_grid"])
    (d / "results.csv").write_text(
        HEADER + "img_a,color_grid,0.1,0.2,0.003,image_guidance\n"
        "img_a,missing_cond,0.1,0.2,0.003,image_guidance\n", encoding="utf-8")
    out = ig_probe.defense_images(d, "img_a")
    assert [c for c, _, _ in out] == ["color_grid"]
    assert "[skip]" in capsys.readouterr().out


def test_別的影像的列不會被算進來(tmp_path):
    d = make_run_dir(tmp_path, "img_a", ["color_grid"])
    assert ig_probe.defense_images(d, "img_b") == []


# ---- regional_displacement ----

def test_分區腳本的遮罩形狀參數與ip2p_run相同():
    """threshold 0.30、dilate 16、feather 24。遮罩不同就不是同一條軸。"""
    args = rd.build_parser().parse_args(
        ["--entry", "a=x", "--images", "i", "--out", "o.csv"])
    assert args.subject_mask_threshold == 0.30
    assert args.subject_mask_dilate == 16
    assert args.subject_mask_feather == 24


def test_主體名詞查不到就拋錯而不是猜(tmp_path):
    cat = tmp_path / "cat.yaml"
    cat.write_text("objects:\n  other_image: [cat]\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="主體名詞"):
        rd.subject_texts(cat, "img_a")


def test_單一名詞也回傳清單(tmp_path):
    cat = tmp_path / "cat.yaml"
    cat.write_text("objects:\n  img_a: a toolbox\n", encoding="utf-8")
    assert rd.subject_texts(cat, "img_a") == ["a toolbox"]


def test_三張圖缺一即跳過(tmp_path, capsys):
    d = make_run_dir(tmp_path, "img_a", ["color_grid"],
                     files=("def", "edit_orig"))
    assert rd.triple(d, "img_a", "color_grid") is None
    assert "[skip]" in capsys.readouterr().out


def test_三張圖齊備時回傳三個路徑(tmp_path):
    d = make_run_dir(tmp_path, "img_a", ["color_grid"],
                     files=("def", "edit_orig", "edit_def"))
    got = rd.triple(d, "img_a", "color_grid")
    assert set(got) == {"def", "edit_orig", "edit_def"}
    assert all(p.exists() for p in got.values())

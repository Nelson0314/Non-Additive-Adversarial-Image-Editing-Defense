"""`phase_drift_diagnosis.py` 的 `--tags`／`--conds`：不給時行為不變。

這一支量的是「裝上去的相位偏移在淨化算子之後還在不在」，原本把四個主線 tag
寫死在 `CONDITIONS`。加旗標的用意是把同一份量測套到新批次上，**預設值必須
逐位元等於原本的行為**，否則已發表的 `runs/phase_drift_diagnosis/` 重跑會變。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import phase_drift_diagnosis as pdd


def test_預設條件表未被旗標改動():
    assert pdd.CONDITIONS == {
        "ours_ph_q": "phase",
        "ours_ph_n": "phase",
        "ours_pg_q20": "phase_gain",
        "dct_aj85": "dct_shield_y",
    }


def test_tags_與_conds_不等長時報錯而不是靜默截斷(tmp_path, capsys):
    imgs = tmp_path / "images.txt"
    imgs.write_text("a\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        pdd.main(["--defended", str(tmp_path), "--images", str(imgs),
                  "--out", str(tmp_path / "out"),
                  "--tags", "ig_d21", "ig_d25", "--conds", "phase_gain"])


def test_只給_tags_不給_conds_也要報錯(tmp_path):
    imgs = tmp_path / "images.txt"
    imgs.write_text("a\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        pdd.main(["--defended", str(tmp_path), "--images", str(imgs),
                  "--out", str(tmp_path / "out"), "--tags", "ig_d21"])

"""有效約束診斷 `e27_binding_check.analyse` 的判定邏輯。

這支腳本是常設步驟：每一輪校準與網格跑完都要用它確認「τ 才是綁住這一格的
那道約束」。本專案已經連續踩到四個誤判的有效約束（`max_dev` 兩次、防禦 margin、
`L_fid` 裡係數為 1 的原始 lpips 項），每一個都讓整批網格變成無效資料，而且
都是事後翻 `history.json` 才發現的。診斷工具本身判錯，等於這道保險不存在。

2026-08-02 補測的直接原因：`HINGES` 原本漏了 `fid_pen_chroma`，色度綁住的
格子會被誤判成 LPIPS hinge 或「沒有任何約束啟動」。
"""

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

_spec = importlib.util.spec_from_file_location(
    "e27_binding_check", ROOT / "scripts" / "e27_binding_check.py"
)
bc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bc)


def _make_run(tmp_path, loss, history, name="run_x", cell="car_00__C__r32"):
    d = tmp_path / name
    (d / cell).mkdir(parents=True)
    (d / "env.json").write_text(json.dumps({"loss": loss}), encoding="utf-8")
    (d / cell / "history.json").write_text(json.dumps(history), encoding="utf-8")
    return d


def _steps(n, **pens):
    """n 步的 history。未指定的懲罰項為 0，`edit_shift` 固定不觸及 margin。"""
    base = {"edit_shift": 0.1, "fid_lpips": 0.04, "fid_acut": 0.01,
            "fid_chroma": 0.5, "fid_pen_lpips": 0.0, "fid_pen_acut": 0.0,
            "fid_pen_chroma": 0.0, "fid_pen_linf": 0.0, "fid_pen_psnr": 0.0}
    return [{**base, **pens} for _ in range(n)]


LOSS = {"margin": 1.0, "tau_lpips": 0.05, "gamma_lpips": 100.0,
        "gamma_acut": 100.0, "gamma_chroma": 100.0, "beta_linf": 0.0,
        "gamma_psnr": 0.0}


def test_色度綁住時判定為色度而非LPIPS(tmp_path):
    """E28 導入的第三道約束必須能被判出來。判準（NEXT_SESSION §5）寫的是
    「必須是 LPIPS hinge，不是色度 hinge」——工具查不到的東西無法構成判準。
    """
    d = _make_run(tmp_path, LOSS, _steps(60, fid_pen_chroma=0.3, fid_pen_lpips=0.0))
    (row,) = bc.analyse(d)
    assert "色度" in row["verdict"]
    assert row["engaged"]["色度"] == 60


def test_兩道同時綁住時並列列出(tmp_path):
    """跨 site 比較要求兩個條件被同一組約束綁住。一條件只有 LPIPS、另一條件
    LPIPS＋色度，是不同的狀態，不可並列，故判定不得只報其中一道。
    """
    d = _make_run(tmp_path, LOSS,
                  _steps(60, fid_pen_lpips=0.2, fid_pen_chroma=0.3))
    (row,) = bc.analyse(d)
    assert "LPIPS" in row["verdict"] and "色度" in row["verdict"]


def test_係數為零的hinge不算約束(tmp_path):
    """`beta_linf=0` 的 L∞ 罰則仍會被計算與記錄，但不進梯度。初版把這種
    hinge 當成有效約束，實測把 site C 判成「PSNR hinge 56/60 步啟動」。
    """
    loss = {**LOSS, "beta_linf": 0.0}
    d = _make_run(tmp_path, loss, _steps(60, fid_pen_linf=0.5))
    (row,) = bc.analyse(d)
    assert "L∞" not in row["engaged"]
    assert "沒有任何約束啟動過" in row["verdict"]


def test_舊run沒有色度係數也沒有色度記錄時跳過(tmp_path):
    """E27 以前的 run 跑在還沒有色度 hinge 的程式上。那些 run 必須仍可
    分析，且不得憑空多出一道「色度 0/60」。
    """
    loss = {k: v for k, v in LOSS.items() if k != "gamma_chroma"}
    hist = [{k: v for k, v in h.items() if k != "fid_pen_chroma"}
            for h in _steps(60, fid_pen_lpips=0.2)]
    d = _make_run(tmp_path, loss, hist)
    (row,) = bc.analyse(d)
    assert "色度" not in row["engaged"]
    assert "LPIPS" in row["verdict"]


def test_有色度記錄卻沒有係數欄位時報錯(tmp_path):
    """env.json 與 history 不一致代表無法判定該道 hinge 有沒有進梯度。
    此時當成零會重演「誤判的有效約束」那一類錯誤，故要求停下來查清楚。
    """
    loss = {k: v for k, v in LOSS.items() if k != "gamma_chroma"}
    d = _make_run(tmp_path, loss, _steps(60, fid_pen_chroma=0.3))
    with pytest.raises(SystemExit):
        bc.analyse(d)


def test_防禦margin飽和優先於保真hinge(tmp_path):
    """margin 一旦飽和，防禦項不再施力，該格量到的不是「τ 下的能力」。
    E27 第三輪就是這種情形。
    """
    hist = _steps(60, fid_pen_lpips=0.2)
    for h in hist:
        h["edit_shift"] = 1.0        # 等於 margin
    d = _make_run(tmp_path, LOSS, hist)
    (row,) = bc.analyse(d)
    assert "margin" in row["verdict"]

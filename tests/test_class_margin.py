"""`scripts/class_margin.py` 的檔名配對 —— 影像與 metrics 必須同一個 (τ, seed)。

這裡不驗 SigLIP 的數值（那要模型），驗的是**配對**：ip3 是第一個四個 τ 都
留在磁碟上的批次，而原本的收集方式把四個 τ 塌成一個鍵，影像取到 τ=0.35、
metrics 取到 τ=0.05。兩者都存在、CSV 也照樣寫得出來，`tau` 欄看起來完全
正常——與 `e9a35a5c6` 修掉的那個覆寫是同一族的漏帶鍵。
"""

import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "class_margin",
    Path(__file__).resolve().parent.parent / "scripts" / "class_margin.py")
cm = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cm)

TAUS = ("0.05", "0.1", "0.2", "0.35")
SEEDS = (0, 1, 2)


@pytest.fixture
def cell(tmp_path):
    """帶 τ 的新格式：四個 τ × 三個 seed。"""
    d = tmp_path / "identity_0"
    d.mkdir()
    for t in TAUS:
        for s in SEEDS:
            (d / f"edit_tau{t}_seed{s}.png").write_bytes(b"")
            (d / f"metrics_tau{t}_seed{s}.json").write_text(
                json.dumps({"tau": float(t), "seed": s, "group": "bird",
                            "prompt": "a cat"}), encoding="utf-8")
    return d


def test_四個tau都被列出而不是塌成一個(cell):
    got = cm.edit_pngs(cell)
    assert len(got) == len(TAUS) * len(SEEDS), (
        f"只取到 {len(got)} 個，四個 τ 被塌成同一個鍵")
    assert {t for t, _, _ in got} == set(TAUS)


def test_影像與metrics取自同一個tau(cell):
    """配對錯開時 CSV 的 `tau` 欄仍然合法，事後查不出來。"""
    for tau, seed, png in cm.edit_pngs(cell):
        m = cm.meta_for(cell, tau, seed)
        assert m["tau"] == pytest.approx(float(tau)), (
            f"影像是 τ={tau} 而 metrics 是 τ={m['tau']}")
        assert m["seed"] == seed


def test_依tau與seed排序(cell):
    got = [(float(t), s) for t, s, _ in cm.edit_pngs(cell)]
    assert got == sorted(got)


def test_舊格式仍讀得到且tau為None(tmp_path):
    """對照側（`control/`）沒有 τ 這個軸，其檔名本來就不帶 τ；舊批次同形。"""
    d = tmp_path / "identity_0"
    d.mkdir()
    for s in SEEDS:
        (d / f"edit_seed{s}.png").write_bytes(b"")
        (d / f"metrics_seed{s}.json").write_text(
            json.dumps({"seed": s}), encoding="utf-8")
    got = cm.edit_pngs(d)
    assert [(t, s) for t, s, _ in got] == [(None, s) for s in SEEDS]
    assert cm.meta_for(d, None, 1)["seed"] == 1


def test_新舊並存時以帶tau的為準(cell):
    """回收舊批次時兩種檔名可能同時出現，混用會讓同一格出現重複列。"""
    for s in SEEDS:
        (cell / f"edit_seed{s}.png").write_bytes(b"")
    got = cm.edit_pngs(cell)
    assert len(got) == len(TAUS) * len(SEEDS)
    assert all(t is not None for t, _, _ in got)


# ---------------------------------------------------------------------------
# 遮罩內讀出量（`HANDOVER_METRICS_2026-08-08` §6.2）
# ---------------------------------------------------------------------------

def _write_mask(path, size=64, box=(16, 48, 8, 40)):
    """(y0, y1, x0, x1) 為 1 的實心矩形遮罩，其餘為 0。"""
    import torch

    from src.experiment.executors import save_image

    m = torch.zeros(1, 3, size, size)
    m[..., box[0]:box[1], box[2]:box[3]] = 1.0
    save_image(m, path)
    return box


def test_外接矩形取自遮罩本身(tmp_path):
    import torch

    p = tmp_path / "bird_02_mask.png"
    box = _write_mask(p)
    assert cm.mask_bbox(p, torch.device("cpu")) == box


def test_非矩形遮罩取其外接矩形(tmp_path):
    """輪廓遮罩下仍然包含全部被重畫的像素，只是多帶一點脈絡——那個方向是
    保守的（訊號被稀釋而非放大）。"""
    import torch

    from src.experiment.executors import save_image

    m = torch.zeros(1, 3, 64, 64)
    m[..., 10:12, 20:22] = 1.0        # 左上一小塊
    m[..., 40:42, 50:52] = 1.0        # 右下一小塊
    p = tmp_path / "x_mask.png"
    save_image(m, p)
    assert cm.mask_bbox(p, torch.device("cpu")) == (10, 42, 20, 52)


def test_全零遮罩立刻拋出(tmp_path):
    import torch

    from src.experiment.executors import save_image

    p = tmp_path / "z_mask.png"
    save_image(torch.zeros(1, 3, 32, 32), p)
    with pytest.raises(ValueError, match="全零遮罩"):
        cm.mask_bbox(p, torch.device("cpu"))

"""`scripts/class_margin.py` 的檔案配對 —— 判定層最容易靜默出錯的地方。

這裡釘住的兩件事在 2026-08-09 的 s3a 批次上同時發生過，而且**數字看起來
完全合理**：

1. `edit_pngs` 以 seed 為鍵、逐 τ 互相覆寫，每格只留下一個 τ。
2. `meta_for` 取排序**第一個** metrics 檔，而 PNG 那一側取的是最後一個。

兩者相加的後果是「圖是 τ=0.5 的、metadata 是 τ=0.05 的」，於是 CSV 的 `tau`
欄全錯，`eval_protocols` 的 ISR 段按錯的 τ 分組。margin 的數值本身反而是對的
（PNG 挑中了主表那一點），純屬字串排序的巧合——τ 若含 0.55 與 0.6 就會挑錯。

不用真的模型：這一層是純檔案配對，SigLIP 前向與它無關。
"""
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load():
    """以檔案路徑載入 `scripts/class_margin.py`（scripts/ 不是套件）。"""
    spec = importlib.util.spec_from_file_location(
        "class_margin", ROOT / "scripts" / "class_margin.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["class_margin"] = mod
    spec.loader.exec_module(mod)
    return mod


cm = _load()

# s3a 實際用的那組 τ。0.5 恰好在字串排序的最後，正是那個巧合的來源。
TAUS = ("0.05", "0.1", "0.2", "0.35", "0.5")
SEEDS = (0, 1, 2)


@pytest.fixture
def cell(tmp_path):
    d = tmp_path / "identity_0"
    d.mkdir()
    for t in TAUS:
        for s in SEEDS:
            (d / f"edit_tau{t}_seed{s}.png").write_bytes(b"")
            (d / f"metrics_tau{t}_seed{s}.json").write_text(
                '{"tau": %s, "group": "bird", "prompt": "a butterfly"}' % t,
                encoding="utf-8")
    return d


def test_逐τ各一列而不是每個seed只留一個(cell):
    got = cm.edit_pngs(cell)
    assert len(got) == len(TAUS) * len(SEEDS), (
        "每個 (seed, τ) 都該有一列；只回傳 seed 數表示 τ 被覆寫掉了")
    assert {t for _, t, _ in got} == set(TAUS)
    for seed, tau, path in got:
        assert f"_tau{tau}_" in path.name
        assert path.name.endswith(f"_seed{seed}.png")


def test_metadata與圖配到同一個τ(cell):
    """before 的症狀：圖是 τ=0.5、metadata 是 τ=0.05，而兩者都「存在」。"""
    for seed, tau, path in cm.edit_pngs(cell):
        m = cm.meta_for(cell, seed, tau)
        assert str(m["tau"]) == tau, (
            f"{path.name} 配到了 τ={m['tau']} 的 metrics")


def test_沒有τ的舊格式仍讀得到(tmp_path):
    """對照側（φ=0）沒有 τ 這個軸，其檔名不帶 τ。"""
    d = tmp_path / "identity_0"
    d.mkdir()
    for s in SEEDS:
        (d / f"edit_seed{s}.png").write_bytes(b"")
        (d / f"metrics_seed{s}.json").write_text(
            '{"group": "bird", "prompt": "a butterfly"}', encoding="utf-8")
    got = cm.edit_pngs(d)
    assert [(s, t) for s, t, _ in got] == [(s, "") for s in SEEDS]
    assert cm.meta_for(d, 0, "")["group"] == "bird"


def test_τ的挑選不依賴字串排序(tmp_path):
    """加入 0.55 與 0.6：舊寫法會挑到 "0.6"（字串最大）而漏掉其餘全部。

    這條測試的用意是讓「剛好挑中主表那一點」這個巧合不能再充當正確性。
    """
    d = tmp_path / "identity_0"
    d.mkdir()
    taus = ("0.5", "0.55", "0.6")
    for t in taus:
        (d / f"edit_tau{t}_seed0.png").write_bytes(b"")
        (d / f"metrics_tau{t}_seed0.json").write_text(
            '{"tau": %s}' % t, encoding="utf-8")
    got = cm.edit_pngs(d)
    assert {t for _, t, _ in got} == set(taus)
    for seed, tau, _ in got:
        assert str(cm.meta_for(d, seed, tau)["tau"]) == tau


def test_條件清單由格點登記表導出(tmp_path):
    """寫死一份清單的症狀是新條件被靜默漏掉，表格看起來仍然完整。"""
    from src.experiment import grid

    assert "N4" in grid.CONDITIONS
    assert "Ra" in grid.CONDITIONS

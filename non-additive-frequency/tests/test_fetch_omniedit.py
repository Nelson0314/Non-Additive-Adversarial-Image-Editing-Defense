"""`scripts/fetch_omniedit.py` 的分層抽樣（DEC-030）。

只測不碰網路的那一半。要釘三件事：

1. **每類張數正確、且不含 style**——論文那句涵蓋的是 style 以外的五類，
   而 dev split 依任務排序，取前 150 列會抽到一整批 style。
2. **同一個 seed 必得同一批**——論文的 150 張永遠取不回來，就是因為沒有這個。
3. **池子不夠時拋錯**，不是默默少給。
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from fetch_omniedit import TASKS, stratified  # noqa: E402


def _rows(n_per_task: int = 40):
    rows = [{"omni_edit_id": f"task_style_{i}", "task": "style"}
            for i in range(100)]
    for t in TASKS:
        rows += [{"omni_edit_id": f"{t}_{i:04d}", "task": t}
                 for i in range(n_per_task)]
    return rows


def test_每類張數正確且不含style():
    got = stratified(_rows(), per_task=3, seed=0)
    assert len(got) == 3 * len(TASKS)
    assert {r["task"] for r in got} == set(TASKS)
    assert all(r["task"] != "style" for r in got)


def test_同seed可重現_不同seed會不同():
    a = [r["omni_edit_id"] for r in stratified(_rows(), 5, seed=0)]
    b = [r["omni_edit_id"] for r in stratified(_rows(), 5, seed=0)]
    c = [r["omni_edit_id"] for r in stratified(_rows(), 5, seed=1)]
    assert a == b
    assert a != c


def test_輸入順序不影響結果():
    # 先依 omni_edit_id 排序才抽，故列的先後不改變抽樣結果。
    rows = _rows()
    a = [r["omni_edit_id"] for r in stratified(rows, 5, seed=0)]
    b = [r["omni_edit_id"] for r in stratified(list(reversed(rows)), 5, seed=0)]
    assert a == b


def test_池子不夠時拋錯():
    with pytest.raises(SystemExit, match="抽不出"):
        stratified(_rows(n_per_task=2), per_task=30, seed=0)

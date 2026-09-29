"""選點測試不載入編輯模型或權重。"""
import ast
from pathlib import Path

import pytest


def select_result(*args):
    source = Path(__file__).parents[1] / "code/style_prompt_defence.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "select_result")
    namespace = {}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), namespace)
    return namespace["select_result"](*args)


@pytest.mark.parametrize("policy,best_feasible,last_feasible,expected_update,expected_feasible", [
    ("last", 1, 0, 10, 0), ("last", 1, 1, 10, 1),
    ("best", 1, 0, 3, 1), ("best", None, 0, 10, 0),
])
def test_selected_feasibility_is_independent_of_policy(
        policy, best_feasible, last_feasible, expected_update, expected_feasible):
    best = (3, "best image", {"feasible": best_feasible}) if best_feasible is not None else None
    last = (10, "last image", {"feasible": last_feasible})
    selected, fields = select_result(best, last, policy)
    assert selected[0] == expected_update
    assert fields == {"selection_policy": policy, "selected_feasible": expected_feasible,
                      "feasible": expected_feasible}


def test_no_updates_reports_explicit_error():
    with pytest.raises(ValueError, match="沒有可選取"):
        select_result(None, None, "last")

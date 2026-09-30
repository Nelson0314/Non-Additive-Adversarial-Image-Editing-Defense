"""選點測試不載入編輯模型或權重。"""
import pytest

from immunization_style.method import select_result


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


def test_styles_come_from_the_registry_file():
    from immunization_style.method import STYLES
    assert STYLES["p_snow"] == "Add some snow."
    assert STYLES["cool_grade"] == "apply subtle cool cinematic grading"
    assert len(STYLES) == 11

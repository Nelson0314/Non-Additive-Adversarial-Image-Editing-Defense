"""兩份指標入口均須傳遞模型建立失敗，且不得觸碰輸出。"""
import ast
from pathlib import Path
import sys
import types

import pytest


@pytest.mark.parametrize("relative", ["scripts/metrics_union.py", "main_table/code/metrics_union.py"])
def test_required_metric_failure_propagates(relative, monkeypatch):
    path = Path(__file__).parents[1] / relative
    tree = ast.parse(path.read_text(encoding="utf-8"))
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "stage_aesthetic")
    namespace = {"AESTHETIC": [("aes_nima", "nima", False)]}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
    error = RuntimeError("checkpoint unavailable")

    def fail(*args, **kwargs):
        raise error

    monkeypatch.setitem(sys.modules, "pyiqa", types.SimpleNamespace(create_metric=fail))
    with pytest.raises(RuntimeError) as raised:
        namespace["stage_aesthetic"]("cpu")
    assert raised.value is error

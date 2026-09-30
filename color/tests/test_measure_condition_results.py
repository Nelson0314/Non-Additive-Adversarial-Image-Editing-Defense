"""讀數 shell 的退出碼契約；以 stub 取代 GPU 程式。"""
import os
from pathlib import Path
import shutil
import subprocess

import pytest


@pytest.mark.parametrize("displacement,retention,expected", [(7, 0, 7), (0, 9, 9), (0, 0, 0)])
def test_readout_propagates_stage_failure(tmp_path, displacement, retention, expected):
    script = Path(__file__).parents[1] / "scripts/measure_condition_results.sh"
    stub = tmp_path / "python-stub"
    stub.write_text('#!/usr/bin/env bash\ncase "$2" in\n'
                    f'*measure_edit_displacement) exit {displacement};;\n'
                    f'*measure_purified_displacement) exit {retention};;\nesac\n',
                    encoding="utf-8", newline="\n")
    stub.chmod(0o755)
    env = dict(os.environ, PY=stub.as_posix())
    result = subprocess.run([shutil.which("bash"), str(script), "0"], env=env,
                            capture_output=True, text=True)
    assert result.returncode == expected, result.stderr
    assert ("[READOUT-DONE]" in result.stdout) == (expected == 0)
    if displacement:
        assert "retention" not in result.stdout

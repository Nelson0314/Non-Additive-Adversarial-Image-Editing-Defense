"""讀數 shell 的退出碼契約；以 stub 取代 GPU 程式。"""
import os
from pathlib import Path
import shutil
import subprocess

import pytest


@pytest.mark.parametrize("displacement,retention,expected", [(7, 0, 7), (0, 9, 9), (0, 0, 0)])
def test_readout_propagates_stage_failure(tmp_path, displacement, retention, expected):
    script = Path(__file__).parents[1] / "scripts/readout.sh"
    text = script.read_text(encoding="utf-8")
    text = text.replace("source ~/env.sh", ":")
    text = text.replace("R=/nfs/home/nelson0314/image-immunization", 'R="$TEST_ROOT"')
    runner = tmp_path / "readout.sh"
    runner.write_text(text, encoding="utf-8", newline="\n")
    stub = tmp_path / "python-stub"
    stub.write_text('#!/usr/bin/env bash\ncase "$1" in\n'
                    f'*edit_displacement.py) exit {displacement};;\n'
                    f'*edit_retention.py) exit {retention};;\nesac\n',
                    encoding="utf-8", newline="\n")
    stub.chmod(0o755)
    env = dict(os.environ, TEST_ROOT=tmp_path.as_posix(), PY=stub.as_posix())
    result = subprocess.run([shutil.which("bash"), str(runner), "0"], env=env,
                            capture_output=True, text=True)
    assert result.returncode == expected, result.stderr
    assert ("[READOUT-DONE]" in result.stdout) == (expected == 0)
    if displacement:
        assert "retention" not in result.stdout

"""派工入口的整合測試；所有 GPU 查詢與 CUDA 檢查均以 stub 取代。"""
import os
from pathlib import Path
import shutil
import subprocess

import pytest


@pytest.mark.parametrize("command_rc", [0, 23])
def test_command_exit_releases_owned_lease(tmp_path, command_rc):
    source = Path(__file__).parents[1] / "scripts"
    scripts = tmp_path / "lab/scripts"
    scripts.mkdir(parents=True)
    for filename in ("gpu_lease.sh", "gpu_policy.sh"):
        shutil.copyfile(source / filename, scripts / filename)
    text = (source / "run_on_card.sh").read_text(encoding="utf-8")
    text = text.replace("source ~/env.sh >/dev/null 2>&1", ":")
    text = text.replace("R=/nfs/home/nelson0314/image-immunization", 'R="$TEST_ROOT"')
    runner = scripts / "run_on_card.sh"
    runner.write_text(text, encoding="utf-8", newline="\n")
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts/free_cards.sh").write_text(
        '#!/usr/bin/env bash\n[ "${1:-}" = --assert ] || echo 0\nexit 0\n', newline="\n")
    stubs = tmp_path / "bin"
    stubs.mkdir()
    commands = {
        "nvidia-smi": '#!/usr/bin/env bash\ncase "$*" in *--query-gpu=uuid*) echo GPU-test;; esac\n',
        "python-stub": '#!/usr/bin/env bash\nexit 0\n',
    }
    for name, text in commands.items():
        path = stubs / name
        path.write_text(text, newline="\n")
        path.chmod(0o755)
    leases = tmp_path / "leases"
    env = dict(os.environ, TEST_ROOT=tmp_path.as_posix(), LEASE=leases.as_posix(), LAB_CAP="",
               PY=(stubs / "python-stub").as_posix(), PATH=str(stubs) + os.pathsep + os.environ["PATH"])
    result = subprocess.run([shutil.which("bash"), str(runner), "test", "bash", "-c", f"exit {command_rc}"],
                            env=env, capture_output=True, text=True, encoding="utf-8", timeout=15)
    assert result.returncode == command_rc, result.stderr
    assert "[CARD]" in result.stderr
    assert not [p for p in leases.iterdir() if not p.name.startswith(".")]
    assert not (leases / ".guard").exists()

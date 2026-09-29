import os
from pathlib import Path
import shutil
import subprocess

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts/gpu_lease.sh"


def run(tmp_path, code, cap=""):
    env = dict(os.environ, SCRIPT=SCRIPT.as_posix(), LEASE=tmp_path.as_posix(),
               LEASE_HOST="test-host", LAB_CAP=cap)
    return subprocess.run([shutil.which("bash"), "-c", 'source "$SCRIPT"; ' + code],
                          env=env, capture_output=True, text=True, encoding="utf-8", timeout=15)


def test_default_is_six(tmp_path):
    result = run(tmp_path, 'gpu_policy_init && gpu_global_cap')
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "6"


def test_parameter_overrides_environment_and_is_shared(tmp_path):
    assert run(tmp_path, 'gpu_policy_init 3', cap="8").returncode == 0
    assert run(tmp_path, 'gpu_global_cap').stdout.strip() == "3"


def test_environment_sets_global_authorization(tmp_path):
    assert run(tmp_path, 'gpu_policy_init', cap="8").returncode == 0
    assert run(tmp_path, 'gpu_global_cap').stdout.strip() == "8"
    assert run(tmp_path, 'gpu_policy_init default && gpu_global_cap').stdout.strip() == "6"


@pytest.mark.parametrize("cap", ["0", "-1", "six", "2.5"])
def test_invalid_authorization_is_rejected(tmp_path, cap):
    assert run(tmp_path, 'gpu_policy_init', cap=cap).returncode == 2


def test_all_job_names_and_hosts_count_toward_authorized_limit(tmp_path):
    (tmp_path / "remote-0").write_text("remote 1 unrelated\n")
    (tmp_path / "remote-1").write_text("remote 2 q_other\n")
    result = run(tmp_path, '''gpu_policy_init 2 || exit $?
lease_card_available() { return 0; }
lease_acquire 0 style_prompt_work 6
''')
    assert result.returncode == 4, result.stderr

"""以 Bash stub 驗證跨程序取卡；不呼叫 CUDA 或 nvidia-smi。"""
import os
from pathlib import Path
import shutil
import subprocess
import time

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts/gpu_lease.sh"
BASH = shutil.which("bash")


def bash(tmp_path, code, **environment):
    env = dict(os.environ, LEASE=tmp_path.as_posix(), LEASE_HOST="test-host",
               SCRIPT=SCRIPT.as_posix(), **environment)
    return subprocess.run([BASH, "-c", 'source "$SCRIPT"; ' + code],
                          env=env, capture_output=True, text=True, encoding="utf-8", timeout=20)


@pytest.mark.parametrize("same_card,cap,expected", [(True, 3, 1), (False, 3, 3)])
def test_concurrent_claims_respect_card_and_global_capacity(tmp_path, same_card, cap, expected):
    leases = tmp_path / "leases"
    leases.mkdir()
    release = tmp_path / "release"
    children = []
    for i in range(8):
        env = dict(os.environ, LEASE=leases.as_posix(), LEASE_HOST="test-host",
                   SCRIPT=SCRIPT.as_posix(), GPU=str(0 if same_card else i), CAP=str(cap),
                   RESULT=(tmp_path / f"result_{i}").as_posix(), RELEASE=release.as_posix())
        code = '''source "$SCRIPT"
lease_card_available() { return 0; }
if lease_acquire "$GPU" "job_$GPU" "$CAP"; then
  trap 'lease_release "$GPU"' EXIT
  echo yes > "$RESULT"
  while [ ! -e "$RELEASE" ]; do sleep 0.05; done
else
  echo no > "$RESULT"
fi
'''
        children.append(subprocess.Popen([BASH, "-c", code], env=env,
                                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8"))
    try:
        deadline = time.monotonic() + 15
        while len(list(tmp_path.glob("result_*"))) != len(children) and time.monotonic() < deadline:
            time.sleep(0.02)
        assert len(list(tmp_path.glob("result_*"))) == len(children)
        assert sum(p.read_text().strip() == "yes" for p in tmp_path.glob("result_*")) == expected
        assert len([p for p in leases.iterdir() if p.is_file() and not p.name.startswith(".")]) == expected
    finally:
        release.touch()
        for child in children:
            stdout, stderr = child.communicate(timeout=20)
            assert child.returncode == 0, stdout + stderr
    assert not list(leases.glob("test-host-*"))


def test_release_never_removes_another_owners_lease(tmp_path):
    result = bash(tmp_path, '''lease_card_available() { return 0; }
lease_acquire 0 first 3 || exit 1
printf '%s %s %s %s\n' "$LEASE_HOST" "$BASHPID" second replacement > "$LEASE/$LEASE_HOST-0"
lease_release 0 && exit 2
test -f "$LEASE/$LEASE_HOST-0"
''')
    assert result.returncode == 0, result.stderr


def test_remote_and_legacy_leases_count_toward_limit(tmp_path):
    (tmp_path / "remote-0").write_text("other-host 1 legacy\n")
    result = bash(tmp_path, 'lease_card_available() { return 0; }; lease_acquire 1 new 1')
    assert result.returncode == 4, result.stderr


def test_card_check_failure_creates_no_lease(tmp_path):
    result = bash(tmp_path, 'lease_card_available() { return 1; }; lease_acquire 0 job 3')
    assert result.returncode == 4
    assert not list(tmp_path.glob("test-host-*"))

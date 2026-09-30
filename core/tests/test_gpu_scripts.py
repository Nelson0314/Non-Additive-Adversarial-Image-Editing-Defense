"""GPU 租約工具的跨程序契約；nvidia-smi 與 CUDA 檢查均以 stub 取代。

來源為 lab/tests 的 test_gpu_leases.py、test_gpu_capacity.py、test_run_on_card_script.py
與 test_queue_validation.py 的 queue 部分；另加注入式 runner／validator／depends 的佇列案例。
"""
import os
from pathlib import Path
import shutil
import subprocess
import time

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
LEASE_SCRIPT = SCRIPTS / "gpu_lease.sh"
BASH = shutil.which("bash")

NVIDIA_SMI = '''#!/usr/bin/env bash
case "$*" in
  *--query-gpu=index,uuid,memory.used*) printf '%s\\n' "0, GPU-a, 0 MiB" "1, GPU-b, 9000 MiB" ;;
  *--query-gpu=uuid*) echo GPU-a ;;
  *--query-compute-apps=gpu_uuid,pid,used_memory*) echo "GPU-b, 999999, 8000 MiB" ;;
  *--query-compute-apps*) echo "GPU-b, 8000" ;;
esac
'''


def stub_bin(tmp_path):
    stubs = tmp_path / "bin"
    stubs.mkdir()
    for name, text in {"nvidia-smi": NVIDIA_SMI,
                       "python-stub": "#!/usr/bin/env bash\nexit 0\n"}.items():
        path = stubs / name
        path.write_text(text, newline="\n")
        path.chmod(0o755)
    return stubs


def stub_env(tmp_path, **extra):
    stubs = stub_bin(tmp_path)
    leases = tmp_path / "leases"
    return dict(os.environ, LEASE=leases.as_posix(), LEASE_HOST="test-host", GPU_CAP="",
                USER=os.environ.get("USER", "root"), PY=(stubs / "python-stub").as_posix(),
                PATH=str(stubs) + os.pathsep + os.environ["PATH"], **extra), leases


def bash(tmp_path, code, cap="", **environment):
    env = dict(os.environ, LEASE=tmp_path.as_posix(), LEASE_HOST="test-host",
               SCRIPT=LEASE_SCRIPT.as_posix(), GPU_CAP=cap, **environment)
    return subprocess.run([BASH, "-c", 'source "$SCRIPT"; ' + code],
                          env=env, capture_output=True, text=True, encoding="utf-8", timeout=20)


def test_scripts_use_lf_line_endings():
    for path in SCRIPTS.glob("*.sh"):
        assert b"\r\n" not in path.read_bytes(), path


# ---- 容量政策 ----

def test_default_is_six(tmp_path):
    result = bash(tmp_path, 'gpu_policy_init && gpu_global_cap')
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "6"


def test_parameter_overrides_environment_and_is_shared(tmp_path):
    assert bash(tmp_path, 'gpu_policy_init 3', cap="8").returncode == 0
    assert bash(tmp_path, 'gpu_global_cap').stdout.strip() == "3"


def test_environment_sets_global_authorization(tmp_path):
    assert bash(tmp_path, 'gpu_policy_init', cap="8").returncode == 0
    assert bash(tmp_path, 'gpu_global_cap').stdout.strip() == "8"
    assert bash(tmp_path, 'gpu_policy_init default && gpu_global_cap').stdout.strip() == "6"


@pytest.mark.parametrize("cap", ["0", "-1", "six", "2.5"])
def test_invalid_authorization_is_rejected(tmp_path, cap):
    assert bash(tmp_path, 'gpu_policy_init', cap=cap).returncode == 2


def test_all_job_names_and_hosts_count_toward_authorized_limit(tmp_path):
    (tmp_path / "remote-0").write_text("remote 1 unrelated\n")
    (tmp_path / "remote-1").write_text("remote 2 q_other\n")
    result = bash(tmp_path, '''gpu_policy_init 2 || exit $?
lease_card_available() { return 0; }
lease_acquire 0 style_prompt_work 6
''')
    assert result.returncode == 4, result.stderr


# ---- 租約 ----

@pytest.mark.parametrize("same_card,cap,expected", [(True, 3, 1), (False, 3, 3)])
def test_concurrent_claims_respect_card_and_global_capacity(tmp_path, same_card, cap, expected):
    leases = tmp_path / "leases"
    leases.mkdir()
    release = tmp_path / "release"
    children = []
    for i in range(8):
        env = dict(os.environ, LEASE=leases.as_posix(), LEASE_HOST="test-host",
                   SCRIPT=LEASE_SCRIPT.as_posix(), GPU=str(0 if same_card else i), CAP=str(cap),
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
        children.append(subprocess.Popen([BASH, "-c", code], env=env, stdout=subprocess.PIPE,
                                         stderr=subprocess.PIPE, text=True, encoding="utf-8"))
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
printf '%s %s %s %s\\n' "$LEASE_HOST" "$BASHPID" second replacement > "$LEASE/$LEASE_HOST-0"
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


# ---- 空卡查詢 ----

def test_free_cards_excludes_foreign_memory_and_asserts(tmp_path):
    env, _ = stub_env(tmp_path)
    run = lambda *a: subprocess.run([BASH, str(SCRIPTS / "free_cards.sh"), *a], env=env,
                                    capture_output=True, text=True, encoding="utf-8", timeout=15)
    listed = run()
    assert listed.returncode == 0, listed.stderr
    assert listed.stdout.split() == ["0"]
    assert run("--assert", "0").returncode == 0
    assert run("--assert", "1").returncode == 3


# ---- 派工入口 ----

@pytest.mark.parametrize("command_rc", [0, 23])
def test_command_exit_releases_owned_lease(tmp_path, command_rc):
    env, leases = stub_env(tmp_path)
    work = tmp_path / "work"
    work.mkdir()
    result = subprocess.run(
        [BASH, str(SCRIPTS / "run_with_gpu_lease.sh"), "--work-dir", str(work), "test",
         "bash", "-c", f'pwd > where; exit {command_rc}'],
        env=env, capture_output=True, text=True, encoding="utf-8", timeout=15)
    assert result.returncode == command_rc, result.stderr
    assert "[CARD]" in result.stderr and "gpu=0" in result.stderr
    assert (work / "where").read_text().strip() == str(work)
    assert not [p for p in leases.iterdir() if not p.name.startswith(".")]
    assert not (leases / ".guard").exists()


def test_missing_workdir_is_rejected(tmp_path):
    env, _ = stub_env(tmp_path)
    result = subprocess.run([BASH, str(SCRIPTS / "run_with_gpu_lease.sh"), "test", "true"],
                            env=env, capture_output=True, text=True, encoding="utf-8", timeout=15)
    assert result.returncode == 2
    assert "--work-dir" in result.stderr


# ---- 佇列 ----

def write_tool(path, body):
    path.write_text("#!/usr/bin/env bash\n" + body, newline="\n")
    path.chmod(0o755)
    return path


def run_queue(tmp_path, jobs, validator_body, depends_body=None):
    env, leases = stub_env(tmp_path, POLL="0.05", LAUNCH_GAP="0", MAXFAIL="1")
    work, tools = tmp_path / "work", tmp_path / "tools"
    work.mkdir()
    tools.mkdir()
    runner = write_tool(tools / "runner", 'echo "$2" > "done_$1"; echo "$1" >> order\n')
    validator = write_tool(tools / "validator", validator_body)
    command = [BASH, str(SCRIPTS / "queue_worker.sh"), "--work-dir", str(work),
               "--state-dir", str(tmp_path / "state"), "--log-dir", str(tmp_path / "logs"),
               "--runner", str(runner), "--validator", str(validator)]
    if depends_body is not None:
        command += ["--depends", str(write_tool(tools / "depends", depends_body))]
    result = subprocess.run(command + ["test", *jobs], env=env, capture_output=True, text=True,
                            encoding="utf-8", timeout=60)
    return result, work, leases


def test_queue_runs_dependencies_first_and_validates(tmp_path):
    depends = 'if [ "$1" = second ]; then echo first; fi\nexit 0\n'
    result, work, leases = run_queue(tmp_path, ["second", "first"],
                                     '[ -e "done_$1" ]\n', depends)
    assert result.returncode == 0, result.stderr
    assert (work / "order").read_text().split() == ["first", "second"]
    assert (tmp_path / "state/first.done").exists() and (tmp_path / "state/second.done").exists()
    assert (work / "done_first").read_text().strip() == "0"
    assert not [p for p in leases.iterdir() if not p.name.startswith(".")]


def test_zero_exit_without_valid_output_is_not_done(tmp_path):
    result, _, _ = run_queue(tmp_path, ["only"], 'exit 9\n')
    assert result.returncode == 1
    assert not (tmp_path / "state/only.done").exists()
    assert (tmp_path / "state/only.GIVEUP").exists()
    assert not (tmp_path / "state/only.lock").exists()


def test_given_up_upstream_blocks_dependents(tmp_path):
    depends = 'if [ "$1" = second ]; then echo first; fi\nexit 0\n'
    result, work, _ = run_queue(tmp_path, ["first", "second"],
                                '[ "$1" != first ] && [ -e "done_$1" ]\n', depends)
    assert result.returncode == 1
    assert (work / "order").read_text().split() == ["first"]


def test_invalid_dependency_status_stops_worker(tmp_path):
    result, work, _ = run_queue(tmp_path, ["only"], 'exit 0\n', 'exit 5\n')
    assert result.returncode == 2
    assert "相依指令" in result.stderr
    assert not (work / "order").exists()


def test_queue_requires_injected_validator(tmp_path):
    env, _ = stub_env(tmp_path)
    result = subprocess.run([BASH, str(SCRIPTS / "queue_worker.sh"), "--work-dir", str(tmp_path),
                             "--state-dir", str(tmp_path / "s"), "--log-dir", str(tmp_path / "l"),
                             "--runner", "true", "test", "job"],
                            env=env, capture_output=True, text=True, encoding="utf-8", timeout=15)
    assert result.returncode == 2
    assert "--validator" in result.stderr


# ---- 暫時性嘗試 ----

def trial_project(tmp_path):
    project = tmp_path / "project"
    (project / "vendor/scripts").mkdir(parents=True)
    (project / "docs").mkdir()
    shutil.copyfile(SCRIPTS / "trial.sh", project / "vendor/scripts/trial.sh")
    (project / "pyproject.toml").write_text("[project]\nname = 'x'\n")
    (project / ".gitignore").write_text("/trials/\n")
    (project / "docs/TRIALS.md").write_text("| 名稱 | 試了什麼 | 設定 | 關鍵數字 | 結論來源 |\n|---|---|---|---|---|\n")
    git = lambda *a: subprocess.run(["git", *a], cwd=project, check=True, capture_output=True)
    git("init", "-q")
    git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "init", "--allow-empty")
    git("add", ".")
    git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "files")
    stubs = tmp_path / "bin"
    stubs.mkdir()
    write_tool(stubs / "ssh", 'echo "$@" >> "$SSH_LOG"\n')
    env = dict(os.environ, PATH=str(stubs) + os.pathsep + os.environ["PATH"],
               SSH_LOG=str(tmp_path / "ssh.log"))
    env.pop("TRIAL_REMOTE", None)
    env.pop("TRIAL_REMOTE_ROOT", None)
    run = lambda *a, **e: subprocess.run([BASH, str(project / "vendor/scripts/trial.sh"), *a],
                                         env=dict(env, **e), capture_output=True, text=True,
                                         encoding="utf-8", timeout=15)
    return project, run, git


def test_trial_new_and_promote_require_committed_content(tmp_path):
    project, run, git = trial_project(tmp_path)
    assert run("new", "bad-name").returncode == 2
    assert run("new", "warm_grade").returncode == 0
    assert (project / "trials/warm_grade/README.md").is_file()
    assert run("new", "warm_grade").returncode == 1
    (project / "configs.yaml").write_text("x: 1\n")
    result = run("promote", "warm_grade")
    assert result.returncode == 1 and "configs.yaml" in result.stderr
    git("add", "configs.yaml")
    git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "promote")
    assert run("promote", "warm_grade").returncode == 0
    assert not (project / "trials/warm_grade").exists()


def test_trial_drop_requires_ledger_row_and_remote(tmp_path):
    project, run, _ = trial_project(tmp_path)
    run("new", "cold_grade")
    result = run("drop", "cold_grade")
    assert result.returncode == 1 and "TRIALS.md" in result.stderr
    with (project / "docs/TRIALS.md").open("a") as ledger:
        ledger.write("| `cold_grade` | 冷色調 | lr 0.02 | LPIPS 0.1 | trials 已刪 |\n")
    assert run("drop", "cold_grade").returncode == 2
    assert (project / "trials/cold_grade").exists()
    result = run("drop", "cold_grade", TRIAL_REMOTE="-p 1 u@h", TRIAL_REMOTE_ROOT="/r/color")
    assert result.returncode == 0, result.stderr
    assert "/r/color/trials/cold_grade" in (tmp_path / "ssh.log").read_text()
    assert not (project / "trials/cold_grade").exists()
    run("new", "cold_grade")
    assert run("drop", "cold_grade", "--local-only").returncode == 0

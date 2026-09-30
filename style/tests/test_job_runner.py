"""`scripts/run_style_prompt_jobs.sh` 的完成判定；GPU、模型與讀數以替身取代，在暫存的專案副本中執行。"""
import csv
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
BASH = shutil.which("bash")

NVIDIA_SMI = '''#!/usr/bin/env bash
case "$*" in
  *--query-gpu=index,uuid,memory.used*) printf '%s\\n' "0, GPU-a, 0 MiB" "1, GPU-b, 0 MiB" ;;
  *--query-gpu=uuid*) echo GPU-a ;;
  *--query-compute-apps*) : ;;
esac
'''

# 替身直譯器：-c（CUDA 檢查）直接成功；evaluate_job_outputs 交給真的模組；其餘三個入口寫出最小輸出。
STUB = r'''
import csv, os, runpy, sys
from pathlib import Path

args = sys.argv[1:]
if args[0] == "-c":
    raise SystemExit(1 if os.environ.get("FAIL_CUDA") else 0)
module, rest = args[1], args[2:]
if module.endswith("evaluate_job_outputs"):
    sys.argv = [module, *rest]
    runpy.run_module(module, run_name="__main__")

def values(flag):
    i = rest.index(flag) + 1
    out = []
    while i < len(rest) and not rest[i].startswith("--"):
        out.append(rest[i]); i += 1
    return out

def write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, list(rows[0]))
        w.writeheader(); w.writerows(rows)

if module.endswith("generate_style_prompt_defenses"):
    out = Path(values("--output-dir")[0]); style = values("--styles")[0]
    if out.name in os.environ.get("FAIL_OPT", "").split(","):
        raise SystemExit(3)
    rows = []
    for name in values("--images"):
        out.mkdir(parents=True, exist_ok=True)
        (out / f"{name}__{style}_si2__def.png").write_bytes(b"png")
        rows.append({"image": name, "style": style})
    write(out / "results.csv", rows)
elif module.endswith("run_edits"):
    out = Path(values("--output-dir")[0]); arm = "ip2p" + values("--suffix")[0]
    rows = []
    for name in values("--images"):
        for k in range(4):
            (out / arm).mkdir(parents=True, exist_ok=True)
            (out / arm / f"{name}__p{k}.png").write_bytes(b"png")
            rows.append({"scenario": "ip2p", "arm": arm, "image": name, "prompt_index": k})
    write(out / "preflight.csv", rows)
elif module.endswith("measure_style_prompt_edits"):
    if os.environ.get("FAIL_READOUT"):
        raise SystemExit(4)
    style = values("--styles")[0]; root = Path(values("--edits-root")[0])
    rows = []
    for strength in ["ref", *values("--strengths")]:
        with (root / f"{strength}_{style}" / "preflight.csv").open(encoding="utf-8") as f:
            for r in csv.DictReader(f):
                rows.append({"style": style, "strength": strength, "image": r["image"],
                             "prompt_index": r["prompt_index"]})
    write(Path(values("--output-csv")[0]), rows)
else:
    raise SystemExit(f"unexpected module {module}")
'''

SPEC = """ref man_01+woman_02 cool_grade --lr 0 --updates 1
alpha man_01 cool_grade --lr 0.1
beta woman_02 cool_grade --lr 0.2
"""


@pytest.fixture
def project(tmp_path):
    copy = tmp_path / "style"
    for part in ("scripts", "src", "vendor"):
        shutil.copytree(ROOT / part, copy / part, ignore=shutil.ignore_patterns("__pycache__"))
    (copy / "data/portraits").mkdir(parents=True)
    shutil.copyfile(ROOT / "data/portraits/prompts.yaml", copy / "data/portraits/prompts.yaml")
    stubs = tmp_path / "bin"
    stubs.mkdir()
    (stubs / "nvidia-smi").write_text(NVIDIA_SMI, newline="\n", encoding="utf-8")
    (stubs / "stub.py").write_text(STUB, encoding="utf-8", newline="\n")
    (stubs / "python-stub").write_text(
        f'#!/usr/bin/env bash\nexec "{Path(sys.executable).as_posix()}" "{(stubs / "stub.py").as_posix()}" "$@"\n',
        newline="\n", encoding="utf-8")
    for name in ("nvidia-smi", "python-stub"):
        (stubs / name).chmod(0o755)
    (copy / "jobs.spec").write_text(SPEC, newline="\n", encoding="utf-8")
    env = dict(os.environ, PY=(stubs / "python-stub").as_posix(), LEASE=(tmp_path / "leases").as_posix(),
               LEASE_HOST="test-host", GPU_CAP="", POLL="0.2",
               PATH=str(stubs) + os.pathsep + os.environ["PATH"])
    env.pop("ENV_FILE", None)
    env.pop("PYTHONPATH", None)

    def run(spec="jobs.spec", **extra):
        return subprocess.run([BASH, (copy / "scripts/run_style_prompt_jobs.sh").as_posix(), "exp", spec, "2"],
                              cwd=copy, env=dict(env, **extra), capture_output=True, text=True,
                              encoding="utf-8", timeout=120)
    return copy, run


def readout(copy):
    with (copy / "artifacts/edits/exp/readout_cool_grade.csv").open(encoding="utf-8") as f:
        return {(r["strength"], r["image"], r["prompt_index"]) for r in csv.DictReader(f)}


def test_complete_run_validates_every_stage(project):
    copy, run = project
    result = run()
    assert result.returncode == 0, result.stdout + result.stderr
    assert "exp_DONE" in result.stdout
    keys = readout(copy)
    assert len(keys) == 4 * (2 + 1 + 1)
    assert (copy / "artifacts/defenses/exp/alpha/job.spec").read_text(encoding="utf-8").split()[-2:] == ["--lr", "0.1"]
    again = run()
    assert again.returncode == 0 and "launched" not in again.stdout


def test_failed_optimization_ends_the_run_with_failure(project):
    copy, run = project
    result = run(FAIL_OPT="beta")
    assert result.returncode == 1, result.stdout + result.stderr
    assert "exp_DONE" not in result.stdout
    assert "beta FAILED" in result.stderr
    assert not {k for k in readout(copy) if k[0] == "beta"}


def test_failed_readout_is_not_reported_done(project):
    _, run = project
    result = run(FAIL_READOUT="1")
    assert result.returncode == 1
    assert "exp_DONE" not in result.stdout
    assert "讀數失敗" in result.stderr


def test_outputs_from_other_settings_stop_the_run(project):
    copy, run = project
    assert run().returncode == 0
    (copy / "jobs.spec").write_text(SPEC.replace("--lr 0.1", "--lr 0.05"), newline="\n", encoding="utf-8")
    result = run()
    assert result.returncode == 2
    assert "job.spec 不符" in result.stderr


def test_spec_without_reference_job_is_rejected(project):
    copy, run = project
    (copy / "noref.spec").write_text("alpha man_01 cool_grade --lr 0.1\n", newline="\n", encoding="utf-8")
    result = run("noref.spec")
    assert result.returncode == 2
    assert "ref" in result.stderr
    assert not (copy / "artifacts/defenses/exp/alpha").exists()


def test_launch_failure_ends_the_run_instead_of_retrying(project):
    """啟動程序結束卻沒有結束碼（此處為 CUDA 檢查失敗）時，工作記為失敗，排程結束而不是反覆重派。"""
    _, run = project
    result = run(FAIL_CUDA="1")
    assert result.returncode == 1, result.stdout + result.stderr
    assert "exp_DONE" not in result.stdout
    assert "啟動失敗" in result.stderr


def test_unusable_detach_command_is_reported(project, tmp_path):
    """setsid 無法執行（例如 Windows 上缺少該指令）時，排程回報啟動失敗並結束，不會等到逾時。"""
    _, run = project
    broken = tmp_path / "broken"
    broken.mkdir()
    (broken / "setsid").write_text('#!/usr/bin/env bash\necho "setsid: command not found" >&2\nexit 127\n',
                                   newline="\n", encoding="utf-8")
    (broken / "setsid").chmod(0o755)
    result = run(PATH=os.pathsep.join([str(broken), str(tmp_path / "bin"), os.environ["PATH"]]))
    assert result.returncode == 1, result.stdout + result.stderr
    assert "啟動失敗" in result.stderr

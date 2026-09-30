import csv
import os
from pathlib import Path
import shutil
import subprocess

import pytest

from immunization_color.cli import validate_queue_job as validation

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def write_table(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


@pytest.mark.parametrize("problem", ["missing", "duplicate", "unexpected", "empty_metric", "nan", "missing_column"])
def test_bad_table_cannot_mark_job_done(tmp_path, problem):
    rows = [{"image": "a", "metric": "0.1"}, {"image": "b", "metric": "0.2"}]
    if problem == "missing":
        rows.pop()
    elif problem == "duplicate":
        rows[1] = rows[0]
    elif problem == "unexpected":
        rows[1]["image"] = "c"
    elif problem == "empty_metric":
        rows[1]["metric"] = ""
    elif problem == "nan":
        rows[1]["metric"] = "nan"
    else:
        rows = [{"image": "a"}, {"image": "b"}]
    path = tmp_path / "result.csv"
    write_table(path, rows)
    with pytest.raises(ValueError):
        validation.validate_table(path, ("image",), {("a",), ("b",)}, ("metric",))


def test_complete_keys_and_required_values_pass(tmp_path):
    path = tmp_path / "result.csv"
    write_table(path, [{"image": "a", "metric": 0, "psnr": "inf"}])
    assert len(validation.validate_table(path, ("image",), {("a",)}, ("metric", "psnr"))) == 1


def test_fidelity_requires_explicit_arms(tmp_path):
    with pytest.raises(ValueError, match="FID_ARMS"):
        validation.validate_job(tmp_path, "fid")


def depends(tmp_path, job, jobs, **environment):
    env = dict(os.environ, **environment)
    return subprocess.run([shutil.which("bash"), str(SCRIPTS / "queue_depends.sh"), job, *jobs],
                          env=env, capture_output=True, text=True, encoding="utf-8", timeout=15)


JOBS = ["pilot:color:man_00:10", "def:color:man_00", "def:color_simple:man_00",
        "chain:color", "chain:color_simple", "readout", "fid"]


@pytest.mark.parametrize("job,expected", [
    ("def:color:man_00", ["pilot:color:man_00:10"]),
    ("chain:color", ["def:color:man_00"]),
    ("readout", ["chain:color", "chain:color_simple"]),
    ("fid", []),
    ("pilot:color:man_00:10", []),
])
def test_queue_dependencies_follow_job_grammar(tmp_path, job, expected):
    result = depends(tmp_path, job, JOBS)
    assert result.returncode == 0, result.stderr
    assert result.stdout.split() == expected


def test_readout_waits_for_external_condition_sentinels(tmp_path):
    state = SCRIPTS.parent / "runtime/state"
    sentinel = state / "queue_test_arm.chain.done"
    assert depends(tmp_path, "readout", JOBS, WAIT_ARMS="queue_test_arm").returncode == 1
    state.mkdir(parents=True, exist_ok=True)
    try:
        sentinel.touch()
        assert depends(tmp_path, "fid", JOBS, WAIT_ARMS="queue_test_arm").returncode == 0
    finally:
        sentinel.unlink()


def test_fid_queue_requires_explicit_arms(tmp_path):
    env = dict(os.environ)
    env.pop("FID_ARMS", None)
    result = subprocess.run([shutil.which("bash"), str(SCRIPTS / "run_queue.sh"), "test", "fid"],
                            env=env, capture_output=True, text=True, encoding="utf-8", timeout=15)
    assert result.returncode == 2
    assert "FID_ARMS" in result.stderr

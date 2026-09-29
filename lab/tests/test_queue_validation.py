import csv
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess

import pytest

path = Path(__file__).parents[1] / "code/validate_job.py"
spec = importlib.util.spec_from_file_location("validate_job_test", path)
validation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validation)


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


@pytest.mark.parametrize("validation_rc,done", [(0, True), (9, False)])
def test_zero_process_exit_still_requires_output_validation(tmp_path, validation_rc, done):
    script = (Path(__file__).parents[1] / "scripts/queue_worker.sh").read_text(encoding="utf-8")
    launch = script[script.index("launch() {"):script.index('\nfor job in "${JOBS[@]}"')]
    queue, logs, leases = tmp_path / "queue", tmp_path / "logs", tmp_path / "leases"
    queue.mkdir()
    logs.mkdir()
    leases.mkdir()
    (queue / "readout.lock").mkdir()
    env = dict(os.environ, Q=queue.as_posix(), LOGDIR=logs.as_posix(), LEASE=leases.as_posix(),
               HOST="test", QNAME="test", CAP="2", MYCAP="2", MAXFAIL="3")
    stubs = f'''key() {{ echo "$1"; }}
lease_acquire() {{ return 0; }}
lease_release() {{ return 0; }}
gpu_global_cap() {{ echo 2; }}
log() {{ :; }}
sleep() {{ :; }}
run_job() {{ return 0; }}
validate_job() {{ return {validation_rc}; }}
'''
    result = subprocess.run([shutil.which("bash"), "-c", stubs + launch + '\nlaunch readout 0\nwait\n'],
                            env=env, capture_output=True, text=True, encoding="utf-8", timeout=15)
    assert result.returncode == 0, result.stderr
    assert (queue / "readout.done").exists() == done
    assert (queue / "readout.fails").exists() != done
    assert not (queue / "readout.lock").exists()

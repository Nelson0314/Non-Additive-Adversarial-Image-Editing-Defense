"""`scripts/freeze_env.py`：寫出的 requirements.lock 與目前環境一致，偏離時 --check 失敗。"""
import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/freeze_env.py"
spec = importlib.util.spec_from_file_location("freeze_env", SCRIPT)
freeze_env = importlib.util.module_from_spec(spec)
spec.loader.exec_module(freeze_env)


def test_lock_round_trip_and_drift(tmp_path, capsys):
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'x'\n")
    freeze_env.main(["--project-root", str(tmp_path)])
    lock = tmp_path / "requirements.lock"
    assert b"\r\n" not in lock.read_bytes()
    assert not any(line.lower().startswith("immunization") for line in freeze_env.packages(lock))
    with pytest.raises(SystemExit) as done:
        freeze_env.main(["--project-root", str(tmp_path), "--check"])
    assert done.value.code == 0
    with lock.open("a", encoding="utf-8") as stream:
        stream.write("not-installed-package==0.0.1\n")
    with pytest.raises(SystemExit) as drift:
        freeze_env.main(["--project-root", str(tmp_path), "--check"])
    assert drift.value.code == 1
    assert "- not-installed-package==0.0.1" in capsys.readouterr().out


def test_requires_project_root(tmp_path):
    with pytest.raises(SystemExit, match="pyproject.toml"):
        freeze_env.main(["--project-root", str(tmp_path)])

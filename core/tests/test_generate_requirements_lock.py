"""`scripts/generate_requirements_lock.py`：寫出的 requirements.lock 與目前環境一致，偏離時 --check 失敗。"""
import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/generate_requirements_lock.py"
spec = importlib.util.spec_from_file_location("generate_requirements_lock", SCRIPT)
generate_requirements_lock = importlib.util.module_from_spec(spec)
spec.loader.exec_module(generate_requirements_lock)


def test_lock_round_trip_and_drift(tmp_path, capsys):
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'x'\n", encoding="utf-8")
    generate_requirements_lock.main(["--project-root", str(tmp_path)])
    lock = tmp_path / "requirements.lock"
    assert b"\r\n" not in lock.read_bytes()
    assert not any(line.lower().startswith("immunization") for line in generate_requirements_lock.packages(lock))
    with pytest.raises(SystemExit) as done:
        generate_requirements_lock.main(["--project-root", str(tmp_path), "--check"])
    assert done.value.code == 0
    with lock.open("a", encoding="utf-8") as stream:
        stream.write("not-installed-package==0.0.1\n")
    with pytest.raises(SystemExit) as drift:
        generate_requirements_lock.main(["--project-root", str(tmp_path), "--check"])
    assert drift.value.code == 1
    assert "- not-installed-package==0.0.1" in capsys.readouterr().out


def test_requires_project_root(tmp_path):
    with pytest.raises(SystemExit, match="pyproject.toml"):
        generate_requirements_lock.main(["--project-root", str(tmp_path)])


def test_uses_uv_when_interpreter_has_no_pip(monkeypatch):
    monkeypatch.setattr(generate_requirements_lock.importlib.util, "find_spec", lambda name: None)
    monkeypatch.setattr(generate_requirements_lock.shutil, "which", lambda name: "/opt/bin/uv" if name == "uv" else None)
    assert generate_requirements_lock.freeze_command() == ["/opt/bin/uv", "pip", "freeze", "--python", generate_requirements_lock.sys.executable]
    assert generate_requirements_lock.tool_name() == "uv pip freeze"


def test_prefers_pip_when_available(monkeypatch):
    monkeypatch.setattr(generate_requirements_lock.shutil, "which", lambda name: "/opt/bin/uv")
    assert generate_requirements_lock.freeze_command()[1:] == ["-m", "pip", "freeze", "--all"]


def test_stops_without_pip_or_uv(monkeypatch):
    monkeypatch.setattr(generate_requirements_lock.importlib.util, "find_spec", lambda name: None)
    monkeypatch.setattr(generate_requirements_lock.shutil, "which", lambda name: None)
    with pytest.raises(SystemExit, match="uv"):
        generate_requirements_lock.freeze_command()

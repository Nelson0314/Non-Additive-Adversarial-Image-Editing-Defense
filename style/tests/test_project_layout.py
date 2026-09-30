"""專案自足性：預設目錄只在本專案內，vendor 快照與 lock 一致。"""
import importlib.util
from pathlib import Path

from immunization_style import layout

ROOT = Path(__file__).resolve().parents[1]


def test_default_directories_stay_inside_the_project():
    assert layout.PROJECT == ROOT
    for name in dir(layout):
        value = getattr(layout, name)
        if isinstance(value, Path):
            assert value.is_relative_to(ROOT), (name, value)


def test_vendor_snapshot_matches_lock():
    spec = importlib.util.spec_from_file_location("generate_vendor_snapshot", ROOT / "vendor/scripts/generate_vendor_snapshot.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.check(ROOT)


def test_core_is_imported_from_vendor():
    import immunization_core
    assert Path(immunization_core.__file__).resolve().is_relative_to(ROOT / "vendor")


def test_shell_scripts_use_lf_line_endings():
    for path in list((ROOT / "scripts").glob("*.sh")) + list((ROOT / "vendor/scripts").glob("*.sh")):
        assert b"\r\n" not in path.read_bytes(), path

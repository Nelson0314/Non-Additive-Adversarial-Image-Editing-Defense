"""單獨複製 core 後的匯入契約，不讀兄弟專案或載入模型。"""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def test_every_public_module_imports_from_isolated_copy(tmp_path):
    source = Path(__file__).resolve().parents[1]
    isolated = tmp_path / "delivery"
    shutil.copytree(source, isolated, ignore=shutil.ignore_patterns(
        "__pycache__", "*.egg-info", ".pytest_cache", "pytest-of-*", ".pytest_tmp_*", "build"))
    assert {p.name for p in tmp_path.iterdir()} == {"delivery"}
    assert not (isolated / "anti-purification").exists()
    assert not (isolated / "lab").exists()
    script = r'''
import importlib
import json
from pathlib import Path
import pkgutil
import socket
import sys
import torch

def forbidden(*args, **kwargs):
    raise AssertionError("Module import attempted network access or CUDA initialization")

socket.create_connection = forbidden
socket.socket.connect = forbidden
torch.hub.load_state_dict_from_url = forbidden
torch.cuda.init = forbidden
torch.cuda._lazy_init = forbidden
assert not torch.cuda.is_initialized()
import immunization_core
expected = Path.cwd() / "src" / "immunization_core"
names = ["immunization_core"] + sorted(
    m.name for m in pkgutil.walk_packages(immunization_core.__path__, "immunization_core."))
for name in names:
    module = importlib.import_module(name)
    assert Path(module.__file__).resolve().is_relative_to(expected.resolve()), module.__file__
assert not any(name == "src" or name.startswith("src.") for name in sys.modules)
assert not torch.cuda.is_initialized()
print(json.dumps(names))
'''
    env = dict(os.environ, PYTHONPATH=str(isolated / "src"), CUDA_VISIBLE_DEVICES="",
               PYTHONNOUSERSITE="1", PYTHONIOENCODING="utf-8", HF_HUB_OFFLINE="1",
               TRANSFORMERS_OFFLINE="1")
    result = subprocess.run([sys.executable, "-c", script], cwd=isolated, env=env,
                            capture_output=True, text=True, encoding="utf-8", timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    imported = json.loads(result.stdout)
    assert "immunization_core.editors.stable_diffusion_xl" in imported
    assert "immunization_core.metrics.regional" in imported
    assert "immunization_core.runtime.device" in imported
    for name in ("pipelines.editing", "pipelines.displacement", "pipelines.retention",
                 "color.space", "color.difference", "color.uniformity",
                 "optimization.carrier", "optimization.instruction_free",
                 "optimization.attention", "color.shift"):
        assert "immunization_core." + name in imported
    assert len(imported) == len(list((isolated / "src/immunization_core").rglob("*.py")))

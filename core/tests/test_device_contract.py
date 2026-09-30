"""TF32 授權變數與骨幹／VAE 精度契約。"""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from immunization_core.runtime.device import resolve_precision


@pytest.mark.parametrize("requested,expected", [
    (torch.float32, (torch.float32, torch.float32)),
    (torch.float16, (torch.float16, torch.float32)),
    (torch.bfloat16, (torch.bfloat16, torch.bfloat16)),
])
def test_precision_keeps_vae_contract(requested, expected):
    assert resolve_precision(requested) == expected


def test_unsupported_precision_is_rejected():
    with pytest.raises(ValueError):
        resolve_precision(torch.float64)


@pytest.mark.parametrize("value,expected", [(None, False), ("0", False), ("1", True), ("true", False)])
def test_tf32_uses_only_explicit_new_environment_variable(value, expected):
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src"),
               WACV_ALLOW_TF32="1", CUDA_VISIBLE_DEVICES="", PYTHONIOENCODING="utf-8")
    env.pop("IMMUNIZATION_ALLOW_TF32", None)
    if value is not None:
        env["IMMUNIZATION_ALLOW_TF32"] = value
    code = ("import json,torch; from immunization_core.runtime.device import tf32_enabled; "
            "print(json.dumps([tf32_enabled(),torch.backends.cuda.matmul.allow_tf32,torch.cuda.is_initialized()]))")
    result = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True,
                            text=True, encoding="utf-8", timeout=30)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == [expected, expected, False]

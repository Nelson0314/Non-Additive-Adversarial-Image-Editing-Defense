"""擴散淨化的不動點項。

沒有權重的機器上只能測**接線**：預設關閉、旗鈕存在、缺權重時明確拋錯。
需要權重的行為測試以 `skipif` 標出，不靜默跳過。
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch

from src.defense.fixedpoint_loss import (
    _null_embedding, _scheduler_of, make_manifold_term,
)

ROOT = Path(__file__).resolve().parents[1]


def _load_runner():
    spec = importlib.util.spec_from_file_location(
        "ip2p_run_manifold", ROOT / "scripts" / "ip2p_run.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_defaults_are_off():
    args = _load_runner().build_parser().parse_args(["--out", "o", "--data", "d"])
    assert args.manifold_weight == 0.0
    assert args.manifold_only is False
    assert args.manifold_t == 100


def test_invalid_timestep_range_rejected():
    with pytest.raises(ValueError, match="t_min <= t_max"):
        make_manifold_term(object(), t_min=50, t_max=10)


def test_scheduler_lookup_refuses_to_guess():
    """**找不到排程要拋錯，不可以自己算 beta 表**——那會與攻擊方實際用的
    噪聲尺度不一致，而且不會有症狀。"""
    class Bare:
        pass
    with pytest.raises(AttributeError, match="不要用自己算的 beta 表"):
        _scheduler_of(Bare())


def test_null_embedding_refuses_without_tokenizer():
    class Bare:
        device = "cpu"
    with pytest.raises(AttributeError, match="找不到 tokenizer"):
        _null_embedding(Bare())

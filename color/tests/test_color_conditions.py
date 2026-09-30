"""保留條件的派工參數與共用選項契約；不載入或執行 GPU 模型。"""
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from immunization_color.cli.generate_color_defenses import build_parser

ROOT = Path(__file__).resolve().parents[1]
CONDITIONS = {
    "color": [],
    "color_simple": ["--caps", "simple"],
    "color_simple_skinbox": ["--caps", "simple", "--box", "skin"],
    "color_simple_xattn": ["--caps", "simple", "--objective", "xattn"],
}


def dispatch(tmp_path, arm, extra=(), output=None):
    stub = tmp_path / "capture-python"
    stub.write_text('#!/usr/bin/env bash\n'
                    f'[ "$2" = immunization_color.conditions ] && exec "{Path(sys.executable).as_posix()}" "$@"\n'
                    'printf "%s\\n" "$@"\n', newline="\n", encoding="utf-8")
    stub.chmod(0o755)
    env = dict(os.environ, PY=stub.as_posix())
    env.pop("DEF_OUT", None)
    if output is not None:
        env["DEF_OUT"] = output
    return subprocess.run([shutil.which("bash"), (ROOT / "scripts/generate_condition.sh").as_posix(), arm, *extra],
                          env=env, capture_output=True, text=True, encoding="utf-8", timeout=10)


@pytest.mark.parametrize("arm", CONDITIONS)
@pytest.mark.parametrize("override", [False, True])
def test_retained_commands_preserve_arguments_and_overrides(tmp_path, arm, override):
    extra = ["--noise-seed", "7", "--steps", "12", "--lr-final-ratio", "0.4", "--lam-every", "9"] if override else []
    output = "custom output/shard" if override else None
    result = dispatch(tmp_path, arm, extra, output)
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [
        "-m", "immunization_color.cli.generate_color_defenses", "--arm", arm,
        "--output-dir", output or f"artifacts/defenses/{arm}", "--data-root", "data/portraits",
        *CONDITIONS[arm], *extra,
    ]


def test_registry_file_matches_the_recorded_conditions():
    from immunization_color.conditions import CONDITIONS as REGISTRY
    assert REGISTRY == CONDITIONS


@pytest.mark.parametrize("arm", ["color_xattn", "color_simple_dayn", "color_lut3d_dayn"])
def test_retired_commands_are_rejected_before_python_runs(tmp_path, arm):
    result = dispatch(tmp_path, arm)
    assert result.returncode == 2
    assert result.stdout == ""
    assert arm in result.stderr


def parse_options(monkeypatch, options):
    """正式入口的 argparse 設定；不初始化模型。"""
    return build_parser().parse_args(["--output-dir", "unused", *options])


@pytest.mark.parametrize("arm", CONDITIONS)
def test_retained_defaults_match_the_recorded_protocol(monkeypatch, arm):
    args = parse_options(monkeypatch, ["--arm", arm, *CONDITIONS[arm]])
    assert (args.steps, args.lr, args.noise_seed, args.lr_final_ratio, args.lam_every) == (900, 0.02, 0, 0.2, 5)
    assert args.lpips_tolerance == 0.0025
    assert args.carrier == "ab"
    assert args.caps == ("full" if arm == "color" else "simple")
    assert args.objective == ("xattn" if arm == "color_simple_xattn" else "comm")
    assert args.box == ("skin" if arm == "color_simple_skinbox" else "uniform")


def test_shared_optimizer_options_remain_available(monkeypatch):
    args = parse_options(monkeypatch, ["--objective", "xattn", "--caps", "simple",
                                      "--carrier", "ab", "--lr-final-ratio", "0.4", "--lam-every", "9"])
    assert (args.lr_final_ratio, args.lam_every) == (0.4, 9)
    assert args.carrier == "ab"


@pytest.mark.parametrize("options", [["--objective", "dayn"], ["--carrier", "lut3d"]])
def test_retired_only_options_are_rejected(monkeypatch, options):
    with pytest.raises(SystemExit) as error:
        parse_options(monkeypatch, options)
    assert error.value.code == 2

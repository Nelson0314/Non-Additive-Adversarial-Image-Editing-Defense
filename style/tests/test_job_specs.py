"""`configs/jobs/*.spec` 的工作清單：每列可由現行 CLI 解析，且對應的結果目錄已入版控。"""
from pathlib import Path

import pytest

from immunization_style.cli.generate_style_prompt_defenses import build_parser
from immunization_style.method import STYLES

ROOT = Path(__file__).resolve().parents[1]
SPECS = sorted((ROOT / "configs/jobs").glob("*.spec"))
BASE = ["--data-root", "data/portraits", "--s-i", "2.0", "--tf32"]  # run_style_prompt_jobs.sh 的 BASE


def jobs(spec: Path):
    for line in spec.read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.startswith("#"):
            name, images, style, *rest = line.split()
            yield name, images.split("+"), style, rest


def test_every_recorded_round_has_a_spec():
    assert {p.stem for p in SPECS} == {p.name for p in (ROOT / "results/defenses").iterdir() if p.is_dir()}


@pytest.mark.parametrize("spec", SPECS, ids=lambda p: p.stem)
def test_spec_rows_parse_and_match_results(spec):
    names = []
    for name, images, style, rest in jobs(spec):
        assert style in STYLES
        args = build_parser().parse_args([*BASE, "--images", *images, "--styles", style, *rest,
                                          "--output-dir", "x"])
        assert args.images == images
        assert (ROOT / "results/defenses" / spec.stem / name / "results.csv").is_file()
        names.append(name)
    assert len(names) == len(set(names))
    assert set(names) == {p.name for p in (ROOT / "results/defenses" / spec.stem).iterdir()}

"""`measure_additional_metrics` 的輸入根與輸出目錄參數：預設值等於 `layout` 常數，自訂時只寫入指定目錄。"""
import csv

import pytest
from PIL import Image

from immunization_baseline import layout
from immunization_baseline.cli import measure_additional_metrics as metrics

STAGES = ("fidelity", "displacement", "retention", "aesthetic", "vmaf")


@pytest.mark.parametrize("stage", STAGES)
def test_path_defaults_equal_layout_constants(stage):
    args = metrics.build_parser().parse_args(["--stage", stage])
    assert args.stage == stage
    assert args.defenses == layout.DEFENSES
    assert args.displacement == layout.RESULTS / "displacement.csv"
    assert args.retention == layout.RESULTS / "retention.csv"
    assert args.purified_edits == layout.PURIFIED_EDITS
    assert args.path_root == layout.PROJECT
    assert args.out == layout.RESULTS / "additional_metrics"


def test_relative_csv_paths_resolve_against_path_root(tmp_path):
    assert metrics.resolve_png("a/b.png", tmp_path) == (tmp_path / "a" / "b.png").resolve()
    absolute = (tmp_path / "c.png").resolve()
    assert metrics.resolve_png(str(absolute), layout.PROJECT) == absolute


def _snapshot(directory):
    """目錄內各檔的 (位元組, mtime)；用來確認自訂輸出時 baseline 的表未被改寫。"""
    if not directory.is_dir():
        return {}
    return {p.name: (p.read_bytes(), p.stat().st_mtime_ns)
            for p in directory.iterdir() if p.is_file()}


def test_fidelity_writes_only_to_custom_output_dir(tmp_path):
    defenses = tmp_path / "defenses"
    condition_dir = defenses / "mist"
    condition_dir.mkdir(parents=True)
    original = condition_dir / "man_00__orig.png"
    defended = condition_dir / "man_00__mist__def.png"
    Image.new("RGB", (32, 32), (120, 90, 60)).save(original)
    Image.new("RGB", (32, 32), (124, 88, 63)).save(defended)
    out_dir = tmp_path / "additional_metrics"

    baseline_tables = layout.RESULTS / "additional_metrics"
    before = _snapshot(baseline_tables)

    metrics.main(["--stage", "fidelity",
                  "--defenses-root", str(defenses),
                  "--output-dir", str(out_dir)])

    assert _snapshot(baseline_tables) == before
    assert sorted(p.name for p in out_dir.iterdir()) == ["fidelity.csv"]
    with (out_dir / "fidelity.csv").open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
    assert reader.fieldnames == ["condition", "image", "fid_fsim", "fid_delta_e00", "fid_mse",
                                 "original_png", "defended_png", "device"]
    assert len(rows) == 1
    row = rows[0]
    assert (row["condition"], row["image"], row["device"]) == ("mist", "man_00", "cpu")
    assert row["original_png"] == original.as_posix()
    assert row["defended_png"] == defended.as_posix()
    assert float(row["fid_mse"]) > 0

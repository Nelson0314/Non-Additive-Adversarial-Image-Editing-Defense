"""`immunization_color.stages`：完成標記綁定解析後設定、輸入雜湊與輸出驗收。"""
import csv
from pathlib import Path
import shutil

import pytest

from immunization_color import stages

ROOT = Path(__file__).resolve().parents[1]
NAMES = ("man_00", "woman_00")


def write_table(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture
def project(tmp_path):
    shutil.copytree(ROOT / "data", tmp_path / "data")
    defenses = tmp_path / "artifacts/defenses/color"
    defenses.mkdir(parents=True)
    for name in NAMES:
        (defenses / f"{name}__color__def.png").write_bytes(b"def-" + name.encode())
        (defenses / f"{name}__orig.png").write_bytes(b"orig")
    write_table(defenses / "results.csv", [{"image": n} for n in NAMES])
    return tmp_path


def edits(project, directory, suffix):
    rows = []
    for name in NAMES:
        for k in range(4):
            arm = f"ip2p{suffix}"
            (directory / arm).mkdir(parents=True, exist_ok=True)
            (directory / arm / f"{name}__p{k}.png").write_bytes(b"edit")
            rows.append({"scenario": "ip2p", "arm": arm, "image": name, "prompt_index": k})
    write_table(directory / "preflight.csv", rows)


def test_defense_marker_follows_inputs_and_outputs(project):
    assert stages.check(project, "color", "defense") == (False, "沒有完成標記")
    stages.write(project, "color", "defense")
    assert stages.check(project, "color", "defense") == (True, "")
    (project / "data/portraits/man/man_00.png").write_bytes(b"changed")
    ok, reason = stages.check(project, "color", "defense")
    assert not ok and "不符" in reason


def test_missing_output_invalidates_marker(project):
    stages.write(project, "color", "defense")
    (project / "artifacts/defenses/color/results.csv").unlink()
    assert not stages.check(project, "color", "defense")[0]


def test_write_refuses_incomplete_outputs(project):
    (project / "artifacts/defenses/color/woman_00__orig.png").unlink()
    with pytest.raises(ValueError):
        stages.write(project, "color", "defense")
    assert not stages.marker(project, "color", "defense").exists()


def test_edit_marker_binds_protocol_settings_and_subset(project):
    edits(project, project / "artifacts/defended_edits/color", "_color")
    record = stages.stage_record(project, "color", "edit_ip2p")
    assert record["settings"]["seed"] is None or "seed" in record["settings"]
    assert {"s_t", "s_i", "scenarios", "suffix"} <= set(record["settings"])
    stages.write(project, "color", "edit_ip2p")
    assert stages.check(project, "color", "edit_ip2p")[0]
    assert not stages.check(project, "color", "edit_ip2p", ["man_00"])[0]
    (project / "artifacts/defenses/color/man_00__color__def.png").write_bytes(b"new defense")
    assert not stages.check(project, "color", "edit_ip2p")[0]


def test_purify_marker_requires_every_protocol_purifier(project):
    purified = project / "artifacts/purified/color"
    labels = stages.purifier_labels()
    for label in labels:
        for name in NAMES:
            (purified / label).mkdir(parents=True, exist_ok=True)
            (purified / label / f"{name}__def.png").write_bytes(b"p")
    stages.write(project, "color", "purify")
    assert stages.check(project, "color", "purify")[0]
    (purified / labels[-1] / "man_00__def.png").unlink()
    assert not stages.check(project, "color", "purify")[0]


def test_chain_tags_cover_every_purifier():
    tags = stages.chain_tags()
    assert tags[:3] == ["defense", "edit_ip2p", "purify"]
    assert tags[3:] == [f"pedit_{p}_ip2p" for p in stages.purifier_labels()]

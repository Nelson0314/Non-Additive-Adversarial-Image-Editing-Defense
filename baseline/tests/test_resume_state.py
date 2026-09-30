"""續跑不得沿用其他協定或刪除缺圖的證據列。"""
import csv
from pathlib import Path

import pytest

from immunization_baseline import resume_state as resume
FIELDS = ["arm", "image", "prompt_index", "protocol_id", "png", "input_png", "input_sha256"]


def state(tmp_path):
    source, png = tmp_path / "input.png", tmp_path / "edit.png"
    source.write_bytes(b"source")
    png.write_bytes(b"edit")
    row = dict(arm="color", image="a", prompt_index="0", protocol_id="protocol",
               png=str(png), input_png=str(source), input_sha256=resume.file_digest(source))
    csv_path = tmp_path / "result.csv"
    resume.write_rows_atomic(csv_path, FIELDS, [row])
    return csv_path, row


def test_valid_state_is_preserved(tmp_path):
    path, row = state(tmp_path)
    assert resume.load_resume_rows(path, FIELDS, "protocol", ("arm", "image", "prompt_index")) == [row]


@pytest.mark.parametrize("problem", ["protocol", "missing_png", "changed_input", "legacy", "duplicate"])
def test_invalid_state_fails_without_rewriting_csv(tmp_path, problem):
    path, row = state(tmp_path)
    protocol = "protocol"
    if problem == "protocol":
        protocol = "other"
    elif problem == "missing_png":
        Path(row["png"]).unlink()
    elif problem == "changed_input":
        Path(row["input_png"]).write_bytes(b"different source")
    elif problem == "legacy":
        path.write_text("image,prompt_index\na,0\n", encoding="utf-8")
    else:
        resume.write_rows_atomic(path, FIELDS, [row, row])
    before = path.read_bytes()
    with pytest.raises((ValueError, FileNotFoundError)):
        resume.load_resume_rows(path, FIELDS, protocol, ("arm", "image", "prompt_index"))
    assert path.read_bytes() == before


@pytest.mark.parametrize("field", ["guidance", "seed", "steps", "prompts", "precision", "image_guidance", "variant"])
def test_protocol_changes_invalidate_digest(field):
    config = {"guidance": 3.5, "seed": 12, "steps": 28, "prompts": ["edit"],
              "precision": "bf16", "image_guidance": 1.5, "variant": "add"}
    assert resume.protocol_digest(config) != resume.protocol_digest({**config, field: "changed"})

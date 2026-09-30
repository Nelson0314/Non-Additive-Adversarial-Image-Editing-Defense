"""固定淨化、幾何遮罩與明確產物根目錄的 CPU 契約。"""
import csv
from pathlib import Path
import sys

from PIL import Image
import pytest
import torch

from immunization_core.artifacts.layout import ArtifactLayout, defended_image
from immunization_core.pipelines import purification
from immunization_core.pipelines.masks import purified_mask, subject_mask
from immunization_core.purifiers.operators import Purifier
from immunization_core.purifiers.protocol import PURIFIERS, label


def test_fixed_protocol_keeps_identity_and_seven_purifiers():
    assert [label(*spec) for spec in PURIFIERS] == [
        "identity", "crop_resize0.1", "jpeg30", "jpeg50", "jpeg80", "blur1", "blur2", "rotate15"]


@pytest.mark.parametrize("tag,kind,strength", [("crop_resize0.1", "crop_resize", 0.1), ("rotate15", "rotate", 15)])
def test_mask_moves_after_polarity_conversion(tag, kind, strength):
    repaint = torch.zeros(1, 1, 32, 32)
    repaint[..., :12, :18] = 1
    subject = subject_mask(repaint)
    got = purified_mask(subject, tag)
    expected = (Purifier(kind, strength).evaluate(subject) >= 0.5).float()
    assert torch.equal(got, expected)
    assert not torch.equal(got, subject)
    assert set(got.unique().tolist()) <= {0.0, 1.0}
    if kind == "rotate":
        assert got[..., -1, -1].item() == 0
        wrong_order = 1 - (Purifier(kind, strength).evaluate(repaint) >= 0.5).float()
        assert not torch.equal(got, wrong_order)


def test_non_geometric_mask_is_not_resampled():
    mask = torch.rand(1, 1, 32, 32)
    assert purified_mask(mask, "jpeg30") is mask


def test_layout_has_no_discovery_or_directory_creation(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    layout = ArtifactLayout(Path("dataset"), Path("output"), Path("tables"))
    assert layout.masks == tmp_path / "dataset/masks"
    assert layout.prompts == tmp_path / "dataset/prompts.yaml"
    assert not list(tmp_path.iterdir())


def test_defense_file_requires_exactly_one_match(tmp_path):
    with pytest.raises(ValueError):
        defended_image(tmp_path, "person")
    first = tmp_path / "person__color__def.png"
    first.touch()
    assert defended_image(tmp_path, "person") == first
    (tmp_path / "person__def.png").touch()
    with pytest.raises(ValueError):
        defended_image(tmp_path, "person")


@pytest.mark.parametrize("source_kind", ["data", "defended"])
def test_pipeline_writes_complete_protocol_outputs(tmp_path, monkeypatch, source_kind):
    source, output = tmp_path / "source", tmp_path / "output"
    image = source / ("people/person.png" if source_kind == "data" else "person__color__def.png")
    image.parent.mkdir(parents=True)
    Image.new("RGB", (32, 32), (90, 120, 160)).save(image)
    monkeypatch.setattr(purification, "RESOLUTION", 32)
    monkeypatch.setattr(sys, "argv", ["purification", "--" + source_kind, str(source), "--out", str(output)])
    purification.main()
    with (output / "purified.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert [r["purifier"] for r in rows] == [label(*spec) for spec in PURIFIERS]
    assert len(rows) == 8
    assert all(Path(r["output_png"]).is_file() for r in rows)
    assert {r["condition"] for r in rows} == {"undefended" if source_kind == "data" else "source"}
    assert [r["purifier"] for r in rows if r["geometric"] == "True"] == ["crop_resize0.1", "rotate15"]


@pytest.mark.parametrize("kind", ["gridpure", "fdpure"])
def test_absent_historical_extension_fails_explicitly(kind, monkeypatch):
    from immunization_core.purifiers import diffpure
    monkeypatch.setattr(diffpure, "has_diffpure_weights", lambda *a: True)
    operator = Purifier(kind)
    assert not operator.available
    with pytest.raises(NotImplementedError, match=kind):
        operator.evaluate(torch.zeros(1, 3, 16, 16))

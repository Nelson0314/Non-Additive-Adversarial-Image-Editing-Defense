"""匯入 manifest 與保真 CSV 的雜湊欄；指標以替身取代，不載入權重。"""
import csv
import hashlib
import json
import sys

from PIL import Image
import pytest

from immunization_baseline.cli import import_defense_artifacts as importer


class FakeSuite:
    def __init__(self, device=None):
        pass

    def pairwise(self, a, b):
        return {"psnr": 30.0, "lpips": 0.1, "rms": 0.01, "linf": 0.05, "ssim": 0.9,
                "vif_p": 0.8, "dists": 0.1}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def setup(tmp_path, monkeypatch):
    data, source, out = tmp_path / "data", tmp_path / "source", tmp_path / "out"
    for name in ("man_00", "woman_00"):
        (data / name.split("_")[0]).mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (16, 16), (100, 100, 100)).save(data / name.split("_")[0] / f"{name}.png")
        source.mkdir(exist_ok=True)
        Image.new("RGB", (16, 16), (90, 110, 130)).save(source / f"{name}__color__defended.png")
    settings = source / "results.csv"
    settings.write_text("image,arm\nman_00,color\nwoman_00,color\n")
    monkeypatch.setattr(importer, "MetricSuite", FakeSuite)
    monkeypatch.setattr(importer, "RESOLUTION", 16)
    argv = ["import", "--source-dir", str(source), "--variant", "color", "--data-root", str(data),
            "--output-dir", str(out), "--source-settings", str(settings)]
    return data, source, out, settings, argv


def test_manifest_and_csv_record_input_hashes(setup, monkeypatch):
    data, source, out, settings, argv = setup
    monkeypatch.setattr(sys, "argv", argv)
    importer.main()
    manifest = json.loads((out / "import_manifest.json").read_text(encoding="utf-8"))
    assert manifest["source_settings"]["sha256"] == sha(settings)
    by_image = {entry["image"]: entry for entry in manifest["images"]}
    assert by_image["man_00"]["source_sha256"] == sha(source / "man_00__color__defended.png")
    assert by_image["man_00"]["original_sha256"] == sha(data / "man/man_00.png")
    assert by_image["man_00"]["defended_sha256"] == sha(out / "man_00__color__def.png")
    with (out / "results_all.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert [r["image"] for r in rows] == ["man_00", "woman_00"]
    assert rows[0]["defended_sha256"] == by_image["man_00"]["defended_sha256"]
    assert rows[0]["source_settings_sha256"] == sha(settings)


def test_missing_settings_or_image_writes_no_manifest(setup, monkeypatch):
    data, source, out, settings, argv = setup
    settings.unlink()
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit):
        importer.main()
    settings.write_text("x\n")
    (source / "woman_00__color__defended.png").unlink()
    with pytest.raises(SystemExit):
        importer.main()
    assert not (out / "import_manifest.json").exists()

import csv

from PIL import Image
import pytest

from immunization_baseline.cli.evaluate_edit_completion import validate_stage


def stage(tmp_path):
    arm = "ip2p_color"
    (tmp_path / arm).mkdir()
    png = tmp_path / arm / "a__p0.png"
    Image.new("RGB", (8, 8)).save(png)
    expected = [{"image": "a", "prompt_index": 0, "arm": arm, "seed": 12, "prompt": "edit"}]
    with (tmp_path / "preflight.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=expected[0])
        writer.writeheader()
        writer.writerows(expected)
    return arm, expected, [png]


def test_absent_stage_can_start(tmp_path):
    assert not validate_stage(tmp_path, "ip2p_color", [], [])


def test_empty_directory_is_not_completion(tmp_path):
    (tmp_path / "ip2p_color").mkdir()
    with pytest.raises(ValueError, match="未完成"):
        validate_stage(tmp_path, "ip2p_color", [{"image": "a", "prompt_index": 0}], [])


def test_complete_stage_has_stable_manifest(tmp_path):
    args = stage(tmp_path)
    assert validate_stage(tmp_path, *args)
    assert validate_stage(tmp_path, *args)
    assert (tmp_path / args[0] / "completion.json").is_file()


@pytest.mark.parametrize("problem", ["seed", "missing", "changed", "partial"])
def test_invalid_stage_is_rejected(tmp_path, problem):
    arm, expected, artifacts = stage(tmp_path)
    validate_stage(tmp_path, arm, expected, artifacts)
    if problem == "seed":
        expected[0]["seed"] = 13
    elif problem == "missing":
        artifacts[0].unlink()
    elif problem == "changed":
        Image.new("RGB", (8, 8), "red").save(artifacts[0])
    else:
        expected.append({**expected[0], "prompt_index": 1})
    with pytest.raises((ValueError, OSError)):
        validate_stage(tmp_path, arm, expected, artifacts)

"""編輯、displacement 與 retention 主流程的 CPU 契約；指標以替身取代，不載入權重。"""
import csv
from pathlib import Path

from PIL import Image
import pytest
import torch

from immunization_core.pipelines import displacement, editing, retention

SIZE = 16


class FakeSuite:
    """回傳固定值的 MetricSuite 替身；只記錄被要求比較的影像。"""

    def __init__(self, device=None):
        self.device = device
        self.lpips_module = object()

    def pairwise(self, a, b):
        return {"lpips": 0.25, "ssim": 0.5, "psnr": 30.0, "vif_p": 0.5,
                "dists": 0.1, "rms": 0.01, "linf": 0.02}

    def semantic(self, x, prompt):
        return {"clip": 0.3}

    def image_similarity(self, a, b):
        return {"clip": 0.9, "siglip": 0.8}


def fake_split(regional, a, b, mask):
    """全圖值為兩圖平均絕對差，主體與背景值為遮罩面積，供驗證遮罩傳遞。"""
    return {"lpips_full": float((a - b).abs().mean()),
            "lpips_subject": float(mask.mean()),
            "lpips_background": float((1 - mask).mean())}


@pytest.fixture
def fakes(monkeypatch):
    for module in (editing, displacement, retention):
        monkeypatch.setattr(module, "MetricSuite", FakeSuite)
        monkeypatch.setattr(module, "RESOLUTION", SIZE)
    for module in (displacement, retention):
        monkeypatch.setattr(module, "RegionalLPIPS", lambda m: None)
        monkeypatch.setattr(module, "split_displacement", fake_split)


def png(path: Path, value=128) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (SIZE, SIZE), (value, value, value)).save(path)
    return path


def mask_png(path: Path) -> Path:
    """左半白（重繪）、右半黑（主體）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("L", (SIZE, SIZE), 0)
    image.paste(255, (0, 0, SIZE // 2, SIZE))
    image.save(path)
    return path


def read(path: Path) -> list:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


@pytest.fixture
def dataset(tmp_path):
    root = tmp_path / "data"
    root.mkdir()
    (root / "prompts.yaml").write_text(
        "man:\n  content: a man\n"
        "edits:\n  ip2p: ['add a hat', 'add glasses']\n"
        "  inpaint: ['{content} on a beach']\n", encoding="utf-8")
    for name in ("man_00", "man_01"):
        png(root / "man" / f"{name}.png")
        mask_png(root / "masks" / f"{name}.png")
    return root


def run_metrics_only(dataset, out, *extra):
    for scenario in ("ip2p", "inpaint"):
        for name in ("man_00", "man_01"):
            for index in range(2):
                png(out / scenario / f"{name}__p{index}.png", 90)
    editing.main(["--data", str(dataset), "--out", str(out), "--metrics-only", *extra])
    return read(out / "preflight.csv")


def test_editing_metrics_only_rows_keep_protocol(fakes, dataset, tmp_path):
    rows = run_metrics_only(dataset, tmp_path / "out")
    assert [(r["scenario"], r["image"], r["prompt_index"]) for r in rows] == [
        ("ip2p", "man_00", "0"), ("ip2p", "man_00", "1"),
        ("ip2p", "man_01", "0"), ("ip2p", "man_01", "1"),
        ("inpaint", "man_00", "0"), ("inpaint", "man_01", "0")]
    assert {r["seed"] for r in rows} == {"20260812"}
    assert {r["steps"] for r in rows} == {"50"}
    assert {r["s_i"] for r in rows if r["scenario"] == "ip2p"} == {"1.8"}
    assert {r["guidance"] for r in rows if r["scenario"] == "inpaint"} == {"7.5"}
    assert rows[4]["prompt"] == "a man on a beach"


def test_editing_subset_runs_with_subset_defended_directory(fakes, dataset, tmp_path):
    defended = tmp_path / "defended"
    png(defended / "man_01__color__def.png")
    rows = run_metrics_only(dataset, tmp_path / "out", "--images", "man_01",
                            "--defended", str(defended), "--scenarios", "inpaint")
    assert [r["image"] for r in rows] == ["man_01"]
    assert rows[0]["defence"] == "defended"
    assert rows[0]["input_png"].endswith("man_01__color__def.png")


def test_editing_rejects_ambiguous_defended_image(fakes, dataset, tmp_path):
    defended = tmp_path / "defended"
    png(defended / "man_01__color__def.png")
    png(defended / "man_01__def.png")
    with pytest.raises(ValueError):
        run_metrics_only(dataset, tmp_path / "out", "--images", "man_01",
                         "--defended", str(defended))


def test_editing_replaces_only_touched_arm(fakes, dataset, tmp_path):
    out = tmp_path / "out"
    run_metrics_only(dataset, out)
    rows = run_metrics_only(dataset, out, "--scenarios", "inpaint")
    assert [r["arm"] for r in rows] == ["ip2p"] * 4 + ["inpaint"] * 2


def displacement_tree(tmp_path, dataset):
    preflight, root = tmp_path / "preflight", tmp_path / "defended_edits"
    for name in ("man_00", "man_01"):
        png(preflight / "ip2p_si18" / f"{name}__p0.png", 100)
    for condition, value in (("color", 140), ("style", 160), ("_scratch", 0)):
        directory = root / condition
        directory.mkdir(parents=True)
        with (directory / "preflight.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, ["scenario", "image", "prompt_index", "arm", "prompt"])
            writer.writeheader()
            for name in ("man_00", "man_01"):
                writer.writerow({"scenario": "ip2p", "image": name, "prompt_index": 0,
                                 "arm": "ip2p", "prompt": "add a hat"})
                png(directory / "ip2p" / f"{name}__p0.png", value)
    return preflight, root


def test_displacement_scans_conditions_and_splits_subject(fakes, dataset, tmp_path):
    preflight, root = displacement_tree(tmp_path, dataset)
    out = tmp_path / "displacement.csv"
    displacement.main(["--defended-root", str(root), "--preflight", str(preflight),
                       "--data", str(dataset), "--out", str(out)])
    rows = read(out)
    assert [(r["condition"], r["image"]) for r in rows] == [
        ("color", "man_00"), ("color", "man_01"), ("style", "man_00"), ("style", "man_01")]
    assert {r["undefended_arm"] for r in rows} == {"ip2p_si18"}
    # 右半為主體：主體遮罩面積 0.5，極性未對調。
    assert {r["disp_lpips_subject"] for r in rows} == {"0.5"}
    assert float(rows[0]["disp_lpips_full"]) == pytest.approx(40 / 255, abs=1e-4)


def test_displacement_condition_filter(fakes, dataset, tmp_path):
    preflight, root = displacement_tree(tmp_path, dataset)
    out = tmp_path / "displacement.csv"
    displacement.main(["--defended-root", str(root), "--preflight", str(preflight),
                       "--data", str(dataset), "--out", str(out), "--conditions", "style"])
    assert {r["condition"] for r in read(out)} == {"style"}


def test_displacement_missing_undefended_side_fails(fakes, dataset, tmp_path):
    preflight, root = displacement_tree(tmp_path, dataset)
    (preflight / "ip2p_si18" / "man_01__p0.png").unlink()
    with pytest.raises(SystemExit, match="man_01__p0"):
        displacement.main(["--defended-root", str(root), "--preflight", str(preflight),
                           "--data", str(dataset), "--out", str(tmp_path / "d.csv")])


def retention_tree(tmp_path, plain="0.2"):
    root = tmp_path / "purified_edits"
    for condition, value in (("undefended", 100), ("color", 140), ("style", 160)):
        for purifier in ("jpeg30", "rotate15"):
            png(root / condition / purifier / f"ip2p_{condition}_{purifier}" / "man_00__p0.png",
                value)
    (root / "_scratch").mkdir()
    base = tmp_path / "displacement.csv"
    with base.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, ["condition", "scenario", "image", "prompt_index",
                                    "disp_lpips_full"])
        writer.writeheader()
        for condition in ("color", "style"):
            writer.writerow({"condition": condition, "scenario": "ip2p", "image": "man_00",
                             "prompt_index": 0, "disp_lpips_full": plain})
    return root, base


def test_retention_pairs_both_purified_sides(fakes, dataset, tmp_path):
    root, base = retention_tree(tmp_path)
    out = tmp_path / "retention.csv"
    retention.main(["--purified-root", str(root), "--displacement", str(base),
                    "--data", str(dataset), "--out", str(out)])
    rows = read(out)
    assert [(r["condition"], r["purifier"], r["geometric"]) for r in rows] == [
        ("color", "jpeg30", "False"), ("color", "rotate15", "True"),
        ("style", "jpeg30", "False"), ("style", "rotate15", "True")]
    purified = 40 / 255
    assert float(rows[0]["disp_purified"]) == pytest.approx(purified, abs=1e-5)
    assert float(rows[0]["net_gain"]) == pytest.approx(0.2 - purified, abs=1e-5)
    assert float(rows[0]["retained"]) == pytest.approx(purified / 0.2, abs=1e-5)
    # 旋轉後黑角歸背景，主體面積小於未變換的 0.5。
    assert float(rows[0]["disp_purified_subject"]) == 0.5
    assert float(rows[1]["disp_purified_subject"]) < 0.5


def test_retention_condition_filter_and_zero_plain(fakes, dataset, tmp_path):
    root, base = retention_tree(tmp_path, plain="0")
    out = tmp_path / "retention.csv"
    retention.main(["--purified-root", str(root), "--displacement", str(base),
                    "--data", str(dataset), "--out", str(out), "--conditions", "color"])
    rows = read(out)
    assert {r["condition"] for r in rows} == {"color"}
    assert {r["retained"] for r in rows} == {""}


def test_retention_missing_undefended_side_fails(fakes, dataset, tmp_path):
    root, base = retention_tree(tmp_path)
    (root / "undefended/jpeg30/ip2p_undefended_jpeg30/man_00__p0.png").unlink()
    with pytest.raises(SystemExit, match="缺檔"):
        retention.main(["--purified-root", str(root), "--displacement", str(base),
                        "--data", str(dataset), "--out", str(tmp_path / "r.csv")])


def test_retention_mask_matches_geometric_purifier(dataset):
    from immunization_core.io import load_image_tensor
    from immunization_core.pipelines.masks import purified_mask, subject_mask
    repaint = load_image_tensor(dataset / "masks/man_00.png", torch.device("cpu"),
                                size=SIZE)[:, :1]
    subject = subject_mask(repaint)
    assert subject[..., :, -1].min() == 1 and subject[..., :, 0].max() == 0
    assert not torch.equal(purified_mask(subject, "rotate15"), subject)

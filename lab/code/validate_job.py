"""CPU 驗收 queue 產物的列數、鍵集合與必要欄位。"""
import argparse
import csv
import math
from pathlib import Path

PURIFIERS = ("jpeg50", "crop_resize0.1", "blur1", "rotate15", "jpeg30", "jpeg80", "blur2")
KEY = ("condition", "scenario", "image", "prompt_index")
DISPLACEMENT = ("disp_lpips", "disp_ssim", "disp_psnr", "disp_vif_p", "disp_dists",
                "disp_rms", "disp_linf", "disp_lpips_full", "disp_lpips_subject",
                "disp_lpips_background", "clip_pair", "siglip_pair", "siglip_blocked_threshold")
RETENTION = ("disp_plain", "disp_purified", "net_gain", "disp_purified_subject",
             "disp_purified_background", "siglip_pair", "siglip_blocked_threshold")


def read_rows(path, required):
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if not set(required) <= set(reader.fieldnames or []):
            raise ValueError(f"{path}: 缺必要欄位 {sorted(set(required) - set(reader.fieldnames or []))}")
        rows = list(reader)
    if not rows or any(None in row or any(v is None for v in row.values()) for row in rows):
        raise ValueError(f"{path}: 空表或不完整 CSV 列")
    return rows


def validate_table(path, key_fields, expected, numeric=(), required=()):
    rows = read_rows(path, (*key_fields, *numeric, *required))
    keys = [tuple(row[k] for k in key_fields) for row in rows]
    if len(keys) != len(set(keys)) or set(keys) != set(expected):
        raise ValueError(f"{path}: 列數或鍵集合不符；預期 {len(expected)} 列，實際 {len(rows)} 列，"
                         f"缺格={sorted(set(expected) - set(keys))[:8]}")
    for row in rows:
        for field in numeric:
            value = float(row[field])
            # 相同影像的 PSNR 可以是正無限大，不新增科學判準。
            if not math.isfinite(value) and not (field in ("psnr", "disp_psnr") and value == math.inf):
                raise ValueError(f"{path}: {field} 不是有效讀數：{row[field]!r}")
        for field in required:
            if row[field] == "":
                raise ValueError(f"{path}: {field} 是空值")
    return rows


def artifact(path):
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"缺少或空的產物：{path}")


def defense_names(root, arm):
    names = {p.name.split("__")[0] for p in root.glob(f"*__{arm}__def.png")}
    if not names:
        raise ValueError(f"{root}: 沒有防禦圖")
    for name in names:
        artifact(root / f"{name}__{arm}__def.png")
        artifact(root / f"{name}__orig.png")
    return names


def edit_keys(lab, condition, directory, suffix):
    import yaml
    spec = yaml.safe_load((lab / "data/portraits/prompts.yaml").read_text(encoding="utf-8"))
    names = defense_names(lab / "runs/defence" / condition, condition)
    rows = read_rows(directory / "preflight.csv", ("scenario", "image", "prompt_index", "arm"))
    scenarios = {"ip2p"} | {r["scenario"] for r in rows}
    expected = {(condition, scenario, name, str(pi)) for scenario in scenarios
                for name in names for pi in range(len(spec["edits"][scenario]))}
    observed = [(condition, r["scenario"], r["image"], r["prompt_index"]) for r in rows]
    if len(observed) != len(set(observed)) or set(observed) != expected:
        raise ValueError(f"{directory}: 編輯 CSV 缺格、多餘格或重複格")
    for row in rows:
        arm = row["scenario"] + suffix
        if row["arm"] != arm:
            raise ValueError(f"{directory}: arm 不符：{row['arm']} != {arm}")
        artifact(directory / arm / f"{row['image']}__p{row['prompt_index']}.png")
    return expected


def validate_job(lab, job, fid_arms=()):
    parts = job.split(":")
    kind = parts[0]
    runs = lab / "runs"
    if kind in ("pilot", "def"):
        _, arm, image, *_ = parts
        directory = runs / ("defence_pilot" if kind == "pilot" else "defence_shards") / arm / image
        names = defense_names(directory, arm)
        if names != {image}:
            raise ValueError(f"{directory}: 影像集合不符 {image}")
        validate_table(directory / "results.csv", ("image",), {(image,)})
    elif kind == "chain":
        arm = parts[1]
        edit_keys(lab, arm, runs / "edit_defended" / arm, f"_{arm}")
        for purifier in PURIFIERS:
            edit_keys(lab, arm, runs / "edit_purified" / arm / purifier, f"_{arm}_{purifier}")
    elif kind == "readout":
        directories = sorted(d for d in (runs / "edit_defended").iterdir()
                             if d.is_dir() and not d.name.startswith("_"))
        if not directories:
            raise ValueError("沒有可驗收的防禦後編輯")
        expected = set()
        for directory in directories:
            expected |= edit_keys(lab, directory.name, directory, f"_{directory.name}")
        displacement = validate_table(lab / "results/displacement.csv", KEY, expected,
                                      DISPLACEMENT, ("blocked", "defended_png", "undefended_png"))
        for row in displacement:
            artifact(Path(row["defended_png"]))
            artifact(Path(row["undefended_png"]))
        retention_keys = {(*key, purifier) for key in expected for purifier in PURIFIERS}
        rows = validate_table(lab / "results/retention.csv", (*KEY, "purifier"), retention_keys,
                              RETENTION, ("blocked", "geometric"))
        for row in rows:
            if "retained" not in row or (float(row["disp_plain"]) != 0 and
                                         not math.isfinite(float(row["retained"]))):
                raise ValueError("retention.csv: retained 缺失或無效")
    elif kind == "fid":
        if not fid_arms:
            raise ValueError("fid 工作必須明確指定 FID_ARMS")
        expected = {(arm, name) for arm in fid_arms
                    for name in defense_names(runs / "defence" / arm, arm)}
        validate_table(lab / "results/fidelity.csv", ("arm", "image"), expected,
                       ("lpips", "lpips_vs_anchor", "deltaE00", "psnr", "linf", "rms", "anchor_lpips"))
    else:
        raise ValueError(f"未知工作：{job}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job")
    parser.add_argument("--lab", type=Path, default=Path("lab"))
    parser.add_argument("--fid-arms", nargs="*", default=[])
    args = parser.parse_args()
    validate_job(args.lab, args.job, args.fid_arms)


if __name__ == "__main__":
    main()

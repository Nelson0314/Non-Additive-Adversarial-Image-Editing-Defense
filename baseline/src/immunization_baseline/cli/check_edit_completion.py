"""驗收編輯格、協定欄位與產物；空目錄或部分 CSV 不代表完成。"""
import argparse
import csv
import json
from pathlib import Path

from immunization_baseline.resume_state import file_digest, protocol_digest


def validate_stage(out, arm, expected, artifacts):
    csv_path = out / "preflight.csv"
    rows = []
    if csv_path.is_file():
        with csv_path.open(encoding="utf-8", newline="") as stream:
            rows = [r for r in csv.DictReader(stream) if r.get("arm") == arm]
    if not rows and not (out / arm).exists():
        return False
    actual = {}
    for row in rows:
        key = (row.get("image"), row.get("prompt_index"))
        if key in actual:
            raise ValueError(f"{csv_path}: 重複格 {key}")
        actual[key] = row
    wanted = {(r["image"], str(r["prompt_index"])): r for r in expected}
    if not wanted or set(actual) != set(wanted):
        raise ValueError(f"{out / arm}: 編輯未完成；缺格={sorted(set(wanted) - set(actual))}，"
                         f"多餘格={sorted(set(actual) - set(wanted))}")
    for key, expected_row in wanted.items():
        for column, value in expected_row.items():
            observed = actual[key].get(column)
            equal = (float(observed) == value if isinstance(value, (int, float))
                     and observed not in (None, "") else observed == str(value))
            if not equal:
                raise ValueError(f"{csv_path}: {key} 的 {column} 不符協定：{observed!r} != {value!r}")
    from PIL import Image
    hashes = {}
    for path in artifacts:
        with Image.open(path) as image:
            image.verify()
        hashes[str(path.resolve())] = file_digest(path)
    manifest = {"protocol_id": protocol_digest(expected), "rows": len(rows), "artifacts": hashes}
    marker = out / arm / "completion.json"
    if marker.exists():
        if json.loads(marker.read_text(encoding="utf-8")) != manifest:
            raise ValueError(f"{marker}: 產物或協定摘要已變更")
    else:
        with marker.open("x", encoding="utf-8") as stream:
            json.dump(manifest, stream, ensure_ascii=False, sort_keys=True, indent=2)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--defended", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--scenario", choices=["ip2p", "inpaint"], required=True)
    parser.add_argument("--suffix", required=True)
    args = parser.parse_args()
    from immunization_core.pipelines.editing import (load_items, defended_image, EDIT_SEED, EDIT_STEPS,
                                IP2P_MODEL, INPAINT_MODEL, IP2P_TEXT_GUIDANCE,
                                IP2P_EDIT_IMAGE_GUIDANCE, INPAINT_GUIDANCE)
    items, edits = load_items(args.data)
    arm = args.scenario + args.suffix
    expected, artifacts = [], []
    for item in items:
        src = defended_image(args.defended, item["name"])
        artifacts.extend([src, args.out / arm / f"{item['name']}__orig.png"])
        if args.scenario == "inpaint":
            artifacts.append(item["mask"])
        for pi, template in enumerate(edits[args.scenario]):
            png = args.out / arm / f"{item['name']}__p{pi}.png"
            expected.append({
                "scenario": args.scenario, "arm": arm, "image": item["name"], "prompt_index": pi,
                "prompt": template.format(content=item["content"]),
                "victim": IP2P_MODEL if args.scenario == "ip2p" else INPAINT_MODEL,
                "seed": EDIT_SEED, "steps": EDIT_STEPS,
                "guidance": IP2P_TEXT_GUIDANCE if args.scenario == "ip2p" else INPAINT_GUIDANCE,
                "sampler": "ip2p_pipeline" if args.scenario == "ip2p" else "official",
                "s_t": IP2P_TEXT_GUIDANCE if args.scenario == "ip2p" else "",
                "s_i": IP2P_EDIT_IMAGE_GUIDANCE if args.scenario == "ip2p" else "",
                "negative_prompt": "", "input_png": src.as_posix(), "output_png": png.as_posix(),
            })
            artifacts.append(png)
            if args.scenario == "inpaint":
                artifacts.append(png.with_name(png.stem + "__raw.png"))
    try:
        complete = validate_stage(args.out, arm, expected, artifacts)
    except (ValueError, OSError) as error:
        parser.exit(2, f"{error}\n既有證據保留；請檢查中斷原因並使用獨立輸出。\n")
    raise SystemExit(0 if complete else 1)


if __name__ == "__main__":
    main()

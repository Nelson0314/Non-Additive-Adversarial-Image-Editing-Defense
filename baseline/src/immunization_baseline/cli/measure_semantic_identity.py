"""編輯結果的指令方向語意、身分與影像語意相似度（未淨化與七道淨化後）。

每格為一張編輯結果（條件 × 場景 × 影像 × 指令 × 淨化狀態），另含 `undefended` 條件
（未防禦圖、經同一淨化後的編輯）。欄位與 color 專案
`trials/lut3d_defense_and_naturalness/measure/measure_semantic_identity.py` 相同，另加偵測與影像相似度：

    clip_dir、siglip_dir  `MetricSuite.direction_similarity(原圖, 編輯結果, 指令)`（FaceLock 的 CLIP-S）
    facenet_id            `identity.embed` 的 FaceNet（VGGFace2）餘弦，原圖對編輯結果
    arcface_id            ArcFace 餘弦，原圖對編輯結果；`arc_det_n`、`arc_det_score` 為編輯結果的
                          偵測臉數與最高偵測分數
    clip_pair、siglip_pair、blocked
                          `MetricSuite.image_similarity(未防禦編輯, 編輯結果)`，兩者經同一淨化；
                          `blocked` 為 `siglip_pair` 低於 `SIGLIP_BLOCKED_THRESHOLD`；`undefended` 列留空

身分與方向皆以未經淨化的原圖為參照；未偵測到臉時身分欄留空，`face_found_*` 記錄偵測結果。
身分下降（未防禦列減防禦列）於彙總時計算。

計畫來源（擇一）：
    --displacement-csv  位移表（`defended_png`、`undefended_png`、`prompt` 欄）；淨化後的編輯取自
                        `defended_png` 中 `defended_edits/` 之前的目錄下的 `purified_edits/`
    --edits-csv         編輯表（`arm`、`image`、`prompt_index`、`prompt`、`png`，可含 `purifier`）；
                        `arm` 為 `undefended` 者為對照，`--scenario` 指定場景名

用法（專案根，CPU；`--shard i --shards n` 分片並行，同一輸出檔中斷後以同一指令接續）：
    python -m immunization_baseline.cli.measure_semantic_identity \\
        --displacement-csv results/displacement.csv --data-root data/portraits \\
        --output-csv <目錄>/semantic_identity_0.csv --shard 0 --shards 4
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from immunization_core.purifiers.protocol import purifier_labels

UNDEFENDED = "undefended"
NONE = "none"
RESOLUTION = 512
FIELDS = ("condition", "scenario", "image", "prompt_index", "purifier",
          "clip_dir", "siglip_dir", "facenet_id", "arcface_id",
          "face_found_facenet", "face_found_arcface", "arc_det_n", "arc_det_score",
          "clip_pair", "siglip_pair", "blocked", "edit_png")


def read_csv(path: Path) -> list:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def purified_path(edit_png: str, condition: str, purifier: str, scenario: str) -> Path:
    """未淨化編輯的路徑 → 同一格經 `purifier` 淨化後的編輯路徑。"""
    marker = "/undefended_edits/" if condition == UNDEFENDED else "/defended_edits/"
    root, sep, _ = edit_png.partition(marker)
    if not sep:
        raise SystemExit(f"{edit_png} 不含 {marker.strip('/')}，無法推得淨化後的路徑")
    name = Path(edit_png).name
    arm = f"{scenario}_{condition}_{purifier}"
    return Path(root) / "purified_edits" / condition / purifier / arm / name


def plan_from_displacement(paths: list, purifiers: list) -> list:
    """位移表 → [(條件, 場景, 影像, 指令序, 淨化, 指令, 編輯路徑, 未防禦編輯路徑)]。"""
    cells, undefended = [], {}
    for path in paths:
        for row in read_csv(path):
            key = (row["scenario"], row["image"], row["prompt_index"])
            undefended.setdefault(key, (row["prompt"], row["undefended_png"]))
            for purifier in [NONE] + purifiers:
                if purifier == NONE:
                    edit, reference = row["defended_png"], row["undefended_png"]
                else:
                    edit = purified_path(row["defended_png"], row["condition"], purifier, key[0])
                    reference = purified_path(row["undefended_png"], UNDEFENDED, purifier, key[0])
                cells.append((row["condition"], *key, purifier, row["prompt"],
                              Path(edit), Path(reference)))
    for (scenario, image, index), (prompt, png) in sorted(undefended.items()):
        for purifier in [NONE] + purifiers:
            edit = png if purifier == NONE else purified_path(png, UNDEFENDED, purifier, scenario)
            cells.append((UNDEFENDED, scenario, image, index, purifier, prompt, Path(edit), None))
    return cells


def plan_from_edits(paths: list, scenario: str, purifiers: list) -> list:
    rows = [row for path in paths for row in read_csv(path)]
    allowed = {NONE, *purifiers}
    rows = [r for r in rows if r.get("purifier", NONE) in allowed]
    reference = {(r["image"], r["prompt_index"], r.get("purifier", NONE)): r["png"]
                 for r in rows if r["arm"] == UNDEFENDED}
    cells = []
    for r in rows:
        purifier = r.get("purifier", NONE)
        key = (r["image"], r["prompt_index"], purifier)
        if r["arm"] != UNDEFENDED and key not in reference:
            raise SystemExit(f"編輯表缺未防禦對照：{key}")
        cells.append((r["arm"], scenario, r["image"], r["prompt_index"], purifier, r["prompt"],
                      Path(r["png"]),
                      None if r["arm"] == UNDEFENDED else Path(reference[key])))
    return cells


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--displacement-csv", nargs="+", type=Path)
    source.add_argument("--edits-csv", nargs="+", type=Path)
    parser.add_argument("--scenario", help="`--edits-csv` 的場景名（例如 ultraedit、flux）")
    parser.add_argument("--data-root", type=Path, required=True,
                        help="原圖所在的資料集根目錄（`<類別>/<影像>.png`）")
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--purifiers", nargs="*", default=purifier_labels(include_identity=False),
                        help="淨化狀態（預設為協定的七道；不給值時只量未淨化）")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--threads", type=int, default=4, help="每個行程的 torch CPU 執行緒數")
    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    if args.edits_csv and not args.scenario:
        raise SystemExit("--edits-csv 需要 --scenario")
    if args.displacement_csv:
        plan = plan_from_displacement(args.displacement_csv, args.purifiers)
    else:
        plan = plan_from_edits(args.edits_csv, args.scenario, args.purifiers)
    plan = plan[args.shard::args.shards]
    missing = [p for c in plan for p in c[6:] if p is not None and not p.is_file()]
    if missing:
        raise SystemExit(f"{len(missing)} 個編輯結果缺檔，首個：{missing[0]}")

    done = set()
    if args.output_csv.is_file():
        done = {(r["condition"], r["scenario"], r["image"], r["prompt_index"], r["purifier"])
                for r in read_csv(args.output_csv)}
    todo = [c for c in plan if c[:5] not in done]

    import torch

    from immunization_core.io import load_image_tensor
    from immunization_core.metrics import arcface
    from immunization_core.metrics.identity import embed, similarity
    from immunization_core.metrics.standard import SIGLIP_BLOCKED_THRESHOLD
    from immunization_core.metrics.suite import MetricSuite

    torch.set_num_threads(args.threads)
    device = torch.device("cpu")
    suite = MetricSuite(device=device)
    source, reference = {}, {}

    def original(image: str):
        if image not in source:
            x = load_image_tensor(args.data_root / image.split("_")[0] / f"{image}.png",
                                  device, size=RESOLUTION)
            source[image] = x
            reference[image] = (embed(x), arcface.embed(x))
        return source[image], reference[image]

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    fresh = not args.output_csv.is_file()
    print(f"[PLAN] {len(plan)} 格，已完成 {len(plan) - len(todo)} → {args.output_csv}", flush=True)
    with open(args.output_csv, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if fresh:
            w.writeheader()
        for n, (condition, scenario, image, k, purifier, prompt, edit, undef) in enumerate(todo, 1):
            x, (r_fn, r_af) = original(image)
            y = load_image_tensor(edit, device, size=RESOLUTION)
            with torch.no_grad():
                direction = suite.direction_similarity(x, y, prompt)
                pair = None if undef is None else suite.image_similarity(
                    load_image_tensor(undef, device, size=RESOLUTION), y)
            e_fn = embed(y)
            found = arcface.faces(y)
            e_af = found[0].normed_embedding if found else None
            fn, af = similarity(r_fn, e_fn), arcface.similarity(r_af, e_af)
            w.writerow({
                "condition": condition, "scenario": scenario, "image": image,
                "prompt_index": k, "purifier": purifier,
                "clip_dir": round(direction["clip"], 5), "siglip_dir": round(direction["siglip"], 5),
                "facenet_id": "" if fn is None else round(fn, 5),
                "arcface_id": "" if af is None else round(af, 5),
                "face_found_facenet": e_fn is not None, "face_found_arcface": e_af is not None,
                "arc_det_n": len(found),
                "arc_det_score": round(float(found[0].det_score), 5) if found else "",
                "clip_pair": "" if pair is None else round(pair["clip"], 5),
                "siglip_pair": "" if pair is None else round(pair["siglip"], 5),
                "blocked": "" if pair is None else pair["siglip"] < SIGLIP_BLOCKED_THRESHOLD,
                "edit_png": edit.as_posix(),
            })
            f.flush()
            if n % 100 == 0:
                print(f"[PROGRESS] {n}/{len(todo)}", flush=True)
    print(f"[ALLDONE] {args.output_csv}（{len(plan)} 格）", flush=True)


if __name__ == "__main__":
    main()

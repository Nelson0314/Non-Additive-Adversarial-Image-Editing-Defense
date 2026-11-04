"""Pexels 人像一個影像組的 ArcFace(E_u, E_d)：未防禦編輯對防禦後編輯，兩者經同一淨化（未淨化、七道、VAE-Trans），CPU。

等失真九個條件與 color3d_pexels。影像路徑取自 scripts/evaluate_pexels_followup.sh 產生的
runtime/pexels_portraits/followup/<根>/results/<aligned|color3d>/semantic_identity{,_vae_trans}.csv 的 edit_png
（`undefended` 列為 E_u）。任一側偵測不到臉時 `arcface_pair` 留空（`immunization_core.metrics.arcface.similarity`）。

輸出：同目錄的 arcface_pair.csv。
`--without-vae-trans` 時只讀 semantic_identity.csv（未淨化與七道）。
通用輸入：`--semantic-csv <CSV>... --output-csv <CSV>`（measure_semantic_identity 的輸出，含 undefended 列）取代影像組。
`--color3d-group color3d_<變體>` 時只做該分組（results/<分組>/）。
用法（遠端 baseline 專案根）：python scripts/measure_pexels_arcface_pair.py <影像組> [--without-vae-trans] [--color3d-group <分組>]
"""
import argparse
import csv
import json
from pathlib import Path

import torch

from immunization_core.io import load_image_tensor
from immunization_core.metrics import arcface


def read(path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def groups_of(color3d_group):
    if color3d_group != "color3d" and not color3d_group.startswith("color3d_"):
        raise SystemExit(f"--color3d-group 須為 color3d 或 color3d_<變體>：{color3d_group}")
    return ("aligned", "color3d") if color3d_group == "color3d" else (color3d_group,)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image_set", nargs="?")
    ap.add_argument("--semantic-csv", nargs="*", default=[], type=Path)
    ap.add_argument("--output-csv", type=Path, default=None)
    ap.add_argument("--without-vae-trans", action="store_true")
    ap.add_argument("--color3d-group", default="color3d", help="color3d（預設，與等失真 aligned 一起）或 color3d_<變體>（只做該分組，條件為 <分組>_pexels）")
    args = ap.parse_args()
    if args.semantic_csv and args.output_csv is None or bool(args.semantic_csv) == bool(args.image_set):
        ap.error("給影像組，或給 --semantic-csv 與 --output-csv")
    device = torch.device("cpu")
    cache = {}

    def embedding(path):
        if path not in cache:
            cache[path] = arcface.embed(load_image_tensor(path, device, size=512).clamp(0, 1))
        return cache[path]

    if args.semantic_csv:
        jobs = [([r for p in args.semantic_csv for r in read(p)], args.output_csv)]
    else:
        sets = {s["name"]: s for s in json.loads(Path("data/pexels_portraits/image_sets.json").read_text(encoding="utf-8"))["sets"]}
        base = Path("runtime/pexels_portraits/followup") / sets[args.image_set]["root"] / "results"
        jobs = []
        for group in groups_of(args.color3d_group):
            rows = read(base / group / "semantic_identity.csv")
            if not args.without_vae_trans:
                rows += read(base / group / "semantic_identity_vae_trans.csv")
            jobs.append((rows, base / group / "arcface_pair.csv"))
    for rows, out_path in jobs:
        reference = {(r["image"], r["prompt_index"], r["purifier"]): r["edit_png"] for r in rows if r["condition"] == "undefended"}
        out = []
        for r in rows:
            if r["condition"] == "undefended":
                continue
            key = (r["image"], r["prompt_index"], r["purifier"])
            similarity = arcface.similarity(embedding(reference[key]), embedding(r["edit_png"]))
            out.append({"condition": r["condition"], "image": key[0], "prompt_index": key[1], "purifier": key[2],
                        "arcface_pair": "" if similarity is None else round(similarity, 5),
                        "undefended_png": reference[key], "defended_png": r["edit_png"], "device": "cpu"})
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(out[0]), lineterminator="\n")
            w.writeheader()
            w.writerows(out)
        print(f"[DONE] {out_path} {len(out)} 列", flush=True)
    print("[ALLDONE]", flush=True)


if __name__ == "__main__":
    main()

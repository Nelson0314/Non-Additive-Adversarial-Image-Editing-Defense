"""Pexels 人像一個影像組的 ArcFace(E_u, E_d)：未防禦編輯對防禦後編輯，兩者經同一淨化（未淨化、七道、VAE-Trans），CPU。

等失真九個條件與 color3d_pexels。影像路徑取自 scripts/evaluate_pexels_followup.sh 產生的
runtime/pexels_portraits/followup/<根>/results/<aligned|color3d>/semantic_identity{,_vae_trans}.csv 的 edit_png
（`undefended` 列為 E_u）。任一側偵測不到臉時 `arcface_pair` 留空（`immunization_core.metrics.arcface.similarity`）。

輸出：同目錄的 arcface_pair.csv。
用法（遠端 baseline 專案根）：python scripts/measure_pexels_arcface_pair.py <影像組>
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


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image_set")
    args = ap.parse_args()
    sets = {s["name"]: s for s in json.loads(Path("data/pexels_portraits/image_sets.json").read_text(encoding="utf-8"))["sets"]}
    root = sets[args.image_set]["root"]
    base = Path("runtime/pexels_portraits/followup") / root / "results"
    device = torch.device("cpu")
    cache = {}

    def embedding(path):
        if path not in cache:
            cache[path] = arcface.embed(load_image_tensor(path, device, size=512).clamp(0, 1))
        return cache[path]

    for group in ("aligned", "color3d"):
        rows = read(base / group / "semantic_identity.csv") + read(base / group / "semantic_identity_vae_trans.csv")
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
        with open(base / group / "arcface_pair.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(out[0]), lineterminator="\n")
            w.writeheader()
            w.writerows(out)
        print(f"[DONE] {group} {len(out)} 列", flush=True)
    print("[ALLDONE]", flush=True)


if __name__ == "__main__":
    main()

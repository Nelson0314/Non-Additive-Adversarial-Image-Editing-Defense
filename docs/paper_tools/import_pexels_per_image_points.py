"""Export per-image distortion and edit displacement of the Pexels main comparison (run on the remote
host from the baseline project root).

    python import_pexels_per_image_points.py --out ~/paper_figures/in_pexels/per_image_points.csv

Sources follow baseline/scripts/generate_pexels_main_table_summary.py: matched-distortion arms from
runtime/pexels_portraits/aligned/<set root>/results/ (fidelity.csv; photoguard_encoder under
results/photoguard_encoder/; displacement_<condition>.csv, InstructPix2Pix rows), color3d_pexels from
runtime/pexels_portraits/followup/<set root>/results/color3d/fidelity.csv and
../color/trials/lut3d_defense_and_naturalness/results/color3d_pexels/displacement.csv. The set roots are
those of data/pexels_portraits/image_sets.json. Each output row is one condition and image: defense
LPIPS against the original, and the edit LPIPS before purification averaged over the four instructions.
"""

import argparse
import csv
import json
from pathlib import Path

ALIGNED = ["danp", "dayn", "dct_shield", "dct_shield_y", "dia_pt", "dia_r", "mist", "photoguard_encoder", "sifm"]
COLOR3D = "color3d_pexels"
CT = Path("../color/trials/lut3d_defense_and_naturalness")


def read(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    sets = json.loads(Path("data/pexels_portraits/image_sets.json").read_text(encoding="utf-8"))["sets"]
    defense = {c: {} for c in ALIGNED + [COLOR3D]}
    edits = {c: {} for c in ALIGNED + [COLOR3D]}
    for s in sets:
        sub = Path(s["root"]) if s["root"] else Path()
        results = Path("runtime/pexels_portraits/aligned") / sub / "results"
        for c in ALIGNED:
            directory = results / c if c == "photoguard_encoder" else results
            for row in read(directory / "fidelity.csv"):
                if row["condition"] == c:
                    defense[c][row["image"]] = float(row["fid_lpips"])
            for row in read(results / f"displacement_{c}.csv"):
                if row["scenario"] == "ip2p":
                    edits[c].setdefault(row["image"], {})[row["prompt_index"]] = float(row["disp_lpips"])
        followup = Path("runtime/pexels_portraits/followup") / sub / "results"
        for row in read(followup / "color3d/fidelity.csv"):
            if row["condition"] == COLOR3D:
                defense[COLOR3D][row["image"]] = float(row["fid_lpips"])
    for row in read(CT / "results/color3d_pexels/displacement.csv"):
        edits[COLOR3D].setdefault(row["image"], {})[row["prompt_index"]] = float(row["disp_lpips"])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["condition", "image", "defense_lpips", "edit_lpips_none"])
        for c in ALIGNED + [COLOR3D]:
            images = sorted(set(defense[c]) & set(edits[c]))
            if len(images) != 306 or any(len(edits[c][i]) != 4 for i in images):
                raise SystemExit(f"{c}: {len(images)} images with defense and edit readings, expected 306 x 4 instructions")
            for i in images:
                w.writerow([c, i, f"{defense[c][i]:.5f}", f"{sum(edits[c][i].values()) / 4:.5f}"])
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

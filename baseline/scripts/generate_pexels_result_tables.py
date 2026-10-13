"""把已完整完成的影像組的讀數合併成逐條件 CSV（遠端執行，於 baseline 專案根）。

完整完成：十個條件皆有 measured_<條件>.done 且有 additional_metrics.done。輸出到 <輸出目錄>：
defense_<條件>.csv、displacement_<條件>.csv、retention_<條件>.csv、fidelity.csv、aesthetic.csv 與 image_sets.txt。
"""
import csv
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
conditions = "diffvax photoguard_encoder mist dia_pt sifm dia_r dct_shield_y dct_shield dayn danp".split()
sets = json.loads(Path("data/pexels_portraits/image_sets.json").read_text(encoding="utf-8"))["sets"]
base = Path("runtime/pexels_portraits")
done = []
for s in sets:
    state = base / s["root"] if s["root"] else base
    if (state / "additional_metrics.done").exists() and all((state / f"measured_{c}.done").exists() for c in conditions):
        done.append((s, state / "results"))


def merge(name):
    rows, fields = [], []
    for s, r in done:
        with (r / name).open(encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for k in reader.fieldnames:
                if k not in fields:
                    fields.append(k)
            keep = set(s["images"])
            for row in reader:
                if row.get("image") in keep or row.get("condition") == "original" and row.get("image") in keep:
                    rows.append(row)
    with (out / name).open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    return len(rows)


n_images = sum(len(s["images"]) for s, _ in done)
for c in conditions:
    a, b, d = merge(f"defense_{c}.csv"), merge(f"displacement_{c}.csv"), merge(f"retention_{c}.csv")
    assert a == n_images and b == n_images * 4 and d == n_images * 28, (c, a, b, d, n_images)
fid, aes = merge("fidelity.csv"), merge("aesthetic.csv")
(out / "image_sets.txt").write_text("\n".join(s["name"] for s, _ in done) + "\n", encoding="utf-8")
print("sets", len(done), "images", n_images, "fidelity rows", fid, "aesthetic rows", aes)

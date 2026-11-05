#!/usr/bin/env bash
# BlurGuard 在 Pexels 100 張子集上的跨影像組彙整（CPU）：合併各組的配對表並加上防禦圖對原圖的配對，
# 以 measure_distribution_metrics --pairs-csv 量 FID／KID（編輯端：淨化前、七道、VAE-Trans；distortion：防禦圖對原圖），
# 再以 scripts/generate_pexels_blurguard_summary.py 產生主表列。
# 輸出 runtime/pexels_portraits/blurguard/results/all/{pairs.csv,distribution/,summary.csv,per_purifier.csv}。
# 已開始的影像組（state/<影像組>/）有任一缺 followup.done 時以結束碼 4 結束；佇列以 AFTER= 保證各組都已排過。
# 用法（遠端 baseline 專案根）：ENV_FILE=~/env.sh bash scripts/evaluate_pexels_blurguard_summary.sh
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
cd "$BASELINE_ROOT"
export CUDA_VISIBLE_DEVICES=
B=runtime/pexels_portraits/blurguard
O=$B/results/all
for d in "$B"/state/*/; do
  [ -f "$d/followup.done" ] || { echo "[NOT READY] $(basename "$d") 的 followup"; exit 4; }
done
mkdir -p "$O"
"$PY" - "$B" "$O/pairs.csv" <<'EOF'
import csv, json, sys
from pathlib import Path
B, out = Path(sys.argv[1]), sys.argv[2]
rows = []
for s in json.load(open("data/pexels_portraits/image_sets.json", encoding="utf-8"))["sets"]:
    R = B / "results" / s["name"]
    if not (R / "pairs.csv").is_file():
        continue
    rows += list(csv.DictReader(open(R / "pairs.csv", encoding="utf-8")))
    d = Path("artifacts/pexels_portraits") / (s["root"] or "") / "defenses/blurguard"
    for p in sorted(d.glob("*__blurguard__def.png")):
        rows.append({"condition": "blurguard", "end": "distortion", "purifier": "", "a_png": p.as_posix(),
                     "b_png": p.with_name(p.name.replace("__blurguard__def.png", "__orig.png")).as_posix()})
with open(out, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["condition", "end", "purifier", "a_png", "b_png"], lineterminator="\n")
    w.writeheader(); w.writerows(rows)
print(f"[PAIRS] {out}（{len(rows)} 對）")
EOF
"$PY" scripts/measure_distribution_metrics.py "$O/distribution" --dataset pexels --pairs-csv "$O/pairs.csv"
"$PY" scripts/generate_pexels_blurguard_summary.py
echo "[STEP] $(date -Iseconds) BLURGUARD SUMMARY DONE"

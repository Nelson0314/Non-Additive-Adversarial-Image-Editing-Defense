#!/usr/bin/env bash
# BlurGuard 在 Pexels 100 張子集上的跨影像組彙整（CPU）：合併各組的配對表並加上防禦圖對原圖的配對，
# 以 measure_distribution_metrics --pairs-csv 量 FID／KID（編輯端：淨化前、七道、VAE-Trans；distortion：防禦圖對原圖），
# 再以 scripts/generate_pexels_blurguard_summary.py 產生主表列。
# 輸出 runtime/pexels_portraits/blurguard/results/all/{pairs.csv,distribution/,summary.csv,per_purifier.csv}。
# 已開始的影像組（state/<影像組>/）有任一缺 followup.done 時以結束碼 4 結束；佇列以 AFTER= 保證各組都已排過。
# BLURGUARD_ROOTS（預設 runtime/pexels_portraits/blurguard）可列多個執行根，合併為一列；BLURGUARD_OUTPUT 為輸出目錄
# （預設 <第一個根>/results/all）。防禦圖對原圖的配對取各單位 results/<單位>/defenses_root/blurguard/ 中、列在該執行根 images.txt 的影像。
# 用法（遠端 baseline 專案根）：[BLURGUARD_ROOTS="<根> ..."] [BLURGUARD_OUTPUT=<目錄>] ENV_FILE=~/env.sh bash scripts/evaluate_pexels_blurguard_summary.sh
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
cd "$BASELINE_ROOT"
export CUDA_VISIBLE_DEVICES=
ROOTS=${BLURGUARD_ROOTS:-runtime/pexels_portraits/blurguard}
first=${ROOTS%% *}
O=${BLURGUARD_OUTPUT:-$first/results/all}
for B in $ROOTS; do
  for d in "$B"/state/*/; do
    [ -f "$d/followup.done" ] || { echo "[NOT READY] $(basename "$d") 的 followup（$B）"; exit 4; }
  done
done
mkdir -p "$O"
"$PY" - "$O/pairs.csv" $ROOTS <<'EOF'
import csv, sys
from pathlib import Path
out, roots = sys.argv[1], [Path(r) for r in sys.argv[2:]]
images = set().union(*(set((r / "images.txt").read_text(encoding="utf-8").split()) for r in roots))
rows = []
for B in roots:
    for p in sorted(B.glob("results/*/pairs.csv")):
        if p.parent.name != "all":
            rows += list(csv.DictReader(open(p, encoding="utf-8")))
for B in roots:
    own = set((B / "images.txt").read_text(encoding="utf-8").split())
    for p in sorted(q for q in B.glob("results/*/defenses_root/blurguard/*__blurguard__def.png") if q.name.split("__")[0] in own):
        rows.append({"condition": "blurguard", "end": "distortion", "purifier": "", "a_png": p.as_posix(),
                     "b_png": p.with_name(p.name.replace("__blurguard__def.png", "__orig.png")).as_posix()})
with open(out, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["condition", "end", "purifier", "a_png", "b_png"], lineterminator="\n")
    w.writeheader(); w.writerows(rows)
print(f"[PAIRS] {out}（{len(rows)} 對）")
EOF
"$PY" scripts/measure_distribution_metrics.py "$O/distribution" --dataset pexels --pairs-csv "$O/pairs.csv"
"$PY" scripts/generate_pexels_blurguard_summary.py --roots $ROOTS --output-dir "$O"
echo "[STEP] $(date -Iseconds) BLURGUARD SUMMARY DONE"

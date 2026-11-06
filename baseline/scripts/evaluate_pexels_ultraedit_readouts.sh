#!/usr/bin/env bash
# Pexels 人像 UltraEdit 50 張的讀數（等失真 photoguard_encoder、dct_shield、dct_shield_y、mist；未防禦分母取 color trial 的 UltraEdit 輸出）。
# 程式與 color trial tools/run_transfer_readouts.sh sd3 相同（皆為 baseline 的量測程式），另加編輯端 SSIM／PSNR；全部 CPU。
#   逐格表   scripts/generate_pexels_ultraedit_cell_tables.py
#   位移     measure_edit_displacement（淨化前）；保留率 measure_purified_displacement（七道與 VAE-Trans）
#   語意     measure_semantic_identity --edits-csv（CLIP-I＝clip_pair、SigLIP、ArcFace(原圖, 編輯)），分 SHARDS 片並行後合併
#   CLIP-S   scripts/measure_clip_score.py --edits-csv；ArcFace(E_u, E_d) scripts/measure_pexels_arcface_pair.py
#   FID／KID scripts/measure_distribution_metrics.py --pairs-csv；SSIM／PSNR scripts/measure_pexels_edit_fidelity.py pairs
# 輸出 runtime/pexels_portraits/ultraedit_readouts/（彙整見 scripts/generate_pexels_ultraedit_summary.py）。
# IMAGES_FILE、CONDS、READOUTS_DIR 可改影像清單、條件與輸出目錄（預設 flux.txt 50 張、上列四條件、runtime/pexels_portraits/ultraedit_readouts）。
# 用法（遠端 baseline 專案根）：[IMAGES_FILE=<檔>] [CONDS="..."] [READOUTS_DIR=<目錄>] ENV_FILE=~/env.sh [SHARDS=8] bash scripts/evaluate_pexels_ultraedit_readouts.sh
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
cd "$BASELINE_ROOT"
export CUDA_VISIBLE_DEVICES=
SHARDS=${SHARDS:-8}
CONDS=${CONDS:-photoguard_encoder dct_shield dct_shield_y mist}
OPS="blur1 blur2 crop_resize0.1 jpeg30 jpeg50 jpeg80 rotate15"
D=data/pexels_portraits
R=${READOUTS_DIR:-runtime/pexels_portraits/ultraedit_readouts}
TABLE_ARGS=(--conditions $CONDS --output-dir "$R")
SUMMARY_ARGS=(--conditions $CONDS --readouts-dir "$R")
if [ -n "${IMAGES_FILE:-}" ]; then TABLE_ARGS+=(--images-file "$IMAGES_FILE"); SUMMARY_ARGS+=(--images-file "$IMAGES_FILE"); fi
UE=../color/trials/lut3d_defense_and_naturalness/artifacts_pexels/cross_model/sd3/edits/undefended_edits
step() { echo "[STEP] $(date -Iseconds) $*"; }

step 逐格表
"$PY" scripts/generate_pexels_ultraedit_cell_tables.py "${TABLE_ARGS[@]}"
EDITS="$R/edits_undefended.csv $(for c in $CONDS; do printf '%s ' "$R/edits_$c.csv"; done)"

step 位移與保留率
"$PY" -m immunization_baseline.cli.measure_edit_displacement --data-root $D --defended-edits-root $R/edits/defended_edits \
    --undefended-edits-root $UE --ip2p-arm ultraedit_undefended --conditions $CONDS --output-csv $R/displacement.csv
"$PY" -m immunization_baseline.cli.measure_purified_displacement --data-root $D --purified-edits-root $R/edits/purified_edits \
    --displacement-csv $R/displacement.csv --conditions $CONDS --output-csv $R/retention.csv

step "語意與身分（$SHARDS 片）"
for i in $(seq 0 $((SHARDS - 1))); do
  OMP_NUM_THREADS=4 "$PY" -m immunization_baseline.cli.measure_semantic_identity --scenario ultraedit --data-root $D \
      --purifiers $OPS vae_trans --edits-csv $EDITS --output-csv $R/semantic_identity_$i.csv --shard $i --shards $SHARDS \
      > $R/semantic_identity_$i.log 2>&1 &
done
wait
"$PY" - "$R" "$SHARDS" <<'EOF'
import csv, sys
from pathlib import Path
R, n = Path(sys.argv[1]), int(sys.argv[2])
rows = []
for i in range(n):
    with open(R / f"semantic_identity_{i}.csv", encoding="utf-8", newline="") as f:
        part = list(csv.DictReader(f))
    rows += part
keys = {(r["condition"], r["image"], r["prompt_index"], r["purifier"]) for r in rows}
if len(keys) != len(rows):
    raise SystemExit(f"分片合併後有重複格：{len(rows)} 列、{len(keys)} 格")
with open(R / "semantic_identity.csv", "w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
    w.writeheader()
    w.writerows(rows)
print(f"[MERGED] semantic_identity.csv {len(rows)} 格")
EOF

step CLIP-S 與 ArcFace
"$PY" scripts/measure_clip_score.py --edits-csv $EDITS --output-csv $R/clip_score.csv --device cpu
"$PY" scripts/measure_pexels_arcface_pair.py --semantic-csv $R/semantic_identity.csv --output-csv $R/arcface_pair.csv

step FID／KID 與 SSIM／PSNR
"$PY" scripts/measure_distribution_metrics.py $R/distribution --dataset pexels --pairs-csv $R/pairs.csv
"$PY" scripts/measure_pexels_edit_fidelity.py pairs --pairs-csv $R/pairs.csv --output-csv $R/edit_fidelity.csv --workers 16
step 彙整
"$PY" scripts/generate_pexels_ultraedit_summary.py "${SUMMARY_ARGS[@]}"
step "READOUTS DONE"

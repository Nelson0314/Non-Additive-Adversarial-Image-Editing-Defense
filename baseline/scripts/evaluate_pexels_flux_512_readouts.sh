#!/usr/bin/env bash
# Pexels 人像 FLUX 512 協定（資料版本 flux_512）的讀數：等失真 photoguard_encoder、dct_shield、dct_shield_y、mist 與 color3d_pexels，
# 分母為同一資料版本的 undefended。程式與 scripts/evaluate_pexels_ultraedit_readouts.sh 相同，位移與保留率改用 FLUX 版面的
# measure_flux_displacement、measure_flux_retention（--resolution 512）。
#   逐格表   scripts/generate_pexels_flux_512_cell_tables.py
#   位移     measure_flux_displacement（淨化前）；保留率 measure_flux_retention（七道與 VAE-Trans）
#   語意     measure_semantic_identity --edits-csv（CLIP-I＝clip_pair、SigLIP、ArcFace(原圖, 編輯)）；CPU 時分 SHARDS 片並行後合併
#   CLIP-S   scripts/measure_clip_score.py --edits-csv；ArcFace(E_u, E_d) scripts/measure_pexels_arcface_pair.py
#   FID／KID scripts/measure_distribution_metrics.py --pairs-csv；SSIM／PSNR scripts/measure_pexels_edit_fidelity.py pairs
# 位移、保留率、語意與 CLIP-S 在 CUDA_VISIBLE_DEVICES 有卡時用 GPU，否則用 CPU；ArcFace(E_u, E_d)、FID／KID、SSIM／PSNR 固定 CPU。
# IMAGES_FILE 指定影像清單（預設 color trial pexels_subsets/flux.txt 的 50 張）；清單內影像的 FLUX 512 完成標記
# （runtime/pexels_portraits/flux_512/done/<影像>.done）不齊時以結束碼 4 結束。SHARDS（預設 8）為 CPU 語意量測的平行片數。
# 輸出 runtime/pexels_portraits/flux_512/readouts/（彙整見 scripts/generate_pexels_flux_512_summary.py）。
# CONDS、READOUTS_DIR 改條件與輸出目錄（預設上列五條件、runtime/pexels_portraits/flux_512/readouts）；EDITOR_NAME（預設 flux_512）改產物與
# 逐格表的根（artifacts/pexels_portraits/<名>/、runtime/pexels_portraits/<名>/），供版面相同的其他編輯器（scripts/evaluate_pexels_nonsd_editor.sh）使用。
# 用法（遠端 baseline 專案根）：[IMAGES_FILE=<檔>] [CONDS="..."] [READOUTS_DIR=<目錄>] [SHARDS=8] ENV_FILE=~/env.sh bash scripts/evaluate_pexels_flux_512_readouts.sh
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
cd "$BASELINE_ROOT"
CONDS=${CONDS:-photoguard_encoder dct_shield dct_shield_y mist color3d_pexels}
OPS="blur1 blur2 crop_resize0.1 jpeg30 jpeg50 jpeg80 rotate15"
D=data/pexels_portraits
EDITOR_NAME=${EDITOR_NAME:-flux_512}
X=artifacts/pexels_portraits/$EDITOR_NAME
S=runtime/pexels_portraits/$EDITOR_NAME
R=${READOUTS_DIR:-$S/readouts}
IMAGES_FILE=${IMAGES_FILE:-../color/trials/lut3d_defense_and_naturalness/pexels_subsets/flux.txt}
SHARDS=${SHARDS:-8}
step() { echo "[STEP] $(date -Iseconds) $*"; }

missing=0
for img in $(cat "$IMAGES_FILE"); do
  for c in $CONDS undefended; do
    for op in none vae_trans $OPS; do [ -f "$S/done/${img}__${c}__$op.done" ] || { missing=$((missing + 1)); break 2; }; done
  done
done
[ "$missing" -eq 0 ] || { step "前置條件不成立：FLUX 512 尚缺 $missing 張"; exit 4; }
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then DEVICE=cuda; else DEVICE=cpu; fi
step "device=$DEVICE（CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}），影像 $(wc -w < "$IMAGES_FILE") 張（$IMAGES_FILE）"

step 逐格表
"$PY" scripts/generate_pexels_flux_512_cell_tables.py --images-file "$IMAGES_FILE" --conditions $CONDS --output-dir "$R" --chunks-dir "$S/chunks"
EDITS="$R/edits_undefended.csv $(for c in $CONDS; do printf '%s ' "$R/edits_$c.csv"; done)"

step 位移與保留率
"$PY" -m immunization_baseline.cli.measure_flux_displacement --data-root $D --edits-csv-dir $R/plain --edits-root $X/edits \
    --conditions $CONDS --resolution 512 --output-csv $R/displacement.csv
"$PY" -m immunization_baseline.cli.measure_flux_retention --data-root $D --purified-edits-root $X/purified_edits \
    --displacement-csv $R/displacement.csv --conditions $CONDS --purifiers $OPS vae_trans --resolution 512 --output-csv $R/retention.csv

if [ "$DEVICE" = cuda ]; then
  step 語意與身分
  "$PY" -m immunization_baseline.cli.measure_semantic_identity --device cuda --scenario flux --data-root $D \
      --purifiers $OPS vae_trans --edits-csv $EDITS --output-csv $R/semantic_identity.csv
else
  step "語意與身分（CPU，$SHARDS 片）"
  pids=()
  for i in $(seq 0 $((SHARDS - 1))); do
    OMP_NUM_THREADS=4 "$PY" -m immunization_baseline.cli.measure_semantic_identity --device cpu --scenario flux --data-root $D \
        --purifiers $OPS vae_trans --edits-csv $EDITS --output-csv $R/semantic_identity_$i.csv --shard $i --shards $SHARDS \
        > $R/semantic_identity_$i.log 2>&1 &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p" || { echo "[FATAL] 語意分片失敗（行程 $p），見 $R/semantic_identity_*.log" >&2; exit 1; }; done
  "$PY" - "$R" "$SHARDS" <<'EOF'
import csv, sys
from pathlib import Path
R, n = Path(sys.argv[1]), int(sys.argv[2])
rows = []
for i in range(n):
    with open(R / f"semantic_identity_{i}.csv", encoding="utf-8", newline="") as f:
        rows += list(csv.DictReader(f))
keys = {(r["condition"], r["image"], r["prompt_index"], r["purifier"]) for r in rows}
if len(keys) != len(rows):
    raise SystemExit(f"分片合併後有重複格：{len(rows)} 列、{len(keys)} 格")
with open(R / "semantic_identity.csv", "w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
    w.writeheader()
    w.writerows(rows)
print(f"[MERGED] semantic_identity.csv {len(rows)} 格")
EOF
fi

step CLIP-S 與 ArcFace
"$PY" scripts/measure_clip_score.py --edits-csv $EDITS --output-csv $R/clip_score.csv --device $DEVICE
CUDA_VISIBLE_DEVICES= "$PY" scripts/measure_pexels_arcface_pair.py --semantic-csv $R/semantic_identity.csv --output-csv $R/arcface_pair.csv

step FID／KID 與 SSIM／PSNR
CUDA_VISIBLE_DEVICES= "$PY" scripts/measure_distribution_metrics.py $R/distribution --dataset pexels --pairs-csv $R/pairs.csv
CUDA_VISIBLE_DEVICES= "$PY" scripts/measure_pexels_edit_fidelity.py pairs --pairs-csv $R/pairs.csv --output-csv $R/edit_fidelity.csv --workers 16

step 彙整
"$PY" scripts/generate_pexels_flux_512_summary.py --images-file "$IMAGES_FILE" --conditions $CONDS --readouts-dir "$R"
step "READOUTS DONE"

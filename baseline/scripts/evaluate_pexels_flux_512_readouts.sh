#!/usr/bin/env bash
# Pexels 人像 FLUX 512 協定（資料版本 flux_512，50 張）的讀數：等失真 photoguard_encoder、dct_shield、dct_shield_y、mist 與 color3d_pexels，
# 分母為同一資料版本的 undefended。程式與 scripts/evaluate_pexels_ultraedit_readouts.sh 相同，位移與保留率改用 FLUX 版面的
# measure_flux_displacement、measure_flux_retention（--resolution 512）。
#   逐格表   scripts/generate_pexels_flux_512_cell_tables.py
#   位移     measure_flux_displacement（淨化前）；保留率 measure_flux_retention（七道與 VAE-Trans）
#   語意     measure_semantic_identity --edits-csv（CLIP-I＝clip_pair、SigLIP、ArcFace(原圖, 編輯)）
#   CLIP-S   scripts/measure_clip_score.py --edits-csv；ArcFace(E_u, E_d) scripts/measure_pexels_arcface_pair.py
#   FID／KID scripts/measure_distribution_metrics.py --pairs-csv；SSIM／PSNR scripts/measure_pexels_edit_fidelity.py pairs
# 位移、保留率、語意與 CLIP-S 在 CUDA_VISIBLE_DEVICES 有卡時用 GPU，否則用 CPU；ArcFace(E_u, E_d)、FID／KID、SSIM／PSNR 固定 CPU。
# 50 張的 FLUX 512 完成標記（runtime/pexels_portraits/flux_512/done/<影像>.done）不齊時以結束碼 4 結束。
# 輸出 runtime/pexels_portraits/flux_512/readouts/（彙整見 scripts/generate_pexels_flux_512_summary.py）。
# 用法（遠端 baseline 專案根）：ENV_FILE=~/env.sh bash scripts/evaluate_pexels_flux_512_readouts.sh
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
cd "$BASELINE_ROOT"
CONDS="photoguard_encoder dct_shield dct_shield_y mist color3d_pexels"
OPS="blur1 blur2 crop_resize0.1 jpeg30 jpeg50 jpeg80 rotate15"
D=data/pexels_portraits
X=artifacts/pexels_portraits/flux_512
S=runtime/pexels_portraits/flux_512
R=$S/readouts
step() { echo "[STEP] $(date -Iseconds) $*"; }

missing=0
for img in $(cat ../color/trials/lut3d_defense_and_naturalness/pexels_subsets/flux.txt); do
  [ -f "$S/done/$img.done" ] || missing=$((missing + 1))
done
[ "$missing" -eq 0 ] || { step "前置條件不成立：FLUX 512 尚缺 $missing 張"; exit 4; }
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then DEVICE=cuda; else DEVICE=cpu; fi
step "device=$DEVICE（CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}）"

step 逐格表
"$PY" scripts/generate_pexels_flux_512_cell_tables.py
EDITS="$R/edits_undefended.csv $(for c in $CONDS; do printf '%s ' "$R/edits_$c.csv"; done)"

step 位移與保留率
"$PY" -m immunization_baseline.cli.measure_flux_displacement --data-root $D --edits-csv-dir $R/plain --edits-root $X/edits \
    --conditions $CONDS --resolution 512 --output-csv $R/displacement.csv
"$PY" -m immunization_baseline.cli.measure_flux_retention --data-root $D --purified-edits-root $X/purified_edits \
    --displacement-csv $R/displacement.csv --conditions $CONDS --purifiers $OPS vae_trans --resolution 512 --output-csv $R/retention.csv

step 語意與身分
"$PY" -m immunization_baseline.cli.measure_semantic_identity --device $DEVICE --scenario flux --data-root $D \
    --purifiers $OPS vae_trans --edits-csv $EDITS --output-csv $R/semantic_identity.csv

step CLIP-S 與 ArcFace
"$PY" scripts/measure_clip_score.py --edits-csv $EDITS --output-csv $R/clip_score.csv --device $DEVICE
CUDA_VISIBLE_DEVICES= "$PY" scripts/measure_pexels_arcface_pair.py --semantic-csv $R/semantic_identity.csv --output-csv $R/arcface_pair.csv

step FID／KID 與 SSIM／PSNR
CUDA_VISIBLE_DEVICES= "$PY" scripts/measure_distribution_metrics.py $R/distribution --dataset pexels --pairs-csv $R/pairs.csv
CUDA_VISIBLE_DEVICES= "$PY" scripts/measure_pexels_edit_fidelity.py pairs --pairs-csv $R/pairs.csv --output-csv $R/edit_fidelity.csv --workers 16

step 彙整
"$PY" scripts/generate_pexels_flux_512_summary.py
step "READOUTS DONE"

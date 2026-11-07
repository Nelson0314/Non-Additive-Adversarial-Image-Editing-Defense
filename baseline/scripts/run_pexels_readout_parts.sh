#!/usr/bin/env bash
# scripts/evaluate_pexels_flux_512_readouts.sh 的分段版（佇列工作），把語意與身分讀數分成多個 GPU 工作，並把 CPU 步驟移出持卡的工作：
#   base            逐格表、位移、保留率（GPU）；前置條件與 evaluate_pexels_flux_512_readouts.sh 相同，不齊時以結束碼 4 結束
#   sem <k> <n>     語意與身分第 k／n 片（GPU），輸出 semantic_identity_<k>.csv
#   finish_gpu <n>  合併 n 片 → semantic_identity.csv，CLIP-S 與 FID／KID（CLIP-IQA、InceptionV3 特徵在 GPU）
#   finish_cpu      ArcFace(E_u, E_d)、SSIM／PSNR（CPU）與彙整
# 各步驟的程式與參數同 evaluate_pexels_flux_512_readouts.sh；輸出到 READOUTS_DIR。
# 用法（遠端 baseline 專案根）：EDITOR_NAME=<名> READOUTS_DIR=<目錄> IMAGES_FILE=<檔> CONDS="..." ENV_FILE=~/env.sh bash scripts/run_pexels_readout_parts.sh <段> ...
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
cd "$BASELINE_ROOT"
: "${EDITOR_NAME:?}" "${READOUTS_DIR:?}" "${IMAGES_FILE:?}" "${CONDS:?}"
OPS="blur1 blur2 crop_resize0.1 jpeg30 jpeg50 jpeg80 rotate15"
D=data/pexels_portraits
X=artifacts/pexels_portraits/$EDITOR_NAME
S=runtime/pexels_portraits/$EDITOR_NAME
R=$READOUTS_DIR
EDITS="$R/edits_undefended.csv $(for c in $CONDS; do printf '%s ' "$R/edits_$c.csv"; done)"
step() { echo "[STEP] $(date -Iseconds) $*"; }
need_gpu() { [ -n "${CUDA_VISIBLE_DEVICES:-}" ] || { echo "[FATAL] $1 需要 GPU（CUDA_VISIBLE_DEVICES 未設定）" >&2; exit 1; }; }
case ${1:?用法：$0 <base|sem|finish_gpu|finish_cpu> ...} in
  base)
    need_gpu base
    missing=0
    for img in $(cat "$IMAGES_FILE"); do
      for c in $CONDS undefended; do
        for op in none vae_trans $OPS; do [ -f "$S/done/${img}__${c}__$op.done" ] || { missing=$((missing + 1)); break 2; }; done
      done
    done
    [ "$missing" -eq 0 ] || { step "前置條件不成立：$EDITOR_NAME 尚缺 $missing 張"; exit 4; }
    step "逐格表（影像 $(wc -w < "$IMAGES_FILE") 張）"
    "$PY" scripts/generate_pexels_flux_512_cell_tables.py --images-file "$IMAGES_FILE" --conditions $CONDS --output-dir "$R" --chunks-dir "$S/chunks"
    step 位移與保留率
    "$PY" -m immunization_baseline.cli.measure_flux_displacement --data-root $D --edits-csv-dir $R/plain --edits-root $X/edits \
        --conditions $CONDS --resolution 512 --output-csv $R/displacement.csv
    "$PY" -m immunization_baseline.cli.measure_flux_retention --data-root $D --purified-edits-root $X/purified_edits \
        --displacement-csv $R/displacement.csv --conditions $CONDS --purifiers $OPS vae_trans --resolution 512 --output-csv $R/retention.csv
    ;;
  sem)
    need_gpu sem
    step "語意分片 $2／$3（GPU $CUDA_VISIBLE_DEVICES）"
    "$PY" -m immunization_baseline.cli.measure_semantic_identity --device cuda --scenario flux --data-root $D \
        --purifiers $OPS vae_trans --edits-csv $EDITS --output-csv $R/semantic_identity_$2.csv --shard "$2" --shards "$3"
    ;;
  finish_gpu)
    need_gpu finish_gpu
    "$PY" - "$R" "$2" <<'EOF'
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
    w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n"); w.writeheader(); w.writerows(rows)
print(f"[MERGED] semantic_identity.csv {len(rows)} 格")
EOF
    step CLIP-S
    "$PY" scripts/measure_clip_score.py --edits-csv $EDITS --output-csv $R/clip_score.csv --device cuda
    step FID／KID
    "$PY" scripts/measure_distribution_metrics.py $R/distribution --dataset pexels --pairs-csv $R/pairs.csv --device cuda
    ;;
  finish_cpu)
    export CUDA_VISIBLE_DEVICES=
    step "ArcFace、SSIM／PSNR"
    "$PY" scripts/measure_pexels_arcface_pair.py --semantic-csv $R/semantic_identity.csv --output-csv $R/arcface_pair.csv
    "$PY" scripts/measure_pexels_edit_fidelity.py pairs --pairs-csv $R/pairs.csv --output-csv $R/edit_fidelity.csv --workers 16
    step 彙整
    "$PY" scripts/generate_pexels_flux_512_summary.py --images-file "$IMAGES_FILE" --conditions $CONDS --readouts-dir "$R"
    step "READOUTS DONE"
    ;;
  *) echo "[FATAL] 未知的段 $1" >&2; exit 2 ;;
esac

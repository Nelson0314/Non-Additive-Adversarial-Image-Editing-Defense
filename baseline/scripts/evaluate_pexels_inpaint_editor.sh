#!/usr/bin/env bash
# SD inpainting 對一張 Pexels 影像的跨編輯器協定：七條件（undefended、等失真 photoguard_encoder、dia_pt、dct_shield、sifm、BlurGuard、
# color3d_scale2p5_pexels）× {淨化前、jpeg50、blur2、VAE-Trans} × 4 句 ＝ 112 次編輯；輸入為 ip2p 流程的產物，不重新求解。
# 編輯協定、輸入解析與輸出版面見 scripts/run_inpaint_edits.py。完成標記 runtime/pexels_portraits/inpaint/done/<影像>.done。
# 用法（遠端 baseline 專案根；CUDA_VISIBLE_DEVICES 由呼叫端的租約設定）：ENV_FILE=~/env.sh bash scripts/evaluate_pexels_inpaint_editor.sh <影像>
set -uo pipefail
[ $# -eq 1 ] || { echo "用法：$0 <影像>" >&2; exit 2; }
IMG=$1
source "$(dirname "${BASH_SOURCE[0]}")/env.sh" || exit 1
cd "$BASELINE_ROOT" || exit 1
: "${CUDA_VISIBLE_DEVICES:?需要由租約設定 CUDA_VISIBLE_DEVICES}"
S=runtime/pexels_portraits/inpaint
mkdir -p "$S/done"
[ -f "$S/done/$IMG.done" ] && { echo "[STEP] inpaint $IMG 已完成"; exit 0; }
echo "[STEP] $(date -Iseconds) inpaint $IMG（GPU $CUDA_VISIBLE_DEVICES）"
HF_HOME=${HF_HOME:-/var/cache/huggingface} "$PY" scripts/run_inpaint_edits.py --images "$IMG" || exit 1
touch "$S/done/$IMG.done"
echo "[STEP] $(date -Iseconds) INPAINT DONE $IMG"

#!/usr/bin/env bash
# 補上缺了整個 session 的對照組：**`latent_norm` 用修好的固定步長跑一次。**
#
# 為什麼這一格不能少
# ────────────────────────────────────────────────────────────────────
# 影像引導消除損失的所有讀數，比較對象都是**沒有用同樣設定跑過的**
# `latent_norm`。現有的唯一對照是 1000 步的舊資料（步長還綁在半徑上，即後來
# 修掉的那個缺陷），步數與步長都不同，兩邊都不能算數。
#
# 沒有這一格，「新損失比舊損失好」這句話在本專案裡沒有證據——不論好或壞。
#
# 設定與 `runs/ip2p_eot_matched/eb_off` **完全相同**（同兩張影像、radius 2.5、
# 固定步長 0.01、8000 步、同樣的 early stop），只換 `--loss`，所以兩者可以
# 逐欄相比。抗淨化沿用 `purify_points.sh`，地板也是同一份。
#
# 用法：bash scripts/latent_norm_control.sh "<卡號>"
set -uo pipefail
ROOT=/nfs/home/nelson0314/WACV-s3
PY="$HOME/venvs/wacv/bin/python"
export PYTHONPATH="$ROOT" HF_HOME="$HOME/hf_cache" PYTHONIOENCODING=utf-8
export TOKENIZERS_PARALLELISM=false
cd "$ROOT" || exit 2

DEV="${1:-}"
[ -z "$DEV" ] && { echo "用法：$0 \"<卡號>\"" >&2; exit 2; }
bash scripts/free_cards.sh --assert "$DEV" || exit 3

OUT=runs/ip2p_latent_norm_ctrl
mkdir -p "$OUT"
IMGS="task_attr_mod_color_11699 task_obj_remove_380621"

CUDA_VISIBLE_DEVICES="$DEV" setsid nohup "$PY" scripts/ip2p_run.py \
    --out "$OUT/ln_fixed" \
    --data data/omniedit150 --conditions phase_gain --quantile 0 \
    --freq-weight jpeg_luma --freq-weight-power 0.25 --hop 8 --gain-ratio 1.0 \
    --loss latent_norm --radius 2.5 \
    --steps 8000 --step-size 0.01 --eval-every 100 --eval-draws 8 \
    --patience 15 --min-delta 0.0002 --save-weights --skip-existing \
    --images $IMGS < /dev/null >> "$OUT/ln_fixed.log" 2>&1 &
disown
sleep 15
echo "[lnctrl] ln_fixed dev=$DEV  process=$(ps -u "$USER" -o cmd | grep -c '[i]p2p_run')"

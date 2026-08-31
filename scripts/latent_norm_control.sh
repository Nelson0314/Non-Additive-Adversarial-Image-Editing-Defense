#!/usr/bin/env bash
# 補上缺了整個 session 的對照組：**`latent_norm` 用修好的固定步長跑滿訓練。**
#
# 為什麼這一格不能少
# ────────────────────────────────────────────────────────────────────
# 影像引導消除損失的所有讀數，比較對象都是**沒有用同樣設定跑過的**
# `latent_norm`。現有的唯一對照是 1000 步的舊資料，而且步長那時還綁在半徑上
# （α = radius/(steps·0.25)，即後來修掉的缺陷——「跑更多步」同時「每步走更
# 小」，兩者互相抵銷）。步數與步長都不同，兩邊都不能算數。
#
# 沒有這一格，「新損失比舊損失好」這句話在本專案裡沒有證據——不論好或壞。
#
# 設定與 `runs/ip2p_eot_matched/eb_off` **完全相同**（同兩張影像、radius 2.5、
# 固定步長 0.01、8000 步、同樣的 early stop、同一顆種子），**只換 `--loss`**，
# 所以兩者可以逐欄相比。
#
# 為什麼一張影像一個 process
# ────────────────────────────────────────────────────────────────────
# 兩張影像在驅動裡是**依序**跑的，一個 process 要 2.2 小時。每張影像的最佳化
# 彼此獨立（`param.reset(x01, seed)` 與損失都逐圖重建），拆成兩個 process
# 逐位元相同而牆鐘減半。**但兩個 process 不可寫同一個輸出目錄**——每張寫檔
# 是整份重寫 `results.csv`，並行會互相截掉（`--skip-existing` 那個坑的同型）。
# 故各自一個目錄，最後合併。
#
# 用法：bash scripts/latent_norm_control.sh "<兩個卡號>"
set -uo pipefail
ROOT=/nfs/home/nelson0314/WACV-s3
PY="$HOME/venvs/wacv/bin/python"
export PYTHONPATH="$ROOT" HF_HOME="$HOME/hf_cache" PYTHONIOENCODING=utf-8
export TOKENIZERS_PARALLELISM=false
cd "$ROOT" || exit 2

DEVS=(${1:-})
[ ${#DEVS[@]} -ne 2 ] && { echo "用法：$0 \"<兩個卡號>\"" >&2; exit 2; }
bash scripts/free_cards.sh --assert "${DEVS[*]}" || exit 3

OUT=runs/ip2p_latent_norm_ctrl
mkdir -p "$OUT"
IMGS=(task_attr_mod_color_11699 task_obj_remove_380621)

BASE="--data data/omniedit150 --conditions phase_gain --quantile 0 \
--freq-weight jpeg_luma --freq-weight-power 0.25 --hop 8 --gain-ratio 1.0 \
--loss latent_norm --radius 2.5 \
--steps 8000 --step-size 0.01 --eval-every 100 --eval-draws 8 \
--patience 15 --min-delta 0.0002 --save-weights --skip-existing"

for i in 0 1; do
  tag="ln_part$((i + 1))"
  CUDA_VISIBLE_DEVICES="${DEVS[$i]}" setsid nohup "$PY" scripts/ip2p_run.py \
      --out "$OUT/$tag" $BASE --images "${IMGS[$i]}" \
      < /dev/null >> "$OUT/$tag.log" 2>&1 &
  disown
  echo "[lnctrl] $tag dev=${DEVS[$i]} image=${IMGS[$i]}"
done
sleep 20
echo "啟動了 $(ps -u "$USER" -o cmd | grep -c '[i]p2p_run') 個 ip2p_run process"

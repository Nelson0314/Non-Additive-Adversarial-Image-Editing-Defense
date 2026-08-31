#!/usr/bin/env bash
# 乙：把 `eot_broad` 放到**等失真**上。四點、兩張影像、8000 步。
#
# 為什麼需要這一輪
# ────────────────────────────────────────────────────────────────────
# `ih_eot`（`runs/ip2p_ig_harden`，十張）在每一個被洗過的欄位都贏對照
# `ig_d25`——模糊 σ1 0.3221 對 0.1787（1.80 倍）、JPEG30 0.3252 對 0.2604
# （1.25 倍）——**但它付得比較多**：DISTS 0.2065 對 0.1996、PSNR 20.56 對
# 22.08、SSIM 0.751 對 0.803、L∞ 0.960 對 0.918。
#
# 於是那些倍數**不可歸因**：分不出是 EOT 買到的，還是多付換來的。三個較低的
# 半徑讓它的失真跨過對照的 0.1996，可直接讀等失真、**不必外插**。
#
# `eb_off` 是同批的無 EOT 基準線，**不可省**——跨批次比對照會把「這兩張」與
# 「那十張」的差混進來。
#
# 其餘旗標與 `ig_d25` 完全相同。
#
# 用法：bash scripts/eot_broad_matched.sh "<四個卡號>"
set -uo pipefail
ROOT=/nfs/home/nelson0314/WACV-s3
PY="$HOME/venvs/wacv/bin/python"
export PYTHONPATH="$ROOT" HF_HOME="$HOME/hf_cache" PYTHONIOENCODING=utf-8
export TOKENIZERS_PARALLELISM=false
cd "$ROOT" || { echo "錯誤：找不到 $ROOT" >&2; exit 2; }

DEVS=(${1:-})
[ ${#DEVS[@]} -ne 4 ] && { echo "用法：$0 \"<四個卡號>\"" >&2; exit 2; }
bash scripts/free_cards.sh --assert "${DEVS[*]}" || exit 3

OUT=runs/ip2p_eot_matched
mkdir -p "$OUT"
IMGS="task_attr_mod_color_11699 task_obj_remove_380621"

BASE="--data data/omniedit150 --conditions phase_gain --quantile 0 \
--freq-weight jpeg_luma --freq-weight-power 0.25 --hop 8 --gain-ratio 1.0 \
--loss image_guidance --ig-zt diffuse_src \
--steps 8000 --step-size 0.01 --eval-every 100 --eval-draws 8 \
--patience 15 --min-delta 0.0002 --save-weights --skip-existing"

POINTS="
eb_off:--radius~2.5
eb_r20:--purify-aware~eot_broad~--radius~2.0
eb_r22:--purify-aware~eot_broad~--radius~2.2
eb_r25:--purify-aware~eot_broad~--radius~2.5
"

i=0
for p in $POINTS; do
  IFS=: read -r tag extra <<< "$p"
  dev=${DEVS[$i]}; i=$((i + 1))
  CUDA_VISIBLE_DEVICES="$dev" setsid nohup "$PY" scripts/ip2p_run.py \
      --out "$OUT/$tag" $BASE $(echo "$extra" | tr '~' ' ') \
      --images $IMGS < /dev/null >> "$OUT/$tag.log" 2>&1 &
  disown
  echo "[eotmatch] $tag dev=$dev $(echo "$extra" | tr '~' ' ')"
done
sleep 20
echo "啟動了 $(ps -u "$USER" -o cmd | grep -c '[i]p2p_run') 個 ip2p_run process"

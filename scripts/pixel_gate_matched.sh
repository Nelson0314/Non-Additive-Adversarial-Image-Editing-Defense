#!/usr/bin/env bash
# 逐像素遮罩加到**等失真**。一張影像、8000 步、四個半徑。
#
# 第二輪留下的唯一混淆
# ────────────────────────────────────────────────────────────────────
# 逐像素遮罩把牆面的波紋清乾淨了，但四個條件的編輯結果都不重畫。問題是
# 那不是等失真的比較：遮罩把擾動限制在紋理像素上，同一個半徑交出去的總
# 改變量因此少很多——DISTS 0.059–0.096 對上對照的 0.151。位移掉下來有一
# 部分只是**付得比較少**。
#
# 加半徑在第二輪沒有解決，而且方向是反的：`px_r35`（半徑 3.5 ＋ quantile
# 0.5）的位移 0.276 低於 `px_only`（半徑 2.5、不開 quantile）的 0.404。
# 兩者只差 `--quantile 0.5`，所以能量分位數那個因子在扣分——本輪一律不開。
#
# 四個半徑跨過對照的 DISTS 0.151，於是可以直接在等失真上比，不必外插。
#   po_r35 / po_r45 / po_r60   σ2，半徑 3.5 / 4.5 / 6.0
#   po_s3r45                   σ3，半徑 4.5（遮罩略放寬的同一失真點）
#
# **對照是 `ig_d25`**（無旗標、8000 步、同一張影像、DISTS 0.1514、位移
# 0.5638、編輯後重畫）。
#
# 一張影像，不是兩張：使用者裁定第二張不必跑，直接進下一輪。
#
# 用法：bash scripts/pixel_gate_matched.sh "<四個卡號>"
set -uo pipefail
ROOT=/nfs/home/nelson0314/WACV-s3
PY="$HOME/venvs/wacv/bin/python"
export PYTHONPATH="$ROOT" HF_HOME="$HOME/hf_cache" PYTHONIOENCODING=utf-8
export TOKENIZERS_PARALLELISM=false
cd "$ROOT" || { echo "錯誤：找不到 $ROOT" >&2; exit 2; }

DEVS=(${1:-})
[ ${#DEVS[@]} -ne 4 ] && { echo "用法：$0 \"<四個卡號>\"" >&2; exit 2; }
bash scripts/free_cards.sh --assert "${DEVS[*]}" || exit 3

OUT=runs/ip2p_pixel_matched
mkdir -p "$OUT"
IMGS="task_attr_mod_color_11699"

BASE="--data data/omniedit150 --conditions phase_gain --quantile 0 \
--freq-weight jpeg_luma --freq-weight-power 0.25 --hop 8 --gain-ratio 1.0 \
--loss image_guidance --ig-zt diffuse_src \
--steps 8000 --step-size 0.01 --eval-every 100 --eval-draws 8 \
--patience 15 --min-delta 0.0002 --save-weights --skip-existing"

POINTS="
po_r35:--pixel-gate-sigma~2~--radius~3.5
po_r45:--pixel-gate-sigma~2~--radius~4.5
po_r60:--pixel-gate-sigma~2~--radius~6.0
po_s3r45:--pixel-gate-sigma~3~--radius~4.5
"

i=0
for p in $POINTS; do
  IFS=: read -r tag extra <<< "$p"
  dev=${DEVS[$i]}; i=$((i + 1))
  CUDA_VISIBLE_DEVICES="$dev" setsid nohup "$PY" scripts/ip2p_run.py \
      --out "$OUT/$tag" $BASE $(echo "$extra" | tr '~' ' ') \
      --images $IMGS < /dev/null >> "$OUT/$tag.log" 2>&1 &
  disown
  echo "[matched] $tag dev=$dev $(echo "$extra" | tr '~' ' ')"
done
sleep 20
echo "啟動了 $(ps -u "$USER" -o cmd | grep -c '[i]p2p_run') 個 ip2p_run process"

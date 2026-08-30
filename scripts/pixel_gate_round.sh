#!/usr/bin/env bash
# 逐像素閘的完整步數驗證。兩張影像、8000 步、四個條件。
#
# 第一輪（`scripts/visual_screen.sh`，2000 步）看到的
# ────────────────────────────────────────────────────────────────────
# 造成整張波紋的是**閘的空間解析度**，不是頻帶也不是能量分位數：
#   `--r-max 0.7`     牆面波紋仍在
#   `--quantile 0.5`  牆面波紋仍在，只稍淡
#   `--pixel-gate-sigma 2`  **牆面完全乾淨**，擾動只落在葉子與臉
# 閘原本逐 32×32 區塊，一個橫跨葉子與牆的區塊會把牆一起畫花；逐像素不會。
#
# 第一輪答不了的問題，以及為什麼
# ────────────────────────────────────────────────────────────────────
# 四個條件的編輯圖**全部是劣化**（人還在、場景完整），包含基準線。而基準線
# 的設定與 `ig_d25` 完全相同、只差步數 2000 對 8000，`ig_d25` 那一張是**重畫**。
# 也就是 2000 步時連基準線自己都還不會重畫，該輪因此無法分辨旗標對重畫的
# 影響。**這是把步數砍到 2000 換速度的代價。** 本輪把步數放回 8000。
#
# 四個條件
# ────────────────────────────────────────────────────────────────────
#   px_r25    逐像素閘 σ2 ＋ quantile 0.5，radius 2.5。牆乾淨的解跑滿步數
#             之後會不會重畫。
#   px_r35    同上但 radius 3.5。逐像素閘讓擾動的作用面積變小，同一個半徑
#             交付的總改變量因此較少；加預算能不能把重畫換回來。
#   px_only   只有 σ2（quantile 0）。拆開兩個旗標，看逐像素閘是不是單獨就夠。
#   px_s4r35  σ4 ＋ radius 3.5。遮罩放粗，在乾淨與強度之間取中間點。
#
# **對照組不必另跑**：`ig_d25`（無旗標、8000 步、同樣的設定）就是它，而本輪
# 這兩張都在它的十張裡。
#
# 用法：bash scripts/pixel_gate_round.sh "<四個卡號>"
set -uo pipefail
ROOT=/nfs/home/nelson0314/WACV-s3
PY="$HOME/venvs/wacv/bin/python"
export PYTHONPATH="$ROOT" HF_HOME="$HOME/hf_cache" PYTHONIOENCODING=utf-8
export TOKENIZERS_PARALLELISM=false
cd "$ROOT" || { echo "錯誤：找不到 $ROOT" >&2; exit 2; }

DEVS=(${1:-})
[ ${#DEVS[@]} -ne 4 ] && { echo "用法：$0 \"<四個卡號>\"" >&2; exit 2; }
bash scripts/free_cards.sh --assert "${DEVS[*]}" || exit 3

OUT=runs/ip2p_pixel_gate
mkdir -p "$OUT"
IMGS="task_attr_mod_color_11699 task_obj_remove_380621"

BASE="--data data/omniedit150 --conditions phase_gain --quantile 0 \
--freq-weight jpeg_luma --freq-weight-power 0.25 --hop 8 --gain-ratio 1.0 \
--loss image_guidance --ig-zt diffuse_src \
--steps 8000 --step-size 0.01 --eval-every 100 --eval-draws 8 \
--patience 15 --min-delta 0.0002 --save-weights --skip-existing"

POINTS="
px_r25:--quantile~0.5~--pixel-gate-sigma~2~--radius~2.5
px_r35:--quantile~0.5~--pixel-gate-sigma~2~--radius~3.5
px_only:--pixel-gate-sigma~2~--radius~2.5
px_s4r35:--quantile~0.5~--pixel-gate-sigma~4~--radius~3.5
"

i=0
for p in $POINTS; do
  IFS=: read -r tag extra <<< "$p"
  dev=${DEVS[$i]}; i=$((i + 1))
  CUDA_VISIBLE_DEVICES="$dev" setsid nohup "$PY" scripts/ip2p_run.py \
      --out "$OUT/$tag" $BASE $(echo "$extra" | tr '~' ' ') \
      --images $IMGS < /dev/null >> "$OUT/$tag.log" 2>&1 &
  disown
  echo "[pxgate] $tag dev=$dev $(echo "$extra" | tr '~' ' ')"
done
sleep 20
echo "啟動了 $(ps -u "$USER" -o cmd | grep -c '[i]p2p_run') 個 ip2p_run process"

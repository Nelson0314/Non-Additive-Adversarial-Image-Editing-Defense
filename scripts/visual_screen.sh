#!/usr/bin/env bash
# 視覺篩選：兩張影像、短步數、一輪四個條件，用**眼睛**判定。
#
# 為什麼是這一支而不是接著跑大批次
# ────────────────────────────────────────────────────────────────────
# 十張 × 8000 步 × 四條件要十一小時，而要判的問題是「防禦圖看起來怎麼樣、
# 編輯結果是重畫還是劣化」——那不需要十張，也不需要跑到收斂就看得出方向。
# 使用者已明確授權由助手的視覺判定，並要求在做出來之前把規模降到兩張。
#
# **這一批的數字不是收斂值，不可與 `ip2p_ig_converge`／`ip2p_ig_harden`
# 的表混在同一列。** 它只用來挑方向，挑到之後再用完整設定重跑。
#
# 要處理的兩個症狀（都是看圖看到的，不是從指標推的）
# ────────────────────────────────────────────────────────────────────
# 1. 防禦圖上有一層**整張、空間上連貫的波紋**，連平坦的牆面都有浮雕質感。
#    L∞ 投影壓不掉它：`ih_l15`（L∞ 0.150）與對照（0.973）肉眼幾乎一樣。
#    L∞ 管的是單一像素的最大偏移，而這是每個像素中等幅度、方向協調的形變。
# 2. `ih_l15` 的編輯結果是**劣化**（人還在、只是被蓋上紋理）而不是**重畫**，
#    但 SigLIP 代理把它標成 blocked。擋下與否要用眼睛判。
#
# 四個條件對應四個假說，全部是既有旗標，不需要新程式
# ────────────────────────────────────────────────────────────────────
#   sc_base   無新旗標。**同步數的基準線**，沒有它就無法分辨改善來自旗標
#             還是來自步數不同。
#   sc_q05    `--quantile 0.5`。紋理閘的能量因子目前是**關著的**
#             （`--quantile 0` 使該因子恆為 1，見 ip2p_run.py 的說明），
#             於是平坦的牆和有紋理的葉子拿到一樣多的預算。開回來會把擾動
#             趕向高梯度的區塊。
#   sc_px     再加 `--pixel-gate-sigma 2`。閘目前是逐 32×32 區塊的，一個
#             橫跨葉子與牆的區塊會把牆一起畫花；逐像素的遮罩不會。
#   sc_rmax   `--r-max 0.7`。浮雕般的細質感住在最高頻帶，砍掉上界看它是不是
#             那一帶造成的。本模組原本只有下界。
#
# 用法：bash scripts/visual_screen.sh "<四個卡號>"
set -uo pipefail
ROOT=/nfs/home/nelson0314/WACV-s3
PY="$HOME/venvs/wacv/bin/python"
export PYTHONPATH="$ROOT" HF_HOME="$HOME/hf_cache" PYTHONIOENCODING=utf-8
export TOKENIZERS_PARALLELISM=false
cd "$ROOT" || { echo "錯誤：找不到 $ROOT" >&2; exit 2; }

DEVS=(${1:-})
[ ${#DEVS[@]} -ne 4 ] && { echo "用法：$0 \"<四個卡號>\"" >&2; exit 2; }
bash scripts/free_cards.sh --assert "${DEVS[*]}" || exit 3

OUT=runs/ip2p_visual_screen
mkdir -p "$OUT"

# 兩張：一張大面積平坦牆面（波紋最明顯），一張場景較滿的。
IMGS="task_attr_mod_color_11699 task_obj_remove_380621"
STEPS="${STEPS:-2000}"

BASE="--data data/omniedit150 --conditions phase_gain --quantile 0 \
--freq-weight jpeg_luma --freq-weight-power 0.25 --hop 8 --gain-ratio 1.0 \
--loss image_guidance --ig-zt diffuse_src --radius 2.5 \
--steps $STEPS --step-size 0.01 --eval-every 100 --eval-draws 8 \
--patience 15 --min-delta 0.0002 --save-weights --skip-existing"

# `--quantile` 在 BASE 裡已是 0，後面再給一次會覆蓋（argparse 取最後一個）。
POINTS="
sc_base:
sc_q05:--quantile~0.5
sc_px:--quantile~0.5~--pixel-gate-sigma~2
sc_rmax:--r-max~0.7
"

i=0
for p in $POINTS; do
  IFS=: read -r tag extra <<< "$p"
  dev=${DEVS[$i]}; i=$((i + 1))
  CUDA_VISIBLE_DEVICES="$dev" setsid nohup "$PY" scripts/ip2p_run.py \
      --out "$OUT/$tag" $BASE $(echo "$extra" | tr '~' ' ') \
      --images $IMGS < /dev/null >> "$OUT/$tag.log" 2>&1 &
  disown
  echo "[screen] $tag dev=$dev $(echo "$extra" | tr '~' ' ')"
done
sleep 20
echo "啟動了 $(ps -u "$USER" -o cmd | grep -c '[i]p2p_run') 個 ip2p_run process"

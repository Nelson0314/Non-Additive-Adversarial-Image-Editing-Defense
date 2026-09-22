#!/usr/bin/env bash
# 臂名 → 產防禦圖的指令。**所有臂的定義集中在這一個檔**，
# 鏈的腳本只認臂名，不認參數；參數散在多處時改一個臂會漏掉另一處而沒有症狀。
#
# 用法：bash lab/scripts/defence_cmd.sh <臂名>
#        產出寫到 lab/runs/defence/<臂名>/
set -uo pipefail
ARM="$1"
OUT="lab/runs/defence/$ARM"
DATA="lab/data/portraits"

case "$ARM" in
  # ---- 生成載體：直接交付 SDEdit 的輸出 ----
  # 兩臂只差 strength，其餘逐項相同（同一顆噪聲、同一個風格、同樣 20 步）。
  style_random)
    exec "$PY" lab/code/style_defence.py --arm style_random --out "$OUT" \
        --data "$DATA" --style film --strength 0.35 --num-steps 20 --guidance 7.5
    ;;
  style_low)
    exec "$PY" lab/code/style_defence.py --arm style_low --out "$OUT" \
        --data "$DATA" --style film --strength 0.15 --num-steps 20 --guidance 7.5
    ;;
  # ---- 生成載體：只取低頻色彩，結構留在原圖 ----
  style_filter)
    exec "$PY" lab/code/style_filter_defence.py --arm style_filter --out "$OUT" \
        --data "$DATA" --style film --strength 0.6 --num-steps 20 --guidance 7.5 \
        --sigma 12.0 --frame-cap 16.0 --face-cap 8.0
    ;;
  # ---- 色調曲線：臉與背景不同預算 ----
  curve_dual_spatial)
    exec "$PY" lab/code/curve_budget_defence.py --arm curve_dual_spatial --out "$OUT" \
        --data "$DATA" --budget-mode spatial \
        --frame-cap 16.0 --face-cap 8.0 --band-tv-cap 2.0 --feather 0.35
    ;;
  curve_dual_chroma)
    exec "$PY" lab/code/curve_budget_defence.py --arm curve_dual_chroma --out "$OUT" \
        --data "$DATA" --budget-mode chroma \
        --frame-cap 16.0 --face-cap 8.0
    ;;
  *)
    echo "未知的臂：$ARM" >&2; exit 2 ;;
esac

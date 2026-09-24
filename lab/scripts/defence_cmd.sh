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
  # ---- 已退役：直接交付 SDEdit 的輸出 ----
  # **這兩個臂已停止發展，逐格圖已刪**（見 lab/docs/DESIGN.md「已退役」一節）。
  # 配方留著，因為它是「SDEdit 直接交付會換人」那個結論的產生方式，刪掉就
  # 重現不了；防禦圖也留著（lab/runs/defence/ 各 8 張）當那個結論的證據。
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
  style_filter_guided)
    exec "$PY" lab/code/style_filter_defence.py --arm style_filter_guided --out "$OUT"         --data "$DATA" --style film --strength 0.6 --num-steps 20 --guidance 7.5         --smoother guided --radius 32 --eps 0.01 --frame-cap 16.0 --face-cap 8.0
    ;;
  # ---- 生成載體：局部仿射色彩轉移（身分由 photorealism prior 保證）----
  style_affine)
    exec "$PY" lab/code/style_filter_defence.py --arm style_affine --out "$OUT"         --data "$DATA" --style film --strength 0.6 --num-steps 20 --guidance 7.5         --smoother affine --radius 48 --eps 0.05 --frame-cap 16.0 --face-cap 8.0
    ;;
  # ---- 生成載體：風格由最佳化在字典凸包上選，交付仍是局部仿射色彩轉移 ----
  style_opt)
    exec "$PY" lab/code/style_opt_defence.py --arm style_opt --out "$OUT"         --data "$DATA" --strength 0.6 --num-steps 10 --guidance 7.5         --radius 48 --eps 0.05 --max-scale 3.0         --frame-cap 16.0 --face-cap 8.0 --skin-radius 12.0 --chroma-gain 1.15         --steps 60 --lr 0.08
    ;;
  # ---- 生成載體：只重繪一塊，受保護的那一塊逐位元保留 ----
  inpaint_bg)
    exec "$PY" lab/code/inpaint_region_defence.py --arm inpaint_bg --out "$OUT"         --data "$DATA" --region background --style autumn --steps 50 --guidance 7.5
    ;;
  inpaint_outside_face)
    exec "$PY" lab/code/inpaint_region_defence.py --arm inpaint_outside_face         --out "$OUT" --data "$DATA" --region outside_face --style autumn         --steps 50 --guidance 7.5 --feather 0.35
    ;;
  # ---- spatial 那條線：從該張自己的膚色色調起步，另加彩度與同色上界 ----
  curve_dual_spatial_anchored)
    exec "$PY" lab/code/curve_budget_defence.py --arm curve_dual_spatial_anchored         --out "$OUT" --data "$DATA" --budget-mode spatial         --frame-cap 16.0 --face-cap 8.0 --band-tv-cap 2.0 --feather 0.35         --init skin_tone --init-strength 1.0 --skin-radius 12.0 --chroma-gain 1.15
    ;;
  # ---- 色度平面的全域扭曲（chroma 那條線的強化）----
  ab_warp)
    exec "$PY" lab/code/ab_warp_defence.py --arm ab_warp --out "$OUT"         --data "$DATA" --grid 7 --extent 90 --warp-radius 30 --pieces 16         --frame-cap 16.0 --face-cap 8.0 --skin-radius 12.0 --chroma-gain 1.15
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

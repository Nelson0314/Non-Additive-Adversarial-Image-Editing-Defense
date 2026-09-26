#!/usr/bin/env bash
# 臂名 → 產防禦圖的指令。所有臂的參數只定義在這一個檔。
#
# 用法：bash lab/scripts/defence_cmd.sh <臂名> [額外參數…]
#        產出寫到 lab/runs/defence/<臂名>/；DEF_OUT 可覆寫（分片執行用，
#        results.csv 每寫一列就整份重寫，不可共用目錄）。額外參數附在指令最後。
set -uo pipefail
ARM="$1"
OUT="${DEF_OUT:-lab/runs/defence/$ARM}"
DATA="lab/data/portraits"
# ab_warp 族的共同設定。_s12／_s16／_ch 的額度由使用者看預覽後定（同色 12–16 可接受）。
WARP_WIDE=(--grid 7 --extent 90 --warp-radius 80 --pieces 16 --frame-cap 32.0 --skin-radius 12.0 --chroma-gain 2.0)
# 分通道 Lab 位移 p95 上限：紅／洋紅（a*＋）與黃（b*＋）緊，綠青（a*－）與藍（b*－）寬。
CH_CAPS=(--a-pos-cap 4 --a-neg-cap 15 --b-pos-cap 4 --b-neg-cap 25 --l-abs-cap 15)

case "$ARM" in
  # ---- 不碰受保護區：只重繪一塊，受保護的那一塊逐位元保留 ----
  inpaint_bg)
    exec "$PY" lab/code/inpaint_region_defence.py --arm inpaint_bg --out "$OUT" \
        --data "$DATA" --region background --style autumn --steps 50 --guidance 7.5 ;;
  inpaint_outside_face)
    exec "$PY" lab/code/inpaint_region_defence.py --arm inpaint_outside_face --out "$OUT" \
        --data "$DATA" --region outside_face --style autumn --steps 50 --guidance 7.5 --feather 0.35 ;;
  # ---- 色調曲線：單一全域曲線，預算分在色度 ----
  curve_dual_chroma)
    exec "$PY" lab/code/curve_budget_defence.py --arm curve_dual_chroma --out "$OUT" \
        --data "$DATA" --frame-cap 16.0 --face-cap 8.0 ;;
  # ---- CIELAB (a,b) 平面的全域 RBF 位移 ＋ 單調亮度曲線 ----
  ab_warp)
    exec "$PY" lab/code/ab_warp_defence.py --arm ab_warp --out "$OUT" --data "$DATA" \
        --grid 7 --extent 90 --warp-radius 30 --pieces 16 \
        --frame-cap 16.0 --face-cap 8.0 --skin-radius 12.0 --chroma-gain 1.15 "${@:2}" ;;
  ab_warp_s12|ab_warp_s16)
    exec "$PY" lab/code/ab_warp_defence.py --arm "$ARM" --out "$OUT" --data "$DATA" \
        "${WARP_WIDE[@]}" --face-cap "${ARM#ab_warp_s}" "${@:2}" ;;
  ab_warp_ch)
    exec "$PY" lab/code/ab_warp_defence.py --arm "$ARM" --out "$OUT" --data "$DATA" \
        "${WARP_WIDE[@]}" --face-cap 16 "${CH_CAPS[@]}" "${@:2}" ;;
  # 等變殘差目標（_comm）與現行 FreeObjective（_free）：ab_warp_ch 的載體與上限，
  # 同一個非恆等起點，另加逐張輸入 LPIPS ≤ ab_warp_ch 防禦圖 ＋ 0.0025。
  ab_warp_ch_comm|ab_warp_ch_free)
    exec "$PY" lab/code/ab_warp_defence.py --arm "$ARM" --out "$OUT" --data "$DATA" \
        "${WARP_WIDE[@]}" --face-cap 16 "${CH_CAPS[@]}" \
        --objective "${ARM#ab_warp_ch_}" --init random --init-std 0.1 \
        --lpips-ref-arm ab_warp_ch --lpips-tolerance 0.0025 --noise-seed 0 "${@:2}" ;;
  # ---- 全域三角式 Lab 映射（112 參數），輸入 LPIPS ≤ ab_warp 逐張值 ----
  ab_prism)
    exec "$PY" -B lab/code/ab_prism_defence.py --arm ab_prism --out "$OUT" --data "$DATA" \
        --grid 5 --extent 90 --slope-span 1.5 --angle-degrees 12 --chroma-span 1.25 \
        --bias-radius 20 --init-std 0.03 --frame-cap 16 --face-cap 8 --skin-radius 12 \
        --chroma-gain 1.15 --lpips-targets lab/results/fidelity.csv --lpips-reference ab_warp \
        --lpips-tolerance 0.0025 --steps 900 --lr 0.02 --lr-final-ratio 0.2 --rho 10 \
        --lam-every 5 --check-every 10 --probe-every 50 --log-every 100 --noise-seed 0 \
        --no-lpips-lower "${@:2}" ;;
  # ---- 生成模型只選風格（SDEdit film，取自遠端既有的 style_affine/*__sdedit_raw.png），
  # 交付為 ab_warp 族的全域映射 ----
  style_warp)
    exec "$PY" -B lab/code/style_warp_defence.py --arm style_warp --out "$OUT" --data "$DATA" \
        --style-root lab/runs/defence/style_affine --grid 7 --extent 90 --warp-radius 30 \
        --pieces 16 --frame-cap 16.0 --face-cap 8.0 --skin-radius 12.0 --chroma-gain 1.15 \
        --steps 400 --lr 0.02 "${@:2}" ;;
  *)
    echo "未知的臂：$ARM" >&2; exit 2 ;;
esac

#!/usr/bin/env bash
# 臂名 → 產防禦圖的指令。**所有臂的定義集中在這一個檔**，
# 鏈的腳本只認臂名，不認參數；參數散在多處時改一個臂會漏掉另一處而沒有症狀。
#
# 用法：bash lab/scripts/defence_cmd.sh <臂名>
#        產出寫到 lab/runs/defence/<臂名>/
set -uo pipefail
ARM="$1"
# DEF_OUT 讓分片執行把輸出寫到各自的目錄（results.csv 每寫一列就整份重寫，不可共用）。
OUT="${DEF_OUT:-lab/runs/defence/$ARM}"
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
  # ---- 生成載體：局部仿射色彩轉移（身分由 photorealism prior 保證）----
  style_affine)
    exec "$PY" lab/code/style_filter_defence.py --arm style_affine --out "$OUT"         --data "$DATA" --style film --strength 0.6 --num-steps 20 --guidance 7.5         --radius 48 --eps 0.05 --frame-cap 16.0 --face-cap 8.0
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
  # ---- 色度平面的全域扭曲（chroma 那條線的強化）----
  ab_warp)
    exec "$PY" lab/code/ab_warp_defence.py --arm ab_warp --out "$OUT"         --data "$DATA" --grid 7 --extent 90 --warp-radius 30 --pieces 16         --frame-cap 16.0 --face-cap 8.0 --skin-radius 12.0 --chroma-gain 1.15
    ;;
  # Global monotone Lab coupling, calibrated to each ab_warp input LPIPS.
  ab_prism)
    exec "$PY" -B lab/code/ab_prism_defence.py --arm ab_prism --out "$OUT" \
        --data "$DATA" --grid 5 --extent 90 --slope-span 1.5 \
        --angle-degrees 12 --chroma-span 1.25 --bias-radius 20 --init-std 0.03 \
        --frame-cap 16 --face-cap 8 --skin-radius 12 --chroma-gain 1.15 \
        --lpips-targets lab/results/fidelity.csv --lpips-reference ab_warp \
        --lpips-tolerance 0.0025 --steps 900 --lr 0.02 --lr-final-ratio 0.2 \
        --rho 10 --lam-every 5 --check-every 10 --probe-every 50 \
        --log-every 100 --noise-seed 0 --no-lpips-lower "${@:2}"
    ;;
  # ab_prism 的同函數族隨機對照（AB_WARP_NEXT.md 6.3）。一次呼叫產出 r1..r5，
  # 已完成的影像跳過；共用輸出根目錄，必須序列執行（queue_worker 的 gen 工作）。
  # 隨機候選對齊的是 ab_prism 最佳化解實際達到的逐張 LPIPS（先量出來）。
  ab_prism_random_r[1-5])
    "$PY" lab/code/defence_fidelity.py --arms ab_prism         --out lab/results/fidelity_ab_prism.csv || exit 1
    exec "$PY" -B lab/code/ab_prism_random.py --out lab/runs/defence         --lpips-targets lab/results/fidelity_ab_prism.csv --lpips-reference ab_prism         --replicates 3 --max-candidates 4096 "${@:2}"
    ;;
  # ---- 備案：生成模型只選風格，交付為 ab_warp 那一族的全域映射 ----
  style_warp)
    exec "$PY" -B lab/code/style_warp_defence.py --arm style_warp --out "$OUT"         --data "$DATA" --style-root lab/runs/defence/style_affine         --grid 7 --extent 90 --warp-radius 30 --pieces 16         --frame-cap 16.0 --face-cap 8.0 --skin-radius 12.0 --chroma-gain 1.15         --steps 400 --lr 0.02 "${@:2}"
    ;;
  # ---- ab_warp 放寬額度（2026-09-25 使用者看預覽後定：同色 12–16 可接受）----
  # 臉框與同色上限同值（--face-cap 兩道共用）；整圖放到 32（預覽中碰不到）；
  # 位移半徑 30 → 80，否則推不到新額度；彩度 p95 上限 2.0×（預覽 12 級最高 1.98×）。
  ab_warp_s12|ab_warp_s16)
    exec "$PY" lab/code/ab_warp_defence.py --arm "$ARM" --out "$OUT"         --data "$DATA" --grid 7 --extent 90 --warp-radius 80 --pieces 16         --frame-cap 32.0 --face-cap "${ARM#ab_warp_s}" --skin-radius 12.0         --chroma-gain 2.0 "${@:2}"
    ;;
  # ---- ab_warp 分通道預算（2026-09-25）：紅／粉／洋紅（a*＋）與黃／暖黃（b*＋）緊，
  # 綠／青（a*－）與藍（b*－）寬。上限是逐像素 Lab 位移的 p95。同色 16。
  # 依據見 lab/docs/DESIGN.md「分通道預算」。
  ab_warp_ch)
    exec "$PY" lab/code/ab_warp_defence.py --arm "$ARM" --out "$OUT"         --data "$DATA" --grid 7 --extent 90 --warp-radius 80 --pieces 16         --frame-cap 32.0 --face-cap 16 --skin-radius 12.0 --chroma-gain 2.0         --a-pos-cap 4 --a-neg-cap 15 --b-pos-cap 4 --b-neg-cap 25 --l-abs-cap 15         "${@:2}"
    ;;
  # ---- 等變殘差目標（docs/NEXT_PLAN.md 方向二）：ab_warp_ch 的載體與上限，
  # 非恆等起點（種子 0），再加逐張輸入 LPIPS ≤ ab_warp_ch 防禦圖 ＋ 0.0025。
  # _comm 用 comm 目標，_free 用現行 FreeObjective，其餘逐項相同。
  ab_warp_ch_comm|ab_warp_ch_free)
    exec "$PY" lab/code/ab_warp_defence.py --arm "$ARM" --out "$OUT" \
        --data "$DATA" --grid 7 --extent 90 --warp-radius 80 --pieces 16 \
        --frame-cap 32.0 --face-cap 16 --skin-radius 12.0 --chroma-gain 2.0 \
        --a-pos-cap 4 --a-neg-cap 15 --b-pos-cap 4 --b-neg-cap 25 --l-abs-cap 15 \
        --objective "${ARM#ab_warp_ch_}" --init random --init-std 0.1 \
        --lpips-ref-arm ab_warp_ch --lpips-tolerance 0.0025 --noise-seed 0 \
        "${@:2}"
    ;;
  # ---- 同族隨機對照（NEXT_PLAN.md B4）：對齊最佳化臂逐張實際 LPIPS ± 0.0025。
  # 一次產出 r1..r3；由 queue_worker 的 gen 工作序列執行。
  ab_warp_ch_comm_random_r[1-3]|ab_warp_ch_free_random_r[1-3])
    ref="${ARM%_random_r[0-9]*}"
    exec "$PY" -B lab/code/ab_warp_random.py --ref-arm "$ref" \
        --prefix "${ref}_random" --replicates 3 --max-candidates 4096 "${@:2}"
    ;;
  # ---- 色調曲線：單一全域曲線，預算分在色度 ----
  curve_dual_chroma)
    exec "$PY" lab/code/curve_budget_defence.py --arm curve_dual_chroma --out "$OUT" \
        --data "$DATA" \
        --frame-cap 16.0 --face-cap 8.0
    ;;
  *)
    echo "未知的臂：$ARM" >&2; exit 2 ;;
esac

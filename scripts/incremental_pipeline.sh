#!/usr/bin/env bash
# 逐張串接：一張影像在所有條件上都訓練完 → 立刻跑淨化與編輯 → 重建表與報告。
#
# 為什麼逐張而不是等整批
# ────────────────────────────────────────────────────────────────────
# 一個工作點十張要十一小時，等整批跑完才出圖等於十一小時內看不到任何東西。
# 逐張串接讓報告在第一張完成後（約一小時）就有內容，之後每完成一張更新一次。
# 未完成的格子在報告上畫成占位，一眼看得出還缺哪幾張。
#
# 淨化的 CSV 命名成 `<tag>_img<NN>.csv`
# ────────────────────────────────────────────────────────────────────
# `mainline_tables.py` 用 `rsplit("_", 1)[0]` 還原 tag，所以**分片名不可含
# 底線**——用影像名當分片會把 tag 切錯（`ig_d21_task_attr_mod_color`），
# 而那不會拋錯，只會讓整張表對不上。用 `img01`…`img10` 這種序號安全。
#
# 空白地板重用
# ────────────────────────────────────────────────────────────────────
# 地板量的是**乾淨影像**過算子之後的編輯位移，與用哪個條件的防禦圖無關，
# 所以直接沿用既有的（同一組十張、同一組算子、同樣一顆種子）。
#
# **執行中不可覆寫這個檔案。** bash 是邊讀邊執行的，換掉檔案內容會讓它跳到
# 錯的位置。要改就先停、改完再啟動；或改寫到新檔名再切換。
#
# 用法：bash scripts/incremental_pipeline.sh "<淨化用的卡號>"
set -uo pipefail
ROOT=/nfs/home/nelson0314/WACV-s3
PY="$HOME/venvs/wacv/bin/python"
export PYTHONPATH="$ROOT" HF_HOME="$HOME/hf_cache" PYTHONIOENCODING=utf-8
export TOKENIZERS_PARALLELISM=false
cd "$ROOT" || { echo "錯誤：找不到 $ROOT" >&2; exit 2; }

DEV="${1:-}"
[ -z "$DEV" ] && { echo "用法：$0 \"<淨化用的卡號>\"" >&2; exit 2; }

SRC=runs/ip2p_ig_converge
OUT=runs/ip2p_ig_purify
GAL=runs/gallery_ig
TABLES=$SRC/tables
CONV=$SRC/convergence
REPORT=$SRC/report.html
FLOOR_SRC=runs/ip2p_dispersion_purify
mkdir -p "$OUT" "$GAL" "$TABLES" "$CONV"

TAGS=(ig_d21 ig_d25 ig_n30 ig_n35)
SHORT=("d21" "d25" "n30" "n35")
PUR="identity jpeg90 jpeg75 jpeg50 jpeg30 blur1 blur2 crop_resize0.1 crop_resize0.15"

# 地板重用一次
for f in floor_color floor_object; do
  [ -f "$OUT/$f.csv" ] || cp "$FLOOR_SRC/$f.csv" "$OUT/$f.csv" 2>/dev/null
done

mapfile -t IMAGES < runs/ip2p_fair_comparison/images10.txt

rebuild () {
  "$PY" scripts/mainline_tables.py --defense "$SRC" --purify "$OUT" \
      --ours-tags "${TAGS[@]}" --out "$TABLES" > /dev/null 2>&1
  "$PY" scripts/convergence_summary.py --runs "$SRC" --out "$CONV" \
      > /dev/null 2>&1
  "$PY" scripts/batch_report.py --tables "$TABLES" --convergence "$CONV" \
      --gallery "$GAL" --defense "$SRC" \
      --pipeline image_guidance --tags "${TAGS[@]}" \
      --conds phase_gain phase_gain phase_gain phase_gain \
      --short "${SHORT[@]}" \
      --title "影像引導消除損失：四個工作點" \
      --subtitle "$1" --out "$REPORT" 2>&1 | tail -1
}

echo "$(date +%H:%M) 逐張串接啟動（淨化用卡 $DEV）"
done_n=0
for idx in "${!IMAGES[@]}"; do
  nm="${IMAGES[$idx]}"
  tag_img=$(printf 'img%02d' $((idx + 1)))
  # 等這張圖在**所有**條件上都訓練完
  while true; do
    have=0
    for t in "${TAGS[@]}"; do
      [ -f "$SRC/$t/${nm}__phase_gain__def.png" ] && have=$((have + 1))
    done
    [ "$have" -eq "${#TAGS[@]}" ] && break
    sleep 120
  done
  echo "$(date +%H:%M) $nm 四個條件都訓練完，跑淨化與編輯"
  for t in "${TAGS[@]}"; do
    [ -f "$OUT/${t}_${tag_img}.csv" ] && continue
    CUDA_VISIBLE_DEVICES="$DEV" "$PY" scripts/phase_retention.py \
        --run "$SRC/$t" --images "$nm" --data data/omniedit150 \
        --attacker ip2p --seeds 1 --purifiers $PUR \
        --gallery "$GAL/$t" --out "$OUT/${t}_${tag_img}.csv" \
        >> "$OUT/${t}_${tag_img}.log" 2>&1
  done
  done_n=$((done_n + 1))
  echo "$(date +%H:%M) 重建報告（已完成 $done_n/${#IMAGES[@]} 張）"
  rebuild "已完成 $done_n/${#IMAGES[@]} 張 · 4 條件 · 9 算子 · 1 種子 · image_guidance · 8000 步上限 · 固定步長 0.01"
done
echo "$(date +%H:%M) 十張全部完成"

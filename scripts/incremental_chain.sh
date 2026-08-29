#!/usr/bin/env bash
# 逐張串接（通用版）：一張影像在所有條件上都訓練完 → 立刻跑淨化與編輯 →
# 重建表與報告。批次的設定全部由環境變數給，**腳本本身不綁任何一批**。
#
# 為什麼要一支通用的
# ────────────────────────────────────────────────────────────────────
# `incremental_pipeline.sh` 把 tag、來源目錄、標題寫死在檔案裡。要串下一批就
# 只能改那個檔，而**它正在被 bash 邊讀邊執行**——改掉會讓執行中的那一份跳到
# 錯的位置（實際踩過：改完之後執行中的 `rebuild()` 少了 `--pipeline`）。
# 這一支把設定拉到環境變數，於是同時跑兩批不必動到任何正在執行的檔案。
#
# 淨化 CSV 命名成 `<tag>_img<NN>.csv`
# ────────────────────────────────────────────────────────────────────
# `mainline_tables.py` 用 `rsplit("_", 1)[0]` 還原 tag，**分片名不可含底線**
# ——用影像名當分片會把 tag 切錯，而那不會拋錯，只會讓整張表對不上。
#
# 空白地板重用：地板量的是**乾淨影像**過算子之後的編輯位移，與用哪個條件的
# 防禦圖無關，故沿用既有的（同一組十張、同一組算子、同一顆種子）。
#
# **執行中不可覆寫這個檔案。**
#
# 用法：
#   SRC=runs/ip2p_ig_harden OUT=runs/ip2p_harden_purify GAL=runs/gallery_harden \
#   TAGS="a b" SHORT="a b" COND=phase_gain TITLE="…" SUB="…" \
#   bash scripts/incremental_chain.sh "<淨化用的卡號>"
set -uo pipefail
ROOT=/nfs/home/nelson0314/WACV-s3
PY="$HOME/venvs/wacv/bin/python"
export PYTHONPATH="$ROOT" HF_HOME="$HOME/hf_cache" PYTHONIOENCODING=utf-8
export TOKENIZERS_PARALLELISM=false
cd "$ROOT" || { echo "錯誤：找不到 $ROOT" >&2; exit 2; }

DEV="${1:-}"
[ -z "$DEV" ] && { echo "用法：$0 \"<淨化用的卡號>\"" >&2; exit 2; }
for v in SRC OUT GAL TAGS SHORT COND TITLE SUB; do
  [ -z "${!v:-}" ] && { echo "錯誤：環境變數 $v 未設" >&2; exit 2; }
done

FLOOR_SRC="${FLOOR_SRC:-runs/ip2p_dispersion_purify}"
IMAGES_FILE="${IMAGES_FILE:-runs/ip2p_fair_comparison/images10.txt}"
PUR="${PUR:-identity jpeg90 jpeg75 jpeg50 jpeg30 blur1 blur2 crop_resize0.1 crop_resize0.15}"
TABLES=$SRC/tables
CONV=$SRC/convergence
REPORT=$SRC/report.html
mkdir -p "$OUT" "$GAL" "$TABLES" "$CONV"

read -r -a TAG_ARR <<< "$TAGS"
read -r -a SHORT_ARR <<< "$SHORT"
[ "${#TAG_ARR[@]}" -ne "${#SHORT_ARR[@]}" ] && {
  echo "錯誤：TAGS 與 SHORT 長度不同" >&2; exit 2; }
CONDS=""
for _ in "${TAG_ARR[@]}"; do CONDS="$CONDS $COND"; done

for f in floor_color floor_object; do
  [ -f "$OUT/$f.csv" ] || cp "$FLOOR_SRC/$f.csv" "$OUT/$f.csv" 2>/dev/null
done

mapfile -t IMAGES < "$IMAGES_FILE"

rebuild () {
  "$PY" scripts/mainline_tables.py --defense "$SRC" --purify "$OUT" \
      --ours-tags "${TAG_ARR[@]}" --out "$TABLES" > /dev/null 2>&1
  "$PY" scripts/convergence_summary.py --runs "$SRC" --out "$CONV" \
      > /dev/null 2>&1
  "$PY" scripts/batch_report.py --tables "$TABLES" --convergence "$CONV" \
      --gallery "$GAL" --defense "$SRC" \
      --pipeline image_guidance --tags "${TAG_ARR[@]}" \
      --conds $CONDS --short "${SHORT_ARR[@]}" \
      --title "$TITLE" --subtitle "$1" --out "$REPORT" 2>&1 | tail -1
}

echo "$(date +%H:%M) 逐張串接啟動（$SRC，淨化用卡 $DEV）"
done_n=0
for idx in "${!IMAGES[@]}"; do
  nm="${IMAGES[$idx]}"
  tag_img=$(printf 'img%02d' $((idx + 1)))
  while true; do
    have=0
    for t in "${TAG_ARR[@]}"; do
      [ -f "$SRC/$t/${nm}__${COND}__def.png" ] && have=$((have + 1))
    done
    [ "$have" -eq "${#TAG_ARR[@]}" ] && break
    sleep 120
  done
  echo "$(date +%H:%M) $nm 四個條件都訓練完，跑淨化與編輯"
  for t in "${TAG_ARR[@]}"; do
    [ -f "$OUT/${t}_${tag_img}.csv" ] && continue
    CUDA_VISIBLE_DEVICES="$DEV" "$PY" scripts/phase_retention.py \
        --run "$SRC/$t" --images "$nm" --data data/omniedit150 \
        --attacker ip2p --seeds 1 --purifiers $PUR \
        --gallery "$GAL/$t" --out "$OUT/${t}_${tag_img}.csv" \
        >> "$OUT/${t}_${tag_img}.log" 2>&1
  done
  done_n=$((done_n + 1))
  echo "$(date +%H:%M) 重建報告（已完成 $done_n/${#IMAGES[@]} 張）"
  rebuild "已完成 $done_n/${#IMAGES[@]} 張 · $SUB"
done
echo "$(date +%H:%M) 十張全部完成"

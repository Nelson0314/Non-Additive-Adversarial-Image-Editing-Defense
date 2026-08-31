#!/usr/bin/env bash
# 把一個批次的防禦圖跑過九個淨化算子，出抗淨化的讀數。**不重訓。**
#
# 空白地板沿用 `runs/ip2p_harden_purify` 的——地板量的是**乾淨影像**過算子
# 之後的編輯位移，與用哪個條件的防禦圖無關，同樣兩張、同一組算子、同一顆
# 種子，故可重用。
#
# 用法：SRC=runs/ip2p_eot_matched OUT=runs/ip2p_eot_matched_purify \
#       TAGS="eb_off eb_r20 eb_r22 eb_r25" bash scripts/purify_points.sh "<卡號>"
set -uo pipefail
ROOT=/nfs/home/nelson0314/WACV-s3
PY="$HOME/venvs/wacv/bin/python"
export PYTHONPATH="$ROOT" HF_HOME="$HOME/hf_cache" PYTHONIOENCODING=utf-8
export TOKENIZERS_PARALLELISM=false
cd "$ROOT" || { echo "錯誤：找不到 $ROOT" >&2; exit 2; }

DEV="${1:-}"
[ -z "$DEV" ] && { echo "用法：SRC=… OUT=… TAGS=… $0 \"<卡號>\"" >&2; exit 2; }
for v in SRC OUT TAGS; do
  [ -z "${!v:-}" ] && { echo "錯誤：環境變數 $v 未設" >&2; exit 2; }
done
bash scripts/free_cards.sh --assert "$DEV" || exit 3

GAL="${GAL:-runs/gallery_$(basename "$OUT")}"
mkdir -p "$OUT" "$GAL"
IMGS="${IMGS:-task_attr_mod_color_11699 task_obj_remove_380621}"
PUR="${PUR:-identity jpeg90 jpeg75 jpeg50 jpeg30 blur1 blur2 crop_resize0.1 crop_resize0.15}"

for f in floor_color floor_object; do
  [ -f "$OUT/$f.csv" ] || cp "runs/ip2p_harden_purify/$f.csv" "$OUT/$f.csv"
done

for t in $TAGS; do
  [ -f "$OUT/$t.csv" ] && { echo "$t 已有結果，跳過"; continue; }
  echo "$(date +%H:%M) $t 開始"
  CUDA_VISIBLE_DEVICES="$DEV" "$PY" scripts/phase_retention.py \
      --run "$SRC/$t" --images $IMGS --data data/omniedit150 \
      --attacker ip2p --seeds 1 --purifiers $PUR \
      --gallery "$GAL/$t" --out "$OUT/$t.csv" >> "$OUT/$t.log" 2>&1
  echo "$(date +%H:%M) $t 完成"
done
echo "$(date +%H:%M) $SRC 的抗淨化跑完"

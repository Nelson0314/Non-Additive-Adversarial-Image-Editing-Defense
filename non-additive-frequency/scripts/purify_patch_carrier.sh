#!/usr/bin/env bash
# 補丁族（衣服載體與固定形狀浮水印）的抗淨化。**只讀已存的防禦圖，不重跑防禦。**
#
# 為什麼是這些格
# `runs/ip2p_patch_purify/` 量到補集補丁的 JPEG 30 保留 76%（相位族是 11%），
# 但**裁切只有 46%**——補丁是局部的，裁掉一部分就少一部分。這一批問兩件事：
#
#   1. 把支撐限制在衣服上（更小、更集中）之後，裁切那一欄掉到哪裡。
#   2. `crop_safe` 擺放（整塊落在中央 80% 內）有沒有把裁切那一欄拉回來。
#      這是 B 組存在的理由，也是它唯一要回答的問題。
#
# 算子取現行主組六個，外加 `jpeg_then_resize75`——`start.md` §5 標它為
# 「針對性最強」且從未在現行條件下測過，而它正好是 C&R 串接，與這一批的
# 裁切問題同軸。
#
# 空白地板不可省：淨化算子自己就會把編輯推開。
#
# 用法：bash scripts/purify_patch_carrier.sh "<卡號…>"
set -uo pipefail
ROOT=/nfs/home/nelson0314/WACV-s3
PY="$HOME/venvs/wacv/bin/python"
export PYTHONPATH="$ROOT" HF_HOME="$HOME/hf_cache" PYTHONIOENCODING=utf-8
export TOKENIZERS_PARALLELISM=false
cd "$ROOT" || { echo "錯誤：找不到 $ROOT" >&2; exit 2; }

DEVS=(${1:-})
[ ${#DEVS[@]} -lt 1 ] && { echo "用法：$0 \"<卡號…>\"" >&2; exit 2; }
bash scripts/free_cards.sh --assert "${DEVS[*]}" || exit 3

OUT=runs/ip2p_patch_carrier_purify
mkdir -p "$OUT"
MARIO=task_attr_mod_color_6205
PUR="identity jpeg75 jpeg30 blur1 blur2 crop_resize0.1 jpeg_then_resize75"
COMMON="--data data/omniedit150 --attacker ip2p --seeds 3 --purifiers $PUR --codefense"

i=0
launch() {                     # $1 tag  $2 來源  $3 額外
  local tag="$1" run="$2" extra="${3:-}"
  local dev=${DEVS[$(( i % ${#DEVS[@]} ))]}; i=$((i + 1))
  if [ ! -f "$run/results.csv" ] || ! ls "$run"/*__def.png >/dev/null 2>&1; then
    echo "[skip] $run 缺防禦圖或 results.csv" >&2
    return
  fi
  CUDA_VISIBLE_DEVICES="$dev" setsid nohup "$PY" scripts/phase_retention.py \
      --run "$run" $COMMON --images $MARIO $extra \
      --out "$OUT/${tag}_all.csv" --gallery "$OUT/gallery_${tag}" \
      < /dev/null >> "$OUT/${tag}.log" 2>&1 &
  disown
  echo "[purify] $tag dev=$dev src=$run"
}

# 衣服載體：最強的一格、最美化的一格、只有褲子的一格，以及同幾何隨機對照。
launch clothes_plain  runs/ip2p_patch_carrier/clothes_plain
launch clothes_tile_a runs/ip2p_patch_carrier/clothes_tile_a
launch pants_tile     runs/ip2p_patch_carrier/pants_tile
launch clothes_rand   runs/ip2p_patch_carrier/clothes_rand
# 美化組最極端的一格。
launch pretty         runs/ip2p_patch_shape/pretty
# 固定形狀：一塊對兩塊，同一個總面積。裁切那一欄的答案在這裡。
launch sq1_a04        runs/ip2p_patch_shape/sq1_a04
launch sq2_a04        runs/ip2p_patch_shape/sq2_a04
# 空白地板。來源目錄只用來定位設定，必須含齊要量的每一張影像。
launch floor          runs/ip2p_patch_carrier/clothes_plain --floor

sleep 20
echo "啟動了 $(ps -u "$USER" -o cmd | grep -c '[p]hase_retention') 個 phase_retention（$(date)）"

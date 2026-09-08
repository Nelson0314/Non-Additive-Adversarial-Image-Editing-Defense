#!/usr/bin/env bash
# 十二張擴增影像的抗淨化。**只讀已存的防禦圖，不重跑防禦。**
#
# 為什麼要它
# `runs/ip2p_patch_carrier_purify/` 的七個算子只有瑪利歐一張的數，而那一張
# 正是十三張裡唯一沒有出現輸出毀壞的（見 `runs/ip2p_patch_breadth/README.md`）。
# 裁切與 `jpeg_then_resize` 那兩欄要不要據以下結論，取決於它們在其餘十二張
# 上長什麼樣。
#
# **逐張分片**：`runs/ip2p_patch_breadth/plain_<img>/` 每個目錄只有一張影像，
# 一個 process 一個目錄。輸出檔名帶影像名——兩個 process 不可寫同一個目錄，
# 而每寫一列是整份重寫 CSV。
#
# 用法：
#   bash scripts/purify_patch_breadth.sh "<卡號…>"          # 防禦圖
#   bash scripts/purify_patch_breadth.sh "<卡號…>" floor     # 空白地板
set -uo pipefail
ROOT=/nfs/home/nelson0314/WACV-s3
PY="$HOME/venvs/wacv/bin/python"
export PYTHONPATH="$ROOT" HF_HOME="$HOME/hf_cache" PYTHONIOENCODING=utf-8
export TOKENIZERS_PARALLELISM=false
cd "$ROOT" || { echo "錯誤：找不到 $ROOT" >&2; exit 2; }

DEVS=(${1:-})
[ ${#DEVS[@]} -lt 1 ] && { echo "用法：$0 \"<卡號…>\" [floor]" >&2; exit 2; }
bash scripts/free_cards.sh --assert "${DEVS[*]}" || exit 3

MODE="${2:-}"
OUT=runs/ip2p_patch_breadth_purify
mkdir -p "$OUT"
PUR="identity jpeg75 jpeg30 blur1 blur2 crop_resize0.1 jpeg_then_resize75"
COMMON="--data data/omniedit150 --attacker ip2p --seeds 3 --purifiers $PUR"

#  會同時吃到 plain_<img>.log，數量會變兩倍。只取目錄。
SRCS=$(find runs/ip2p_patch_breadth -maxdepth 1 -type d -name "plain_*" | sort)

# rand 模式：來源是**同一個目錄**（十二張都在 rand_all/），逐張分片時每個
# process 讀同一個目錄但寫不同的輸出檔——讀是安全的，寫才不可共用。
if [ "${2:-}" = "rand" ]; then
  RAND_SRC=runs/ip2p_patch_breadth/rand_all
  if [ ! -f "$RAND_SRC/results.csv" ]; then
    echo "錯誤：找不到 $RAND_SRC/results.csv" >&2; exit 2
  fi
  IMGS=$(cut -d, -f1 "$RAND_SRC/results.csv" | tail -n +2 | sort -u)
  N=$(echo "$IMGS" | wc -l)
  if [ "$N" -gt $(( ${#DEVS[@]} * 2 )) ]; then
    echo "錯誤：$N 張影像需要至少 $(( (N + 1) / 2 )) 張卡，只給了 ${#DEVS[@]} 張。" >&2
    exit 2
  fi
  i=0
  for img in $IMGS; do
    dev=${DEVS[$(( i % ${#DEVS[@]} ))]}
    i=$(( i + 1 ))
    CUDA_VISIBLE_DEVICES="$dev" setsid nohup "$PY" scripts/phase_retention.py         --run "$RAND_SRC" $COMMON --images "$img"         --conditions patch_rand         --out "$OUT/rand_$img.csv"         < /dev/null > "$OUT/rand_$img.log" 2>&1 &
    disown
    echo "[purify] rand_$img dev=$dev"
  done
  sleep 20
  echo "啟動了 $(ps -u "$USER" -o cmd | grep -c '[p]hase_retention') 個 phase_retention（$(date)）"
  exit 0
fi
N=$(echo "$SRCS" | wc -l)
if [ "$N" -gt $(( ${#DEVS[@]} * 2 )) ]; then
  echo "錯誤：$N 個來源需要至少 $(( (N + 1) / 2 )) 張卡，只給了 ${#DEVS[@]} 張。" >&2
  exit 2
fi

i=0
for run in $SRCS; do
  img=$(basename "$run" | sed 's/^plain_//')
  dev=${DEVS[$(( i % ${#DEVS[@]} ))]}
  i=$(( i + 1 ))
  extra=""; tag="$img"
  if [ "$MODE" = "floor" ]; then extra="--floor"; tag="floor_$img"; fi
  if [ ! -f "$run/results.csv" ] || ! ls "$run"/*__def.png >/dev/null 2>&1; then
    echo "[skip] $run 缺防禦圖或 results.csv" >&2
    continue
  fi
  CUDA_VISIBLE_DEVICES="$dev" setsid nohup "$PY" scripts/phase_retention.py \
      --run "$run" $COMMON --images "$img" $extra \
      --out "$OUT/${tag}.csv" \
      < /dev/null > "$OUT/${tag}.log" 2>&1 &
  disown
  echo "[purify] $tag dev=$dev"
done

sleep 20
echo "啟動了 $(ps -u "$USER" -o cmd | grep -c '[p]hase_retention') 個 phase_retention（$(date)）"

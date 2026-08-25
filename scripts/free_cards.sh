#!/usr/bin/env bash
# 列出**真正空著**的卡號。派工之前一律先跑這一支，不要自己讀 nvidia-smi。
#
# 為什麼要有這支：`nvidia-smi --query-gpu=memory.used` 只給總量，看不出那些
# 記憶體是誰的。實際踩過一次——只看了輸出的前三行、把別人 16 GB 的工作當成
# 自己的殘留，於是把兩個 process 疊到別人正在用的卡上，那張卡變成四個 process。
#
# 判定「空」的兩個條件（兩個都要成立）：
#   1. 卡上**沒有任何別人的** compute app；
#   2. 已用記憶體低於 `--max-used`（預設 1024 MiB），擋掉沒有列出 app 但仍被
#      佔住的情況。
#
# 用法：
#     bash scripts/free_cards.sh              # 印出空卡號，空白分隔
#     bash scripts/free_cards.sh --verbose    # 連每張卡的狀態一起印
#     DEVS=$(bash scripts/free_cards.sh)      # 拿去餵派工腳本
set -uo pipefail

MAX_USED=1024
VERBOSE=0
ASSERT=""
while [ $# -gt 0 ]; do
  case "$1" in
    --max-used) MAX_USED="$2"; shift 2 ;;
    --verbose) VERBOSE=1; shift ;;
    --assert) ASSERT="$2"; shift 2 ;;
    *) echo "未知參數 $1" >&2; exit 2 ;;
  esac
done

MINE=$(ps -u "$USER" -o pid --no-headers | tr -d ' ' | paste -sd'|')
[ -z "$MINE" ] && MINE="__none__"
APPS=$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader)

FREE=""
while IFS=, read -r idx uuid used; do
  idx=$(echo "$idx" | tr -d ' ')
  uuid=$(echo "$uuid" | tr -d ' ')
  used=$(echo "$used" | tr -d ' MiB')
  foreign=$(echo "$APPS" | grep "$uuid" | awk -F, '{print $2}' | tr -d ' ' \
            | grep -vcE "^(${MINE})$" || true)
  mine=$(echo "$APPS" | grep "$uuid" | awk -F, '{print $2}' | tr -d ' ' \
         | grep -cE "^(${MINE})$" || true)
  if [ "$foreign" -eq 0 ] && [ "$used" -le "$MAX_USED" ]; then
    FREE="$FREE $idx"
    state="空"
  elif [ "$foreign" -gt 0 ]; then
    state="別人的（$foreign 個）"
  else
    state="我的（$mine 個）或已佔用"
  fi
  [ "$VERBOSE" -eq 1 ] && printf "卡%s 用了 %s MiB  別人 %s 個  我的 %s 個  → %s\n" \
      "$idx" "$used" "$foreign" "$mine" "$state" >&2
done < <(nvidia-smi --query-gpu=index,uuid,memory.used --format=csv,noheader)

FREE="${FREE# }"

# `--assert "<卡號>"`：**指定的卡只要有一張是別人的就拒絕啟動**，回傳 3。
# 派工腳本一律呼叫這一個模式，不要只呼叫列印模式——列印了卻不擋，等於沒擋。
# （實際踩過：等待器印出空卡清單、清單是空的，但派工照樣送出去，兩張卡上
# 各疊到一個別人的 process。）
if [ -n "$ASSERT" ]; then
  bad=""
  for c in $ASSERT; do
    case " $FREE " in *" $c "*) ;; *) bad="$bad $c" ;; esac
  done
  if [ -n "$bad" ]; then
    echo "錯誤：卡$bad 不是空的（有別人的 process 或已用記憶體超過 ${MAX_USED} MiB），拒絕啟動。" >&2
    echo "      目前真正空著的是：${FREE:-（沒有）}" >&2
    exit 3
  fi
  exit 0
fi

echo "$FREE"

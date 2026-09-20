#!/usr/bin/env bash
# 唯讀複製自主 repo：`non-additive-frequency/scripts/free_cards.sh`
#（遠端 `/nfs/home/nelson0314/WACV-s4/scripts/free_cards.sh`），逐位元相同。
# 列出**真正空著**的卡號。派工之前一律先跑這一支，不要自己讀 nvidia-smi。
#
# 為什麼要有這支：`nvidia-smi --query-gpu=memory.used` 只給總量，看不出那些
# 記憶體是誰的。實際踩過一次——只看了輸出的前三行、把別人 16 GB 的工作當成
# 自己的殘留，於是把兩個 process 疊到別人正在用的卡上，那張卡變成四個 process。
#
# 判定「空」的兩個條件（兩個都要成立）：
#   1. 別人的 compute app **實際佔用**的記憶體總和不超過 `--foreign-max`
#      （預設 512 MiB）；
#   2. 扣掉那些被忽略的佔用之後，已用記憶體低於 `--max-used`（預設 1024
#      MiB），擋掉沒有列出 app 但仍被佔住的情況。
#
# 為什麼第 1 條看的是記憶體而不是「有沒有別人的 process」
# ────────────────────────────────────────────────────────────────────
# 原本只要卡上有別人的任何一個 compute app 就判定佔用。實務上會遇到別人的
# **單一** process 在八張卡上各開一個約 256 MiB 的 CUDA context——那是列舉
# 裝置留下的，不是在算——於是十六張卡全部被判定為佔用，什麼都送不出去。
# 使用者裁定：只擋高於 512 MiB 的佔用。
#
# **保護的意圖沒有改變**：真的在算的工作至少要幾 GB，遠高於門檻，照樣擋得住。
# 不確定時把 `--foreign-max` 調小，不要調大。
#
# 用法：
#     bash scripts/free_cards.sh              # 印出空卡號，空白分隔
#     bash scripts/free_cards.sh --verbose    # 連每張卡的狀態一起印
#     bash scripts/free_cards.sh --max-cards 8  # 超過五張，需使用者授意
#     DEVS=$(bash scripts/free_cards.sh)      # 拿去餵派工腳本
set -uo pipefail

MAX_USED=1024
FOREIGN_MAX=512
VERBOSE=0
ASSERT=""
MAX_CARDS=5
while [ $# -gt 0 ]; do
  case "$1" in
    --max-used) MAX_USED="$2"; shift 2 ;;
    --foreign-max) FOREIGN_MAX="$2"; shift 2 ;;
    --verbose) VERBOSE=1; shift ;;
    --assert) ASSERT="$2"; shift 2 ;;
    --max-cards) MAX_CARDS="$2"; shift 2 ;;
    *) echo "未知參數 $1" >&2; exit 2 ;;
  esac
done

MINE=$(ps -u "$USER" -o pid --no-headers | tr -d ' ' | paste -sd'|')
[ -z "$MINE" ] && MINE="__none__"
APPS=$(nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader)

FREE=""
while IFS=, read -r idx uuid used; do
  idx=$(echo "$idx" | tr -d ' ')
  uuid=$(echo "$uuid" | tr -d ' ')
  used=$(echo "$used" | tr -d ' MiB')
  same=$(echo "$APPS" | grep "$uuid" || true)
  # **空輸入要當成零個，不是一個。** `echo ""` 會送出一個空行，而空行不匹配
  # pid 樣式，於是 `grep -vc` 把它算成「別人 1 個」——沒有任何 compute app 的
  # 卡在 --verbose 上看起來像是有人在用。判定本身看的是 fmem 所以是對的，
  # 但印出來的東西與判定不一致，讀的人會不知道該信哪一個。
  if [ -z "$same" ]; then
    foreign=0; mine=0
  else
    foreign=$(echo "$same" | awk -F', ' '{print $2}' | tr -d ' ' \
              | grep -vcE "^(${MINE})$" || true)
    mine=$(echo "$same" | awk -F', ' '{print $2}' | tr -d ' ' \
           | grep -cE "^(${MINE})$" || true)
  fi
  # 別人**實際佔用**多少。判定看的是這個量，不是 process 的個數——一個只留
  # CUDA context 的 process 佔約 256 MiB，真的在算的至少幾 GB。
  fmem=$(echo "$same" | awk -F', ' -v m="^(${MINE})\$" \
         '{pid=$2; gsub(/ /,"",pid); if (pid !~ m) {n=$3; gsub(/[^0-9]/,"",n); s+=n}}
          END {print s+0}')
  if [ "$fmem" -le "$FOREIGN_MAX" ] && [ "$(( used - fmem ))" -le "$MAX_USED" ]; then
    FREE="$FREE $idx"
    state="空"
  elif [ "$fmem" -gt "$FOREIGN_MAX" ]; then
    state="別人的（$foreign 個，佔 ${fmem} MiB）"
  else
    state="我的（$mine 個）或已佔用"
  fi
  [ "$VERBOSE" -eq 1 ] && printf "卡%s 用了 %s MiB  別人 %s 個／%s MiB  我的 %s 個  → %s\n" \
      "$idx" "$used" "$foreign" "$fmem" "$mine" "$state" >&2
done < <(nvidia-smi --query-gpu=index,uuid,memory.used --format=csv,noheader)

FREE="${FREE# }"

# **一次最多五張卡。** 這是使用者定的規則（CLAUDE.md）：機器是多人共用的，
# 把八張全佔滿會讓別人完全排不進來。**上限做在這裡而不是各派工腳本裡**——
# 寫在文件或個別腳本上的規則遲早會有一支漏掉，而漏掉不會報錯。
#
# `--assert` 不受此限：那是在檢查「指定的卡是不是空的」，與取幾張無關。
# 要用超過五張時給 `--max-cards N`，那必須是使用者明確授意。
if [ -z "$ASSERT" ] && [ "$MAX_CARDS" -gt 0 ]; then
  CAPPED=""
  n=0
  for c in $FREE; do
    [ "$n" -ge "$MAX_CARDS" ] && break
    CAPPED="$CAPPED $c"
    n=$(( n + 1 ))
  done
  if [ "$(echo $FREE | wc -w)" -gt "$MAX_CARDS" ]; then
    echo "（空卡有 $(echo $FREE | wc -w) 張，依規定只取 $MAX_CARDS 張；" \
         "要更多請給 --max-cards 並確認使用者已授意）" >&2
  fi
  FREE="${CAPPED# }"
fi

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
    echo "錯誤：卡$bad 不是空的（別人佔用超過 ${FOREIGN_MAX} MiB，或扣除後仍超過 ${MAX_USED} MiB），拒絕啟動。" >&2
    echo "      目前真正空著的是：${FREE:-（沒有）}" >&2
    exit 3
  fi
  exit 0
fi

echo "$FREE"

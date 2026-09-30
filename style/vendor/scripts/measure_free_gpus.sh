#!/usr/bin/env bash
# 列出空閒的 GPU 卡號。候選清單不等於取卡；派工須另經 gpu_lease.sh 取得租約。
#
# 空閒的兩個條件須同時成立：
#   1. 其他使用者的 compute app 實際佔用的記憶體總和不超過 `--foreign-max`
#      （預設 512 MiB）；只建立 CUDA context 的程序約佔 256 MiB，不視為佔用。
#   2. 扣除上述佔用後，已用記憶體不超過 `--max-used`（預設 1024 MiB），
#      涵蓋未列出 app 但記憶體已被佔用的情況。
#
# 用法：
#     bash measure_free_gpus.sh                  # 印出空卡號，空白分隔
#     bash measure_free_gpus.sh --verbose        # 另於 stderr 印出每張卡的狀態
#     bash measure_free_gpus.sh --max-cards 8    # 只限制候選清單長度
#     bash measure_free_gpus.sh --assert "0 3"   # 指定卡任一非空閒即回傳 3
set -uo pipefail

MAX_USED=1024
FOREIGN_MAX=512
VERBOSE=0
ASSERT=""
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/gpu_policy.sh" || exit 1
MAX_CARDS=$(gpu_global_cap) || exit $?
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
  # 沒有 compute app 時計為零個；`echo ""` 的空行會被 `grep -vc` 計為一個。
  if [ -z "$same" ]; then
    foreign=0; mine=0
  else
    foreign=$(echo "$same" | awk -F', ' '{print $2}' | tr -d ' ' \
              | grep -vcE "^(${MINE})$" || true)
    mine=$(echo "$same" | awk -F', ' '{print $2}' | tr -d ' ' \
           | grep -cE "^(${MINE})$" || true)
  fi
  # 其他使用者實際佔用的記憶體；判定依此量而非程序個數。
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

# 候選清單不等於取卡；全局容量由 gpu_lease.sh 在取租約時驗證。
# `--assert` 只檢查指定卡是否空閒，不保留卡，也不核准額外容量。
if [ -z "$ASSERT" ] && [ "$MAX_CARDS" -gt 0 ]; then
  CAPPED=""
  n=0
  for c in $FREE; do
    [ "$n" -ge "$MAX_CARDS" ] && break
    CAPPED="$CAPPED $c"
    n=$(( n + 1 ))
  done
  if [ "$(echo $FREE | wc -w)" -gt "$MAX_CARDS" ]; then
    echo "（空卡有 $(echo $FREE | wc -w) 張，候選清單只列 $MAX_CARDS 張；派工須另取全局租約）" >&2
  fi
  FREE="${CAPPED# }"
fi

# `--assert "<卡號>"`：指定的卡任一張非空閒即拒絕，回傳 3。派工入口使用此模式。
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

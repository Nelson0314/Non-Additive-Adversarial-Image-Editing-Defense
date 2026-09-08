#!/usr/bin/env bash
# 色彩載體的抗淨化：**與補丁族同一支程式、同一組十二個算子、同一個空白地板**。
#
# 為什麼要重跑一次色彩族的抗淨化
#   `runs/ip2p_color_capacity_purify` 已經量到十個算子保留 93–116%，但那是在
#   **全圖載體、半徑 0.10、舊讀數（LPIPS 位移）** 下量的。補丁族那張表是
#   **衣物載體、身分讀數、十二個算子**。`docs/DIRECTION.md` §3.1 的互補性主張
#   踩在這兩張表上，而它們不可並列。
#
#   這一支用新的色彩工作點（衣物載體、大半徑）跑同一個協定，補上那張表。
#
# 三個臂
#   r20    半徑 2.00 的色彩網格（`colour_budget_round.sh` 的最大預算）
#   r06    半徑 0.60，接得回既有掃描
#   r20_rand  同半徑隨機、零最佳化。**不可省**——色彩族已知抗淨化對隨機沒有
#             優勢（1.05–1.30），那正是要與補丁族的 1.82 對照的數字
#
# 空白地板由補丁族那批共用（`runs/ip2p_purify_identity` 的 floor 臂，
# 同一組原圖、同一組算子、同一個種子），不重跑。
#
# 用法：bash scripts/colour_purify_round.sh "<卡號…>"
set -uo pipefail
ROOT=/nfs/home/nelson0314/WACV-s3
PY="$HOME/venvs/wacv/bin/python"
export PYTHONPATH="$ROOT" HF_HOME="$HOME/hf_cache" PYTHONIOENCODING=utf-8
export TOKENIZERS_PARALLELISM=false
# 權重在系統共用快取，`ssh <host> "bash script"` 不載入 profile（見
# `colour_budget_round.sh` 的同一段說明）。
export HF_HUB_CACHE="${HF_HUB_CACHE:-/var/cache/huggingface/hub}"
export HF_ASSETS_CACHE="${HF_ASSETS_CACHE:-/var/cache/huggingface/assets}"
cd "$ROOT" || { echo "錯誤：找不到 $ROOT" >&2; exit 2; }

DEVS=(${1:-1 2 3 4 5})
[ ${#DEVS[@]} -gt 5 ] && DEVS=("${DEVS[@]:0:5}")

SRC=runs/ip2p_colour_budget
OUT=runs/ip2p_colour_purify
LOG=runs/execution_logs/colour_purify.log
mkdir -p "$OUT" runs/execution_logs

ARMS="r20 r06 r20_rand"
CATS="clothing accessory background"

bash scripts/free_cards.sh --assert "${DEVS[*]}" || exit 3

# 來源目錄檢查：缺哪一個就整批拒絕，不要跑到第七個 job 才發現。
for arm in $ARMS; do
  for cat in $CATS; do
    n=$(find "$SRC" -maxdepth 1 -type d -name "${arm}_${cat}_task_*" 2>/dev/null | wc -l)
    [ "$n" -eq 0 ] && { echo "錯誤：$SRC 底下沒有 ${arm}_${cat}_task_*" >&2; exit 2; }
  done
done
echo "=== [$(date '+%F %T')] 色彩族抗淨化：$(echo $ARMS | wc -w) 臂 × $(echo $CATS | wc -w) 類 ===" | tee -a "$LOG"

SLOTS=$(( ${#DEVS[@]} * 2 ))
running() {
  ps -u "$USER" -o comm=,cmd= | awk '$1 ~ /^python/ && /purify_identity\.py/' | wc -l
}

i=0
for arm in $ARMS; do
  for cat in $CATS; do
    while [ "$(running)" -ge "$SLOTS" ]; do sleep 30; done
    dev=${DEVS[$(( i % ${#DEVS[@]} ))]}
    i=$(( i + 1 ))
    echo "[colour-purify] $arm/$cat dev=$dev" | tee -a "$LOG"
    CUDA_VISIBLE_DEVICES="$dev" setsid nohup "$PY" scripts/purify_identity.py \
      --arm "colour_$arm" --cells "$SRC/${arm}_${cat}_task_*" --out "$OUT" \
      --skip-existing > "$OUT/${arm}_${cat}.log" 2>&1 < /dev/null &
    sleep 5
  done
done

while [ "$(running)" -gt 0 ]; do sleep 60; done
echo "=== [$(date '+%F %T')] 收工 ===" | tee -a "$LOG"

ENT=""
for d in "$OUT"/colour_*/; do
  [ -d "$d" ] && ENT="$ENT --entry $(basename "$d")=$d"
done
# shellcheck disable=SC2086
"$PY" scripts/identity_probe.py --out "$OUT/identity.csv" $ENT 2>&1 | tee -a "$LOG"
echo "=== [$(date '+%F %T')] 身分讀數寫入 $OUT/identity.csv ===" | tee -a "$LOG"

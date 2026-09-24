#!/usr/bin/env bash
# 收尾用的單一排程器：等一張卡，依序做完三件事，然後放掉。
#
#   1. 跨臂讀數（現有十一個臂）        ——先做，確保這一批的交付面先落地
#   2. `style_opt`（唯一還沒建過的臂）  ——先用一張圖三步試跑，過了才跑正式的
#   3. 跨臂讀數（十二個臂）
#
# 為什麼是同一支腳本而不是三個排程器
# ────────────────────────────────────────────────────────────────────
# 三件事都會寫 `lab/results/*.csv`。兩個排程器同時跑會同時寫同一份檔案，
# 而那不會報錯——本專案踩過「兩個行程靜默蓋掉同一份結果」。序列化在同一支
# 腳本裡，就不需要另外做互斥。
#
# `style_opt` 是**沒有在卡上試跑過**的程式（寫的時候一張空卡都沒有），所以
# 正式跑之前先跑一張圖三步。試跑失敗就寫 `STYLE_OPT_NEEDS_DECISION` 並停住，
# 不啟動那幾個 GPU 小時——第 1 步的讀數已經落地，不受影響。
# 這與主表那條線的 `PG_LINF_NEEDS_DECISION` 是同一條規則：查不到就寫明缺什麼，
# 不填一個看起來合理的值。
#
# 用法：
#   nohup setsid bash lab/scripts/finish_batch.sh > log 2>&1 < /dev/null &
set -uo pipefail
source ~/env.sh
R=/nfs/home/nelson0314/image-immunization
cd "$R" || { echo "[FATAL] 進不去 $R"; exit 1; }

HOST=$(hostname)
LEASE=$HOME/lab_leases
POLL=180
CAP=5
mkdir -p "$LEASE" lab/runs/state lab/runs/logs

claim() {
  local gpu=""
  while [ -z "$gpu" ]; do
    local held
    held=$(ls -1 "$LEASE" 2>/dev/null | wc -l)
    if [ "$held" -ge "$CAP" ]; then
      echo "[WAIT] $(date -Is) 租約已滿（$held/$CAP）"; sleep "$POLL"; continue
    fi
    for c in $(bash scripts/free_cards.sh 2>/dev/null); do
      [ -e "$LEASE/${HOST}-${c}" ] && continue
      bash scripts/free_cards.sh --assert "$c" >/dev/null 2>&1 || continue
      gpu="$c"; break
    done
    [ -z "$gpu" ] && { echo "[WAIT] $(date -Is) 沒有空卡"; sleep "$POLL"; }
  done
  echo "$gpu"
}

echo "[BATCH-START] $(date -Is) host=$HOST"
GPU=$(claim)
LEASEFILE="$LEASE/${HOST}-${GPU}"
echo "$HOST $$ lab_finish_batch" > "$LEASEFILE"
trap 'rm -f "$LEASEFILE"' EXIT
echo "[CLAIM] $(date -Is) gpu=$GPU"

# ---- 1. 十一個臂的讀數 ----
echo "[STEP1] $(date -Is) 跨臂讀數（十一個臂）"
bash lab/scripts/readout.sh "$GPU"
echo "[STEP1-EXIT] $(date -Is) rc=$?"
wc -l lab/results/*.csv

# ---- 2. style_opt ----
export PYTHONPATH="$R" TOKENIZERS_PARALLELISM=false PYTHONIOENCODING=utf-8
export CUDA_VISIBLE_DEVICES="$GPU" CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

if [ ! -f lab/runs/state/style_opt.defence.done ]; then
  echo "[SMOKE] $(date -Is) style_opt 一張圖三步試跑"
  rm -rf lab/runs/_opt_smoke
  "$PY" lab/code/style_opt_defence.py --arm _opt_smoke \
      --out lab/runs/_opt_smoke --data lab/data/portraits --images man_00 \
      --steps 3 --probe-every 2 --log-every 1 > lab/runs/logs/style_opt_smoke.log 2>&1
  rc=$?
  echo "[SMOKE-EXIT] $(date -Is) rc=$rc"
  if [ "$rc" -ne 0 ]; then
    {
      echo "style_opt 的試跑失敗（rc=$rc），正式跑沒有啟動。"
      echo "第 1 步的十一個臂讀數已經落地，不受影響。"
      echo "--- log 尾 ---"
      tail -40 lab/runs/logs/style_opt_smoke.log
    } > lab/runs/STYLE_OPT_NEEDS_DECISION
    echo "[GIVE-UP] $(date -Is) 見 lab/runs/STYLE_OPT_NEEDS_DECISION"
    exit 1
  fi
  rm -rf lab/runs/_opt_smoke
fi

echo "[STEP2] $(date -Is) style_opt 完整鏈"
bash lab/scripts/arm_chain.sh "$GPU" style_opt
echo "[STEP2-EXIT] $(date -Is) rc=$?"

# ---- 3. 十二個臂的讀數 ----
if [ "$(ls -1 lab/runs/state/style_opt.*.done 2>/dev/null | wc -l)" -ge 18 ]; then
  echo "[STEP3] $(date -Is) 跨臂讀數（十二個臂）"
  bash lab/scripts/readout.sh "$GPU"
  echo "[STEP3-EXIT] $(date -Is) rc=$?"
  wc -l lab/results/*.csv
else
  echo "[SKIP-STEP3] $(date -Is) style_opt 的鏈沒跑完（sentinel $(ls -1 lab/runs/state/style_opt.*.done 2>/dev/null | wc -l)/18），"
  echo "             不重跑讀數——第 1 步的十一臂版本留著，不要用半套的十二臂覆蓋它。"
fi

echo "[BATCH-DONE] $(date -Is)"

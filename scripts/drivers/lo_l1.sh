#!/bin/bash
# L1：在 L∞ ≤ 0.06 上重現 Lo et al. (CVPR 2024) Table 1 的三根柱子。
#
# 用法（在 repo 根目錄）：
#   bash scripts/drivers/lo_l1.sh              # 開始或接續
#   bash scripts/drivers/lo_l1.sh --fg         # 前景執行，看得到即時輸出
#
# 為什麼要 detached
# ─────────────────────────────────────────────────────────────────────
# 24 張 × 3 攻擊約三小時（實測，見規格 §4.1）。直接在 SSH 前景跑的話，
# 連線一斷行程就被 SIGHUP 殺掉。此處用 setsid + nohup 讓它脫離終端機，
# 斷線後仍繼續，重連用 `tail -f` 看進度。
#
# 接續
# ─────────────────────────────────────────────────────────────────────
# 一律帶 `--resume`。腳本以 summary.csv 判定哪些 (影像, 攻擊) 已完成並略過，
# 每格算完就落地。中途死掉重跑同一行指令即可，不會重算已完成的部分，
# 也不會覆寫既有結果。
#
# TF32
# ─────────────────────────────────────────────────────────────────────
# `WACV_ALLOW_TF32=1`。專案預設關閉（E28），理由是跨機器數值可比與 BDIA
# 精確反演的精度；**本協定不走反演**，而實測開啟後快 1.6–1.8 倍、
# edit_lpips 只變動 0.002–0.006，遠小於要分辨的效果量 0.059。
# 這是明確的取捨，報告中須寫明。見規格 §4.1。
set -euo pipefail

cd "$(dirname "$0")/../.."

PY="${PY:-/home/zeus/miniconda3/envs/cloudspace/bin/python3}"
OUT="${OUT:-runs/lo_baseline}"
export HF_HOME="${HF_HOME:-/teamspace/studios/this_studio/hf_cache}"
export WACV_ALLOW_TF32="${WACV_ALLOW_TF32:-1}"

mkdir -p runs/logs
LOG="runs/logs/lo_l1.log"

CMD=("$PY" scripts/run_lo_baseline.py
     --data data/lo_aligned
     --out "$OUT"
     --attacks pg_encoder,pg_diffusion,semantic
     --eval_seeds 20
     --resume)

echo "輸出目錄 : $OUT"
echo "日誌     : $LOG"
echo "TF32     : $WACV_ALLOW_TF32"
echo "指令     : ${CMD[*]}"

if [ "${1:-}" = "--fg" ]; then
  exec "${CMD[@]}" 2>&1 | tee -a "$LOG"
fi

setsid nohup "${CMD[@]}" >> "$LOG" 2>&1 < /dev/null &
PID=$!
echo "已在背景啟動，PID $PID"
echo
echo "看進度：  tail -f $LOG"
echo "看已完成：wc -l < $OUT/summary.csv    # 含表頭，總共會是 73 列"
echo "中止：    kill $PID"

#!/bin/bash
# L2：本專案的殘差模塊，走與 L1 完全相同的評測路徑。
#
# 用法（在 repo 根目錄）：
#   bash scripts/drivers/ours_l2.sh              # 開始或接續
#   bash scripts/drivers/ours_l2.sh --after-l1   # 等 L1 跑完再開始
#
# 兩個 site 交錯而非依序跑完
# ─────────────────────────────────────────────────────────────────────
# 驅動腳本的迴圈是「逐影像，內層逐 site」。機器時間若中途用盡，拿到的會是
# 前 k 張影像的**兩個 site**，而不是全部影像的其中一個 site。前者是 n=k 的
# 有效配對比較，後者比不了東西。
#
#   PF — 加性全秩。本專案在 τ_lpips 綁定下最強的一組（E8）。
#   S  — 空間變形，非加性。研究問題本身。
#
# --rank 32 對 PF 無作用（全秩不吃 rank），對 S 被重新解讀為位移場的控制
# 網格邊長，與 e21／e22 相同。
#
# 學習率逐 site 不同，不由本檔指定
# ─────────────────────────────────────────────────────────────────────
# 2026-08-04：本檔原本不傳 --lr，於是吃到腳本當時的單一預設 0.03；
# runs/ours_lo/ 那一批則是人工以 --lr 0.008 執行，兩個 site 共用同一個值。
# 那是錯的：φ 的量綱逐 site 不同（PF 是像素值、S 是位移像素），E14 校準出
# 來的是 S=0.1、P/PF=0.008，相差 12.5 倍。後果見 docs/LEDGER.md 6.16。
# 腳本現在預設就逐 site 取校準值，本檔因此**刻意不傳 --lr**。
#
# 失真未匹配
# ─────────────────────────────────────────────────────────────────────
# 基準在 κ = 0.06 上的擾動 LPIPS 實測 0.49–0.54，本條件綁在 τ_lpips = 0.10。
# 兩邊的數字不是同一條軸上的點，報表必須併看 pert_lpips 與 pert_linf。
# 匹配掃描是後續工作，此處刻意不做。
#
# τ_acut 與 τ_chroma 同樣不由本檔指定：腳本會依 --tau_lpips 去
# runs/p14_budget_thresholds/thresholds.csv 查該預算的值（0.10 → 0.0598 與
# 1.2965）。LossConfig 的預設 0.04／0.8 是 τ_lpips=0.05 的量級上定的，
# 沿用會讓副約束變成真正的有效約束，見 docs/LEDGER.md 6.17。
set -euo pipefail

cd "$(dirname "$0")/../.."

PY="${PY:-/home/zeus/miniconda3/envs/cloudspace/bin/python3}"
OUT="${OUT:-runs/ours_lo}"
export HF_HOME="${HF_HOME:-/teamspace/studios/this_studio/hf_cache}"
export WACV_ALLOW_TF32="${WACV_ALLOW_TF32:-1}"

mkdir -p runs/logs
LOG="runs/logs/ours_l2.log"

CMD=("$PY" scripts/run_ours_lo_eval.py
     --data data/lo_aligned
     --out "$OUT"
     --sites PF,S
     --rank 32
     --tau_lpips 0.10
     --eval_seeds 20
     --resume)

echo "輸出目錄 : $OUT"
echo "日誌     : $LOG"
echo "指令     : ${CMD[*]}"

# 等 L1 結束再開始。實測併行無用：三個行程同時跑時每個要 28.6s，序列
# 只要 10.5s，吞吐量只有 1.10 倍——單一行程已經把 SM 佔滿。
WAIT=""
if [ "${1:-}" = "--after-l1" ]; then
  WAIT='while pgrep -f "run_lo_base[l]ine" > /dev/null; do sleep 60; done; '
  echo "先等 L1 結束"
fi

setsid nohup bash -c "${WAIT}exec $(printf '%q ' "${CMD[@]}")" \
  >> "$LOG" 2>&1 < /dev/null &
echo "已在背景啟動，PID $!"
echo
echo "看進度：tail -f $LOG"

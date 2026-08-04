#!/bin/bash
# L1：在 L∞ ≤ 0.06 上重現 Lo et al. (CVPR 2024) Table 1 的三個基準方法。
#
# 用法（在 repo 根目錄）：
#   bash scripts/drivers/lo_l1.sh 0            # 編輯 prompt 1，開始或接續
#   bash scripts/drivers/lo_l1.sh 1            # 編輯 prompt 2
#   bash scripts/drivers/lo_l1.sh 0 --fg       # 前景執行，看得到即時輸出
#
# 兩個 prompt 都要跑
# ─────────────────────────────────────────────────────────────────────
# 論文補充材料 §A：每個物件有兩個編輯 prompt，Table 1 是兩者一起平均。
#
#   prompt 1（索引 0）改掉指定的內容，c_a **不**出現在 prompt 裡（A dog → "A cat"）
#   prompt 2（索引 1）改動其他區域，c_a 出現在 prompt 裡（"A dog in the park"）
#
# 只跑索引 0 會系統性低估 semantic attack：正文 §4.3 把「c_a 未出現在編輯
# prompt 中仍有效」列為額外的優點，代表那是較難的一半。兩個 PhotoGuard 變體 基準方法
# 完全不用 c_a，不受影響——這正是 2026-08-03 首批 37 格觀察到的模式
# （PhotoGuard 重現、semantic 反而低於論文）。兩半各自寫進不同的 --out，
# 由 scripts/report_table1.py 合併。
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
export HF_HOME="${HF_HOME:-/teamspace/studios/this_studio/hf_cache}"
export WACV_ALLOW_TF32="${WACV_ALLOW_TF32:-1}"

PI="${1:-0}"
case "$PI" in
  0) SUF="" ;;
  1) SUF="_p1" ;;
  *) echo "prompt 索引只能是 0 或 1，收到 '$PI'" >&2; exit 2 ;;
esac
shift || true

# 索引 0 沿用原本的目錄名，2026-08-03 之前跑的 37 格才接得上。
OUT="${OUT:-runs/lo_baseline$SUF}"

mkdir -p runs/logs
LOG="runs/logs/lo_l1$SUF.log"

CMD=("$PY" scripts/run_lo_baseline.py
     --data data/lo_aligned
     --out "$OUT"
     --attacks pg_encoder,pg_diffusion,semantic
     --prompt_index "$PI"
     --eval_seeds 20
     --resume)

echo "編輯 prompt : 索引 $PI"
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

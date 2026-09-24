#!/usr/bin/env bash
# 一個臂的完整鏈，固定跑在一張卡上：
#   防禦圖 → 防禦後編輯（兩場景）→ 淨化（七道）→ 淨化後編輯（七道 × 兩場景）
#
# 每一階段跑完寫一個 sentinel 到 lab/runs/state/，重跑時已完成的階段直接跳過。
# **sentinel 只在 rc=0 時才寫**：中途被砍掉的階段下次會重跑，不會被當成已完成。
#
# 用法：bash lab/scripts/arm_chain.sh <GPU> <臂名>
set -uo pipefail
GPU="$1"; ARM="$2"
source ~/env.sh
R=/nfs/home/nelson0314/image-immunization
cd "$R" || { echo "[FATAL] 進不去 $R"; exit 1; }
export PYTHONPATH="$R"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES="$GPU"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTHONIOENCODING=utf-8

L=lab/runs
# 攻擊端只跑 ip2p（整張圖編輯）。inpaint 場景的遮罩是資料集由原圖切出的那一張，
# 攻擊者拿不到，而且幾何淨化後遮罩不跟著轉；使用者裁定不再跑。已經跑過的
# inpaint 讀數與逐格圖留著當參考。
SCENARIOS="ip2p"
DATA=lab/data/portraits
mkdir -p "$L/state"

step() {
  local tag="$1"; shift
  local flag="$L/state/${ARM}.${tag}.done"
  if [ -f "$flag" ]; then echo "[SKIP] $ARM/$tag"; return 0; fi
  echo "[START] $(date -Is) $ARM/$tag gpu=$GPU"
  "$@"
  local rc=$?
  echo "[EXIT] $(date -Is) $ARM/$tag rc=$rc"
  if [ "$rc" -eq 0 ]; then touch "$flag"; else return "$rc"; fi
}

step defence bash lab/scripts/defence_cmd.sh "$ARM" || exit 1

for SC in $SCENARIOS; do
  step "edit_$SC" "$PY" lab/code/edit_preflight.py --data "$DATA" \
      --defended "$L/defence/$ARM" --out "$L/edit_defended/$ARM" \
      --scenarios "$SC" --suffix "_$ARM" || exit 1
done

step purify "$PY" lab/code/purify_run.py --defended "$L/defence/$ARM" \
    --out "$L/purified/$ARM" || exit 1

for PUR in jpeg50 crop_resize0.1 blur1 rotate15 jpeg30 jpeg80 blur2; do
  for SC in $SCENARIOS; do
    step "pedit_${PUR}_${SC}" "$PY" lab/code/edit_preflight.py --data "$DATA" \
        --defended "$L/purified/$ARM/$PUR" \
        --out "$L/edit_purified/$ARM/$PUR" --scenarios "$SC" \
        --suffix "_${ARM}_${PUR}" || exit 1
  done
done

echo "[ARM-DONE] $(date -Is) $ARM gpu=$GPU"

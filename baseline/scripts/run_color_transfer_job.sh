#!/usr/bin/env bash
# 在 baseline 佇列中執行 color 專案的遷移工作（未防禦分母與 color3d 的 transfer，Pexels 人像），沿用 color 的工具、
# 輸出位置與完成標記；GPU 由 baseline 佇列的租約給一張卡（CUDA_VISIBLE_DEVICES），color 工具不另外取卡。
#   sd3   bash $T/tools/run_pexels_ultraedit_job.sh <工作>（basic-1）
#   flux  bash $T/tools/run_pexels_cross_model_job.sh flux <工作>（basic-2）
# T=~/image-immunization/color/trials/lut3d_defense_and_naturalness；狀態在 $T/runtime/cross_model_pexels/<sd3|flux>/：
#   jobs/<工作>.done 已存在 → 結束碼 0；jobs/<工作>.claim 已存在（color 的 lane 執行中）→ 結束碼 4；
#   否則建立 .claim、執行工具並把輸出附加到 logs/<工作>.log，工具結束碼 0 且本次輸出含「[JOB DONE] <sd3|flux> <工作>」
#   才寫 .done；結束時移除 .claim。工作清單與相依由 color 提供（$T/runtime/cross_model_pexels/<sd3|flux>/baseline_queue.txt）。
# 用法（遠端 baseline 專案根，由 scripts/run_pexels_queue_worker.sh 呼叫）：bash scripts/run_color_transfer_job.sh <sd3|flux> <工作>
set -uo pipefail
[ $# -eq 2 ] || { echo "用法：$0 <sd3|flux> <工作>" >&2; exit 2; }
E=$1 JOB=$2
: "${CUDA_VISIBLE_DEVICES:?需要由租約設定 CUDA_VISIBLE_DEVICES}"
T=$HOME/image-immunization/color/trials/lut3d_defense_and_naturalness
case $E in
  sd3) TOOL=(bash "$T/tools/run_pexels_ultraedit_job.sh" "$JOB") ;;
  flux) TOOL=(bash "$T/tools/run_pexels_cross_model_job.sh" flux "$JOB") ;;
  *) echo "[FATAL] 未知的編輯器 $E" >&2; exit 2 ;;
esac
J=$T/runtime/cross_model_pexels/$E/jobs
L=$T/runtime/cross_model_pexels/$E/logs/$JOB.log
mkdir -p "$J" "$(dirname "$L")" || exit 1
[ -f "$J/$JOB.done" ] && { echo "[SKIP] $E $JOB 已完成"; exit 0; }
mkdir "$J/$JOB.claim" 2>/dev/null || { echo "[NOT READY] $J/$JOB.claim 已存在"; exit 4; }
trap 'rm -rf "$J/$JOB.claim"' EXIT
echo "[STEP] $(date -Iseconds) color $E $JOB（GPU $CUDA_VISIBLE_DEVICES，log $L）"
offset=$(( $(stat -c %s "$L" 2>/dev/null || echo 0) + 1 ))
(cd "$HOME/image-immunization/color" && source scripts/env.sh && "${TOOL[@]}") >> "$L" 2>&1 < /dev/null
rc=$?
if [ $rc -ne 0 ] || ! tail -c +"$offset" "$L" | grep -qF "[JOB DONE] $E $JOB"; then
  echo "[FATAL] color $E $JOB 結束碼 $rc，或本次 log 無 [JOB DONE]；末段："; tail -n 20 "$L"
  exit 1
fi
touch "$J/$JOB.done"
echo "[DONE] $(date -Iseconds) color $E $JOB"

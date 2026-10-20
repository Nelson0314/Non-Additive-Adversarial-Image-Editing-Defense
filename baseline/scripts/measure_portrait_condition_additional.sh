#!/usr/bin/env bash
# scripts/evaluate_portrait_condition.sh 之後的 CPU 讀數：fidelity、aesthetic（<CHAIN_ROOT>/defenses 下全部條件）與該條件的
# displacement、retention（FSIM／MSE）、vmaf、edit_aesthetic。輸出到 <CHAIN_WORK>/results/additional/。
# 用法（專案根，已 source scripts/env.sh）：[CHAIN_ROOT=…] [CHAIN_WORK=…] bash scripts/measure_portrait_condition_additional.sh <條件>
set -euo pipefail
C=$1
R=${CHAIN_WORK:-runtime/aligned_recalibration/$C}/results
A=${CHAIN_ROOT:-artifacts/aligned}
O=$R/additional
mkdir -p $O
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
M="$PY -m immunization_baseline.cli.measure_additional_metrics"
$M --stage fidelity --defenses-root $A/defenses --output-dir $O
$M --stage aesthetic --defenses-root $A/defenses --output-dir $O
$M --stage displacement --displacement-csv $R/displacement.csv --output-dir $O
$M --stage retention --retention-csv $R/retention.csv --purified-edits-root $A/purified_edits --output-dir $O
$M --stage edit_aesthetic --displacement-csv $R/displacement.csv --output-dir $O
$M --stage vmaf --defenses-root $A/defenses --displacement-csv $R/displacement.csv \
   --retention-csv $R/retention.csv --purified-edits-root $A/purified_edits --output-dir $O
echo "[STEP] $(date -Is) CPU 讀數完成 $C"

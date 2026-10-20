#!/usr/bin/env bash
# data/portraits 8 張上單一條件的完整評測（ip2p 與 inpaint 兩場景）。在租約給定的單卡上執行：
#   舊產物移到 artifacts/aligned/superseded/<條件>_eps_scale_<舊倍率>/ → 以新倍率求解（或沿用 --defenses-from 的解）→
#   兩場景編輯 → 七道淨化與淨化後編輯 → displacement、retention → semantic_identity（GPU）；
#   其後 CPU：additional_metrics 的 displacement、retention、vmaf、edit_aesthetic。
# 部分讀數寫到 runtime/aligned_recalibration/<條件>/results/，由呼叫端併入 results/aligned/。
# 用法（baseline 專案根，已 source scripts/env.sh 並設 CUDA_VISIBLE_DEVICES）：
#   bash scripts/evaluate_portrait_condition.sh <條件> <倍率> <舊倍率字串或 none>
# 預設對象為等失真臂（artifacts/aligned、臂名後綴 _aligned）；主表原生列以 CHAIN_ROOT=artifacts ARM_SUFFIX= 執行。
# 防禦圖由其他程式產生時（例如 generate_blurguard_defenses），先放進 <CHAIN_ROOT>/defenses/<條件>/ 並建立
# <CHAIN_WORK>/state/moved.done 與 defenses.done。其後的 CPU 讀數見 scripts/measure_portrait_condition_additional.sh。
# CHAIN_ROOT、ARM_SUFFIX、CHAIN_WORK 可覆寫產物根（預設 artifacts/aligned）、防禦後編輯的臂名後綴（預設 _aligned）與狀態目錄。
# HELPER=1 時只參與七道淨化後編輯（以 state/claim_<算子> 目錄分工，可多卡並行），等主行程完成淨化後才開始。
set -euo pipefail
C=$1; SCALE=$2; OLD=$3
A=${CHAIN_ROOT:-artifacts/aligned}
SUF=${ARM_SUFFIX-_aligned}
W=${CHAIN_WORK:-runtime/aligned_recalibration/$C}
R=$W/results
S=$W/state
mkdir -p $R $S
step() { echo "[STEP] $(date -Is) $*"; }
PURIFIERS=$($PY -m immunization_core.purifiers.protocol --exclude-identity)
if [ -n "${HELPER:-}" ]; then
  until [ -f $S/purified.done ]; do sleep 30; done
else
if [ ! -f $S/moved.done ]; then
  step "舊產物移到 superseded"
  SUP=$A/superseded/${C}_eps_scale_$OLD
  mkdir -p $SUP/defenses $SUP/defended_edits $SUP/purified_edits
  for d in defenses defended_edits purified_edits; do [ -d $A/$d/$C ] && mv $A/$d/$C $SUP/$d/$C; done
  touch $S/moved.done
fi
if [ ! -f $S/defenses.done ]; then
  step "防禦生成 $C eps_scale=$SCALE"
  $PY -m immunization_baseline.cli.generate_defenses --data-root data/portraits --output-dir $A/defenses/$C \
      --conditions $C --eps-scale $SCALE --tag aligned
  touch $S/defenses.done
fi
if [ ! -f $S/purified.done ]; then
  step "淨化 $C"
  $PY -m immunization_baseline.cli.apply_purifiers --defenses-dir $A/defenses/$C --output-dir $A/purified/$C
  touch $S/purified.done
fi
if [ ! -f $S/edits.done ]; then
  step "防禦後編輯 $C"
  $PY -m immunization_baseline.cli.run_edits --data-root data/portraits --defenses-dir $A/defenses/$C \
      --output-dir $A/defended_edits/$C --scenarios ip2p inpaint --suffix _${C}${SUF}
  touch $S/edits.done
fi
fi
for P in $PURIFIERS; do
  [ -f $S/edits_$P.done ] && continue
  mkdir $S/claim_$P 2>/dev/null || continue
  step "淨化後編輯 $C $P"
  $PY -m immunization_baseline.cli.run_edits --data-root data/portraits --defenses-dir $A/purified/$C/$P \
      --output-dir $A/purified_edits/$C/$P --scenarios ip2p inpaint --suffix _${C}_$P
  touch $S/edits_$P.done
done
[ -n "${HELPER:-}" ] && { step "helper 完成 $C"; exit 0; }
for P in $PURIFIERS; do until [ -f $S/edits_$P.done ]; do sleep 30; done; done
if [ ! -f $S/measured.done ]; then
  step "量測 $C"
  $PY -m immunization_baseline.cli.measure_edit_displacement --defended-edits-root $A/defended_edits \
      --undefended-edits-root artifacts/undefended_edits --data-root data/portraits \
      --output-csv $R/displacement.csv --conditions $C
  $PY -m immunization_baseline.cli.measure_purified_displacement --purified-edits-root $A/purified_edits \
      --displacement-csv $R/displacement.csv --data-root data/portraits --output-csv $R/retention.csv --conditions $C
  $PY - $A/defenses/$C $R/defense_$C.csv <<'PY'
import csv, glob, sys
rows = [r for p in sorted(glob.glob(f"{sys.argv[1]}/results_*.csv")) for r in csv.DictReader(open(p, encoding="utf-8"))]
rows.sort(key=lambda r: r["image"])
if len(rows) != 8 or len({r["image"] for r in rows}) != 8:
    raise SystemExit(f"{sys.argv[1]} 的求解列不是 8 張各一列：{[r['image'] for r in rows]}")
with open(sys.argv[2], "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator=chr(10)); w.writeheader(); w.writerows(rows)
PY
  touch $S/measured.done
fi
if [ ! -f $S/semantic.done ]; then
  step "semantic_identity $C"
  $PY -m immunization_baseline.cli.measure_semantic_identity --displacement-csv $R/displacement.csv \
      --data-root data/portraits --output-csv $R/semantic_identity.csv --device cuda
  touch $S/semantic.done
fi
step "GPU 部分完成 $C"

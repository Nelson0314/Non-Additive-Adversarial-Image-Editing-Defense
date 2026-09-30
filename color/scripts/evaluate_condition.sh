#!/usr/bin/env bash
# 一個條件的完整鏈，固定跑在一張卡上：
#   防禦圖 → 防禦後編輯 → 淨化（七道）→ 淨化後編輯（七道）
#
# 每一階段跑完寫一個完成標記到 runtime/state/，重跑時已完成的階段直接略過；
# 標記只在 rc=0 時才寫，中途中止的階段下次會重跑。全部完成時寫 <條件>.chain.done。
# 七道淨化取自 immunization_core 的協定正本（purifiers/protocol.json）。
#
# 用法：bash scripts/evaluate_condition.sh <GPU> <條件名>
set -uo pipefail
GPU="$1"; ARM="$2"
source "$(dirname "${BASH_SOURCE[0]}")/env.sh" || exit 1
cd "$COLOR_ROOT" || exit 1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES="$GPU"
export CUDA_DEVICE_ORDER=PCI_BUS_ID

A=artifacts
S=runtime/state
# 攻擊端只跑 ip2p（整張圖編輯）。inpaint 場景的遮罩由原圖切出，攻擊者拿不到，
# 且幾何淨化後遮罩不跟著轉；使用者裁定不再跑。
SCENARIOS="ip2p"
# 防禦圖不滿 8 張的條件只編輯現有的影像，由 artifacts/defenses/<條件> 中的防禦圖決定。
IMGS=()
NDEF=$(ls -1 "$A/defenses/$ARM"/*__"$ARM"__def.png 2>/dev/null | wc -l)
if [ "$NDEF" -gt 0 ] && [ "$NDEF" -lt 8 ]; then
  IMGS=(--images $(ls -1 "$A/defenses/$ARM"/*__"$ARM"__def.png | xargs -n1 basename | sed "s/__${ARM}__def.png//"))
  echo "[SUBSET] $ARM 只有 $NDEF 張防禦圖：${IMGS[*]:1}"
fi
DATA=data/portraits
CLI=immunization_color.cli
PURIFIERS=$("$PY" -m immunization_core.purifiers.protocol --exclude-identity) \
  || { echo "[FATAL] 無法讀取淨化協定" >&2; exit 1; }
mkdir -p "$S"

step() {
  local tag="$1"; shift
  local flag="$S/${ARM}.${tag}.done"
  if [ -f "$flag" ]; then echo "[SKIP] $ARM/$tag"; return 0; fi
  echo "[START] $(date -Is) $ARM/$tag gpu=$GPU"
  "$@"
  local rc=$?
  echo "[EXIT] $(date -Is) $ARM/$tag rc=$rc"
  if [ "$rc" -eq 0 ]; then touch "$flag"; else return "$rc"; fi
}

step defense bash scripts/generate_condition.sh "$ARM" || exit 1

for SC in $SCENARIOS; do
  step "edit_$SC" "$PY" -m $CLI.run_edits --data "$DATA" \
      --defended "$A/defenses/$ARM" --out "$A/defended_edits/$ARM" \
      --scenarios "$SC" --suffix "_$ARM" "${IMGS[@]}" || exit 1
done

step purify "$PY" -m $CLI.apply_purifiers --defended "$A/defenses/$ARM" \
    --out "$A/purified/$ARM" || exit 1

for PUR in $PURIFIERS; do
  for SC in $SCENARIOS; do
    step "pedit_${PUR}_${SC}" "$PY" -m $CLI.run_edits --data "$DATA" \
        --defended "$A/purified/$ARM/$PUR" \
        --out "$A/purified_edits/$ARM/$PUR" --scenarios "$SC" \
        --suffix "_${ARM}_${PUR}" "${IMGS[@]}" || exit 1
  done
done

touch "$S/${ARM}.chain.done"
echo "[ARM-DONE] $(date -Is) $ARM gpu=$GPU"

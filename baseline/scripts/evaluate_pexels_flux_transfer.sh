#!/usr/bin/env bash
# Pexels 人像遷移性實驗的 FLUX 部分，模型每張影像只載入一次：協定、輸入、輸出、逐格表與完成標記與
# scripts/evaluate_pexels_transfer.sh flux 相同（四個條件 × {淨化前、VAE-Trans、七道} × 4 句 = 144 格），
# 36 次 run_flux_edits 改由 run_flux_edit_jobs 在同一次載入下依序執行。
#
# 位置：產物 artifacts/pexels_portraits/transfer/flux/；逐格表 runtime/pexels_portraits/transfer/flux/chunks/<影像>/；
#       該次的參數清單 chunks/<影像>/flux_jobs.txt；完成標記 runtime/pexels_portraits/transfer/flux/done/<影像>{,__<條件>__<道>}.done。
# 前置條件：該影像所屬影像組的等失真臂（chain.done、chain_photoguard_encoder.done）與 VAE-Trans 輸入；不成立時以結束碼 4 結束。
# 用法（遠端 baseline 專案根，basic-2；CUDA_VISIBLE_DEVICES 由呼叫端的租約設定）：
#   ENV_FILE=~/env.sh bash scripts/evaluate_pexels_flux_transfer.sh <影像>
set -uo pipefail
[ $# -eq 1 ] || { echo "用法：$0 <影像>" >&2; exit 2; }
IMG=$1
source "$(dirname "${BASH_SOURCE[0]}")/env.sh" || exit 1
cd "$BASELINE_ROOT" || exit 1
source scripts/run_evaluation_chain.sh
: "${CUDA_VISIBLE_DEVICES:?需要由租約設定 CUDA_VISIBLE_DEVICES}"
export TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CONDS="photoguard_encoder dct_shield dct_shield_y mist"
OPS="blur1 blur2 crop_resize0.1 jpeg30 jpeg50 jpeg80 rotate15"
PD=data/pexels_portraits
read -r SET root < <("$PY" - "$IMG" <<'EOF'
import json, sys
for s in json.load(open("data/pexels_portraits/image_sets.json", encoding="utf-8"))["sets"]:
    if sys.argv[1] in s["images"]:
        print(s["name"], s["root"])
        break
else:
    raise SystemExit(f"image_sets.json 沒有影像 {sys.argv[1]}")
EOF
) || exit 1
AS=runtime/pexels_portraits/aligned${root:+/$root}
AA=artifacts/pexels_portraits/aligned${root:+/$root}
VI=artifacts/pexels_portraits/vae_trans${root:+/$root}/inputs/aligned
X=artifacts/pexels_portraits/transfer/flux
S=runtime/pexels_portraits/transfer/flux
R=$S/chunks/$IMG
mkdir -p "$S/done" "$R"
not_ready() { chain_step "前置條件不成立：$*"; exit 4; }
[ -f "$S/done/$IMG.done" ] && { chain_step "flux $IMG 已完成"; exit 0; }
[ -f "$AS/chain.done" ] && [ -f "$AS/chain_photoguard_encoder.done" ] || not_ready "等失真 $SET"
for c in $CONDS; do
  [ -f "$AA/defenses/$c/${IMG}__${c}__def.png" ] || { echo "[FATAL] 缺防禦圖 $AA/defenses/$c/${IMG}__${c}__def.png" >&2; exit 1; }
  [ -f "$VI/$c/vae_trans/ip2p_${c}_vae_trans/${IMG}__orig.png" ] || not_ready "$c 的 VAE-Trans 輸入（$SET）"
  for op in $OPS; do
    [ -f "$AA/purified_edits/$c/$op/ip2p_${c}_$op/${IMG}__orig.png" ] || { echo "[FATAL] 缺 $c $op 的淨化後輸入" >&2; exit 1; }
  done
done
"$PY" -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 'CUDA 不可用')" || exit 1
chain_step "遷移 flux $IMG（$SET，GPU $CUDA_VISIBLE_DEVICES，單次載入）"

# 參數清單：編輯順序同 evaluate_pexels_transfer.sh（每條件依序淨化前、VAE-Trans、七道），已有完成標記的設定不列入
J=$R/flux_jobs.txt
: > "$J.tmp"
pending=""
for c in $CONDS; do
  for op in none vae_trans $OPS; do
    [ -f "$S/done/${IMG}__${c}__$op.done" ] && continue
    pending="$pending ${c}__$op"
    if [ "$op" = none ]; then
      echo "--arm $c --data-root $PD --images $IMG --defenses-dir $AA/defenses/$c --output-dir $X/edits/$c --output-csv $R/edits_${c}__none.csv" >> "$J.tmp"
    else
      root_dir=$AA/purified_edits; [ "$op" = vae_trans ] && root_dir=$VI
      echo "--arm $c --data-root $PD --images $IMG --purifier $op --purified-edits-root $root_dir --output-dir $X/purified_edits/$c/$op --output-csv $R/edits_${c}__$op.csv" >> "$J.tmp"
    fi
  done
done
mv "$J.tmp" "$J"
if [ -n "$pending" ]; then
  chain_step "FLUX $(wc -l < "$J") 項（$J）"
  "$PY" -m $CHAIN_CLI.run_flux_edit_jobs --jobs-file "$J" || exit 1
  for k in $pending; do touch "$S/done/${IMG}__$k.done"; done
fi
touch "$S/done/$IMG.done"
chain_step "TRANSFER DONE flux $IMG"

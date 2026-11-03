#!/usr/bin/env bash
# Pexels 人像的遷移性實驗：ip2p 求解的等失真臂防禦圖直接送 FLUX.1-Kontext 或 UltraEdit 編輯（不以替代編輯器的 VAE 重新求解）。
# 對一張影像執行四個條件（等失真版的 photoguard_encoder、dct_shield、dct_shield_y、mist）的全協定：淨化前、七道、VAE-Trans，
# 每條件 4 句 × 9 = 36 格；編輯順序為淨化前、VAE-Trans、七道。全部 144 格完成才寫該影像的完成標記。
# 淨化後輸入重用 ip2p 流程送進編輯器的輸入：七道為 artifacts/pexels_portraits/aligned/<根>/purified_edits/<條件>/<道>/ip2p_<條件>_<道>/，
# VAE-Trans 為 artifacts/pexels_portraits/vae_trans/<根>/inputs/aligned/<條件>/vae_trans/ip2p_<條件>_vae_trans/。
# 協定同 results/flux（guidance 3.5、28 步、NF4、1024²）與 results/ultraedit（guidance 2.5、image guidance 1.5、variant add、512²、50 步）。
# 未防禦分母與 color 的列由 color 專案產生。
#
# 位置：產物 artifacts/pexels_portraits/transfer/<編輯器>/；逐格表 runtime/pexels_portraits/transfer/<編輯器>/chunks/<影像>/；
#       完成標記 runtime/pexels_portraits/transfer/<編輯器>/done/<影像>.done。
# 前置條件：該影像所屬影像組的等失真臂（chain.done、chain_photoguard_encoder.done）與 VAE-Trans 輸入；不成立時以結束碼 4 結束。
# 用法（遠端 baseline 專案根；CUDA_VISIBLE_DEVICES 由呼叫端的租約設定；FLUX 只在 basic-2、UltraEdit 只在 basic-1）：
#   [TRANSFER_DEADLINE=<ISO 時刻>] ENV_FILE=~/env.sh bash scripts/evaluate_pexels_transfer.sh <flux|ultraedit> <影像>
set -uo pipefail
[ $# -eq 2 ] || { echo "用法：$0 <flux|ultraedit> <影像>" >&2; exit 2; }
M=$1 IMG=$2
source "$(dirname "${BASH_SOURCE[0]}")/env.sh" || exit 1
cd "$BASELINE_ROOT" || exit 1
source scripts/run_evaluation_chain.sh
: "${CUDA_VISIBLE_DEVICES:?需要由租約設定 CUDA_VISIBLE_DEVICES}"
export TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CONDS="photoguard_encoder dct_shield dct_shield_y mist"
OPS="blur1 blur2 crop_resize0.1 jpeg30 jpeg50 jpeg80 rotate15"
UE="--guidance 2.5 --image-guidance 1.5 --variant add"
PD=data/pexels_portraits
case $M in flux|ultraedit) ;; *) echo "[FATAL] 未知的編輯器 $M" >&2; exit 2 ;; esac
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
X=artifacts/pexels_portraits/transfer/$M
S=runtime/pexels_portraits/transfer/$M
R=$S/chunks/$IMG
mkdir -p "$S/done" "$R"
not_ready() { chain_step "前置條件不成立：$*"; exit 4; }
[ -f "$S/done/$IMG.done" ] && { chain_step "$M $IMG 已完成"; exit 0; }
# TRANSFER_DEADLINE（例如 2026-11-04T20:00:00+08:00）之後不再開始新影像；進行中的影像不受影響
if [ -n "${TRANSFER_DEADLINE:-}" ] && [ "$(date +%s)" -ge "$(date -d "$TRANSFER_DEADLINE" +%s)" ]; then
  not_ready "已過 TRANSFER_DEADLINE=$TRANSFER_DEADLINE"
fi
[ -f "$AS/chain.done" ] && [ -f "$AS/chain_photoguard_encoder.done" ] || not_ready "等失真 $SET"
for c in $CONDS; do
  [ -f "$AA/defenses/$c/${IMG}__${c}__def.png" ] || { echo "[FATAL] 缺防禦圖 $AA/defenses/$c/${IMG}__${c}__def.png" >&2; exit 1; }
  [ -f "$VI/$c/vae_trans/ip2p_${c}_vae_trans/${IMG}__orig.png" ] || not_ready "$c 的 VAE-Trans 輸入（$SET）"
  for op in $OPS; do
    [ -f "$AA/purified_edits/$c/$op/ip2p_${c}_$op/${IMG}__orig.png" ] || { echo "[FATAL] 缺 $c $op 的淨化後輸入" >&2; exit 1; }
  done
done
"$PY" -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 'CUDA 不可用')" || exit 1
chain_step "遷移 $M $IMG（$SET，GPU $CUDA_VISIBLE_DEVICES）"

if [ "$M" = flux ]; then
  for c in $CONDS; do
    for op in none vae_trans $OPS; do
      [ -f "$S/done/${IMG}__${c}__$op.done" ] && continue
      chain_step "FLUX $c $op"
      if [ "$op" = none ]; then
        "$PY" -m $CHAIN_CLI.run_flux_edits --arm "$c" --data-root "$PD" --images "$IMG" --defenses-dir "$AA/defenses/$c" \
            --output-dir "$X/edits/$c" --output-csv "$R/edits_${c}__none.csv" || exit 1
      else
        root_dir=$AA/purified_edits; [ "$op" = vae_trans ] && root_dir=$VI
        "$PY" -m $CHAIN_CLI.run_flux_edits --arm "$c" --data-root "$PD" --images "$IMG" --purifier "$op" \
            --purified-edits-root "$root_dir" --output-dir "$X/purified_edits/$c/$op" --output-csv "$R/edits_${c}__$op.csv" || exit 1
      fi
      touch "$S/done/${IMG}__${c}__$op.done"
    done
  done
else
  # run_ultraedit_edits 處理資料根內的全部影像：以只含本影像的資料子集（指向資料組的符號連結）呼叫
  D=$S/data/$IMG
  if [ ! -f "$D/prompts.yaml" ]; then
    cls=${IMG%%_*}
    mkdir -p "$D/$cls" "$D/masks" || exit 1
    ln -sfn "$BASELINE_ROOT/$PD/$cls/$IMG.png" "$D/$cls/$IMG.png" || exit 1
    ln -sfn "$BASELINE_ROOT/$PD/masks/$IMG.png" "$D/masks/$IMG.png" || exit 1
    ln -sfn "$BASELINE_ROOT/$PD/prompts.yaml" "$D/prompts.yaml" || exit 1
  fi
  for c in $CONDS; do
    for part in plain vae_trans seven; do
      [ -f "$S/done/${IMG}__${c}__$part.done" ] && continue
      chain_step "UltraEdit $c $part"
      case $part in
        plain) args=(--purifiers none --purified-edits-root "$AA/purified_edits") ;;
        vae_trans) args=(--purifiers vae_trans --purified-edits-root "$VI") ;;
        seven) # shellcheck disable=SC2206
               args=(--purifiers $OPS --purified-edits-root "$AA/purified_edits") ;;
      esac
      # shellcheck disable=SC2086
      "$PY" -m $CHAIN_CLI.run_ultraedit_edits --data-root "$D" --arms "$c" "${args[@]}" $UE \
          --defenses-root "$AA/defenses" --output-dir "$X/edits" --output-csv-dir "$R/$part" || exit 1
      touch "$S/done/${IMG}__${c}__$part.done"
    done
  done
fi
touch "$S/done/$IMG.done"
chain_step "TRANSFER DONE $M $IMG"

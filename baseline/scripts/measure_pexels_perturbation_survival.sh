#!/usr/bin/env bash
# Pexels 人像的擾動存續比 LPIPS(P(x̃), P(x)) ÷ LPIPS(x̃, x)（`measure_perturbation_survival`，交付 PNG、CPU）。
# 對象為等失真九個條件（evaluate_pexels_aligned.sh 的八個與 photoguard_encoder）與 color3d_pexels。
#
#   <影像組> seven      等失真九個條件的七道；P(x̃)、P(x) 為評測鏈送進編輯器的淨化後輸入
#                       （artifacts/pexels_portraits/aligned/<根>/purified_edits/<條件|undefended>/<算子>/ip2p_*/<影像>__orig.png）
#                       前置條件：該影像組的 chain.done 與 chain_photoguard_encoder.done
#   <影像組> vae_trans  等失真九個條件與 color3d_pexels 的 VAE-Trans；P(x̃)、P(x) 為 apply_vae_trans 的輸出
#                       （artifacts/pexels_portraits/vae_trans/<根>/inputs/<分組>/<條件>/vae_trans/ip2p_<條件>_vae_trans/<影像>__orig.png，
#                       分母為 inputs/native/undefended/）；前置條件：evaluate_pexels_followup.sh 的 aligned 與 color3d 分組
#   color3d seven       color3d_pexels 全部 306 張的七道；影像取自 color trial artifacts_pexels/（defenses/color3d_pexels/、
#                       purified_edits/，只讀）
# 前置條件不成立時以結束碼 4 結束。
#
# 輸出：runtime/pexels_portraits/followup/<根>/results/<aligned|color3d>/survival_<seven|vae_trans>.csv；
#       color3d seven 為 runtime/pexels_portraits/followup/results/color3d_all/survival_seven.csv。
# 用法（遠端 baseline 專案根，不需 GPU）：ENV_FILE=~/env.sh bash scripts/measure_pexels_perturbation_survival.sh <影像組|color3d> <seven|vae_trans>
set -uo pipefail
[ $# -eq 2 ] || { echo "用法：$0 <影像組|color3d> <seven|vae_trans>" >&2; exit 2; }
SET=$1 KIND=$2
source "$(dirname "${BASH_SOURCE[0]}")/env.sh" || exit 1
cd "$BASELINE_ROOT" || exit 1
source scripts/run_evaluation_chain.sh
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=${OMP_NUM_THREADS:-8} MKL_NUM_THREADS=${MKL_NUM_THREADS:-8}
M="$PY -m $CHAIN_CLI.measure_perturbation_survival"
ALIGNED_CONDITIONS="mist dia_pt sifm dia_r dct_shield_y dct_shield dayn danp photoguard_encoder"
CP=../color/trials/lut3d_defense_and_naturalness/artifacts_pexels
not_ready() { chain_step "前置條件不成立：$*"; exit 4; }

if [ "$SET" = color3d ]; then
  [ "$KIND" = seven ] || { echo "[FATAL] color3d 只有 seven（VAE-Trans 依影像組計算）" >&2; exit 2; }
  R=runtime/pexels_portraits/followup/results/color3d_all
  mkdir -p "$R"
  chain_step "擾動存續比 color3d_pexels 七道（306 張）"
  # shellcheck disable=SC2086
  $M --defenses-root "$CP/defenses" --purified-edits-root "$CP/purified_edits" --conditions color3d_pexels \
     --output-csv "$R/survival_seven.csv" || exit 1
  chain_step "SURVIVAL DONE color3d seven"
  exit 0
fi

root=$("$PY" - "$SET" <<'EOF'
import json, sys
sets = {s["name"]: s for s in json.load(open("data/pexels_portraits/image_sets.json", encoding="utf-8"))["sets"]}
if sys.argv[1] not in sets:
    raise SystemExit(f"image_sets.json 沒有影像組 {sys.argv[1]}")
print(sets[sys.argv[1]]["root"])
EOF
) || exit 1
AS=runtime/pexels_portraits/aligned${root:+/$root}
AA=artifacts/pexels_portraits/aligned${root:+/$root}
V=artifacts/pexels_portraits/vae_trans${root:+/$root}
F=runtime/pexels_portraits/followup${root:+/$root}
mkdir -p "$F/results/aligned" "$F/results/color3d"

case $KIND in
  seven)
    [ -f "$AS/chain.done" ] || not_ready "等失真 $SET"
    [ -f "$AS/chain_photoguard_encoder.done" ] || not_ready "等失真 $SET 的 photoguard_encoder"
    chain_step "擾動存續比 $SET 等失真七道"
    # shellcheck disable=SC2086
    $M --defenses-root "$AA/defenses" --purified-edits-root "$AA/purified_edits" --conditions $ALIGNED_CONDITIONS \
       --output-csv "$F/results/aligned/survival_seven.csv" || exit 1
    ;;
  vae_trans)
    [ -f "$F/done/aligned.done" ] || not_ready "$SET 的 followup aligned"
    [ -f "$F/done/color3d.done" ] || not_ready "$SET 的 followup color3d"
    reference="$V/inputs/native/undefended/vae_trans/ip2p_undefended_vae_trans/{image}__orig.png"
    chain_step "擾動存續比 $SET VAE-Trans"
    # shellcheck disable=SC2086
    $M --defenses-root "$AA/defenses" --purified-edits-root "$AA/purified_edits" --conditions $ALIGNED_CONDITIONS --purifiers \
       --extra-purifier vae_trans "$V/inputs/aligned/{condition}/vae_trans/ip2p_{condition}_vae_trans/{image}__orig.png" "$reference" \
       --output-csv "$F/results/aligned/survival_vae_trans.csv" || exit 1
    $M --defenses-root "$V/color3d/defenses" --purified-edits-root "$AA/purified_edits" --conditions color3d_pexels --purifiers \
       --extra-purifier vae_trans "$V/inputs/color3d/{condition}/vae_trans/ip2p_{condition}_vae_trans/{image}__orig.png" "$reference" \
       --output-csv "$F/results/color3d/survival_vae_trans.csv" || exit 1
    ;;
  *) echo "[FATAL] 未知的種類 $KIND" >&2; exit 2 ;;
esac
chain_step "SURVIVAL DONE $SET $KIND"

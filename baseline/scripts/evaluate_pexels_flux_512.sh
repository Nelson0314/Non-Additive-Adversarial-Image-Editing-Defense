#!/usr/bin/env bash
# Pexels 人像的 FLUX.1-Kontext 編輯，512×512 協定（資料版本 flux_512；與 1024 版的差異見 results/pexels_portraits/flux_512/README.md）。
# ip2p 上求得的防禦圖直接送 FLUX；一張影像六個條件：undefended（分母）、等失真 photoguard_encoder、dct_shield、dct_shield_y、mist，
# 以及 color3d_pexels（color trial 的防禦圖與淨化後輸入，只讀）。每條件 {淨化前、VAE-Trans、七道} × 4 句 = 36 格，共 216 格，
# 由 run_flux_edit_jobs 在同一次載入下執行（--generation-size 512；guidance 3.5、28 步、NF4、指令與種子同 results/flux）。
#
# 輸入（<根> 為 image_sets.json 的 root 欄）：
#   防禦圖     等失真 artifacts/pexels_portraits/aligned/<根>/defenses/<條件>/；color3d_pexels 為 color trial artifacts_pexels/defenses/color3d_pexels/
#   七道       等失真與 undefended artifacts/pexels_portraits/aligned/<根>/purified_edits/<條件>/<道>/ip2p_<條件>_<道>/<影像>__orig.png；
#              color3d_pexels 為 color trial artifacts_pexels/purified_edits/
#   VAE-Trans  artifacts/pexels_portraits/vae_trans/<根>/inputs/{aligned,native,color3d}/<條件>/vae_trans/ip2p_<條件>_vae_trans/<影像>__orig.png
#              （等失真、undefended、color3d_pexels 依序）
# 輸出：產物 artifacts/pexels_portraits/flux_512/{edits,purified_edits}/<條件>/…；逐格表 runtime/pexels_portraits/flux_512/chunks/<影像>/；
#       參數清單 chunks/<影像>/flux_jobs.txt；完成標記 runtime/pexels_portraits/flux_512/done/<影像>{,__<條件>__<設定>}.done。
# COLOR3D_GROUP＝color3d_<變體> 時只跑條件 color3d_<變體>_pexels（防禦圖與七道取 color trial 合併後的產物，VAE-Trans 取
# scripts/evaluate_pexels_followup.sh 該分組的輸入），影像完成標記為 done/<影像>__<分組>.done；FLUX_WITHOUT_COLOR3D=1 時
# 預設的六條件去掉 color3d_pexels。
# 前置條件不成立時以結束碼 4 結束；缺必要輸入時失敗。
# 用法（遠端 baseline 專案根，basic-2；CUDA_VISIBLE_DEVICES 由呼叫端的租約設定）：
#   ENV_FILE=~/env.sh bash scripts/evaluate_pexels_flux_512.sh <影像>
set -uo pipefail
[ $# -eq 1 ] || { echo "用法：$0 <影像>" >&2; exit 2; }
IMG=$1
source "$(dirname "${BASH_SOURCE[0]}")/env.sh" || exit 1
cd "$BASELINE_ROOT" || exit 1
source scripts/run_evaluation_chain.sh
: "${CUDA_VISIBLE_DEVICES:?需要由租約設定 CUDA_VISIBLE_DEVICES}"
export TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

ALIGNED="photoguard_encoder dct_shield dct_shield_y mist"
CONDS="undefended $ALIGNED color3d_pexels"
[ "${FLUX_WITHOUT_COLOR3D:-0}" = 1 ] && CONDS="undefended $ALIGNED"
G=${COLOR3D_GROUP:-}
if [ -n "$G" ]; then
  case $G in color3d_*) ;; *) echo "[FATAL] COLOR3D_GROUP 須為 color3d_<變體>：$G" >&2; exit 2 ;; esac
  CONDS=${G}_pexels
fi
OPS="blur1 blur2 crop_resize0.1 jpeg30 jpeg50 jpeg80 rotate15"
PD=data/pexels_portraits
CT=../color/trials/lut3d_defense_and_naturalness/artifacts_pexels
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
VT=artifacts/pexels_portraits/vae_trans${root:+/$root}/inputs
X=artifacts/pexels_portraits/flux_512
S=runtime/pexels_portraits/flux_512
R=$S/chunks/$IMG
mkdir -p "$S/done" "$R"
not_ready() { chain_step "前置條件不成立：$*"; exit 4; }
IMG_DONE=$S/done/$IMG${G:+__$G}.done
[ -f "$IMG_DONE" ] && { chain_step "flux_512 $IMG ${G:-} 已完成"; exit 0; }
[ -f "$AS/chain.done" ] && [ -f "$AS/chain_photoguard_encoder.done" ] || not_ready "等失真 $SET"

defenses_dir() { case $1 in color3d_*) echo "$CT/defenses/$1" ;; *) echo "$AA/defenses/$1" ;; esac; }
seven_root() { case $1 in color3d_*) echo "$CT/purified_edits" ;; *) echo "$AA/purified_edits" ;; esac; }
vae_root() { case $1 in color3d_pexels) echo "$VT/color3d" ;; color3d_*) echo "$VT/${1%_pexels}" ;; undefended) echo "$VT/native" ;; *) echo "$VT/aligned" ;; esac; }
for c in $CONDS; do
  if [ "$c" != undefended ]; then
    [ -f "$(defenses_dir $c)/${IMG}__${c}__def.png" ] || { echo "[FATAL] 缺防禦圖 $(defenses_dir $c)/${IMG}__${c}__def.png" >&2; exit 1; }
  fi
  [ -f "$(vae_root $c)/$c/vae_trans/ip2p_${c}_vae_trans/${IMG}__orig.png" ] || not_ready "$c 的 VAE-Trans 輸入（$SET）"
  for op in $OPS; do
    f=$(seven_root $c)/$c/$op/ip2p_${c}_$op/${IMG}__orig.png
    [ -f "$f" ] || { echo "[FATAL] 缺淨化後輸入 $f" >&2; exit 1; }
  done
done
"$PY" -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 'CUDA 不可用')" || exit 1
chain_step "flux_512 $IMG（$SET，GPU $CUDA_VISIBLE_DEVICES）"

# 參數清單：每條件依序淨化前、VAE-Trans、七道；已有完成標記的設定不列入
J=$R/flux_jobs.txt
: > "$J.tmp"
pending=""
for c in $CONDS; do
  for op in none vae_trans $OPS; do
    [ -f "$S/done/${IMG}__${c}__$op.done" ] && continue
    pending="$pending ${c}__$op"
    common="--arm $c --data-root $PD --images $IMG --generation-size 512"
    if [ "$op" = none ]; then
      out=$X/edits/$c
      [ "$c" = undefended ] && extra="" || extra="--defenses-dir $(defenses_dir $c)"
      echo "$common $extra --output-dir $out --output-csv $R/edits_${c}__none.csv" >> "$J.tmp"
    else
      root_dir=$(seven_root $c); [ "$op" = vae_trans ] && root_dir=$(vae_root $c)
      echo "$common --purifier $op --purified-edits-root $root_dir --output-dir $X/purified_edits/$c/$op --output-csv $R/edits_${c}__$op.csv" >> "$J.tmp"
    fi
  done
done
mv "$J.tmp" "$J"
if [ -n "$pending" ]; then
  chain_step "FLUX 512 $(wc -l < "$J") 項（$J）"
  "$PY" -m $CHAIN_CLI.run_flux_edit_jobs --jobs-file "$J" || exit 1
  for k in $pending; do touch "$S/done/${IMG}__$k.done"; done
fi
touch "$IMG_DONE"
chain_step "FLUX 512 DONE $IMG ${G:-}"

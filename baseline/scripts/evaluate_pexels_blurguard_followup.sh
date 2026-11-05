#!/usr/bin/env bash
# BlurGuard 在 Pexels 100 張子集上的後續量測，一個影像組一次（GPU 1 張）：VAE-Trans 淨化與淨化後 ip2p 編輯、VAE-Trans 位移保留、
# 語意與身分（未淨化、七道、VAE-Trans）、CLIP-S、ArcFace(E_u, E_d)、擾動存續比（七道與 VAE-Trans）、編輯端 SSIM／PSNR。
# 協定同 scripts/evaluate_pexels_followup.sh（encoder、ip2p_si18、四句、主種子）與 scripts/measure_pexels_followup_cpu.sh 系列。
#
# 輸入：runtime/pexels_portraits/blurguard/ 的子集資料根 data/<影像組>/、評測鏈讀數 results/<影像組>/displacement_blurguard.csv
#       （runtime/pexels_portraits/blurguard/run_blurguard_set.sh 產生，完成標記 state/<影像組>/chain.done）；
#       防禦圖與七道淨化後編輯在原生評測的產物根 artifacts/pexels_portraits/<根>/{defenses,purified_edits}/blurguard/。
# 輸出：artifacts/pexels_portraits/vae_trans/<根>/{inputs,blurguard/purified_edits}/blurguard/…；
#       runtime/pexels_portraits/blurguard/results/<影像組>/：vae_trans_fidelity_blurguard.csv、retention_vae_trans_blurguard.csv、
#       semantic_identity{,_vae_trans}.csv、clip_score.csv、arcface_pair.csv、survival.csv、edit_fidelity.csv
# 前置條件不成立時以結束碼 4 結束。
# 用法（遠端 baseline 專案根；CUDA_VISIBLE_DEVICES 由呼叫端的租約設定）：ENV_FILE=~/env.sh bash scripts/evaluate_pexels_blurguard_followup.sh <影像組>
set -uo pipefail
[ $# -eq 1 ] || { echo "用法：$0 <影像組>" >&2; exit 2; }
SET=$1
source "$(dirname "${BASH_SOURCE[0]}")/env.sh" || exit 1
cd "$BASELINE_ROOT" || exit 1
source scripts/run_evaluation_chain.sh
: "${CUDA_VISIBLE_DEVICES:?需要由租約設定 CUDA_VISIBLE_DEVICES}"
export TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
C=blurguard
ENC=../color/trials/lut3d_defense_and_naturalness/artifacts/purification/vae_trans/model/encoder.pt
PURIFIERS=$("$PY" -m immunization_core.purifiers.protocol --exclude-identity) || exit 1
root=$("$PY" -c "import json; print({s['name']: s['root'] for s in json.load(open('data/pexels_portraits/image_sets.json'))['sets']}['$SET'])") || exit 1
B=runtime/pexels_portraits/blurguard
D=$B/data/$SET S=$B/state/$SET R=$B/results/$SET
NA=artifacts/pexels_portraits${root:+/$root}
V=artifacts/pexels_portraits/vae_trans${root:+/$root}
step() { echo "[STEP] $(date -Iseconds) $*"; }
[ -f "$S/chain.done" ] || { step "前置條件不成立：$SET 的評測鏈"; exit 4; }
[ -f "$R/displacement_$C.csv" ] || { step "[SKIP] $SET 沒有選中的影像"; touch "$S/followup.done"; exit 0; }
[ -d "$V/inputs/native/undefended/vae_trans/ip2p_undefended_vae_trans" ] || { echo "[FATAL] 缺 VAE-Trans 未防禦分母 $V/inputs/native/undefended" >&2; exit 1; }
"$PY" -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 'CUDA 不可用')" || exit 1
n_images=$(ls "$D"/man/*.png "$D"/woman/*.png 2>/dev/null | wc -l)
step "BlurGuard followup $SET（$n_images 張，GPU $CUDA_VISIBLE_DEVICES）"

in=$V/inputs/$C/$C/vae_trans/ip2p_${C}_vae_trans
out=$V/$C/purified_edits/$C/vae_trans
if [ ! -f "$S/vae_trans_inputs.done" ]; then
  step "VAE-Trans 淨化"
  "$PY" -m $CHAIN_CLI.apply_vae_trans --defenses-dir "$NA/defenses/$C" --condition "$C" --encoder "$ENC" \
      --output-root "$V/inputs/$C" --fidelity-csv "$R/vae_trans_fidelity_$C.csv" || exit 1
  for f in "$in"/*__orig.png; do ln -f "$f" "${f%__orig.png}__vae_trans__def.png" || exit 1; done
  [ "$(ls "$in"/*__vae_trans__def.png | wc -l)" -eq "$n_images" ] || { echo "[FATAL] $in 張數不等於 $n_images" >&2; exit 1; }
  touch "$S/vae_trans_inputs.done"
fi
if [ ! -f "$S/vae_trans_edits.done" ]; then
  step "VAE-Trans 淨化後編輯"
  chain_edit_arm "$D" "$in" "$out" "_${C}_vae_trans" || exit 1
  touch "$S/vae_trans_edits.done"
fi
mkdir -p "$V/$C/purified_edits"
[ -L "$V/$C/purified_edits/undefended" ] || ln -s ../../native/purified_edits/undefended "$V/$C/purified_edits/undefended" || exit 1
if [ ! -f "$S/vae_trans_retention.done" ]; then
  step "VAE-Trans 位移保留"
  "$PY" -m $CHAIN_CLI.measure_purified_displacement --purified-edits-root "$V/$C/purified_edits" \
      --displacement-csv "$R/displacement_$C.csv" --data-root "$D" --output-csv "$R/retention_vae_trans_$C.csv" --conditions "$C" || exit 1
  touch "$S/vae_trans_retention.done"
fi
if [ ! -f "$S/semantic.done" ]; then
  step "語意與身分（未淨化與七道）"
  # shellcheck disable=SC2086
  "$PY" -m $CHAIN_CLI.measure_semantic_identity --displacement-csv "$R/displacement_$C.csv" --data-root "$D" \
      --output-csv "$R/semantic_identity.csv" --purifiers $PURIFIERS --device cuda || exit 1
  touch "$S/semantic.done"
fi
if [ ! -f "$S/semantic_vae_trans.done" ]; then
  step "語意與身分（VAE-Trans）"
  "$PY" - "$V" "$R/displacement_$C.csv" "$R/vae_trans_edits.csv" <<'EOF' || exit 1
import csv, sys
from pathlib import Path
v, table, out = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
def png(group, c, image, k):
    p = v / group / "purified_edits" / c / "vae_trans" / f"ip2p_{c}_vae_trans" / f"{image}__p{k}.png"
    if not p.is_file():
        raise SystemExit(f"缺 VAE-Trans 編輯：{p}")
    return p.as_posix()
rows = []
for r in csv.DictReader(open(table, encoding="utf-8")):
    if r["scenario"] != "ip2p":
        continue
    key = (r["image"], r["prompt_index"])
    rows.append({"arm": "undefended", "image": key[0], "prompt_index": key[1], "prompt": r["prompt"],
                 "png": png("native", "undefended", *key), "purifier": "vae_trans"})
    rows.append({"arm": r["condition"], "image": key[0], "prompt_index": key[1], "prompt": r["prompt"],
                 "png": png("blurguard", r["condition"], *key), "purifier": "vae_trans"})
with open(out, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n"); w.writeheader(); w.writerows(rows)
print(f"[PLAN] {out}（{len(rows)} 列）")
EOF
  "$PY" -m $CHAIN_CLI.measure_semantic_identity --edits-csv "$R/vae_trans_edits.csv" --scenario ip2p \
      --data-root "$D" --output-csv "$R/semantic_identity_vae_trans.csv" --purifiers vae_trans --device cuda || exit 1
  touch "$S/semantic_vae_trans.done"
fi
if [ ! -f "$S/clip_score.done" ]; then
  step CLIP-S
  "$PY" scripts/measure_clip_score.py --semantic-csv "$R/semantic_identity.csv" "$R/semantic_identity_vae_trans.csv" \
      --data-root data/pexels_portraits --output-csv "$R/clip_score.csv" --device cuda || exit 1
  touch "$S/clip_score.done"
fi
export CUDA_VISIBLE_DEVICES=
if [ ! -f "$S/arcface_pair.done" ]; then
  step "ArcFace(E_u, E_d)"
  "$PY" scripts/measure_pexels_arcface_pair.py --semantic-csv "$R/semantic_identity.csv" "$R/semantic_identity_vae_trans.csv" \
      --output-csv "$R/arcface_pair.csv" || exit 1
  touch "$S/arcface_pair.done"
fi
if [ ! -f "$S/survival.done" ]; then
  step 擾動存續比
  "$PY" -m $CHAIN_CLI.measure_perturbation_survival --defenses-root "$NA/defenses" --purified-edits-root "$NA/purified_edits" --conditions "$C" \
      --extra-purifier vae_trans "$V/inputs/$C/{condition}/vae_trans/ip2p_{condition}_vae_trans/{image}__orig.png" \
      "$V/inputs/native/undefended/vae_trans/ip2p_undefended_vae_trans/{image}__orig.png" --output-csv "$R/survival.csv" || exit 1
  touch "$S/survival.done"
fi
if [ ! -f "$S/edit_fidelity.done" ]; then
  step "編輯端 SSIM／PSNR"
  "$PY" - "$R" <<'EOF' || exit 1
import csv, sys
from pathlib import Path
R = Path(sys.argv[1])
rows = [r for p in ("semantic_identity.csv", "semantic_identity_vae_trans.csv") for r in csv.DictReader(open(R / p, encoding="utf-8"))]
ref = {(r["image"], r["prompt_index"], r["purifier"]): r["edit_png"] for r in rows if r["condition"] == "undefended"}
pairs = [{"condition": r["condition"], "end": "displacement", "purifier": r["purifier"], "a_png": r["edit_png"],
          "b_png": ref[(r["image"], r["prompt_index"], r["purifier"])]} for r in rows if r["condition"] != "undefended"]
with open(R / "pairs.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(pairs[0]), lineterminator="\n"); w.writeheader(); w.writerows(pairs)
print(f"[PAIRS] {len(pairs)} 對")
EOF
  "$PY" scripts/measure_pexels_edit_fidelity.py pairs --pairs-csv "$R/pairs.csv" --output-csv "$R/edit_fidelity.csv" --workers 8 || exit 1
  touch "$S/edit_fidelity.done"
fi
touch "$S/followup.done"
step "BLURGUARD FOLLOWUP DONE $SET"

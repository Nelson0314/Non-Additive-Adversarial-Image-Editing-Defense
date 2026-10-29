#!/usr/bin/env bash
# Pexels 人像的後續量測（GPU 部分），對一個影像組的一個分組執行：VAE-Trans 淨化與淨化後 ip2p 編輯、
# VAE-Trans 的位移保留表，以及語意與身分（未淨化、七道、VAE-Trans）。
#
# 分組（第二個參數）：
#   undefended  VAE-Trans 淨化未防禦原圖並編輯（其餘三組的分母）；需原生評測已完成該影像組的未防禦步驟
#   native      原生預算十個條件（scripts/evaluate_pexels_portraits.sh 的預設）；需該影像組十個條件的 measured_<條件>.done
#   aligned     等失真九個條件：scripts/evaluate_pexels_aligned.sh 的八個（需該影像組的 chain.done）與
#               photoguard_encoder（scripts/evaluate_pexels_aligned_condition.sh；需 chain_photoguard_encoder.done）
#   color3d     color 專案的 color3d_pexels（color trial artifacts_pexels/，只讀）；需該影像組每張影像的四句編輯
# native、aligned、color3d 需 undefended 已完成（done/undefended.done）。前置條件不成立時以結束碼 4 結束。
#
# 協定：VAE-Trans 為 `apply_vae_trans`（encoder 為 color trial 訓練的 artifacts/purification/vae_trans/model/encoder.pt，
# 與跨模型移植相同）；編輯同原生評測（ip2p_si18，四句，主種子）；語意與身分為 `measure_semantic_identity --device cuda`
# （與 results/pexels_portraits/semantic_identity_<條件>.csv 相同）。
#
# 位置（<根> 為 image_sets.json 的 root 欄，空字串表示第一批 97 張）：
#   產物  artifacts/pexels_portraits/vae_trans/<根>/inputs/<分組>/<條件>/vae_trans/ip2p_<條件>_vae_trans/
#             （`<影像>__orig.png` 為 apply_vae_trans 的輸出；`<影像>__vae_trans__def.png` 為同一檔案的硬連結，供 run_edits 讀取）
#         artifacts/pexels_portraits/vae_trans/<根>/<分組>/purified_edits/<條件>/vae_trans/ip2p_<條件>_vae_trans/
#             （分母在 native/purified_edits/undefended/；aligned、color3d 的 purified_edits/undefended 為指向它的相對符號連結）
#   狀態  runtime/pexels_portraits/followup/<根>/done/
#   讀數  runtime/pexels_portraits/followup/<根>/results/<分組>/
#           vae_trans_fidelity_<條件>.csv  VAE-Trans 淨化圖對淨化前輸入的 PSNR 與 LPIPS
#           retention_vae_trans_<條件>.csv  VAE-Trans 淨化後的編輯結果 LPIPS（measure_purified_displacement）
#           semantic_identity.csv           未淨化與七道（含 undefended 列）
#           semantic_identity_vae_trans.csv VAE-Trans（含 undefended 列）
#         color3d 另有 displacement_color3d_pexels.csv：color 的 results/color3d_pexels/displacement.csv 中本影像組的列，
#         影像路徑加上 `../color/` 前綴（相對 baseline 專案根）。
#
# 用法（遠端 baseline 專案根；CUDA_VISIBLE_DEVICES 由呼叫端的租約設定）：
#   ENV_FILE=~/env.sh bash scripts/evaluate_pexels_followup.sh <影像組> <undefended|native|aligned|color3d>
set -uo pipefail
[ $# -eq 2 ] || { echo "用法：$0 <影像組> <undefended|native|aligned|color3d>" >&2; exit 2; }
SET=$1 GROUP=$2
source "$(dirname "${BASH_SOURCE[0]}")/env.sh" || exit 1
cd "$BASELINE_ROOT" || exit 1
source scripts/run_evaluation_chain.sh
: "${CUDA_VISIBLE_DEVICES:?需要由租約設定 CUDA_VISIBLE_DEVICES}"
export TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

NATIVE_CONDITIONS="diffvax photoguard_encoder mist dia_pt sifm dia_r dct_shield_y dct_shield dayn danp"
ALIGNED_CONDITIONS="mist dia_pt sifm dia_r dct_shield_y dct_shield dayn danp photoguard_encoder"
ENC=../color/trials/lut3d_defense_and_naturalness/artifacts/purification/vae_trans/model/encoder.pt
CT=../color/trials/lut3d_defense_and_naturalness
CP=$CT/artifacts_pexels
PURIFIERS=$("$PY" -m immunization_core.purifiers.protocol --exclude-identity) \
  || { echo "[FATAL] 無法讀取淨化協定" >&2; exit 1; }

root=$("$PY" - "$SET" <<'EOF'
import json, sys
sets = {s["name"]: s for s in json.load(open("data/pexels_portraits/image_sets.json", encoding="utf-8"))["sets"]}
if sys.argv[1] not in sets:
    raise SystemExit(f"image_sets.json 沒有影像組 {sys.argv[1]}")
print(sets[sys.argv[1]]["root"])
EOF
) || exit 1
D=runtime/pexels_portraits/data/$SET
NS=runtime/pexels_portraits${root:+/$root}          # 原生評測的狀態與讀數
AS=runtime/pexels_portraits/aligned${root:+/$root}  # 等失真臂的狀態與讀數
AA=artifacts/pexels_portraits/aligned${root:+/$root}
NA=artifacts/pexels_portraits${root:+/$root}
V=artifacts/pexels_portraits/vae_trans${root:+/$root}
F=runtime/pexels_portraits/followup${root:+/$root}
R=$F/results/$GROUP
mkdir -p "$F/done" "$R"
[ -d "$D/man" ] || [ -d "$D/woman" ] || { echo "[FATAL] 缺資料子集 $D" >&2; exit 1; }
IMAGES=$(ls "$D"/man/*.png "$D"/woman/*.png 2>/dev/null | xargs -n1 basename | sed 's/\.png$//' | sort)
[ -n "$IMAGES" ] || { echo "[FATAL] $D 沒有影像" >&2; exit 1; }
not_ready() { chain_step "前置條件不成立：$*"; exit 4; }

# VAE-Trans 淨化一個防禦圖目錄（或未防禦原圖），建立 run_edits 讀取的 __def.png 硬連結，再做 ip2p 編輯。
vae_trans_arm() {  # <分組> <條件> <防禦圖目錄或空字串>
  local group=$1 c=$2 defended=$3 in out n
  in=$V/inputs/$group/$c/vae_trans/ip2p_${c}_vae_trans
  out=$V/$group/purified_edits/$c/vae_trans
  if [ ! -f "$F/done/vae_trans_inputs_${group}_$c.done" ]; then
    chain_step "VAE-Trans 淨化 $group $c"
    if [ -n "$defended" ]; then
      "$PY" -m $CHAIN_CLI.apply_vae_trans --defenses-dir "$defended" --condition "$c" --encoder "$ENC" \
          --output-root "$V/inputs/$group" --fidelity-csv "$R/vae_trans_fidelity_$c.csv" || return 1
    else
      "$PY" -m $CHAIN_CLI.apply_vae_trans --data-root "$D" --condition "$c" --encoder "$ENC" \
          --output-root "$V/inputs/$group" --fidelity-csv "$R/vae_trans_fidelity_$c.csv" || return 1
    fi
    for f in "$in"/*__orig.png; do ln -f "$f" "${f%__orig.png}__vae_trans__def.png" || return 1; done
    n=$(ls "$in"/*__vae_trans__def.png | wc -l)
    [ "$n" -eq "$(echo $IMAGES | wc -w)" ] || { echo "[FATAL] $in 有 $n 張，影像組為 $(echo $IMAGES | wc -w) 張" >&2; return 1; }
    touch "$F/done/vae_trans_inputs_${group}_$c.done"
  fi
  if [ ! -f "$F/done/vae_trans_edits_${group}_$c.done" ]; then
    chain_step "VAE-Trans 淨化後編輯 $group $c"
    chain_edit_arm "$D" "$in" "$out" "_${c}_vae_trans" || return 1
    touch "$F/done/vae_trans_edits_${group}_$c.done"
  fi
}

# 位移保留表：purified_edits 下只有 vae_trans 一道，measure_purified_displacement 因此只產生 VAE-Trans 列。
vae_trans_retention() {  # <分組> <條件> <位移表>
  local group=$1 c=$2 disp=$3
  [ -f "$F/done/vae_trans_retention_${group}_$c.done" ] && return 0
  chain_step "VAE-Trans 位移保留 $group $c"
  "$PY" -m $CHAIN_CLI.measure_purified_displacement --purified-edits-root "$V/$group/purified_edits" \
      --displacement-csv "$disp" --data-root "$D" --output-csv "$R/retention_vae_trans_$c.csv" --conditions "$c" || return 1
  touch "$F/done/vae_trans_retention_${group}_$c.done"
}

link_denominator() {  # <分組>
  mkdir -p "$V/$1/purified_edits" || return 1
  [ -L "$V/$1/purified_edits/undefended" ] || ln -s ../../native/purified_edits/undefended "$V/$1/purified_edits/undefended" || return 1
  [ -d "$V/$1/purified_edits/undefended/vae_trans/ip2p_undefended_vae_trans" ] \
    || { echo "[FATAL] 分母連結無效：$V/$1/purified_edits/undefended" >&2; return 1; }
}

# 語意與身分：未淨化與七道以位移表為計畫；VAE-Trans 以本腳本產生的編輯表為計畫（arm、image、prompt_index、prompt、png、purifier）。
semantic() {  # <分組> <位移表...>
  local group=$1; shift
  if [ ! -f "$F/done/semantic_$group.done" ]; then
    chain_step "語意與身分 $group（未淨化與七道）"
    # shellcheck disable=SC2086
    "$PY" -m $CHAIN_CLI.measure_semantic_identity --displacement-csv "$@" --data-root "$D" \
        --output-csv "$R/semantic_identity.csv" --purifiers $PURIFIERS --device cuda || return 1
    touch "$F/done/semantic_$group.done"
  fi
  if [ ! -f "$F/done/semantic_vae_trans_$group.done" ]; then
    chain_step "語意與身分 $group（VAE-Trans）"
    "$PY" - "$V" "$group" "$R/vae_trans_edits.csv" "$@" <<'EOF' || return 1
import csv, sys
from pathlib import Path
v, group, out, tables = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3]), sys.argv[4:]
rows, seen = [], set()
def png(root, c, image, k):
    p = root / "purified_edits" / c / "vae_trans" / f"ip2p_{c}_vae_trans" / f"{image}__p{k}.png"
    if not p.is_file():
        raise SystemExit(f"缺 VAE-Trans 編輯：{p}")
    return p.as_posix()
for t in tables:
    for r in csv.DictReader(open(t, encoding="utf-8")):
        if r["scenario"] != "ip2p":
            continue
        key = (r["image"], r["prompt_index"])
        if key not in seen:
            seen.add(key)
            rows.append({"arm": "undefended", "image": r["image"], "prompt_index": r["prompt_index"], "prompt": r["prompt"],
                         "png": png(v / "native", "undefended", *key), "purifier": "vae_trans"})
        rows.append({"arm": r["condition"], "image": r["image"], "prompt_index": r["prompt_index"], "prompt": r["prompt"],
                     "png": png(v / group, r["condition"], *key), "purifier": "vae_trans"})
with open(out, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n"); w.writeheader(); w.writerows(rows)
print(f"[PLAN] {out}（{len(rows)} 列）")
EOF
    "$PY" -m $CHAIN_CLI.measure_semantic_identity --edits-csv "$R/vae_trans_edits.csv" --scenario ip2p \
        --data-root "$D" --output-csv "$R/semantic_identity_vae_trans.csv" --purifiers vae_trans --device cuda || return 1
    touch "$F/done/semantic_vae_trans_$group.done"
  fi
}

run_group() {
  local c tables
  case $GROUP in
    undefended)
      [ -f "$NS/undefended_edits.done" ] || not_ready "原生 $SET 的未防禦編輯"
      vae_trans_arm native undefended "" || return 1
      ;;
    native)
      [ -f "$F/done/undefended.done" ] || not_ready "$SET 的 VAE-Trans 分母"
      for c in $NATIVE_CONDITIONS; do [ -f "$NS/measured_$c.done" ] || not_ready "原生 $SET 的 $c"; done
      tables=()
      for c in $NATIVE_CONDITIONS; do
        vae_trans_arm native "$c" "$NA/defenses/$c" || return 1
        vae_trans_retention native "$c" "$NS/results/displacement_$c.csv" || return 1
        tables+=("$NS/results/displacement_$c.csv")
      done
      semantic native "${tables[@]}" || return 1
      ;;
    aligned)
      [ -f "$F/done/undefended.done" ] || not_ready "$SET 的 VAE-Trans 分母"
      [ -f "$AS/chain.done" ] || not_ready "等失真 $SET"
      [ -f "$AS/chain_photoguard_encoder.done" ] || not_ready "等失真 $SET 的 photoguard_encoder"
      link_denominator aligned || return 1
      tables=()
      for c in $ALIGNED_CONDITIONS; do
        vae_trans_arm aligned "$c" "$AA/defenses/$c" || return 1
        vae_trans_retention aligned "$c" "$AS/results/displacement_$c.csv" || return 1
        tables+=("$AS/results/displacement_$c.csv")
      done
      semantic aligned "${tables[@]}" || return 1
      ;;
    color3d)
      [ -f "$F/done/undefended.done" ] || not_ready "$SET 的 VAE-Trans 分母"
      link_denominator color3d || return 1
      c=color3d_pexels
      # 本影像組的 color3d_pexels 防禦圖與原圖：指向 color trial 交付 PNG 的符號連結（只讀）
      mkdir -p "$V/color3d/defenses/$c" || return 1
      for im in $IMAGES; do
        for f in "${im}__${c}__def.png" "${im}__orig.png"; do
          [ -f "$CP/defenses/$c/$f" ] || { echo "[FATAL] 缺 $CP/defenses/$c/$f" >&2; return 1; }
          ln -sfn "$BASELINE_ROOT/$CP/defenses/$c/$f" "$V/color3d/defenses/$c/$f" || return 1
        done
      done
      "$PY" - "$CT/results/color3d_pexels/displacement.csv" "$R/displacement_$c.csv" $IMAGES <<'EOF' || return 1
import csv, sys
src, out, images = sys.argv[1], sys.argv[2], set(sys.argv[3:])
rows = [r for r in csv.DictReader(open(src, encoding="utf-8")) if r["image"] in images and r["scenario"] == "ip2p"]
if len(rows) != 4 * len(images) or {r["image"] for r in rows} != images:
    raise SystemExit(f"{src} 中本影像組的列不齊：{len(rows)} 列，應為 {4 * len(images)}")
for r in rows:
    for k in ("defended_png", "undefended_png"):
        r[k] = "../color/" + r[k]
with open(out, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n"); w.writeheader(); w.writerows(rows)
print(f"[SUBSET] {out}（{len(rows)} 列）")
EOF
      vae_trans_arm color3d "$c" "$V/color3d/defenses/$c" || return 1
      vae_trans_retention color3d "$c" "$R/displacement_$c.csv" || return 1
      semantic color3d "$R/displacement_$c.csv" || return 1
      ;;
    *) echo "[FATAL] 未知的分組 $GROUP" >&2; exit 2 ;;
  esac
}

chain_step "後續量測 $SET $GROUP（GPU $CUDA_VISIBLE_DEVICES）"
"$PY" -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 'CUDA 不可用')" || exit 1
run_group || { echo "[FATAL] $SET $GROUP 失敗" >&2; exit 1; }
touch "$F/done/$GROUP.done"
chain_step "FOLLOWUP DONE $SET $GROUP"

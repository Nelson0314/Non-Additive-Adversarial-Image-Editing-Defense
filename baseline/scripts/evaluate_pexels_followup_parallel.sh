#!/usr/bin/env bash
# 一個影像組的等失真臂後續量測（scripts/evaluate_pexels_followup.sh 的 aligned 分組）拆成可並行的工作。
# 產物、完成標記與讀數的位置、協定與該腳本相同（同一個 $F/done/ 標記，已完成的步驟不重做）；另加逐條件鎖
# runtime/pexels_portraits/followup/<根>/locks/aligned_<條件>，同一條件不會被兩個行程同時處理。
#
# 工作（第二個參數）：
#   condition <條件>         該條件的 VAE-Trans 淨化、淨化後編輯與位移保留表
#   semantic <i> <n>         語意與身分（未淨化與七道）的第 i 片（共 n 片），輸出 results/aligned/semantic_identity_<i>.csv
#   semantic_vae_trans <i> <n>  VAE-Trans 的語意與身分第 i 片；需九個條件的 VAE-Trans 編輯皆已完成
#   finalize <n>             合併 n 片為 semantic_identity.csv、semantic_identity_vae_trans.csv，寫該分組的完成標記
#                            （semantic_aligned.done、semantic_vae_trans_aligned.done、aligned.done）
# 前置條件不成立時以結束碼 4 結束。
#
# 用法（遠端 baseline 專案根；CUDA_VISIBLE_DEVICES 由呼叫端的租約設定，finalize 不需 GPU）：
#   ENV_FILE=~/env.sh bash scripts/evaluate_pexels_followup_parallel.sh <影像組> <工作> [<參數>...]
set -uo pipefail
[ $# -ge 2 ] || { echo "用法：$0 <影像組> <condition|semantic|semantic_vae_trans|finalize> [...]" >&2; exit 2; }
SET=$1 JOB=$2; shift 2
source "$(dirname "${BASH_SOURCE[0]}")/env.sh" || exit 1
cd "$BASELINE_ROOT" || exit 1
source scripts/run_evaluation_chain.sh
export TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

ALIGNED_CONDITIONS="mist dia_pt sifm dia_r dct_shield_y dct_shield dayn danp photoguard_encoder"
ENC=../color/trials/lut3d_defense_and_naturalness/artifacts/purification/vae_trans/model/encoder.pt
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
AS=runtime/pexels_portraits/aligned${root:+/$root}
AA=artifacts/pexels_portraits/aligned${root:+/$root}
V=artifacts/pexels_portraits/vae_trans${root:+/$root}
F=runtime/pexels_portraits/followup${root:+/$root}
R=$F/results/aligned
mkdir -p "$F/done" "$F/locks" "$R"
N_IMAGES=$(ls "$D"/man/*.png "$D"/woman/*.png 2>/dev/null | wc -l)
[ "$N_IMAGES" -gt 0 ] || { echo "[FATAL] $D 沒有影像" >&2; exit 1; }
not_ready() { chain_step "前置條件不成立：$*"; exit 4; }
need_gpu() {
  : "${CUDA_VISIBLE_DEVICES:?需要由租約設定 CUDA_VISIBLE_DEVICES}"
  "$PY" -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 'CUDA 不可用')" || exit 1
}
[ -f "$F/done/undefended.done" ] || not_ready "$SET 的 VAE-Trans 分母"
[ -f "$AS/chain.done" ] && [ -f "$AS/chain_photoguard_encoder.done" ] || not_ready "等失真 $SET"
[ -L "$V/aligned/purified_edits/undefended" ] || { mkdir -p "$V/aligned/purified_edits"; ln -s ../../native/purified_edits/undefended "$V/aligned/purified_edits/undefended" || exit 1; }

case $JOB in
  condition)
    c=$1
    case " $ALIGNED_CONDITIONS " in *" $c "*) ;; *) echo "[FATAL] 不是等失真條件：$c" >&2; exit 2 ;; esac
    lock=$F/locks/aligned_$c
    mkdir "$lock" 2>/dev/null || { echo "[FATAL] $c 已被鎖定：$(cat "$lock/owner" 2>/dev/null)" >&2; exit 1; }
    echo "$(hostname) card ${CUDA_VISIBLE_DEVICES:-} pid $$ $(date -Is)" > "$lock/owner"
    trap 'rm -rf "$lock"' EXIT
    need_gpu
    chain_step "後續量測 $SET aligned $c（GPU $CUDA_VISIBLE_DEVICES）"
    in=$V/inputs/aligned/$c/vae_trans/ip2p_${c}_vae_trans
    if [ ! -f "$F/done/vae_trans_inputs_aligned_$c.done" ]; then
      chain_step "VAE-Trans 淨化 aligned $c"
      "$PY" -m $CHAIN_CLI.apply_vae_trans --defenses-dir "$AA/defenses/$c" --condition "$c" --encoder "$ENC" \
          --output-root "$V/inputs/aligned" --fidelity-csv "$R/vae_trans_fidelity_$c.csv" || exit 1
      for f in "$in"/*__orig.png; do ln -f "$f" "${f%__orig.png}__vae_trans__def.png" || exit 1; done
      n=$(ls "$in"/*__vae_trans__def.png | wc -l)
      [ "$n" -eq "$N_IMAGES" ] || { echo "[FATAL] $in 有 $n 張，影像組為 $N_IMAGES 張" >&2; exit 1; }
      touch "$F/done/vae_trans_inputs_aligned_$c.done"
    fi
    if [ ! -f "$F/done/vae_trans_edits_aligned_$c.done" ]; then
      chain_step "VAE-Trans 淨化後編輯 aligned $c"
      chain_edit_arm "$D" "$in" "$V/aligned/purified_edits/$c/vae_trans" "_${c}_vae_trans" || exit 1
      touch "$F/done/vae_trans_edits_aligned_$c.done"
    fi
    if [ ! -f "$F/done/vae_trans_retention_aligned_$c.done" ]; then
      chain_step "VAE-Trans 位移保留 aligned $c"
      "$PY" -m $CHAIN_CLI.measure_purified_displacement --purified-edits-root "$V/aligned/purified_edits" \
          --displacement-csv "$AS/results/displacement_$c.csv" --data-root "$D" \
          --output-csv "$R/retention_vae_trans_$c.csv" --conditions "$c" || exit 1
      touch "$F/done/vae_trans_retention_aligned_$c.done"
    fi
    ;;
  semantic)
    i=$1 n=$2; need_gpu
    [ -f "$F/done/semantic_aligned.done" ] && { chain_step "semantic 已完成"; exit 0; }
    tables=(); for c in $ALIGNED_CONDITIONS; do tables+=("$AS/results/displacement_$c.csv"); done
    chain_step "語意與身分 $SET aligned（未淨化與七道）第 $i/$n 片"
    # shellcheck disable=SC2086
    "$PY" -m $CHAIN_CLI.measure_semantic_identity --displacement-csv "${tables[@]}" --data-root "$D" \
        --output-csv "$R/semantic_identity_$i.csv" --purifiers $PURIFIERS --device cuda --shard "$i" --shards "$n" || exit 1
    ;;
  semantic_vae_trans)
    i=$1 n=$2; need_gpu
    [ -f "$F/done/semantic_vae_trans_aligned.done" ] && { chain_step "semantic_vae_trans 已完成"; exit 0; }
    for c in $ALIGNED_CONDITIONS; do [ -f "$F/done/vae_trans_edits_aligned_$c.done" ] || not_ready "$c 的 VAE-Trans 編輯"; done
    plan=$R/vae_trans_edits.csv
    tables=(); for c in $ALIGNED_CONDITIONS; do tables+=("$AS/results/displacement_$c.csv"); done
    "$PY" - "$V" "$plan.$i.incoming" "${tables[@]}" <<'EOF' || exit 1
import csv, sys
from pathlib import Path
v, out, tables = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3:]
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
                     "png": png(v / "aligned", r["condition"], *key), "purifier": "vae_trans"})
with open(out, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n"); w.writeheader(); w.writerows(rows)
print(f"[PLAN] {out}（{len(rows)} 列）")
EOF
    mv "$plan.$i.incoming" "$plan.$i"
    chain_step "語意與身分 $SET aligned（VAE-Trans）第 $i/$n 片"
    "$PY" -m $CHAIN_CLI.measure_semantic_identity --edits-csv "$plan.$i" --scenario ip2p --data-root "$D" \
        --output-csv "$R/semantic_identity_vae_trans_$i.csv" --purifiers vae_trans --device cuda --shard "$i" --shards "$n" || exit 1
    ;;
  finalize)
    n=$1
    for c in $ALIGNED_CONDITIONS; do [ -f "$F/done/vae_trans_retention_aligned_$c.done" ] || not_ready "$c 的 VAE-Trans 位移保留"; done
    for kind in "" _vae_trans; do
      files=(); for i in $(seq 0 $((n - 1))); do
        [ -f "$R/semantic_identity${kind}_$i.csv" ] || not_ready "semantic_identity${kind}_$i.csv"
        files+=("$R/semantic_identity${kind}_$i.csv")
      done
      { head -n 1 "${files[0]}"; for f in "${files[@]}"; do tail -n +2 "$f"; done; } > "$R/semantic_identity$kind.csv.incoming" || exit 1
      mv "$R/semantic_identity$kind.csv.incoming" "$R/semantic_identity$kind.csv"
    done
    "$PY" - "$R" "$N_IMAGES" <<'EOF' || exit 1
import csv, sys
r, n = sys.argv[1], int(sys.argv[2])
plain = list(csv.DictReader(open(f"{r}/semantic_identity.csv", encoding="utf-8")))
vae = list(csv.DictReader(open(f"{r}/semantic_identity_vae_trans.csv", encoding="utf-8")))
want_plain, want_vae = 10 * n * 4 * 8, 10 * n * 4
keys = {(x["condition"], x["image"], x["prompt_index"], x["purifier"]) for x in plain}
if len(plain) != want_plain or len(keys) != want_plain or len(vae) != want_vae:
    raise SystemExit(f"合併後列數不符：未淨化與七道 {len(plain)}（唯一 {len(keys)}，應 {want_plain}），VAE-Trans {len(vae)}（應 {want_vae}）")
print(f"[MERGED] {len(plain)} + {len(vae)} 列")
EOF
    touch "$F/done/semantic_aligned.done" "$F/done/semantic_vae_trans_aligned.done" "$F/done/aligned.done"
    ;;
  *) echo "[FATAL] 未知的工作 $JOB" >&2; exit 2 ;;
esac
chain_step "FOLLOWUP PARALLEL DONE $SET $JOB $*"

#!/usr/bin/env bash
# 非 SD 系編輯器（scripts/run_nonsd_edits.py）對一張 Pexels 影像的跨編輯器全協定：六條件（undefended、等失真 photoguard_encoder、
# dct_shield、mist、BlurGuard、color3d_scale2p5_pexels）× {淨化前、七道、VAE-Trans} × 4 句 ＝ 216 次編輯；輸入為 ip2p 流程的產物，不重新求解：
#   undefended     原圖；七道 artifacts/pexels_portraits/<根>/purified_edits/undefended/；VAE-Trans vae_trans/<根>/inputs/native/undefended/
#   等失真三臂     artifacts/pexels_portraits/aligned/<根>/{defenses,purified_edits}/；VAE-Trans vae_trans/<根>/inputs/aligned/
#   color3d_scale2p5_pexels  color trial artifacts_pexels/{defenses,purified_edits}/；VAE-Trans vae_trans/<根>/inputs/color3d_scale2p5/
#   blurguard      第一批（runtime/pexels_portraits/blurguard/images.txt）在原生產物根 artifacts/pexels_portraits/<根>/…/blurguard/ 與
#                  vae_trans/<根>/inputs/blurguard/；第二批在 runtime/pexels_portraits/blurguard_rest/{artifacts,vae_trans}/<單位>/
# 輸出：產物 artifacts/pexels_portraits/<編輯器>/{edits,purified_edits}/；逐格表 runtime/pexels_portraits/<編輯器>/chunks/<影像>/；
#       完成標記 runtime/pexels_portraits/<編輯器>/done/<影像>{,__<條件>__<設定>}.done。版面同 FLUX 512，讀數沿用
#       scripts/evaluate_pexels_flux_512_readouts.sh（EDITOR_NAME＝<編輯器>）。缺輸入時失敗。
# VIBE 系以 ~/venvs/vibe（torch 2.6.0、diffusers 0.33.1、transformers 4.57.1、vibe）執行；klein 系以 $PY。
# 用法（遠端 baseline 專案根；CUDA_VISIBLE_DEVICES 由呼叫端的租約設定）：ENV_FILE=~/env.sh bash scripts/evaluate_pexels_nonsd_editor.sh <編輯器> <影像>
set -uo pipefail
[ $# -eq 2 ] || { echo "用法：$0 <編輯器> <影像>" >&2; exit 2; }
ED=$1 IMG=$2
source "$(dirname "${BASH_SOURCE[0]}")/env.sh" || exit 1
cd "$BASELINE_ROOT" || exit 1
: "${CUDA_VISIBLE_DEVICES:?需要由租約設定 CUDA_VISIBLE_DEVICES}"
case $ED in vibe|vibe_distilled) RUN=~/venvs/vibe/bin/python ;; klein|klein_base) RUN=$PY ;; *) echo "[FATAL] 未知的編輯器 $ED" >&2; exit 2 ;; esac
X=artifacts/pexels_portraits/$ED
S=runtime/pexels_portraits/$ED
R=$S/chunks/$IMG
mkdir -p "$S/done" "$R"
[ -f "$S/done/$IMG.done" ] && { echo "[STEP] $ED $IMG 已完成"; exit 0; }
"$PY" - "$IMG" "$R/manifest.csv" <<'EOF' || exit 1
import csv, json, sys
from pathlib import Path
img, out = sys.argv[1], Path(sys.argv[2])
OPS = ("blur1", "blur2", "crop_resize0.1", "jpeg30", "jpeg50", "jpeg80", "rotate15")
sets = json.load(open("data/pexels_portraits/image_sets.json", encoding="utf-8"))["sets"]
s = next(s for s in sets if img in s["images"])
root = s["root"] or ""
cls = img.split("_")[0]
NA = Path("artifacts/pexels_portraits") / root
AA = Path("artifacts/pexels_portraits/aligned") / root
VT = Path("artifacts/pexels_portraits/vae_trans") / root / "inputs"
CT = Path("../color/trials/lut3d_defense_and_naturalness/artifacts_pexels")
B1 = set(Path("runtime/pexels_portraits/blurguard/images.txt").read_text(encoding="utf-8").split())
if img in B1:
    bd, bp, bv = NA / "defenses", NA / "purified_edits", VT / "blurguard"
else:
    unit = next(p.stem for p in sorted(Path("runtime/pexels_portraits/blurguard_rest/units").glob("*.txt"))
                if img in p.read_text(encoding="utf-8").split())
    br = Path("runtime/pexels_portraits/blurguard_rest")
    bd, bp, bv = br / "artifacts" / unit / "defenses", br / "artifacts" / unit / "purified_edits", br / "vae_trans" / unit / "inputs" / "blurguard"
roots = {  # 條件 -> (防禦圖目錄, 七道根, VAE-Trans 輸入根)
    "undefended": (None, NA / "purified_edits", VT / "native"),
    "photoguard_encoder": (AA / "defenses", AA / "purified_edits", VT / "aligned"),
    "dct_shield": (AA / "defenses", AA / "purified_edits", VT / "aligned"),
    "blurguard": (bd, bp, bv),
    "mist": (AA / "defenses", AA / "purified_edits", VT / "aligned"),
    "color3d_scale2p5_pexels": (CT / "defenses", CT / "purified_edits", VT / "color3d_scale2p5"),
}
rows = []
for c, (d, p, v) in roots.items():
    none = Path("data/pexels_portraits") / cls / f"{img}.png" if d is None else d / c / f"{img}__{c}__def.png"
    rows.append((c, "none", none))
    rows.append((c, "vae_trans", v / c / "vae_trans" / f"ip2p_{c}_vae_trans" / f"{img}__orig.png"))
    rows += [(c, op, p / c / op / f"ip2p_{c}_{op}" / f"{img}__orig.png") for op in OPS]
missing = [str(x) for *_, x in rows if not x.is_file()]
if missing:
    raise SystemExit(f"{img} 缺 {len(missing)} 個輸入，例如 {missing[:3]}")
with open(out, "w", encoding="utf-8", newline="") as f:
    w = csv.writer(f, lineterminator="\n"); w.writerow(["arm", "purifier", "image", "input_png"])
    w.writerows((c, op, img, x.as_posix()) for c, op, x in rows)
print(f"[MANIFEST] {out}（{len(rows)} 組）")
EOF
echo "[STEP] $(date -Iseconds) $ED $IMG（GPU $CUDA_VISIBLE_DEVICES）"
HF_HOME=${HF_HOME:-/var/cache/huggingface} "$RUN" scripts/run_nonsd_edits.py --editor "$ED" --manifest "$R/manifest.csv" \
    --output-root "$X" --chunks-dir "$R" --done-dir "$S/done" || exit 1
touch "$S/done/$IMG.done"
echo "[STEP] $(date -Iseconds) NONSD DONE $ED $IMG"

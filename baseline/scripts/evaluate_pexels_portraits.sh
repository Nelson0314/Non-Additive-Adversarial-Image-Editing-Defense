#!/usr/bin/env bash
# Pexels 人像資料組（data/pexels_portraits）的 ip2p 評測：在一張卡上依序處理指定的影像組。
#
# 影像組定義於 data/pexels_portraits/image_sets.json（名稱為全域影像序的索引範圍，例如 images_0097_0106）。
# 每個影像組是一次完整的評測鏈（scripts/run_evaluation_chain.sh）：未防禦分母與各條件的防禦生成、ip2p 編輯、
# 七道淨化、淨化後編輯、量測與補充指標。影像組逐一完成，因此已完成的影像組對全部條件都有讀數。
#
# 每個影像組的位置（<根> 由 image_sets.json 的 root 欄決定；空字串表示頂層，即第一批 97 張的既有產物）：
#   資料子集   runtime/pexels_portraits/data/<影像組>/（指向 data/pexels_portraits 的符號連結）
#   產物       artifacts/pexels_portraits/<根>/
#   狀態       runtime/pexels_portraits/<根>/*.done
#   讀數       runtime/pexels_portraits/<根>/results/
#
# 條件預設為十個（不含 photoguard_c、photoguard_linf 與 color）；以環境變數 CONDITIONS 覆寫。
# 指令、種子、步數、淨化協定與主表相同（data/pexels_portraits/prompts.yaml 逐字沿用主表）。
#
# 用法（遠端；卡號由呼叫端以 measure_free_gpus.sh 與 nvidia-smi 的 compute app 清單確認後指定，
# 全局上限依 GPU_CAP 或租約目錄的既有紀錄）：
#   ENV_FILE=~/env.sh bash scripts/evaluate_pexels_portraits.sh <卡號> <影像組> [<影像組>...]
set -uo pipefail
card=${1:?需要卡號}; shift
[ $# -ge 1 ] || { echo "需要至少一個影像組" >&2; exit 2; }
source "$(dirname "${BASH_SOURCE[0]}")/env.sh" || exit 1
cd "$BASELINE_ROOT" || exit 1
source "$GPU_TOOLS/gpu_lease.sh"
source scripts/run_evaluation_chain.sh
gpu_policy_init || exit $?
lease_acquire "$card" pexels_portraits "$(gpu_global_cap)" || { echo "CARD $card NOT FREE OR CAP REACHED"; exit 3; }
trap 'lease_release "$card"' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
export CUDA_VISIBLE_DEVICES=$card TOKENIZERS_PARALLELISM=false \
       PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

DATA=data/pexels_portraits
CONDITIONS=${CONDITIONS:-"diffvax photoguard_encoder mist dia_pt sifm dia_r dct_shield_y dct_shield dayn danp"}

# 建立影像組的資料子集並輸出其 root 欄；影像組不存在或影像缺檔即失敗。
prepare_image_set() {
  "$PY" - "$DATA" "$1" <<'EOF'
import json, os, sys
from pathlib import Path
data, name = Path(sys.argv[1]).resolve(), sys.argv[2]
sets = {s["name"]: s for s in json.loads((data / "image_sets.json").read_text(encoding="utf-8"))["sets"]}
if name not in sets:
    raise SystemExit(f"image_sets.json 沒有影像組 {name}")
subset = Path("runtime/pexels_portraits/data") / name
for sub in ("man", "woman", "masks"):
    (subset / sub).mkdir(parents=True, exist_ok=True)
wanted = set()
for image in sets[name]["images"]:
    cls = image.rsplit("_", 1)[0]
    for sub in (cls, "masks"):
        source = data / sub / f"{image}.png"
        if not source.is_file():
            raise SystemExit(f"缺檔：{source}")
        link = subset / sub / f"{image}.png"
        wanted.add(link)
        if not link.is_symlink():
            os.symlink(source, link)
for sub in ("man", "woman", "masks"):
    extra = [p for p in (subset / sub).iterdir() if p not in wanted]
    if extra:
        raise SystemExit(f"{subset / sub} 有不屬於 {name} 的檔案：{[p.name for p in extra]}")
prompts = subset / "prompts.yaml"
if not prompts.is_symlink():
    os.symlink(data / "prompts.yaml", prompts)
print(sets[name]["root"])
EOF
}

for SET in "$@"; do
  root=$(prepare_image_set "$SET") || exit 1
  A=artifacts/pexels_portraits${root:+/$root}
  S=runtime/pexels_portraits${root:+/$root}
  chain_step "影像組 $SET（產物 $A）"
  # shellcheck disable=SC2086
  run_evaluation_chain "runtime/pexels_portraits/data/$SET" "$A" "$S" "$S/results" $CONDITIONS || exit 1
done
chain_step "ALL SETS DONE $*"

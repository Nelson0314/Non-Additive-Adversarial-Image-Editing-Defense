#!/usr/bin/env bash
# 量化交付批與帶限批的抗淨化。**只讀已存的防禦圖，不重訓。**
#
# 為什麼非跑這一段不可
# ────────────────────────────────────────────────────────────────────
# 那兩批的存在理由都在淨化之後，不在未淨化那一側：
#
#   - 量化交付要問的是「攻擊方再壓一次時擾動還在不在」。派工那一批只量到
#     `deliver_retention`（擾動撐過**我方自己**那一次壓縮的比例），那是機制
#     有沒有生效，**不是抗淨化**。
#   - 帶限參數化要問的是模糊與重取樣那三欄（現行 1%／1%／4%）。未淨化的
#     讀數對這個問題完全沒有資訊。
#
# 也就是說：**只跑派工不跑這一段，那兩批等於沒有回答任何問題。**
#
# 用同一支 `purify_identity.py`、同一組十二個算子、同一個空白地板，
# 所以結果與 `runs/ip2p_purify_identity`（補丁族 `free`／`lowproj`／`rand`）
# 可以並列——那正是要比的對象。
#
# 身分讀數為什麼逐臂寫一個檔
# ────────────────────────────────────────────────────────────────────
# 輸出目錄是 `{arm}_{cat}_{pur}`，三批合計 11 臂 × 3 類 × 12 算子 = **396 個**。
# `identity_probe.py` 只有可重複的 `--entry`，而 `docs/DEFECTS.md` 記著
# 「argv 太長時大型 C 擴充在載入期直接 segfault」。逐臂呼叫讓 argv 恆為
# 常數量級，同時與 `runs/ip2p_content_constraint` 的擺法一致
# （`scripts/readout_parallel.py` 會把 `identity*.csv` 全部收進來）。
#
# 用法：bash scripts/jpeg_band_purify_round.sh "<卡號…>" <來源批>
#       來源批取 deliver_jpeg／patch_res／voronoi
set -uo pipefail
ROOT=/nfs/home/nelson0314/WACV-s3
PY="$HOME/venvs/wacv/bin/python"
export PYTHONPATH="$ROOT" HF_HOME="$HOME/hf_cache" PYTHONIOENCODING=utf-8
export TOKENIZERS_PARALLELISM=false
export HF_HUB_CACHE="${HF_HUB_CACHE:-/var/cache/huggingface/hub}"
export HF_ASSETS_CACHE="${HF_ASSETS_CACHE:-/var/cache/huggingface/assets}"
cd "$ROOT" || { echo "錯誤：找不到 $ROOT" >&2; exit 2; }

DEVS=(${1:-1 2 3 4 5})
# 預設仍是五張（`CLAUDE.md`：多人共用，佔滿會讓別人排不進來）。
# 超過五張**必須使用者明確授意**，故做成要另外給 `MAX_DEVS` 才放行，
# 不改預設值——改預設會讓之後每一批都靜默佔六張。
MAX_DEVS="${MAX_DEVS:-5}"
[ ${#DEVS[@]} -gt "$MAX_DEVS" ] && DEVS=("${DEVS[@]:0:$MAX_DEVS}")
WHICH="${2:-deliver_jpeg}"

case "$WHICH" in
  deliver_jpeg)
    SRC=runs/ip2p_deliver_jpeg_patch
    OUT=runs/ip2p_deliver_jpeg_patch_purify
    PURIFY_ARMS="qd85 qd65 qd45" ;;
  patch_res)
    SRC=runs/ip2p_patch_res
    OUT=runs/ip2p_patch_res_purify
    PURIFY_ARMS="s08 s16 s24 s32 s16_rand" ;;
  voronoi)
    SRC=runs/ip2p_voronoi
    OUT=runs/ip2p_voronoi_purify
    PURIFY_ARMS="v024 v064 v064_rand" ;;
  *) echo "未知的來源批 $WHICH（可用：deliver_jpeg／patch_res／voronoi）" >&2; exit 2 ;;
esac
# 第三個引數可以只取一部分的臂。存在理由：來源批可能只跑完一半，而上面的
# 守門對缺臂是 exit 2（那是對的——不擋的話 glob 會空而安靜地什麼都不做）。
# 沒有這個覆寫時，「先量已經跑完的那幾臂」就得改腳本才做得到。
PURIFY_ARMS="${3:-$PURIFY_ARMS}"
LOG=runs/execution_logs/${WHICH}_purify.log
mkdir -p "$OUT" runs/execution_logs

CATS="clothing accessory background"

# 派工前先確認每一格的來源目錄都在。缺一個的話 `purify_identity.py` 會拿到
# 空的 glob 而**安靜地什麼都不做**，跑完的目錄數少一個但不報錯。
for arm in $PURIFY_ARMS; do
  for cat in $CATS; do
    n=$(find "$SRC" -maxdepth 1 -type d -name "${arm}_${cat}_task_*" 2>/dev/null | wc -l)
    [ "$n" -eq 0 ] && {
      echo "錯誤：$SRC 底下沒有 ${arm}_${cat}_task_*（來源批還沒跑完？）" >&2
      exit 2; }
  done
done

bash scripts/free_cards.sh --assert "${DEVS[*]}" || exit 3

SLOTS=$(( ${#DEVS[@]} * 2 ))
# 只數 python，不數外殼（`docs/DEFECTS.md`）。
running() {
  ps -u "$USER" -o comm=,cmd= | awk '$1 ~ /^python/ && /purify_identity\.py/' | wc -l
}

echo "=== [$(date '+%F %T')] $WHICH 抗淨化：$(echo $PURIFY_ARMS | wc -w) 臂 × 3 類 ===" | tee -a "$LOG"
i=0
for arm in $PURIFY_ARMS; do
  for cat in $CATS; do
    while [ "$(running)" -ge "$SLOTS" ]; do sleep 30; done
    dev=${DEVS[$(( i % ${#DEVS[@]} ))]}
    i=$(( i + 1 ))
    echo "[$WHICH-purify] $arm/$cat dev=$dev" | tee -a "$LOG"
    CUDA_VISIBLE_DEVICES="$dev" setsid nohup "$PY" scripts/purify_identity.py \
      --arm "$arm" --cells "$SRC/${arm}_${cat}_task_*" --out "$OUT" \
      --skip-existing > "$OUT/${arm}_${cat}.log" 2>&1 < /dev/null &
    sleep 5
  done
done

while [ "$(running)" -gt 0 ]; do sleep 60; done
echo "=== [$(date '+%F %T')] 淨化收工，目錄 $(ls -d "$OUT"/*/ 2>/dev/null | wc -l) 個 ===" | tee -a "$LOG"

for arm in $PURIFY_ARMS; do
  ENT=""
  for d in "$OUT"/${arm}_*/; do
    [ -d "$d" ] || continue
    ENT="$ENT --entry $(basename "$d")=$d"
  done
  [ -z "$ENT" ] && { echo "[$arm] 沒有輸出目錄，跳過身分讀數" | tee -a "$LOG"; continue; }
  # shellcheck disable=SC2086
  "$PY" scripts/identity_probe.py --out "$OUT/identity_${arm}.csv" $ENT 2>&1 | tee -a "$LOG"
done
echo "=== [$(date '+%F %T')] 身分讀數寫入 $OUT/identity_*.csv ===" | tee -a "$LOG"

#!/usr/bin/env bash
# 只跑未防禦編輯，用來把「指令確實被執行」的影像從 13 張補上來。
#
# 為什麼要這一批
# ────────────────────────────────────────────────────────────────────
# 主線 25 張裡只有 13 張的未防禦編輯真的執行了指令（`runs/obedience_audit/`），
# 其餘 12 張沒有攻擊可防。今晚所有的視覺結論都建立在那 13 張上，n 偏小。
#
# `data/omniedit150` 還有 125 張沒進過主線批次。本批只跑**未防禦編輯**——
# 不做任何防禦，故成本約是防禦批次的五分之一（實測 phase 一格 39 s，其中
# 編輯本身約 19 s）。跑完之後用服從率篩選，再決定哪些進擴充批次。
#
# 服從的判定仍以人眼為準；自動篩選只是把明顯沒動的先剔掉，減少要看的張數。
set -u
source ~/env.sh >/dev/null 2>&1
cd /nfs/home/nelson0314/WACV-s3
export PYTHONPATH=/nfs/home/nelson0314/WACV-s3 TOKENIZERS_PARALLELISM=false
PY=$HOME/venvs/wacv/bin/python
ROOT=runs/ip2p_obedience_expand
mkdir -p $ROOT
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a $ROOT/log.txt; }

LIST=$ROOT/rest_images.txt
[ -s "$LIST" ] || { log "$LIST 不存在或是空的，中止"; exit 1; }
# 影像清單常常是在 Windows 上產生再上傳的，會帶 CRLF。攤成一行時若只換掉
# LF，每個名字尾端會掛著一個 CR，`--images` 一張都對不上，而錯誤訊息是
# 「底下沒有符合 --images 的影像」——看起來像清單錯了，其實是行尾。
# 用八進位  而不是 $''：後者在跨語言產生腳本時很容易被多吃一層跳脫，
# 寫出真正的 CR 位元組，反而讓這支腳本自己的引號跨行。無條件跑，不先檢查。
tr -d '' < "$LIST" > "$LIST.tmp" && mv "$LIST.tmp" "$LIST"
N=$(wc -l < $LIST)
log "服從率篩選開始，$N 張，只跑未防禦編輯"

# 分成 8 片，一片一張卡。`--check-only` 只跑編輯不做防禦。
SHARD=$(( (N + 7) / 8 ))
for g in 0 1 2 3 4 5 6 7; do
  IMGS=$(sed -n "$((g * SHARD + 1)),$(((g + 1) * SHARD))p" $LIST | tr '\n' ' ')
  [ -n "$IMGS" ] || continue
  CUDA_VISIBLE_DEVICES=$g nohup $PY scripts/ip2p_run.py \
    --out "$ROOT/g$g" --data data/omniedit150 --images $IMGS --check-only \
    > "$ROOT/g$g.log" 2>&1 &
done
wait
for d in $ROOT/g*/; do
  [ -d "$d" ] && log "$(basename $d)：$(( $(wc -l < $d/check.csv 2>/dev/null || echo 1) - 1 )) 列"
done
log "服從率篩選完成"

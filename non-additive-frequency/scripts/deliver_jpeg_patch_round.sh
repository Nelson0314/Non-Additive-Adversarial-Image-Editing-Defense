#!/usr/bin/env bash
# 量化交付接到補丁載體上：QD 掃描。**只多 `--deliver-jpeg`，其餘與 `free` 臂
# 逐項相同**，故 `free`（不自壓）就是這條掃描的第一個點，不必重跑。
#
# 為什麼是這一批
# ────────────────────────────────────────────────────────────────────
# `docs/DIRECTION.md` §6.1b 的處境：補丁載體有效（+0.538、14/25）但
# **JPEG 75 只剩 12%、JPEG 60 剩 9%**。
#
# 相位族已經量到一個能改變那一欄的機制，而且它與已否決的 `--purify-aware`
# 只差一件事——**交付什麼**（`runs/ip2p_deliver_jpeg/README.md`）：
#
#                       DISTS    未淨化   blur    jpeg75   jpeg30   crop
#     量化交付 QD.85    0.1908   0.617   0.188    0.493    0.238   0.110
#     不自壓（主線）    0.1447   0.671   0.118    0.362    0.146   0.091
#
# 人眼判定的差距更大：jpeg75 之後擋下 5/13 對 0/13。代價也量過：**等失真下
# 未淨化位移 −21%、擋下 11→3**，這是取捨不是免費的改良。
#
# 補丁族此前跑不了這個旗標——`ip2p_run.py` 的守門把它擋在
# `PHASE_CONDS + DCT_NONADD_CONDS` 之內，而那條守門寫在補丁族加進來之前。
# 現在補丁族在 `DELIVER_JPEG_CONDS` 裡（`tests/test_deliver_jpeg.py` 釘住）。
#
# QD 往哪個方向掃
# ────────────────────────────────────────────────────────────────────
# 兩條證據都指向**訓練壓得比攻擊狠**：
#   - CVIU 2025 那篇抗 JPEG 浮水印訓練用 q=35、評測 Q=75（SURVEY_FREQUENCY §1.13）。
#   - 本專案自己的相位族：QD 0.85→0.45 讓 jpeg90 掉 24%、jpeg75 掉 35%，
#     而 **jpeg30 漲 68%**。
# 現行的 0.85 高於多數攻擊品質，方向是反的。故三個臂由高走低。
#
# 一個必須寫在報表上的代價
# ────────────────────────────────────────────────────────────────────
# JPEG 往返壓的是**整張圖**，含受保護的臉。於是「受保護主體逐位元不動」在這
# 個組合下變成「受保護主體只被壓到 QD」。`identity_probe` 的 `id_orig` 欄
# 會把那個代價量出來——它是攻擊在**原圖**上的身分讀數，與防禦無關，若 QD
# 讓它明顯下移，那就是壓縮本身動到臉了。
#
# 用法：bash scripts/deliver_jpeg_patch_round.sh "<卡號…>" [smoke]
set -uo pipefail
ROOT=/nfs/home/nelson0314/WACV-s3
PY="$HOME/venvs/wacv/bin/python"
export PYTHONPATH="$ROOT" HF_HOME="$HOME/hf_cache" PYTHONIOENCODING=utf-8
export TOKENIZERS_PARALLELISM=false
export HF_HUB_CACHE="${HF_HUB_CACHE:-/var/cache/huggingface/hub}"
export HF_ASSETS_CACHE="${HF_ASSETS_CACHE:-/var/cache/huggingface/assets}"
cd "$ROOT" || { echo "錯誤：找不到 $ROOT" >&2; exit 2; }

DEVS=(${1:-1 2 3 4 5})
MODE="${2:-full}"
[ ${#DEVS[@]} -gt 5 ] && DEVS=("${DEVS[@]:0:5}")

OUT=runs/ip2p_deliver_jpeg_patch
LOG=runs/execution_logs/deliver_jpeg_patch.log
mkdir -p "$OUT" runs/execution_logs

CAT_TRAIN=clothing
CATS_REPLAY="accessory background"

arm_flags() {  # $1=臂 → "COND QD"
  case "$1" in
    qd85) echo "patch 0.85" ;;
    qd65) echo "patch 0.65" ;;
    qd45) echo "patch 0.45" ;;
    *) return 1 ;;
  esac
}

NIMG="${NIMG:-6}"
IMGS=$(find runs/ip2p_face_defence -maxdepth 1 -type d -name 'clothing_plain_task_*' \
       2>/dev/null | sed 's|.*/clothing_plain_||' | sort | head -n "$NIMG")
[ -z "$IMGS" ] && { echo "錯誤：推導不出影像清單" >&2; exit 2; }

COMMON="--data data/omniedit150 --attack-prompts data/attack_prompts.yaml \
--subject-source face --loss image_guidance --ig-zt diffuse_src \
--patch-carrier clothes --carrier-refine 4 --carrier-erode 3 --carrier-feather 8 \
--patch-placement complement --radius 0.05 \
--eval-every 200 --eval-draws 8 --patience 10 --min-delta 0.0002"

if [ "$MODE" = "smoke" ]; then
  IMG=$(echo $IMGS | awk '{print $1}')
  read -r COND QD <<< "$(arm_flags qd65)"
  echo "=== [$(date '+%F %T')] 燒測 qd65（20 步）===" | tee -a "$LOG"
  bash scripts/free_cards.sh --assert "${DEVS[0]}" || exit 3
  # shellcheck disable=SC2086
  CUDA_VISIBLE_DEVICES="${DEVS[0]}" "$PY" scripts/ip2p_run.py \
    --out "$OUT/_smoke" --images "$IMG" --attack-category "$CAT_TRAIN" \
    --conditions "$COND" --deliver-jpeg "$QD" --steps 20 $COMMON 2>&1 \
    | tee -a "$LOG" | tail -12
  rc=${PIPESTATUS[0]}
  [ "$rc" -ne 0 ] && { echo "燒測失敗（rc=$rc）" | tee -a "$LOG"; exit 4; }
  [ -f "$OUT/_smoke/results.csv" ] || {
    echo "燒測沒有產出 results.csv" | tee -a "$LOG"; exit 4; }
  # 這一批的存在理由就是那一欄；沒寫出來的話整批是白跑的。
  head -1 "$OUT/_smoke/results.csv" | tr ',' '\n' | grep -qx deliver_retention || {
    echo "燒測的 results.csv 沒有 deliver_retention 欄" | tee -a "$LOG"; exit 4; }
  echo "✓ 燒測通過" | tee -a "$LOG"
  exit 0
fi

ARMS="${3:-qd85 qd65 qd45}"
bash scripts/free_cards.sh --assert "${DEVS[*]}" || exit 3

SLOTS=$(( ${#DEVS[@]} * 2 ))
running() {
  ps -u "$USER" -o comm=,cmd= | awk '$1 ~ /^python/ && /ip2p_run\.py/' | wc -l
}

echo "=== [$(date '+%F %T')] 量化交付 × 補丁：$(echo $ARMS | wc -w) 臂 × $(echo $IMGS | wc -w) 張 ===" | tee -a "$LOG"
i=0
for arm in $ARMS; do
  read -r COND QD <<< "$(arm_flags "$arm")"
  for img in $IMGS; do
    while [ "$(running)" -ge "$SLOTS" ]; do sleep 45; done
    dev=${DEVS[$(( i % ${#DEVS[@]} ))]}
    i=$(( i + 1 ))
    TRAIN_OUT="$OUT/${arm}_${CAT_TRAIN}_$img"
    echo "[qd] $arm QD=$QD $img dev=$dev" | tee -a "$LOG"
    CUDA_VISIBLE_DEVICES="$dev" setsid nohup bash -c "
      set -e
      '$PY' scripts/ip2p_run.py --out '$TRAIN_OUT' --images '$img' \
        --attack-category $CAT_TRAIN --conditions $COND --deliver-jpeg $QD \
        --steps 6000 --step-size 0.01 --save-weights $COMMON
      for cat in $CATS_REPLAY; do
        '$PY' scripts/ip2p_run.py --out '$OUT/${arm}_'\$cat'_$img' --images '$img' \
          --attack-category \$cat --conditions $COND --deliver-jpeg $QD \
          --steps 0 --resume-weights '$TRAIN_OUT' $COMMON
      done
    " > "$OUT/${arm}_$img.log" 2>&1 < /dev/null &
    sleep 4
  done
done

while [ "$(running)" -gt 0 ]; do sleep 60; done
echo "=== [$(date '+%F %T')] 收工，results $(find $OUT -name results.csv | wc -l) 份 ===" | tee -a "$LOG"

ENT=""
for arm in $ARMS; do
  for d in "$OUT"/${arm}_*/; do
    [ -d "$d" ] && [ -f "$d/results.csv" ] || continue
    ENT="$ENT --entry $(basename "$d")=$d"
  done
done
# shellcheck disable=SC2086
"$PY" scripts/identity_probe.py --out "$OUT/identity.csv" $ENT 2>&1 | tee -a "$LOG"
echo "=== [$(date '+%F %T')] 身分讀數寫入 $OUT/identity.csv ===" | tee -a "$LOG"

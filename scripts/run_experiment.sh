#!/usr/bin/env bash
# 雲端一鍵實驗 runner：全程 log 落檔 + 結束（含崩潰/被關機）自動 push。
#
# 用途：實驗丟下去跑完機器會自動關機、看不到終端機，故：
#   1. 全部 stdout/stderr tee 到 lab/<時間戳>_<label>/run.log
#   2. 關鍵產出（summary.md/results.csv/*.png/config/env）複製進同資料夾
#   3. trap EXIT：不論成功、失敗、被 kill，都把 lab/ commit 並 push 到 origin
#
# 命名：lab/<YYYYMMDD_HHMMSS>_<label>/  —— label 由第一個參數指定（如 quick、full）。
#
# 用法：bash scripts/run_experiment.sh <label>
# 可用環境變數覆寫（皆有預設）：
#   MODEL METHODS MAXIMG NSEEDS STRENGTH PURIFY RUN_STAGE0 WITH_FID
#
# 前置（雲端一次性，讓 headless push 可用）：見 RUN_CLOUD.md 附錄 A。
set -uo pipefail

LABEL="${1:-run}"
MODEL="${MODEL:-CompVis/stable-diffusion-v1-4}"
METHODS="${METHODS:-pg_enc,advdiff,apa,hybrid}"
MAXIMG="${MAXIMG:-6}"
NSEEDS="${NSEEDS:-3}"
STRENGTH="${STRENGTH:-0.8}"
PURIFY="${PURIFY:-jpeg,blur,crop_resize,advclean_bf,advclean_bfgf}"
RUN_STAGE0="${RUN_STAGE0:-0}"      # 1=先跑 stage0 校準
WITH_FID="${WITH_FID:-0}"          # 1=stage1 計 FID（n 小時 FID 意義有限，預設關）

TS="$(date +%Y%m%d_%H%M%S)"
LAB="lab/${TS}_${LABEL}"
mkdir -p "$LAB"
LOG="$LAB/run.log"
RUNDIRS="$LAB/rundirs.txt"
: > "$RUNDIRS"

# 結束時（EXIT trap）：複製關鍵產出 + commit + push。冪等、永不中斷。
finish() {
  local code=$?
  {
    echo ""
    echo "=== FINISHED exit=$code @ $(date +%Y-%m-%d\ %H:%M:%S) ==="
  } >> "$LOG" 2>&1
  while read -r d; do
    [ -d "$d" ] || continue
    local sub="$LAB/$(basename "$(dirname "$d")")__$(basename "$d")"
    mkdir -p "$sub"
    cp "$d"/summary.md "$d"/results.csv "$d"/calibration.csv \
       "$d"/*.png "$d"/config_snapshot.yaml "$d"/env.json "$sub"/ 2>/dev/null
  done < "$RUNDIRS"
  git add -A "$LAB" 2>/dev/null
  git -c user.email=exp@local -c user.name=exp-runner \
      commit -m "exp($LABEL) $TS exit=$code" >/dev/null 2>&1 || true
  git push origin main >/dev/null 2>&1 || true
}
trap finish EXIT

# 最新 run 目錄（stage 腳本以時間戳建目錄；跑完取最新者記入 rundirs.txt）
latest() { ls -dt experiments/"$1"/*/ 2>/dev/null | head -1 | sed 's:/*$::'; }

{
  echo "=== run_experiment: label=$LABEL @ $TS ==="
  echo "model=$MODEL methods=$METHODS max-images=$MAXIMG n-seeds=$NSEEDS strength=$STRENGTH"
  echo "purify=$PURIFY run_stage0=$RUN_STAGE0 with_fid=$WITH_FID"
  echo "commit=$(git rev-parse --short HEAD)"
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "no nvidia-smi"
  echo ""

  if [ "$RUN_STAGE0" = "1" ]; then
    echo "=== STAGE0 @ $(date +%H:%M:%S) ==="
    python -u scripts/stage0_calibrate.py --model "$MODEL" --max-images "$MAXIMG" --skip-pg-diff
    latest stage0 >> "$RUNDIRS"
  fi

  echo "=== STAGE1 @ $(date +%H:%M:%S) ==="
  FID_FLAG="--no-fid"; [ "$WITH_FID" = "1" ] && FID_FLAG=""
  python -u scripts/stage1_clean.py \
      --protect-model "$MODEL" --model "$MODEL" \
      --methods "$METHODS" --edit-methods sdedit \
      --max-images "$MAXIMG" --n-seeds "$NSEEDS" --sdedit-strength "$STRENGTH" \
      --with-clip $FID_FLAG
  S1="$(latest stage1)"
  echo "$S1" >> "$RUNDIRS"

  echo "=== STAGE2 @ $(date +%H:%M:%S) ==="
  python -u scripts/stage2_purify.py \
      --stage1-dir "$S1" --purify-methods "$PURIFY"
  latest stage2 >> "$RUNDIRS"

  echo "=== ALL STAGES DONE @ $(date +%H:%M:%S) ==="
} 2>&1 | tee -a "$LOG"

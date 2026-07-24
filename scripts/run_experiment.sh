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
STAGE0_ONLY="${STAGE0_ONLY:-0}"    # 1=只跑 stage0（Phase A 公平性閘門），跑完 push 即停
STAGE0_SKIP_PGDIFF="${STAGE0_SKIP_PGDIFF:-0}"  # 1=stage0 省略 pg_diff 基準（公平性表會缺 pg_diff）
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
    cp "$d"/summary.md "$d"/results.csv "$d"/calibration.csv "$d"/fairness.csv \
       "$d"/report_v2_tables.csv "$d"/RESULTS_v2.md "$d"/*.png \
       "$d"/config_snapshot.yaml "$d"/env.json "$sub"/ 2>/dev/null
  done < "$RUNDIRS"
  git add -A "$LAB" 2>/dev/null
  # stage0 會改寫 configs（校準值）；一併提交，確保機器關機後 Phase B 仍取得
  if [ "$RUN_STAGE0" = "1" ] || [ "$STAGE0_ONLY" = "1" ]; then
    git add configs/nonadditive.yaml configs/nonadditive_calibrated.yaml 2>/dev/null
  fi
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

  if [ "$RUN_STAGE0" = "1" ] || [ "$STAGE0_ONLY" = "1" ]; then
    echo "=== STAGE0 @ $(date +%H:%M:%S) ==="
    S0_FLAGS=""; [ "$STAGE0_SKIP_PGDIFF" = "1" ] && S0_FLAGS="--skip-pg-diff"
    python -u scripts/stage0_calibrate.py --model "$MODEL" --max-images "$MAXIMG" $S0_FLAGS
    latest stage0 >> "$RUNDIRS"
  fi

  if [ "$STAGE0_ONLY" = "1" ]; then
    echo "=== STAGE0-ONLY (Phase A 公平性閘門) DONE @ $(date +%H:%M:%S) ==="
    echo "檢視 lab 內 stage0 fairness.csv／summary.md 之公平性表；若公平再跑 Phase B。"
  else
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
    S2="$(latest stage2)"
    echo "$S2" >> "$RUNDIRS"

    echo "=== REPORT_V2 @ $(date +%H:%M:%S) ==="
    S0="$(latest stage0)"
    if [ -n "$S0" ]; then
      python -u scripts/report_v2.py --stage1-dir "$S1" --stage2-dir "$S2" \
          --stage0-dir "$S0" --out "$LAB/RESULTS_v2.md"
    else
      python -u scripts/report_v2.py --stage1-dir "$S1" --stage2-dir "$S2" \
          --out "$LAB/RESULTS_v2.md"
    fi

    echo "=== ALL STAGES DONE @ $(date +%H:%M:%S) ==="
  fi
} 2>&1 | tee -a "$LOG"

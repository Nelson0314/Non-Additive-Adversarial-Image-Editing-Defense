#!/usr/bin/env bash
# 逐影像分片：用多張卡把段 1–3 平行跑完，**不改任何 Python 程式**。
#
#   bash scripts/shard.sh calib   <影像1> <影像2> ...     # 段 0，一次，單卡
#   bash scripts/shard.sh fanout  <影像1> <影像2> ...     # 段 1–3，每張圖一卡
#   bash scripts/shard.sh watch                           # 看各分片進度
#   bash scripts/shard.sh merge   <影像1> <影像2> ...     # 合併分片並跑段 4
#
# ---------------------------------------------------------------------------
# 為什麼這樣可行（三個前提，改動任一個之前先確認它們還成立）
#
# 1. `ProgressWriter._acquire()` 的鎖是**逐批次目錄**的，不是全域的。
#    不同批次目錄＝不同寫入者，完全合法。該鎖存在的理由是「同一批資料
#    不可有兩個寫入者」，分片後每個寫入者動的是不相交的資料。
#
# 2. 校準表可跨分片共用：`Calibration.REQUIRED_CONTEXT` 是
#    (model, resolution, guidance, steps, gpu, precision)，**不含 image**。
#    `calibrate_lr` 本來就只探測 `next(iter(res.images.values()))` 一張圖，
#    也就是說學習率在設計上與影像無關。故段 0 只跑一次，把
#    `calib/calibration.json` 複製進每個分片目錄即可。
#
#    **這一點是整個作法的關鍵。** 若改成每個分片各跑自己的段 0，各分片會
#    得到不同的學習率，而那是一個無法歸因的變因——跨影像的比較就毀了。
#
# 3. 合併無衝突：產物在 `batch_dir/<條件>/<影像>/`，逐格紀錄的檔名是
#    `<段>__<條件>__<影像>[__tau…].json`，兩者都帶影像。`run_report` 只讀
#    `_cells/*.json` 且只寫相對路徑的 `<img src>`，故把各分片的檔案疊進
#    同一個目錄再跑段 4，得到的就是完整的 compare.html。
#
# ---------------------------------------------------------------------------
# 分片的粒度就是影像數：N 張圖用 N 張卡，牆鐘時間 ≈ 總時數 / N。
# 影像是唯一有 CLI 入口的軸，條件與 τ 都沒有，故不要試圖再細分。
set -euo pipefail

MODE=${1:?用法見檔頭}
shift || true

# shellcheck disable=SC1090
source "$HOME/env.sh"

BATCH=${BATCH:-b1}
GPU_TAG=RTX-3090
PRECISION=bf16
COMMON="--gpu-tag $GPU_TAG --precision $PRECISION --mist-target data/targets/MIST.png"
RUNS=$HOME/WACV/runs

shard_dir() { echo "$RUNS/${BATCH}_$1"; }

# 空閒的卡，記憶體用量由低到高。排除他人正在用的。
free_gpus() {
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
    | awk -F', ' '$2 < 1000 {print $1}'
}

case "$MODE" in

calib)
  IMAGES=("$@")
  [ ${#IMAGES[@]} -gt 0 ] || { echo "要給影像 id" >&2; exit 1; }
  GPU=$(free_gpus | head -1)
  echo "段 0 在 GPU $GPU 上跑，影像：${IMAGES[*]}"
  tmux new-session -d -s "wacv-calib" \
    "cd $HOME/WACV && PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=$GPU \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
$PY scripts/run_stage.py calib --batch $BATCH $COMMON --images ${IMAGES[*]} \
> $RUNS/$BATCH/calib.log 2>&1; echo \"[exit \$?]\" >> $RUNS/$BATCH/calib.log"
  echo "log: $RUNS/$BATCH/calib.log"
  ;;

fanout)
  IMAGES=("$@")
  [ -f "$RUNS/$BATCH/calib/calibration.json" ] || {
    echo "找不到 $RUNS/$BATCH/calib/calibration.json——段 0 還沒跑完" >&2; exit 1; }
  mapfile -t GPUS < <(free_gpus)
  [ ${#GPUS[@]} -ge ${#IMAGES[@]} ] || {
    echo "空閒卡 ${#GPUS[@]} 張，少於影像 ${#IMAGES[@]} 張" >&2; exit 1; }

  for i in "${!IMAGES[@]}"; do
    IMG=${IMAGES[$i]}; GPU=${GPUS[$i]}; D=$(shard_dir "$IMG")
    mkdir -p "$D"
    # 共用同一份校準表：這是不可省的一步，理由見檔頭第 2 點
    cp -r "$RUNS/$BATCH/calib" "$D/"
    LOG="$D/run.log"
    echo "分片 $IMG → GPU $GPU   $LOG"
    tmux new-session -d -s "wacv-$IMG" \
      "cd $HOME/WACV && export PYTHONIOENCODING=utf-8 \
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; \
for S in train rayscale eval; do \
  echo \"===== \$S =====\"; \
  $PY scripts/run_stage.py \$S --batch ${BATCH}_$IMG $COMMON --images $IMG || break; \
done > $LOG 2>&1; echo \"[exit \$?]\" >> $LOG"
  done
  sleep 3; tmux ls
  ;;

watch)
  for D in "$RUNS/${BATCH}_"*; do
    [ -d "$D" ] || continue
    printf '%-28s ' "$(basename "$D")"
    $PY scripts/dashboard.py "$D" --json 2>/dev/null \
      | $PY -c 'import json,sys
d=json.load(sys.stdin)["summary"]
print(f"done={d[\"done\"]:5d} failed={d[\"failed\"]:3d} skipped={d[\"skipped\"]:4d} / {d[\"total\"]:5d}")' \
      || echo "(尚無進度)"
  done
  ;;

merge)
  IMAGES=("$@")
  OUT=$RUNS/${BATCH}_merged
  rm -rf "$OUT"; mkdir -p "$OUT/_cells"
  cp -r "$RUNS/$BATCH/calib" "$OUT/"
  for IMG in "${IMAGES[@]}"; do
    D=$(shard_dir "$IMG")
    cp "$D/_cells/"*.json "$OUT/_cells/"
    # 產物樹：<條件>/<影像>/…，逐影像互斥故直接疊上去
    for C in "$D"/*/; do
      B=$(basename "$C")
      case "$B" in _cells|calib) continue;; esac
      mkdir -p "$OUT/$B"; cp -r "$C"/* "$OUT/$B/" 2>/dev/null || true
    done
  done
  echo "合併完成：$(ls "$OUT/_cells" | wc -l) 格"
  PYTHONIOENCODING=utf-8 $PY scripts/run_stage.py report \
    --batch "${BATCH}_merged" $COMMON
  echo "compare.html → $OUT/compare.html"
  ;;

*) echo "未知模式 $MODE" >&2; exit 1;;
esac

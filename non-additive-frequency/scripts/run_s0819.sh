#!/usr/bin/env bash
# 2026-08-19 的批次。DEC-026 的三個對照組 ＋ DEC-027 的針對淨化最佳化。
#
# 這份腳本是在**校內網路中斷期間**寫好的（03:20 起連不上遠端），目的是網路一
# 恢復就能一次啟動，不用臨時拼指令。每一段都可以單獨跑。
#
# 用法： bash scripts/run_s0819.sh <stage> <gpu> [strength]
#   stage ∈ pull | advdrop | advdrop_al | blurguard | blurguard_al
#           | diffguard | diffguard_al | pa_jpeg | ret | merge
#   strength 預設 0.7（與 runs/s0817 的主線一致，見 DEC-024）
#
# 前置：
#   * `aret_*.csv`（預算對齊版 DCT-Shield 的抗淨化）應該已經跑完並躺在
#     runs/freqret/。先 `bash scripts/run_s0819.sh pull` 確認。
#   * BlurGuard 需要 SAM：`pip install segment-anything` ＋ 檢查點。
#     沒有時 blurguard 那兩段會自己跳過並印出原因，**不會**改用替代分割。
#   * `DIFFPURE_CKPT` 必須設，否則 gridpure 會被標成相依不齊而靜默跳過
#     （2026-08-19 踩過這個坑）。
#
# 分片：七張影像分四張卡，與 s0817 同一套分法，方便逐圖對照。

set -eu
source "$HOME/env.sh" >/dev/null 2>&1 || true
cd /nfs/home/nelson0314/WACV-s3
export PYTHONPATH=/nfs/home/nelson0314/WACV-s3
export HF_HOME=$HOME/hf_cache
export DIFFPURE_CKPT=$HOME/thirdparty/diffpure/256x256_diffusion_uncond.pt
PY=$HOME/venvs/wacv/bin/python
DATA=data/set0817
ROOT=runs/s0819
STAGE=$1
export CUDA_VISIBLE_DEVICES=${2:-0}
STRENGTH=${3:-0.7}
G=$CUDA_VISIBLE_DEVICES
SAM_CKPT=${SAM_CKPT:-$HOME/thirdparty/sam/sam_vit_h_4b8939.pth}
mkdir -p $ROOT

# 與 s0817 相同的分片
case $G in
  0) IMGS="person_a_00 person_b_00" ;;
  1) IMGS="person_c_00 cat_00" ;;
  2) IMGS="shiba_00 raccoon_00" ;;
  3) IMGS="parrot_00" ;;
  *) IMGS="person_a_00 person_b_00 person_c_00 cat_00 shiba_00 raccoon_00 parrot_00" ;;
esac

# DEC-025 指定的五個淨化算子。identity 是 retention 的分母，不可拿掉。
PUR="identity blur1 crop_resize0.1 jpeg75 gridpure"

sam_arg() {
  if [ -f "$SAM_CKPT" ]; then echo "--sam-ckpt $SAM_CKPT"; else echo ""; fi
}

case $STAGE in
  pull)
    echo "== 中斷期間應已跑完的東西 =="
    ls -la runs/freqret/aret_*.csv 2>/dev/null || echo "  aret_*.csv 不存在"
    ls -d runs/dctshield/al* 2>/dev/null || echo "  runs/dctshield/al* 不存在"
    echo "== SAM =="
    if [ -f "$SAM_CKPT" ]; then echo "  檢查點在 $SAM_CKPT"; else
      echo "  缺 SAM 檢查點（$SAM_CKPT）——blurguard 會被跳過"; fi
    $PY -c "import segment_anything" 2>/dev/null \
      && echo "  segment_anything 已安裝" || echo "  segment_anything 未安裝"
    ;;

  # ---- 原生設定 ----
  advdrop)
    $PY scripts/freq_baselines_run.py --out $ROOT/g$G --data $DATA \
      --conditions advdrop --mode paper --edit-strength $STRENGTH --images $IMGS ;;

  blurguard)
    $PY scripts/freq_baselines_run.py --out $ROOT/g$G --data $DATA \
      --conditions blurguard --mode paper --edit-strength $STRENGTH \
      $(sam_arg) --images $IMGS ;;

  # DiffusionGuard：唯一明確為抗淨化設計的編輯防護。**這是 img2img 的移植，
  # 不是原文的重現**（原文是 inpainting，遮罩增強完全拿掉），見模組 docstring。
  # 每步一次 UNet 前向加反向，800 步，比其餘條件貴一個數量級。
  diffguard)
    $PY scripts/freq_baselines_run.py --out $ROOT/g$G --data $DATA \n      --conditions diffusionguard --mode paper \n      --edit-strength $STRENGTH --images $IMGS ;;

  # ---- 預算對齊到相位臂的 DISTS 0.0349 ----
  advdrop_al)
    $PY scripts/freq_baselines_run.py --out $ROOT/al$G --data $DATA \
      --conditions advdrop --mode aligned --budget 0.0349 \
      --edit-strength $STRENGTH --images $IMGS ;;

  blurguard_al)
    $PY scripts/freq_baselines_run.py --out $ROOT/al$G --data $DATA \
      --conditions blurguard --mode aligned --budget 0.0349 \
      --edit-strength $STRENGTH $(sam_arg) --images $IMGS ;;

  diffguard_al)
    $PY scripts/freq_baselines_run.py --out $ROOT/al$G --data $DATA \n      --conditions diffusionguard --mode aligned --budget 0.0349 \n      --edit-strength $STRENGTH --images $IMGS ;;

  # ---- DEC-027：把可微分 JPEG 放進最佳化迴圈 ----
  # 只跑相位臂與加性對照——這一段要回答的是「針對淨化最佳化能買到多少」，
  # 不是再多一個 baseline。
  pa_jpeg)
    $PY scripts/phase_ablation.py --out $ROOT/pa$G --data $DATA \
      --conditions add phase --human-threshold --purify-aware jpeg \
      --edit-strength $STRENGTH --images $IMGS ;;

  # ---- 抗淨化 ----
  ret)
    for D in $ROOT/g$G $ROOT/al$G $ROOT/pa$G; do
      [ -f "$D/results.csv" ] || continue
      $PY scripts/phase_retention.py --run $D --data $DATA \
        --edit-strength $STRENGTH --seeds 3 --purifiers $PUR \
        --out runs/freqret/s0819_$(basename $D).csv --images $IMGS
    done ;;

  merge)
    $PY scripts/merge_runs.py --glob "$ROOT/g*" --out $ROOT/merged
    $PY scripts/night_report.py ;;

  *) echo "未知的 stage: $STAGE"; exit 1 ;;
esac

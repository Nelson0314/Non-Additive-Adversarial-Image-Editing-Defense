#!/usr/bin/env bash
# 2026-08-20 的批次：DCT-Shield 的工作點對齊掃描（DEC-029）。
#
# 目的只有一個——**讓 DCT-Shield 與紋理重相位變成可比較的對象**。兩者的強度
# 參數不同單位（ε 是量化階、θ 是相位半徑），論文的 baseline 又走 L∞=16/255
# 而本專案走原生預算，沒有任何一個軸天然對齊。作法與四種文獻先例的查證見
# docs/reference/BASELINE_ALIGNMENT.md。
#
# 這份腳本在**校內網路中斷期間**寫好（2026-08-19 03:20 起連不上遠端），
# 網路一恢復即可啟動。每一段可單獨跑。
#
# 用法： bash scripts/run_s0820.sh <stage> <gpu>
#   stage ∈ sweep_dct | sweep_phase | curve | frechet | ret_anchor | merge
#
# 前置：
#   * **資料集必須先擴到 150 張**（見 §資料集），否則 FID 這一欄永遠是空的。
#     未擴之前 sweep 仍可跑，只是 frechet 段會全部跳過。
#   * runs/dctshield 的防禦圖只有 CSV 入庫、PNG 留在遠端。網路恢復後**第一件
#     事**是把它們拉回本機入版控（資料保全規定）。

set -eu
source "$HOME/env.sh" >/dev/null 2>&1 || true
cd /nfs/home/nelson0314/WACV-s3
export PYTHONPATH=/nfs/home/nelson0314/WACV-s3
export HF_HOME=$HOME/hf_cache
export DIFFPURE_CKPT=$HOME/thirdparty/diffpure/256x256_diffusion_uncond.pt
PY=$HOME/venvs/wacv/bin/python
DATA=${DATA:-data/set0817}
ROOT=runs/s0820
STAGE=$1
export CUDA_VISIBLE_DEVICES=${2:-0}
G=$CUDA_VISIBLE_DEVICES
STRENGTH=${STRENGTH:-0.7}
mkdir -p $ROOT

# 掃描網格（DEC-029 §3.1）。ε 的前四點是論文 §6.1 圖 5 自己用的取捨曲線，
# 後兩點是本專案往下延伸到我們的失真區間所加——**ε<1 時論文的抗 JPEG 條件
# 失效**，driver 會自動把 modified_from_paper 標成真。
EPS_GRID=${EPS_GRID:-"0.4 0.6 0.8 1.0 1.2 1.4"}
THETA_GRID=${THETA_GRID:-"0.8 1.0 1.30 1.6 2.0"}

case $STAGE in

  # ---- DCT-Shield：逐 ε 一個輸出目錄，目錄名帶 ε，合併時 glob 不會撈到自己
  # （FND-062 記過：al* 同時匹配 al0..al3 與輸出目錄 aligned，把結果追加給自己）
  sweep_dct)
    for E in $EPS_GRID; do
      OUT=$ROOT/dct_e${E}_g$G
      $PY scripts/dct_shield_run.py --out $OUT --data $DATA \
          --mode paper --eps $E --conditions dct_shield \
          --edit-strength $STRENGTH 2>&1 | tee $ROOT/dct_e${E}_g$G.log
    done
    ;;

  sweep_phase)
    for T in $THETA_GRID; do
      OUT=$ROOT/phase_t${T}_g$G
      $PY scripts/phase_ablation.py --out $OUT --data $DATA \
          --conditions phase --human-threshold --phase-radius $T \
          --edit-strength $STRENGTH 2>&1 | tee $ROOT/phase_t${T}_g$G.log
    done
    ;;

  # ---- 以下兩段不需要 GPU，本機也跑得動 ----

  curve)
    # 兩條曲線 ＋ 兩個錨點。--ref-radius 1.30 是紋理重相位的定案值。
    $PY scripts/tradeoff_curve.py --run $ROOT/dct_e*_g* $ROOT/phase_t*_g* \
        --x fid_dists --ref phase --ref-radius 1.30 --out $ROOT/curve_dists
    $PY scripts/tradeoff_curve.py --run $ROOT/dct_e*_g* $ROOT/phase_t*_g* \
        --x fid_lpips --ref phase --ref-radius 1.30 --out $ROOT/curve_lpips
    ;;

  frechet)
    # n<150 時整批跳過並印出原因——小樣本 FID 不可解讀，不得以 --allow-small
    # 充數進正式報表。
    $PY scripts/fid_batch.py --run $ROOT/dct_e*_g* $ROOT/phase_t*_g* \
        --out $ROOT/frechet.csv
    ;;

  # ---- 抗淨化只在兩個錨點上跑，全曲線都跑淨化沒有必要 ----
  ret_anchor)
    for D in $ROOT/dct_e*_g$G $ROOT/phase_t*_g$G; do
      [ -f "$D/results.csv" ] || continue
      $PY scripts/phase_retention.py --run $D --seeds 3 \
          --purifiers identity blur1 crop_resize0.1 jpeg75 gridpure \
          --edit-strength $STRENGTH --out $ROOT/ret_$(basename $D).csv
    done
    ;;

  merge)
    # 輸出檔一律落在被 glob 匹配的路徑之外（FND-062 的教訓）。
    mkdir -p $ROOT/merged
    head -1 $(ls $ROOT/dct_e*_g*/results.csv | head -1) > $ROOT/merged/dct.csv
    for f in $ROOT/dct_e*_g*/results.csv; do tail -n +2 $f >> $ROOT/merged/dct.csv; done
    head -1 $(ls $ROOT/phase_t*_g*/results.csv | head -1) > $ROOT/merged/phase.csv
    for f in $ROOT/phase_t*_g*/results.csv; do tail -n +2 $f >> $ROOT/merged/phase.csv; done
    wc -l $ROOT/merged/*.csv
    ;;

  *) echo "未知的 stage：$STAGE"; exit 1 ;;
esac

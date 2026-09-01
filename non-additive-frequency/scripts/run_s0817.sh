#!/usr/bin/env bash
# 2026-08-18 的批次。七張影像、prompt 0。
#
# 只有兩組實驗：
#   1. 七個條件的攻擊    apa_weak / mist / dia_r / photoguard_c / add / phase / phase_rand
#   2. 上述防禦圖過九個淨化算子（含 jpeg->resize 串接，不含 IMPRESS）
#
# 相位用人眼門檻 theta = 1.30（phase_ablation.HUMAN_RADIUS）。2026-08-17 使用者
# 判定 theta = pi 失真過大而撤回。不要再加 --phase-radius——在 --human-threshold
# 模式下 radius 就是預算本身，那個旗標會直接蓋掉人眼門檻。
#
# 遮罩已撤回（2026-08-17）：改為在 prompt 裡寫出本人姓名讓模型自己把臉生回同
# 一個人。`data/set0817/headmasks/` 已刪，`load_dataset` 找不到遮罩就不做 latent
# 混合。攻擊不讀 prompt（PGD 用 encoder-targeted 損失、APA 階段一用 class），
# 所以改 prompt 時可以沿用既有的 `*__def.png`，只跑 reeval 與 ret*——那省下
# photoguard_c 的 16.1 GPU 小時。
#
# IMPRESS 依使用者指示不跑。它是這組候選中唯一針對防護擾動設計的淨化方法，
# 也是唯一的瓶頸（實測佔一格 616 s 裡的 505 s）。要補跑的話：
#   phase_retention.py --purifiers identity impress --out <另一個檔>
#
# 用法： bash scripts/run_s0817.sh <stage> <gpu> [strength]
#   stage ∈ pxext | px | ext | pg0..pg6 | merge | reeval | ret0..ret6
#   strength 預設 0.8。只有 photoguard_c 的攻擊會讀它，其餘六個條件的防禦圖
#   與強度無關，故同一批攻擊可以在多個強度上重複評測與淨化。
#
# 分片理由：photoguard_c 實測 6429 s/張，佔全批 94% 的機時。七張影像各吃一張
# 卡（1.8 小時），其餘六個條件（合計 428 s/張）串在第八張卡上跑 pxext
# （約 1.9 小時），兩條路徑等長。同一個 --out 只允許一個寫入者——write_csv
# 每次呼叫整份覆寫。

set -euo pipefail
source "$HOME/env.sh" >/dev/null 2>&1 || true   # env.sh 會 cd 到舊 repo，故 cd 放它後面
cd /nfs/home/nelson0314/WACV-s3
export PYTHONPATH=/nfs/home/nelson0314/WACV-s3
PY=$HOME/venvs/wacv/bin/python
DATA=data/set0817
ROOT=runs/s0817
STAGE=$1
export CUDA_VISIBLE_DEVICES=$2
STRENGTH=${3:-0.8}
S="--edit-strength $STRENGTH"
mkdir -p $ROOT

# 九個便宜算子。identity 是 retention 的分母，不能拿掉。
FAST="identity blur1 noise0.05 quantize16 jpeg75 jpeg30 crop_resize0.1 jpeg_then_resize75 adverse_cleaner"

# photoguard_c：七張影像一張卡一張。
IMGS="person_a_00 person_b_00 person_c_00 cat_00 shiba_00 raccoon_00 parrot_00"

case $STAGE in
  # ---- 像素臂與三個便宜 baseline 串在同一張卡上 ----
  pxext)
    bash "$0" px "$2" "$STRENGTH"
    bash "$0" ext "$2" "$STRENGTH" ;;

  # ---- 像素臂：add / phase / phase_rand，各自用自己的人眼門檻半徑 ----
  px)
    $PY scripts/phase_ablation.py --out $ROOT/px --data $DATA --conditions add phase phase_rand --human-threshold $S ;;

  # ---- 三個便宜的已發表 baseline，七張一起 ----
  ext) $PY scripts/apa_baseline.py --out $ROOT/ext --data $DATA --conditions apa_weak mist dia_r $S ;;

  # ---- photoguard_c，逐影像一片 ----
  pg0|pg1|pg2|pg3|pg4|pg5|pg6)
    IM=$(echo $IMGS | cut -d" " -f$(( ${STAGE#pg} + 1 )))
    $PY scripts/apa_baseline.py --out $ROOT/$STAGE --data $DATA --conditions photoguard_c --images $IM $S ;;

  # ---- 合併，供淨化階段讀 ----
  merge)
    $PY scripts/merge_runs.py --out $ROOT/merged \
        --src $ROOT/px $ROOT/ext \
              $ROOT/pg0 $ROOT/pg1 $ROOT/pg2 $ROOT/pg3 $ROOT/pg4 $ROOT/pg5 $ROOT/pg6 ;;

  # ---- 重算評測：prompt 改了，但攻擊不讀 prompt，防禦圖沿用 ----
  reeval)
    $PY scripts/reeval_edits.py --run $ROOT/merged --data $DATA $S ;;

  # ---- 淨化：跑在已存的防禦圖上，不重跑攻擊。逐影像分片，一張卡一張圖 ----
  ret0|ret1|ret2|ret3|ret4|ret5|ret6)
    N=${STAGE#ret}
    IM=$(echo $IMGS | cut -d" " -f$(( N + 1 )))
    $PY scripts/phase_retention.py --run $ROOT/merged --data $DATA --seeds 3 --purifiers $FAST --images $IM $S --out $ROOT/merged/retention_$N.csv ;;

  *) echo "unknown stage $STAGE"; exit 1 ;;
esac

#!/usr/bin/env bash
# 2026-08-17 的新起點批次。九張新影像、prompt 0（人物換衣物／動物戴帽子）。
#
# 只有兩組實驗：
#   1. 七個條件的攻擊    apa_weak / mist / dia_r / photoguard_c / add / phase / phase_rand
#   2. 上述防禦圖過十個淨化算子（含 jpeg->resize 串接）
#
# 相位的預算開滿：theta = pi（構造上的上限），phase_rand 同半徑。
#
# 用法： bash scripts/run_s0817.sh <stage> <gpu>
#   stage ∈ px | ext1 ext2 ext3 | pg0 pg1 pg2 pg3 | ret0 ret1 ret2 ret3
#
# 分片理由：photoguard_c 實測 6183 s/張，佔全批 94% 的機時，故它自己吃四張卡；
# 其餘六個條件加起來只有 428 s/張，塞在有空檔的卡上。
# 同一個 --out 只允許一個寫入者——write_csv 每次呼叫整份覆寫。

set -euo pipefail
source "$HOME/env.sh" >/dev/null 2>&1 || true   # env.sh 會 cd 到舊 repo，故 cd 放它後面
cd /nfs/home/nelson0314/WACV-s3
export PYTHONPATH=/nfs/home/nelson0314/WACV-s3
PY=$HOME/venvs/wacv/bin/python
DATA=data/set0817
ROOT=runs/s0817
STAGE=$1
export CUDA_VISIBLE_DEVICES=$2
mkdir -p $ROOT

G1="obama_00 lebron_00 ronaldo_00"
G2="musk_00 trump_00 parrot_00"
G3="cat_00 raccoon_00 shiba_00"

# photoguard_c 的四片：3/2/2/2。拿三張的那片不做別的事，它是關鍵路徑。
P0="obama_00 lebron_00 ronaldo_00"
P1="musk_00 trump_00"
P2="parrot_00 cat_00"
P3="raccoon_00 shiba_00"

case $STAGE in
  # ---- 像素臂：add / phase / phase_rand，相位半徑開滿 ----
  px)
    $PY scripts/phase_ablation.py --out $ROOT/px --data $DATA \
        --conditions add phase phase_rand \
        --human-threshold --phase-radius 3.14159265 ;;

  # ---- 三個已發表的便宜 baseline ----
  ext1) $PY scripts/apa_baseline.py --out $ROOT/ext1 --data $DATA \
            --conditions apa_weak mist dia_r --images $G1 ;;
  ext2) $PY scripts/apa_baseline.py --out $ROOT/ext2 --data $DATA \
            --conditions apa_weak mist dia_r --images $G2 ;;
  ext3) $PY scripts/apa_baseline.py --out $ROOT/ext3 --data $DATA \
            --conditions apa_weak mist dia_r --images $G3 ;;

  # ---- photoguard_c，四片 ----
  pg0) $PY scripts/apa_baseline.py --out $ROOT/pg0 --data $DATA \
           --conditions photoguard_c --images $P0 ;;
  pg1) $PY scripts/apa_baseline.py --out $ROOT/pg1 --data $DATA \
           --conditions photoguard_c --images $P1 ;;
  pg2) $PY scripts/apa_baseline.py --out $ROOT/pg2 --data $DATA \
           --conditions photoguard_c --images $P2 ;;
  pg3) $PY scripts/apa_baseline.py --out $ROOT/pg3 --data $DATA \
           --conditions photoguard_c --images $P3 ;;

  # ---- 合併，供淨化階段讀 ----
  merge)
    $PY scripts/merge_runs.py --out $ROOT/merged \
        --src $ROOT/px $ROOT/ext1 $ROOT/ext2 $ROOT/ext3 \
              $ROOT/pg0 $ROOT/pg1 $ROOT/pg2 $ROOT/pg3 ;;

  # ---- 淨化：跑在已存的防禦圖上，不重跑攻擊 ----
  ret0) $PY scripts/phase_retention.py --run $ROOT/merged --data $DATA --seeds 3 \
            --images obama_00 lebron_00 --out $ROOT/merged/retention_0.csv ;;
  ret1) $PY scripts/phase_retention.py --run $ROOT/merged --data $DATA --seeds 3 \
            --images ronaldo_00 musk_00 --out $ROOT/merged/retention_1.csv ;;
  ret2) $PY scripts/phase_retention.py --run $ROOT/merged --data $DATA --seeds 3 \
            --images trump_00 parrot_00 cat_00 --out $ROOT/merged/retention_2.csv ;;
  ret3) $PY scripts/phase_retention.py --run $ROOT/merged --data $DATA --seeds 3 \
            --images raccoon_00 shiba_00 --out $ROOT/merged/retention_3.csv ;;

  *) echo "unknown stage $STAGE"; exit 1 ;;
esac

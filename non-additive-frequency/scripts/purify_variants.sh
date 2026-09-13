#!/usr/bin/env bash
# 完整淨化評估：四道淨化 × 五類指令 × 三顆種子 × 兩臂，逐變體排隊。
#
# 接在 scripts/screen_variants.sh 後面跑。篩選那一支只跑 identity，回答的是
# 「這個載體有沒有東西」；這一支回答「撐不撐得過攻擊者的前處理」。幾何 warp
# 與顏色的抗淨化形狀不同——JPEG、模糊、裁切都不是 warp 的逆變換——所以不可以
# 從顏色線的數字推定，要逐變體實測。
#
# 順序與篩選相同。跑到哪算哪：每個變體一份完整 CSV。
set -uo pipefail
cd /nfs/home/nelson0314/WACV-s4
source ~/env.sh >/dev/null 2>&1
cd /nfs/home/nelson0314/WACV-s4

CARDS=(2 3 4 5 6)
ORDER=(flow_02 flow_04 flow_08 lab_field flow_16)

echo "[purify] 等篩選鏈跑完"
while pgrep -u "$(whoami)" -f screen_variants.sh > /dev/null; do sleep 120; done
echo "[purify] 篩選已結束 $(date -Is)"

for V in "${ORDER[@]}"; do
  RUN=runs/field_ladder/by_variant/$V
  ls $RUN/*__immunised.png > /dev/null 2>&1 || { echo "[purify] $V 無產物，跳過"; continue; }
  mkdir -p runs/field_purify/$V
  for i in 1 2 3 4 5; do
    G=${CARDS[$((i-1))]}
    CUDA_VISIBLE_DEVICES=$G $PY scripts/evaluate_defence.py       --config configs/evaluate_heldout.json --run "$RUN"       --out runs/field_purify/$V/shard$i --shard $i/5       > runs/field_purify/${V}_shard$i.log 2>&1 &
  done
  wait
  echo "[purify] $V 完成 $(date -Is) 列數 $(cat runs/field_purify/$V/shard*/*.csv 2>/dev/null | grep -vc '^arm,')"
done
echo "[purify] 全部結束 $(date -Is)"

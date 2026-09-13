#!/usr/bin/env bash
# 第二批位移場：臉框內的自然度約束。接在淨化鏈後面，產圖完再自己接篩選。
#
# 第一批的 flow_16 在 task_env_weather_114555 上把鼻子局部拉長，那是 warp 的
# 痕跡不是另一張臉。這一批測的假設是：不自然的是**局部**形變而不是整體形變，
# 所以把臉框內壓回近仿射、同時讓仿射那一段走得更遠，可以在同樣的自然度下
# 拿到更強的防禦。
set -uo pipefail
cd /nfs/home/nelson0314/WACV-s4
source ~/env.sh >/dev/null 2>&1
cd /nfs/home/nelson0314/WACV-s4

CARDS=(2 3 4 5 6)
ORDER=(face_rigid_16 face_rigid_32 coarse_16 coarse_32 face_rigid_08)

echo "[affine] 等淨化鏈跑完 $(date -Is)"
while pgrep -u "$(whoami)" -f purify_variants.sh > /dev/null; do sleep 60; done
echo "[affine] 淨化鏈已結束 $(date -Is)"

for i in 1 2 3 4 5; do
  G=${CARDS[$((i-1))]}
  CUDA_VISIBLE_DEVICES=$G $PY scripts/immunise_field.py     --config configs/immunise_field_affine.json     --out runs/field_affine/shard$i --shard $i/5     > runs/field_affine_log_shard$i.txt 2>&1 &
done
wait
echo "[affine] 產圖完成 $(date -Is)"

for V in "${ORDER[@]}"; do
  RUN=runs/field_affine/by_variant/$V
  mkdir -p "$RUN"
  n=0
  for f in runs/field_affine/shard*/*__${V}__immunised.png; do
    [ -e "$f" ] || continue
    b=$(basename "$f"); img=${b%__${V}__immunised.png}
    ln -sf "$(readlink -f "$f")" "$RUN/${img}__immunised.png"
    n=$((n+1))
  done
  echo "[affine] $V 有 $n 張防禦圖 $(date -Is)"
  [ "$n" -eq 0 ] && continue
  mkdir -p runs/field_affine_screen/$V
  for i in 1 2 3 4 5; do
    G=${CARDS[$((i-1))]}
    CUDA_VISIBLE_DEVICES=$G $PY scripts/evaluate_defence.py       --config configs/evaluate_screen.json --run "$RUN"       --out runs/field_affine_screen/$V/shard$i --shard $i/5       > runs/field_affine_screen/${V}_shard$i.log 2>&1 &
  done
  wait
  echo "[affine] $V 篩選完成 $(date -Is) 列數 $(cat runs/field_affine_screen/$V/shard*/*.csv 2>/dev/null | grep -vc '^arm,')"
done
echo "[affine] 全部結束 $(date -Is)"

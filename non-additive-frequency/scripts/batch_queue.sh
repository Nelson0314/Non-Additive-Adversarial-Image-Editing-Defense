#!/usr/bin/env bash
# 一支總排程：五張卡一次只服務一批，一批完整跑完（產圖＋評估）再換下一批。
#
# 為什麼不是每批一條鏈各自等待：多條鏈同時等同一個前置，前置一結束就會一起
# 醒來搶同五張卡。而且殺掉等待中的鏈**不會帶走它已經啟動的子行程**——踩過一次，
# 兩個孤兒各佔 19.8 GiB 跑了 26 分鐘，把同卡上的評估擠到 CUDA OOM，
# runs/field_affine_screen/coarse_16 與 coarse_32 因此只寫出 90 列而不是 150。
# 一支腳本依序跑，佔用永遠是五張，也只有一個要殺的對象。
#
# 順序就是優先序，時間不夠時損失的是排在後面的批次。
set -uo pipefail
cd /nfs/home/nelson0314/WACV-s4
source ~/env.sh >/dev/null 2>&1
cd /nfs/home/nelson0314/WACV-s4

CARDS=(2 3 4 5 6)

repair () {
  # 補跑被 OOM 打斷的分片。只動指名的那幾個目錄，而且只在它們沒有 CSV 時動：
  # 沒有結果檔的目錄可能只是還沒寫完，這裡是已經確認寫入行程死亡的那幾個。
  local NAME=$1; shift
  for spec in "$@"; do
    local V=${spec%%:*} I=${spec##*:}
    local OUT=runs/field_${NAME}_screen/$V/shard$I
    if ls $OUT/*.csv >/dev/null 2>&1; then
      echo "[repair] $V shard$I 已有結果，跳過"
      continue
    fi
    rm -rf $OUT
    local G=${CARDS[$(( (I-1) % 5 ))]}
    CUDA_VISIBLE_DEVICES=$G $PY scripts/evaluate_defence.py       --config configs/evaluate_screen.json       --run runs/field_${NAME}/by_variant/$V       --out $OUT --shard $I/5       > runs/field_${NAME}_screen/${V}_shard${I}_repair.log 2>&1 &
  done
  wait
  echo "[repair] $NAME 補跑完成 $(date -Is)"
}

run_batch () {
  local NAME=$1 CONFIG=$2 EVAL_CONFIG=$3; shift 3
  local ORDER=("$@")
  echo "[queue] $NAME 產圖開始 $(date -Is)"
  rm -rf runs/field_${NAME} runs/field_${NAME}_eval
  for i in 1 2 3 4 5; do
    CUDA_VISIBLE_DEVICES=${CARDS[$((i-1))]} $PY scripts/immunise_field.py       --config $CONFIG --out runs/field_${NAME}/shard$i --shard $i/5       > runs/field_${NAME}_log_shard$i.txt 2>&1 &
  done
  wait
  echo "[queue] $NAME 產圖完成 $(date -Is)"
  for V in "${ORDER[@]}"; do
    local RUN=runs/field_${NAME}/by_variant/$V
    mkdir -p "$RUN"
    local n=0
    for f in runs/field_${NAME}/shard*/*__${V}__immunised.png; do
      [ -e "$f" ] || continue
      local b=$(basename "$f"); local img=${b%__${V}__immunised.png}
      ln -sf "$(readlink -f "$f")" "$RUN/${img}__immunised.png"
      n=$((n+1))
    done
    echo "[queue] $NAME/$V 有 $n 張防禦圖 $(date -Is)"
    [ "$n" -eq 0 ] && continue
    mkdir -p runs/field_${NAME}_eval/$V
    for i in 1 2 3 4 5; do
      CUDA_VISIBLE_DEVICES=${CARDS[$((i-1))]} $PY scripts/evaluate_defence.py         --config $EVAL_CONFIG --run "$RUN"         --out runs/field_${NAME}_eval/$V/shard$i --shard $i/5         > runs/field_${NAME}_eval/${V}_shard$i.log 2>&1 &
    done
    wait
    local rows=$(cat runs/field_${NAME}_eval/$V/shard*/*.csv 2>/dev/null | grep -vc '^arm,')
    echo "[queue] $NAME/$V 評估完成 $(date -Is) 列數 $rows"
    [ "$rows" -ne 150 ] && echo "[queue] !! $NAME/$V 列數不是 150，有分片失敗，看 ${V}_shard*.log"
  done
  echo "[queue] $NAME 全部結束 $(date -Is)"
}

echo "[queue] 等第二批跑完 $(date -Is)"
while pgrep -u "$(whoami)" -f affine_chain.sh > /dev/null; do sleep 60; done
echo "[queue] 第二批已結束 $(date -Is)"

repair affine coarse_16:1 coarse_16:4 coarse_32:1 coarse_32:4

run_batch margin configs/immunise_field_margin.json configs/evaluate_screen.json margin_64 margin_64_norigid margin_32 margin_64_loose margin_00
run_batch coarse_margin configs/immunise_field_coarse_margin.json configs/evaluate_screen.json coarse_margin_48 coarse_margin_32 coarse_margin_96 coarse_margin_48_tight coarse_margin_00
run_batch grid configs/immunise_field_grid.json configs/evaluate_screen.json grid_06 grid_08 grid_05 grid_10 grid_04
run_batch objective configs/immunise_field_objective.json configs/evaluate_screen.json obj_targeted obj_diffusion obj_all obj_longchain obj_baseline
run_batch eot configs/immunise_field_eot.json configs/evaluate_heldout.json eot_crop eot_crop_16 eot_both eot_none
echo "[queue] 佇列全部結束 $(date -Is)"

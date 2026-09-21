#!/usr/bin/env bash
# 把 runs/objective_full 已經解好的 24 個解接完：編輯 → 位移 → 配對差。
#
# 為什麼要重寫一支：`wait_and_run_objective_full.sh` 在**派解的時候**挑好卡，
# 四小時後解完才拿那張卡去編輯，而那時候卡早就被別人佔走了，六次 edit_preflight
# 全部 CUDA OOM。這一支改成**每一次編輯之前重新挑一張當下空的卡**。
set -u
L=/nfs/home/nelson0314/WACV-colour-lab
cd "$L" || exit 1
export PYTHONPATH=$L
export HF_HOME=/var/cache/huggingface
PY=/nfs/home/nelson0314/venvs/wacv/bin/python
SOLVE=runs/objective_full
EDITS=runs/objective_full_edits
VARIANTS="operating_point operating_point_jittered out_no_id"

# 「空」的門檻是 1500 MiB：別人開著、實算在別張卡的 256 MiB 空 context 不算佔用，
# 真正在跑的工作都是 10 GB 以上，分得開。
pick_card() {
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
    | awk -F, '{ if ($2 + 0 < 1500) print $1 }' | head -1
}

wait_for_card() {
  local i=0 c
  while [ "$i" -lt 2880 ]; do
    c=$(pick_card)
    if [ -n "$c" ]; then echo "$c"; return 0; fi
    i=$((i + 1)); sleep 60
  done
  return 1
}

for v in $VARIANTS; do
  def=$SOLVE/defended/$v
  mkdir -p "$def"
  for f in $(find "$SOLVE" -name "*__${v}__defended.png" | sort); do
    b=$(basename "$f" __defended.png)
    cp "$f" "$def/${b%%__*}__def.png"
  done
  cnt=$(ls "$def"/*__def.png 2>/dev/null | wc -l)
  echo "[$v] 防禦圖 $cnt 張"
  if [ "$cnt" -ne 8 ]; then echo "[$v] ABORT 不是八張"; exit 1; fi

  for s in ip2p inpaint; do
    if [ "$s" = ip2p ]; then sfx=_si18; else sfx=_undefended; fi
    if [ -d "$EDITS/$v/${s}${sfx}" ]; then
      echo "[$v/$s] SKIP 已有編輯"
      continue
    fi
    card=$(wait_for_card) || { echo "[ABORT] 等不到空卡"; exit 1; }
    echo "[$v/$s] 用卡 $card $(date +%H:%M:%S)"
    if [ "$s" = ip2p ]; then
      CUDA_VISIBLE_DEVICES=$card "$PY" scripts/edit_preflight.py \
        --data data/portraits --scenarios "$s" --suffix "$sfx" --s-i 1.8 \
        --defended "$def" --out "$EDITS/$v" \
        > "$L/logs/edit_objfull_${v}_${s}.log" 2>&1
    else
      CUDA_VISIBLE_DEVICES=$card "$PY" scripts/edit_preflight.py \
        --data data/portraits --scenarios "$s" --suffix "$sfx" \
        --defended "$def" --out "$EDITS/$v" \
        > "$L/logs/edit_objfull_${v}_${s}.log" 2>&1
    fi
    rc=$?
    echo "[$v/$s] rc=$rc $(date +%H:%M:%S)"
    if [ "$rc" -ne 0 ]; then
      tail -3 "$L/logs/edit_objfull_${v}_${s}.log"
      exit 1
    fi
  done
done

card=$(wait_for_card) || { echo "[ABORT] 彙整等不到空卡"; exit 1; }
export CUDA_VISIBLE_DEVICES=$card
echo "[彙整] 用卡 $card $(date +%H:%M:%S)"
"$PY" scripts/edit_displacement.py --defended-root "$EDITS" \
  --preflight runs/colour_probe_preflight --data data/portraits \
  --ip2p-arm ip2p_si18 --inpaint-arm inpaint_undefended \
  --out "$EDITS/displacement.csv" > "$L/logs/agg_objfull.log" 2>&1
echo "位移 rc=$?"
"$PY" scripts/paired_difference.py --displacement "$EDITS/displacement.csv" \
  --control operating_point --out "$EDITS/paired.csv"
echo "[ALLDONE] $(date +%H:%M:%S)"

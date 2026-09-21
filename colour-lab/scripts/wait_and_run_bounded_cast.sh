#!/usr/bin/env bash
# 等到有四張空卡，把 bounded_cast 那一批跑完：解 → ip2p 與 inpaint 編輯 → 位移 → 配對差。
#
# 與早先那支 wait_and_run_objective_full.sh 的差別：**每一次編輯之前重新挑一張
# 當下空的卡**。那一支在派解的時候挑好卡，四小時後解完才拿去編輯，那時候卡已經
# 被別人佔走，六次 edit_preflight 全部 CUDA OOM。
#
# 這一批是四張影像、五個變體（兩個 advcf 對照臂 ＋ 三組界），共 20 個解。
set -u
L=/nfs/home/nelson0314/WACV-colour-lab
cd "$L" || exit 1
export PYTHONPATH=$L
export HF_HOME=/var/cache/huggingface
PY=/nfs/home/nelson0314/venvs/wacv/bin/python
CFG=configs/bounded_cast.json
SOLVE=runs/bounded_cast_solve
EDITS=runs/bounded_cast_edits
VARIANTS="operating_point operating_point_jittered bounded_cast_C16_d54 bounded_cast_C24_d38 bounded_cast_C32_d24"
IMGS="man_00 man_01 woman_00 woman_01"
WANT=20

# 「空」的門檻是 1500 MiB：別人開著、實算在別張卡的 256 MiB 空 context 不算佔用，
# 真正在跑的工作都是 10 GB 以上，分得開。
free_cards() {
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
    | awk -F, '{ if ($2 + 0 < 1500) print $1 }'
}

wait_for_card() {
  local i=0 c
  while [ "$i" -lt 2880 ]; do
    c=$(free_cards | head -1)
    if [ -n "$c" ]; then echo "$c"; return 0; fi
    i=$((i + 1)); sleep 60
  done
  return 1
}

echo "[WAIT] 等四張空卡 $(date +%H:%M:%S)"
i=0
while [ "$i" -lt 2880 ]; do
  n=$(free_cards | wc -l)
  if [ "$n" -ge 4 ]; then break; fi
  i=$((i + 1)); sleep 60
done
cards=$(free_cards | head -4 | tr '\n' ' ')
if [ "$(echo $cards | wc -w)" -lt 4 ]; then echo "[ABORT] 等不到四張卡"; exit 1; fi
echo "[GO] 用卡 $cards $(date +%H:%M:%S)"

k=1
for c in $cards; do
  case $k in
    1) sh=man_00 ;; 2) sh=man_01 ;; 3) sh=woman_00 ;; 4) sh=woman_01 ;;
  esac
  CUDA_VISIBLE_DEVICES=$c setsid nohup "$PY" scripts/paper_baseline.py \
    --config "$CFG" --out "$SOLVE/$sh" --shard "$k/4" --objective free \
    > "$L/logs/bounded_cast_$sh.log" 2>&1 < /dev/null &
  echo "[LAUNCH] shard $k/4 -> card $c ($sh)"
  k=$((k + 1))
done

while true; do
  m=$(find "$SOLVE" -name '*__defended.png' 2>/dev/null | wc -l)
  if [ "$m" -ge "$WANT" ]; then break; fi
  alive=$(ps -u nelson0314 -o args= | grep -c '[p]aper_baseline.py')
  if [ "$alive" -eq 0 ]; then
    echo "[ABORT] 求解全停，只有 $m/$WANT"
    grep -hiE 'Traceback|Error|ValueError' "$L"/logs/bounded_cast_*.log | tail -5
    exit 1
  fi
  sleep 60
done
echo "[SOLVES DONE] $m/$WANT $(date +%H:%M:%S)"

for v in $VARIANTS; do
  def=$SOLVE/defended/$v
  mkdir -p "$def"
  for f in $(find "$SOLVE" -name "*__${v}__defended.png" | sort); do
    b=$(basename "$f" __defended.png)
    cp "$f" "$def/${b%%__*}__def.png"
  done
  cnt=$(ls "$def"/*__def.png 2>/dev/null | wc -l)
  echo "[$v] 防禦圖 $cnt 張"
  if [ "$cnt" -ne 4 ]; then echo "[$v] ABORT 不是四張"; exit 1; fi

  for s in ip2p inpaint; do
    if [ "$s" = ip2p ]; then sfx=_si18; else sfx=_undefended; fi
    if [ -d "$EDITS/$v/${s}${sfx}" ]; then echo "[$v/$s] SKIP 已有編輯"; continue; fi
    card=$(wait_for_card) || { echo "[ABORT] 等不到空卡"; exit 1; }
    echo "[$v/$s] 用卡 $card $(date +%H:%M:%S)"
    if [ "$s" = ip2p ]; then
      CUDA_VISIBLE_DEVICES=$card "$PY" scripts/edit_preflight.py \
        --data data/portraits --scenarios "$s" --suffix "$sfx" --s-i 1.8 \
        --images $IMGS --defended "$def" --out "$EDITS/$v" \
        > "$L/logs/edit_bcast_${v}_${s}.log" 2>&1
    else
      CUDA_VISIBLE_DEVICES=$card "$PY" scripts/edit_preflight.py \
        --data data/portraits --scenarios "$s" --suffix "$sfx" \
        --images $IMGS --defended "$def" --out "$EDITS/$v" \
        > "$L/logs/edit_bcast_${v}_${s}.log" 2>&1
    fi
    rc=$?
    echo "[$v/$s] rc=$rc $(date +%H:%M:%S)"
    if [ "$rc" -ne 0 ]; then tail -3 "$L/logs/edit_bcast_${v}_${s}.log"; exit 1; fi
  done
done

card=$(wait_for_card) || { echo "[ABORT] 彙整等不到空卡"; exit 1; }
export CUDA_VISIBLE_DEVICES=$card
echo "[彙整] 用卡 $card $(date +%H:%M:%S)"
"$PY" scripts/edit_displacement.py --defended-root "$EDITS" \
  --preflight runs/colour_probe_preflight --data data/portraits \
  --ip2p-arm ip2p_si18 --inpaint-arm inpaint_undefended \
  --out "$EDITS/displacement.csv" > "$L/logs/agg_bcast.log" 2>&1
echo "位移 rc=$?"
"$PY" scripts/paired_difference.py --displacement "$EDITS/displacement.csv" \
  --control operating_point --out "$EDITS/paired.csv"
echo "[ALLDONE] $(date +%H:%M:%S)"

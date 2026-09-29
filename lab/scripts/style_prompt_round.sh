#!/usr/bin/env bash
# style_prompt 的一輪實驗：讀工作清單（每行「名稱 影像 風格 其餘參數」），在 CAP 張卡內派最佳化；
# 每份一跑完就送主種子編輯、重算讀數。用法：bash lab/scripts/style_prompt_round.sh <輪名> <清單檔> [CAP]
cd ~/image-immunization
source ~/env.sh >/dev/null 2>&1
R=$1; SPEC=$2; CAP=${3:-6}; OPT_CAP=${4:-$((CAP - 1))}
O=lab/runs/$R; E=lab/runs/${R}_edit; L=lab/runs/logs
mkdir -p "$O" "$E"
BASE="--data lab/data/portraits --s-i 2.0 --tf32"

declare -A IMG STY ARG
JOBS=""
while read -r name img sty rest; do
  [ -z "$name" ] || [[ $name == \#* ]] && continue
  IMG[$name]=${img//+/ }; STY[$name]=$sty; ARG[$name]=$rest; JOBS="$JOBS $name"  # 影像欄可用 + 串多張
done < "$SPEC"

mine() { cat ~/lab_leases/* 2>/dev/null | awk '$3 ~ /^style_prompt_/' | wc -l; }
try_launch() {  # try_launch <上限> <名稱> <指令>；以 flock 讓各排程的取卡依序進行，避免兩個 run_on_card 同時選到同一張卡
  local cap=$1 name=$2; shift 2
  exec 9> ~/.style_prompt_launch.lock
  flock 9
  if [ "$(mine)" -ge "$cap" ]; then flock -u 9; return 1; fi
  nohup setsid bash lab/scripts/run_on_card.sh "$name" bash -c "$*" > "$L/$name.log" 2>&1 < /dev/null 9>&- &
  local ok=1
  for i in $(seq 1 40); do
    grep -q " $name\$" ~/lab_leases/* 2>/dev/null && { echo "$(date +%H:%M:%S) launched $name"; sleep 3; ok=0; break; }
    grep -q "沒有空卡" "$L/$name.log" 2>/dev/null && break
    sleep 1
  done
  flock -u 9
  return $ok
}
running() { grep -q " $1\$" ~/lab_leases/* 2>/dev/null; }

declare -A OPT EDT
for j in $JOBS; do  # 接手：已在跑或已跑完的最佳化不重派，已有編輯結果的不重編
  { running "${R}_opt_$j" || [ -e "$O/$j/results.csv" ]; } && OPT[$j]=1
  [ -e "$E/${j}_${STY[$j]}/preflight.csv" ] && EDT[$j]=1
done
last=""
while true; do
  for j in $JOBS; do
    [ -n "${EDT[$j]:-}" ] && continue
    [ -n "${OPT[$j]:-}" ] && [ -e "$O/$j/results.csv" ] && ! running "${R}_opt_$j" || continue
    if ls "$O/$j/"*__def.png >/dev/null 2>&1; then
      try_launch "$CAP" "${R}_edit_$j" "\"\$PY\" lab/code/edit_preflight.py --data lab/data/portraits --defended $O/$j --out $E/${j}_${STY[$j]} --scenarios ip2p --suffix _${R}_$j --images ${IMG[$j]}" && EDT[$j]=1
    else
      echo "$(date +%H:%M:%S) $j: no feasible defence image"; EDT[$j]=none
    fi
  done
  # 編輯優先於新的最佳化：防禦圖一產出就送編輯
  for j in $JOBS; do
    [ -n "${OPT[$j]:-}" ] && continue
    try_launch "$OPT_CAP" "${R}_opt_$j" "\"\$PY\" lab/code/style_prompt_defence.py $BASE --images ${IMG[$j]} --styles ${STY[$j]} ${ARG[$j]} --out $O/$j" && OPT[$j]=1
  done
  ndone=0; done_list=""
  for j in $JOBS; do
    if [ "${EDT[$j]:-}" = none ]; then ndone=$((ndone + 1)); continue; fi
    if [ -e "$E/${j}_${STY[$j]}/preflight.csv" ] && ! running "${R}_edit_$j"; then ndone=$((ndone + 1)); done_list="$done_list $j"; fi
  done
  if [ "$done_list" != "$last" ]; then
    last=$done_list
    for st in $(for j in $done_list; do echo "${STY[$j]}"; done | sort -u); do
      js=$(for j in $done_list; do [ "${STY[$j]}" = "$st" ] && echo -n "$j "; done)
      ims=$(for j in $done_list; do [ "${STY[$j]}" = "$st" ] && echo "${IMG[$j]}"; done | tr " " "\n" | sort -u | tr "\n" " ")
      # 對照為 $E/ref_<風格>：由清單中名為 ref 的工作（lr 0 的未最佳化風格圖）產生，或事先放入
      js=$(echo $js | tr " " "\n" | grep -vx ref | tr "\n" " ")
      CUDA_VISIBLE_DEVICES= PYTHONPATH=$PWD "$PY" lab/code/style_prompt_readout.py --edits "$E" \
        --images $ims --styles "$st" --strengths $js \
        --out "$E/readout_$st.csv" > "$L/${R}_readout_$st.log" 2>&1
      echo "$(date +%H:%M:%S) readout $st [$js]:"; grep -v -i warn "$L/${R}_readout_$st.log" | tail -12
    done
  fi
  n=0; for j in $JOBS; do n=$((n + 1)); done
  [ "$ndone" -ge "$n" ] && { echo "${R}_DONE"; break; }
  sleep 20
done

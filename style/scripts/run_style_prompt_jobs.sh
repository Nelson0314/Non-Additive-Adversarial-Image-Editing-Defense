#!/usr/bin/env bash
# style_prompt 的一輪實驗：讀工作清單（每行「名稱 影像 風格 其餘參數」），在 CAP 張卡內派最佳化；
# 每份一跑完就送主種子編輯、重算讀數。
# 用法（style 專案根）：bash scripts/run_style_prompt_jobs.sh <輪名> <清單檔> [全局授權卡數] [最佳化派工上限]
source "$(dirname "${BASH_SOURCE[0]}")/env.sh" || exit 1
cd "$STYLE_ROOT" || exit 1
export PY
source "$GPU_TOOLS/gpu_lease.sh"
gpu_policy_init "${3:-}" || exit $?
R=$1; SPEC=$2; CAP=$(gpu_global_cap); OPT_CAP=${4:-$((CAP > 1 ? CAP - 1 : 1))}
gpu_valid_cap "$OPT_CAP" || { echo "OPT_CAP must be a positive integer" >&2; exit 2; }
O=artifacts/defenses/$R; E=artifacts/edits/$R; L=runtime/logs
mkdir -p "$O" "$E" "$L"
BASE="--data data/portraits --s-i 2.0 --tf32"

declare -A IMG STY ARG
JOBS=""
while read -r name img sty rest; do
  [ -z "$name" ] || [[ $name == \#* ]] && continue
  IMG[$name]=${img//+/ }; STY[$name]=$sty; ARG[$name]=$rest; JOBS="$JOBS $name"  # 影像欄可用 + 串多張
done < "$SPEC"

try_launch() {  # Shared lease acquisition enforces the global limit atomically.
  local cap=$1 name=$2; shift 2
  [ "$(lease_count)" -lt "$cap" ] || return 1
  GPU_CAP= nohup setsid bash "$GPU_TOOLS/run_with_gpu_lease.sh" --workdir "$STYLE_ROOT" --limit "$cap" "$name" \
    bash -c "$*" > "$L/$name.log" 2>&1 < /dev/null &
  local ok=1
  for i in $(seq 1 40); do
    lease_running "$name" && { echo "$(date +%H:%M:%S) launched $name"; sleep 3; ok=0; break; }
    grep -q "\[FATAL\]" "$L/$name.log" 2>/dev/null && break
    sleep 1
  done
  return $ok
}
running() { lease_running "$1"; }

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
      try_launch "$(gpu_global_cap)" "${R}_edit_$j" "\"\$PY\" -m immunization_style.cli.run_edits --data data/portraits --defended $O/$j --out $E/${j}_${STY[$j]} --scenarios ip2p --suffix _${R}_$j --images ${IMG[$j]}" && EDT[$j]=1
    else
      echo "$(date +%H:%M:%S) $j: no feasible defence image"; EDT[$j]=none
    fi
  done
  # 編輯優先於新的最佳化：防禦圖一產出就送編輯
  for j in $JOBS; do
    [ -n "${OPT[$j]:-}" ] && continue
    try_launch "$OPT_CAP" "${R}_opt_$j" "\"\$PY\" -m immunization_style.cli.generate_style_prompt_defenses $BASE --images ${IMG[$j]} --styles ${STY[$j]} ${ARG[$j]} --out $O/$j" && OPT[$j]=1
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
      CUDA_VISIBLE_DEVICES= "$PY" -m immunization_style.cli.measure_style_prompt_edits --edits "$E" \
        --images $ims --styles "$st" --strengths $js \
        --out "$E/readout_$st.csv" > "$L/${R}_readout_$st.log" 2>&1
      echo "$(date +%H:%M:%S) readout $st [$js]:"; grep -v -i warn "$L/${R}_readout_$st.log" | tail -12
    done
  fi
  n=0; for j in $JOBS; do n=$((n + 1)); done
  [ "$ndone" -ge "$n" ] && { echo "${R}_DONE"; break; }
  sleep 20
done

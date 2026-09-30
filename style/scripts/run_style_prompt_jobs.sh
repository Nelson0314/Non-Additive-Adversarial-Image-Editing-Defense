#!/usr/bin/env bash
# style_prompt 的一組實驗：讀工作清單（每行「名稱 影像 風格 其餘參數」，影像以 + 串多張），經租約取卡派最佳化；
# 防禦圖通過驗收即送主種子編輯；全部工作結束後以名為 ref 的工作為參照重算讀數。
# 用法（style 專案根）：bash scripts/run_style_prompt_jobs.sh <實驗名> <清單檔> [全局授權卡數] [最佳化派工上限]
#
# 完成判定：每個階段記錄結束碼（runtime/logs/<工作>.rc），並以 immunization_style.cli.check_job_outputs
# 驗收鍵集合與輸出檔。已存在的輸出只有在 <防禦目錄>/job.spec 與本次設定相同且通過驗收時沿用；
# 設定不同即中止。任一階段失敗，該工作記為失敗、不再派送；結束時有失敗即以結束碼 1 退出，
# 全部通過才印出 <實驗名>_DONE。
source "$(dirname "${BASH_SOURCE[0]}")/env.sh" || exit 1
cd "$STYLE_ROOT" || exit 1
export PY
source "$GPU_TOOLS/gpu_lease.sh"
gpu_policy_init "${3:-}" || exit $?
[ $# -ge 2 ] || { echo "[FATAL] 需要實驗名與清單檔" >&2; exit 2; }
R=$1; SPEC=$2; CAP=$(gpu_global_cap); OPT_CAP=${4:-$((CAP > 1 ? CAP - 1 : 1))}
gpu_valid_cap "$OPT_CAP" || { echo "[FATAL] 最佳化派工上限必須是正整數" >&2; exit 2; }
[ -f "$SPEC" ] || { echo "[FATAL] 找不到清單檔 $SPEC" >&2; exit 2; }
O=artifacts/defenses/$R; E=artifacts/edits/$R; L=runtime/logs
DATA=data/portraits
POLL=${POLL:-20}
mkdir -p "$O" "$E" "$L"
BASE="--data-root $DATA --s-i 2.0 --tf32"
CHECK=("$PY" -m immunization_style.cli.check_job_outputs)

declare -A IMG STY ARG
JOBS=()
while read -r name img sty rest; do
  { [ -z "$name" ] || [[ $name == \#* ]]; } && continue
  [ -z "${IMG[$name]:-}" ] || { echo "[FATAL] 工作名重複：$name" >&2; exit 2; }
  IMG[$name]=${img//+/ }; STY[$name]=$sty; ARG[$name]=$rest; JOBS+=("$name")
done < "$SPEC"
[ -n "${IMG[ref]:-}" ] || { echo "[FATAL] 清單沒有名為 ref 的參照工作；讀數無法計算" >&2; exit 2; }
for j in "${JOBS[@]}"; do
  [ "${STY[$j]}" = "${STY[ref]}" ] || { echo "[FATAL] $j 的風格 ${STY[$j]} 與 ref 的 ${STY[ref]} 不同" >&2; exit 2; }
done
STYLE=${STY[ref]}

spec_line() { printf '%s\n' "$BASE --images ${IMG[$1]} --styles ${STY[$1]} ${ARG[$1]}"; }
for j in "${JOBS[@]}"; do  # 既有輸出須來自同一設定
  if [ -f "$O/$j/job.spec" ] && [ "$(cat "$O/$j/job.spec")" != "$(spec_line "$j")" ]; then
    echo "[FATAL] $O/$j 的既有輸出來自不同設定（job.spec 不符）；換實驗名或移除該目錄" >&2; exit 2
  fi
done

try_launch() {  # 取卡由共用租約原子地計入全局上限；結束碼寫入 <名稱>.rc
  local cap=$1 name=$2; shift 2
  [ "$(lease_count)" -lt "$cap" ] || return 1
  rm -f "$L/$name.rc"
  GPU_CAP= nohup setsid bash "$GPU_TOOLS/run_with_gpu_lease.sh" --work-dir "$STYLE_ROOT" --limit "$cap" "$name" \
    bash -c "$*; echo \$? > '$L/$name.rc'" > "$L/$name.log" 2>&1 < /dev/null &
  local i
  for i in $(seq 1 40); do
    lease_running "$name" && { echo "$(date +%H:%M:%S) launched $name"; return 0; }
    [ -f "$L/$name.rc" ] && return 0
    grep -q "\[FATAL\]" "$L/$name.log" 2>/dev/null && return 1
    sleep 1
  done
  return 1
}
running() { lease_running "$1"; }
finished_ok() { [ -f "$L/$1.rc" ] && [ "$(cat "$L/$1.rc")" = 0 ]; }

defense_ok() { "${CHECK[@]}" defense --dir "$O/$1" --style "$STYLE" --images ${IMG[$1]} >> "$L/${R}_check_$1.log" 2>&1; }
edit_images() { "${CHECK[@]}" feasible --dir "$O/$1" --style "$STYLE" --images ${IMG[$1]}; }
edit_dir() { echo "$E/${1}_$STYLE"; }
edits_ok() {
  local ims; ims=$(edit_images "$1")
  [ -n "$ims" ] && "${CHECK[@]}" edits --dir "$(edit_dir "$1")" --arm "ip2p_${R}_$1" --images $ims \
    --data-root "$DATA" >> "$L/${R}_check_$1.log" 2>&1
}

declare -A STATE  # 空：未派；opt：最佳化中；edit：編輯中；done：通過；failed：失敗
FAILED=()
fail_job() { STATE[$1]=failed; FAILED+=("$1"); echo "$(date +%H:%M:%S) $1 FAILED: $2" >&2; }

for j in "${JOBS[@]}"; do  # 接手：只沿用通過驗收的既有輸出
  if running "${R}_edit_$j"; then STATE[$j]=edit
  elif running "${R}_opt_$j"; then STATE[$j]=opt
  elif [ -f "$O/$j/job.spec" ] && defense_ok "$j"; then
    if edits_ok "$j"; then STATE[$j]=done; else STATE[$j]=defended; fi
  fi
done

while true; do
  pending=0
  for j in "${JOBS[@]}"; do
    case "${STATE[$j]:-}" in
      opt)
        running "${R}_opt_$j" && { pending=1; continue; }
        if ! finished_ok "${R}_opt_$j"; then fail_job "$j" "最佳化結束碼 $(cat "$L/${R}_opt_$j.rc" 2>/dev/null || echo 無)"
        elif ! defense_ok "$j"; then fail_job "$j" "防禦輸出未通過驗收（$L/${R}_check_$j.log）"
        else STATE[$j]=defended; fi ;;
      edit)
        running "${R}_edit_$j" && { pending=1; continue; }
        if ! finished_ok "${R}_edit_$j"; then fail_job "$j" "編輯結束碼 $(cat "$L/${R}_edit_$j.rc" 2>/dev/null || echo 無)"
        elif ! edits_ok "$j"; then fail_job "$j" "編輯輸出未通過驗收（$L/${R}_check_$j.log）"
        else STATE[$j]=done; fi ;;
    esac
  done
  # 編輯優先於新的最佳化：防禦圖一通過驗收就送編輯
  for j in "${JOBS[@]}"; do
    [ "${STATE[$j]:-}" = defended ] || continue
    ims=$(edit_images "$j")
    if [ -z "$ims" ]; then fail_job "$j" "沒有可行的防禦圖"; continue; fi
    try_launch "$(gpu_global_cap)" "${R}_edit_$j" "\"\$PY\" -m immunization_style.cli.run_edits --data-root $DATA --defenses-dir $O/$j --output-dir $(edit_dir "$j") --scenarios ip2p --suffix _${R}_$j --images $ims" \
      && STATE[$j]=edit
    pending=1
  done
  for j in "${JOBS[@]}"; do
    [ -z "${STATE[$j]:-}" ] || continue
    mkdir -p "$O/$j" && spec_line "$j" > "$O/$j/job.spec"
    try_launch "$OPT_CAP" "${R}_opt_$j" "\"\$PY\" -m immunization_style.cli.generate_style_prompt_defenses $BASE --images ${IMG[$j]} --styles ${STY[$j]} ${ARG[$j]} --output-dir $O/$j" \
      && STATE[$j]=opt
    pending=1
  done
  [ "$pending" -eq 0 ] && break
  sleep "$POLL"
done

if [ "${STATE[ref]}" != done ]; then
  echo "[FATAL] 參照工作 ref 未完成；讀數無法計算" >&2
else
  strengths=(); expect=("ref=$(edit_images ref | tr ' ' ',')")
  for j in "${JOBS[@]}"; do
    [ "$j" != ref ] && [ "${STATE[$j]}" = done ] || continue
    strengths+=("$j"); expect+=("$j=$(edit_images "$j" | tr ' ' ',')")
  done
  if [ "${#strengths[@]}" -gt 0 ]; then
    images=$(for j in ref "${strengths[@]}"; do edit_images "$j"; echo; done | tr ' ' '\n' | sort -u | tr '\n' ' ')
    out="$E/readout_$STYLE.csv"
    if CUDA_VISIBLE_DEVICES= "$PY" -m immunization_style.cli.measure_style_prompt_edits --edits-root "$E" \
         --images $images --styles "$STYLE" --strengths "${strengths[@]}" --output-csv "$out" \
         > "$L/${R}_readout_$STYLE.log" 2>&1 \
       && "${CHECK[@]}" readout --csv "$out" --style "$STYLE" --expect "${expect[@]}" --data-root "$DATA" \
         >> "$L/${R}_readout_$STYLE.log" 2>&1; then
      echo "$(date +%H:%M:%S) readout $STYLE [${strengths[*]}] → $out"
    else
      FAILED+=("readout")
      echo "[FATAL] 讀數失敗或未通過驗收（$L/${R}_readout_$STYLE.log）" >&2
    fi
  fi
fi

if [ "${#FAILED[@]}" -gt 0 ] || [ "${STATE[ref]}" != done ]; then
  echo "${R}_FAILED: ${FAILED[*]}" >&2
  exit 1
fi
echo "${R}_DONE"

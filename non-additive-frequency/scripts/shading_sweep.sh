#!/usr/bin/env bash
# 候選二 步驟 1／3：乘性明暗場的效果，以及同失真的隨機對照。
#
# 步驟 0（等 RMS 下乘性對加性的失真）已跑完，見 `runs/shading_field_cost/`：
# 工作點所在的 RMS 上比值只有 1.332，低於該候選自訂的 1.5 門檻。**這一批是
# 效果那一半**——步驟 0 只量了「多明顯」，沒量「有沒有用」。
#
# 步驟 1 的判準（`SURVEY_ARCHITECTURE` 候選二第 6 點）：
#     同 DISTS 下位移低於現行方法的 20%，作為獨立方法無望，
#     只能考慮作為疊加補丁。
#
# 步驟 3 標為**必做**：同失真的隨機明暗場對照。`位移場`（FND-004）的死法正是
# 「與同失真隨機對照無法區分」，而低頻、低自由度的參數化特別容易重蹈覆轍。
# 故每個半徑都配一個 `shading_rand`。
#
# 強度旋鈕是 `--radius`（粗網格上 log 增益的 L-infinity 界）。
#
# 用法：bash scripts/shading_sweep.sh "<卡號>" ["<半徑>"]
set -uo pipefail
ROOT=/nfs/home/nelson0314/WACV-s3
PY="$HOME/venvs/wacv/bin/python"
export PYTHONPATH="$ROOT" HF_HOME="$HOME/hf_cache" PYTHONIOENCODING=utf-8
export TOKENIZERS_PARALLELISM=false
cd "$ROOT"

OUT=runs/ip2p_shading
mkdir -p "$OUT"

IMGS="task_attr_mod_color_11699 task_attr_mod_color_136767 task_attr_mod_color_184837 task_attr_mod_color_32648 task_attr_mod_color_6205 task_env_weather_112463 task_env_weather_246440 task_env_weather_63722 task_obj_add_40931 task_obj_remove_380621 task_obj_swap_joint_mask_276754 task_obj_swap_joint_mask_533428 task_obj_swap_rand_mask_417469"

COMMON="--data data/omniedit150 --loss latent_norm --steps 1000"

DEVS=(${1:-0 1})
RADII=(${2:-0.05 0.10 0.20})

require_slots() {
  local n_points="$1" n_devs="$2"
  if [ "$n_points" -gt $(( n_devs * 2 )) ]; then
    echo "錯誤：$n_points 個工作點需要至少 $(( (n_points + 1) / 2 )) 張卡，" >&2
    echo "      只給了 $n_devs 張。每卡最多 2 個 process。" >&2
    exit 2
  fi
}
# 每個半徑兩格：最佳化的與隨機的。
# **別人的卡一律不碰**。這一道是**強制**的，不是提醒——列印了卻不擋等於沒擋
# （實測踩過兩次：一次只讀 nvidia-smi 的前三行，一次等待器印了空卡清單卻沒有
# 依它決定要不要送）。任何一張指定的卡上有別人的 process 就直接拒絕啟動。
bash scripts/free_cards.sh --assert "${DEVS[*]}" || exit 3

require_slots "$(( ${#RADII[@]} * 2 ))" "${#DEVS[@]}"

i=0
for rad in "${RADII[@]}"; do
  for cond in shading shading_rand; do
    short=$([ "$cond" = shading ] && echo opt || echo rand)
    tag="${short}_r$(echo "$rad" | tr -d '.')"
    dev=${DEVS[$(( i / 2 % ${#DEVS[@]} ))]}
    i=$(( i + 1 ))
    CUDA_VISIBLE_DEVICES="$dev" nohup "$PY" scripts/ip2p_run.py \
        --out "$OUT/$tag" --conditions "$cond" --radius "$rad" \
        --images $IMGS $COMMON > "$OUT/$tag.log" 2>&1 &
    echo "[shading] $tag cond=$cond radius=$rad dev=$dev pid=$!"
  done
done
echo "[shading] 全部送出（$(date)），共 $i 個 process"

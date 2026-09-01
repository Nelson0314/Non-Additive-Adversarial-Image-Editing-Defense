#!/usr/bin/env bash
# 候選一 步驟 1／第三節 H1 的判定實驗：殘差區塊置換。
#
# `docs/reference/SURVEY_ARCHITECTURE.md` 第三節的 H1 是「DCT-Shield 的效果是
# 統計驅動的、本方法的是實現驅動的」——那是「它裁切保留 98.2%、我們 13%」目前
# 唯一的機制解釋，也是候選一（統計驅動的擾動）的前提。判定實驗：
#
#     取已存的防禦圖抽出殘差，做一次 32×32 區塊隨機置換後加回原圖，量效果。
#     預期 DCT-Shield 幾乎不掉、本方法掉光。若兩者都掉光，H1 錯。
#
# 兩種結果都有用：H1 若錯要照實寫，候選一的前提跟著垮。
#
# **這不是淨化算子**（它需要原圖才抽得出殘差，而淨化算子只拿得到防禦圖），
# 所以不進 `src/purify/ops.py`、不進頭對頭的淨化器清單，自己一支腳本。
#
# 用法：bash scripts/residual_permute.sh "<卡號>" ["<分片>"]
set -uo pipefail
ROOT=/nfs/home/nelson0314/WACV-s3
PY="$HOME/venvs/wacv/bin/python"
export PYTHONPATH="$ROOT" HF_HOME="$HOME/hf_cache" PYTHONIOENCODING=utf-8
export TOKENIZERS_PARALLELISM=false
cd "$ROOT"

OUT=runs/residual_permute
mkdir -p "$OUT"

COLOR="task_attr_mod_color_11699 task_attr_mod_color_136767 task_attr_mod_color_184837 task_attr_mod_color_32648 task_attr_mod_color_6205"
SCENE="task_env_weather_112463 task_env_weather_246440 task_env_weather_63722"
OBJECT="task_obj_add_40931 task_obj_remove_380621 task_obj_swap_joint_mask_276754 task_obj_swap_joint_mask_533428 task_obj_swap_rand_mask_417469"

# 兩個條件，就是 H1 要對比的兩邊。用與抗淨化頭對頭相同的兩個工作點，
# 這樣「置換後掉多少」與「裁切後掉多少」是在同一張防禦圖上量的。
SRC="
ours_add:runs/ip2p_axis_necessity/b_pg_r20
dct_e18:runs/ip2p_dct_band_extend/dct_e18
"

# 卡號由參數給，**不寫死**：卡是多人共用的。每張卡放兩個 process。
DEVS=(${1:-0 1 2})
SHARDS=(${2:-color scene object})
for sh in "${SHARDS[@]}"; do
  case "$sh" in color|scene|object) ;; *)
    echo "錯誤：分片名只能是 color / scene / object，收到 $sh" >&2; exit 2 ;;
  esac
done

require_slots() {
  local n_points="$1" n_devs="$2"
  if [ "$n_points" -gt $(( n_devs * 2 )) ]; then
    echo "錯誤：$n_points 個 process 需要至少 $(( (n_points + 1) / 2 )) 張卡，" >&2
    echo "      只給了 $n_devs 張。每卡最多 2 個 process。" >&2
    exit 2
  fi
}

# 防禦圖沒跑完的來源一律報錯，不跳過：少一邊這個實驗就沒有對比。
n_src=0
for s in $SRC; do
  IFS=: read -r tag run <<< "$s"
  n=$(ls -1 "$run"/*__def.png 2>/dev/null | wc -l)
  if [ "$n" -ne 13 ]; then
    echo "錯誤：$tag 的 $run 只有 $n/13 張防禦圖" >&2; exit 3
  fi
  n_src=$(( n_src + 1 ))
done
# **別人的卡一律不碰**。這一道是**強制**的，不是提醒——列印了卻不擋等於沒擋
# （實測踩過兩次：一次只讀 nvidia-smi 的前三行，一次等待器印了空卡清單卻沒有
# 依它決定要不要送）。任何一張指定的卡上有別人的 process 就直接拒絕啟動。
bash scripts/free_cards.sh --assert "${DEVS[*]}" || exit 3

require_slots "$(( n_src * ${#SHARDS[@]} ))" "${#DEVS[@]}"

i=0
for s in $SRC; do
  IFS=: read -r tag run <<< "$s"
  for shard in "${SHARDS[@]}"; do
    case $shard in
      color) imgs="$COLOR" ;; scene) imgs="$SCENE" ;; object) imgs="$OBJECT" ;;
    esac
    dev=${DEVS[$(( i / 2 % ${#DEVS[@]} ))]}
    i=$(( i + 1 ))
    CUDA_VISIBLE_DEVICES="$dev" nohup "$PY" scripts/residual_permute_probe.py \
        --run "$run" --tag "$tag" --data data/omniedit150 --images $imgs \
        --gallery "$OUT/gallery" \
        --out "$OUT/${tag}_${shard}.csv" > "$OUT/${tag}_${shard}.log" 2>&1 &
    echo "[permute] $tag/$shard run=$run dev=$dev pid=$!"
  done
done
echo "[permute] 全部送出（$(date)），共 $i 個 process"

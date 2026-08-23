#!/usr/bin/env bash
# 補跑 GrIDPure 那一格。
#
# 為什麼要分開跑：`DIFFPURE_CKPT` 只寫在 `~/env.sh` 裡，而本專案的驅動一律
# 不 source 它（那支的最後一行會把工作目錄換到舊的 `~/WACV`）。少了那個變數
# `Purifier.available` 會判 gridpure 為「相依不齊」而**靜默跳過**，整批只留
# 一行提示。旗標現已在各驅動裡明寫，這一支是把漏掉的那一格補回來。
#
# `--purifiers` 必須含 `identity`——它是 retention 的分母。補跑會讓 identity
# 出現兩次，`scripts/retention_table.py` 的 dedupe 負責只留一份。
#
# 用法：bash scripts/purify_gridpure.sh "0 1 2 3 4 5 6 7"
set -uo pipefail
ROOT=/nfs/home/nelson0314/WACV-s3
PY="$HOME/venvs/wacv/bin/python"
export PYTHONPATH="$ROOT" HF_HOME="$HOME/hf_cache" PYTHONIOENCODING=utf-8
export DIFFPURE_CKPT="$HOME/thirdparty/diffpure/256x256_diffusion_uncond.pt"
export TOKENIZERS_PARALLELISM=false
cd "$ROOT"

OUT=runs/ip2p_purify_headtohead/gridpure
mkdir -p "$OUT"

COLOR="task_attr_mod_color_11699 task_attr_mod_color_136767 task_attr_mod_color_184837 task_attr_mod_color_32648 task_attr_mod_color_6205"
SCENE="task_env_weather_112463 task_env_weather_246440 task_env_weather_63722"
OBJECT="task_obj_add_40931 task_obj_remove_380621 task_obj_swap_joint_mask_276754 task_obj_swap_joint_mask_533428 task_obj_swap_rand_mask_417469"
ALL="$COLOR $SCENE $OBJECT"

COMMON="--data data/omniedit150 --attacker ip2p --seeds 3 --purifiers identity gridpure"

SRC="
ours_nonadd:runs/ip2p_overlap_sweep/h08_r25
ours_add:runs/ip2p_axis_necessity/b_pg_r20
dct_e14:runs/ip2p_band_calibration/dct_e14
dct_e18:runs/ip2p_dct_band_extend/dct_e18
dct_y_e14:runs/ip2p_dct_band_extend/dct_y_e14
"

DEVS=(${1:-0 1 2 3 4 5 6 7})
i=0
launch() {
  local dev=${DEVS[$(( i / 2 % ${#DEVS[@]} ))]}
  i=$(( i + 1 ))
  CUDA_VISIBLE_DEVICES="$dev" nohup "$PY" scripts/phase_retention.py \
      --run "$2" --images $4 $COMMON $5 --out "$OUT/$1_$3.csv" \
      > "$OUT/$1_$3.log" 2>&1 &
  echo "[gridpure] $1/$3 dev=$dev pid=$!"
}

for s in $SRC; do
  IFS=: read -r tag run <<< "$s"
  [ -f "$run/results.csv" ] || { echo "[gridpure] 跳過 $tag"; continue; }
  launch "$tag" "$run" color  "$COLOR"  ""
  launch "$tag" "$run" scene  "$SCENE"  ""
  launch "$tag" "$run" object "$OBJECT" ""
done
# 空白地板與條件無關，只需一份——但**仍要分片**：不分片的話一個 process 要跑
# 十三張，而其餘每片只跑三到五張，扣地板的表會被這一格決定完成時間。
for shard in color scene object; do
  case $shard in
    color) imgs="$COLOR" ;; scene) imgs="$SCENE" ;; object) imgs="$OBJECT" ;;
  esac
  launch floor runs/ip2p_overlap_sweep/h08_r25 "$shard" "$imgs" "--floor"
done
echo "[gridpure] 全部送出（$(date)），共 $i 個 process"

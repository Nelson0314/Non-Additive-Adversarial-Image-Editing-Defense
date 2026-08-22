#!/usr/bin/env bash
# 抗淨化的頭對頭：兩個主線設定 對 DCT-Shield 的三個工作點。
#
# **只讀已存的防禦圖，不重跑防禦**（scripts/phase_retention.py）。
# 空白地板不可省略：淨化算子自己就會把編輯推開，不扣掉它，「淨化後位移較大」
# 無法排除「該算子本來就把編輯推得比較開」這個平庸解釋。
#
# 五個條件的失真並不相等（DCT-Shield 在 DISTS 上便宜得多），故頭對頭表要與
# matched_distortion_table.py 的等失真內插一起讀，不可單看絕對位移。
#
# 分片依任務族群切，不用流水號：color（5 張）、scene（3 張）、object（5 張）。
set -uo pipefail
ROOT=/nfs/home/nelson0314/WACV-s3
PY="$HOME/venvs/wacv/bin/python"
export PYTHONPATH="$ROOT" HF_HOME="$HOME/hf_cache" PYTHONIOENCODING=utf-8
cd "$ROOT"

OUT=runs/ip2p_purify_headtohead
mkdir -p "$OUT"

COLOR="task_attr_mod_color_11699 task_attr_mod_color_136767 task_attr_mod_color_184837 task_attr_mod_color_32648 task_attr_mod_color_6205"
SCENE="task_env_weather_112463 task_env_weather_246440 task_env_weather_63722"
OBJECT="task_obj_add_40931 task_obj_remove_380621 task_obj_swap_joint_mask_276754 task_obj_swap_joint_mask_533428 task_obj_swap_rand_mask_417469"
ALL="$COLOR $SCENE $OBJECT"

# 八個算子。impress 佔一格約 82% 的機時，另外補跑。
PUR="identity blur1 jpeg75 jpeg30 crop_resize0.1 jpeg_then_resize75 adverse_cleaner gridpure"
COMMON="--data data/omniedit150 --attacker ip2p --seeds 3 --purifiers $PUR"

# tag:防禦圖所在目錄
SRC="
ours_nonadd:runs/ip2p_overlap_sweep/h08_r25
ours_add:runs/ip2p_axis_necessity/b_pg_r20
dct_e14:runs/ip2p_band_calibration/dct_e14
dct_e18:runs/ip2p_dct_band_extend/dct_e18
dct_y_e14:runs/ip2p_dct_band_extend/dct_y_e14
"

i=0
launch() {   # $1 tag  $2 run  $3 shard  $4 imgs  $5 extra
  local dev=$(( i / 2 ))
  i=$(( i + 1 ))
  CUDA_VISIBLE_DEVICES="$dev" nohup "$PY" scripts/phase_retention.py \
      --run "$2" --images $4 $COMMON $5 --out "$OUT/$1_$3.csv" \
      > "$OUT/$1_$3.log" 2>&1 &
  echo "[purify] $1/$3 run=$2 dev=$dev pid=$!"
}

for s in $SRC; do
  IFS=: read -r tag run <<< "$s"
  if [ ! -f "$run/results.csv" ]; then
    echo "[purify] 跳過 $tag：$run/results.csv 不存在"; continue
  fi
  launch "$tag" "$run" color  "$COLOR"  ""
  launch "$tag" "$run" scene  "$SCENE"  ""
  launch "$tag" "$run" object "$OBJECT" ""
done

# 空白地板：防禦圖就是原圖，與條件無關，故整批只跑一次。
launch floor runs/ip2p_overlap_sweep/h08_r25 all "$ALL" "--floor"

echo "[purify] 全部送出（$(date)），共 $i 個 process"

#!/usr/bin/env bash
# 抗淨化的頭對頭：現行的兩個主線設定 對 DCT-Shield 的兩個變體。
#
# **只讀已存的防禦圖，不重跑防禦**（scripts/phase_retention.py）。
# 空白地板不可省略：淨化算子自己就會把編輯推開，不扣掉它「淨化後位移較大」
# 無法排除「該算子本來就把編輯推得比較開」這個平庸解釋。
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

# 七個算子。impress 佔一格約 82% 的機時，另外補跑。
PUR="identity blur1 jpeg75 jpeg30 crop_resize0.1 jpeg_then_resize75 adverse_cleaner gridpure"

COMMON="--data data/omniedit150 --attacker ip2p --seeds 3 --purifiers $PUR"

# tag:防禦圖所在目錄:device
SRC="
ours_nonadd:runs/ip2p_overlap_sweep/h08_r25:0
ours_add:runs/ip2p_axis_necessity/b_pg_r20:1
dct_e14:runs/ip2p_band_calibration/dct_e14:2
dct_e22:runs/ip2p_dct_band_extend/dct_e22:3
dct_y_e14:runs/ip2p_dct_band_extend/dct_y_e14:4
"

for s in $SRC; do
  IFS=: read -r tag run dev <<< "$s"
  if [ ! -f "$run/results.csv" ]; then
    echo "[purify] 跳過 $tag：$run/results.csv 還不存在"; continue
  fi
  for shard in color scene object; do
    case $shard in
      color) imgs="$COLOR" ;; scene) imgs="$SCENE" ;; object) imgs="$OBJECT" ;;
    esac
    CUDA_VISIBLE_DEVICES="$dev" nohup "$PY" scripts/phase_retention.py \
        --run "$run" --images $imgs $COMMON \
        --out "$OUT/${tag}_${shard}.csv" \
        > "$OUT/${tag}_${shard}.log" 2>&1 &
    echo "[purify] $tag/$shard run=$run dev=$dev pid=$!"
  done
done

# 空白地板：防禦圖就是原圖，與條件無關，故只跑一次。
for shard in color scene object; do
  case $shard in
    color) imgs="$COLOR"; dev=5 ;; scene) imgs="$SCENE"; dev=6 ;;
    object) imgs="$OBJECT"; dev=7 ;;
  esac
  CUDA_VISIBLE_DEVICES="$dev" nohup "$PY" scripts/phase_retention.py \
      --run runs/ip2p_overlap_sweep/h08_r25 --floor --images $imgs $COMMON \
      --out "$OUT/floor_${shard}.csv" > "$OUT/floor_${shard}.log" 2>&1 &
  echo "[purify] floor/$shard dev=$dev pid=$!"
done
echo "[purify] 全部送出（$(date)）"

#!/usr/bin/env bash
# 把 DCT-Shield 的取捨曲線延伸到本方法所在的失真區間。
#
# 現有的 13 張曲線只到 eps 1.4（DISTS 0.0771），而本方法的兩個主線設定落在
# DISTS 0.138–0.144。抗淨化要在**等失真**上比，沒有那一段就只能外插，而協定
# 明文拒絕外插。Y-only 那一支同理：既有的只有 eps 2.0／3.0，兩者的 PSNR
# （19.3／16.8）都遠在失真帶之外。
#
# `--q-alg` 一律取 0.95（論文 §5.4 的編輯任務設定），與既有曲線相同——
# 兩支之間唯一的差別因此是通道集合而不是品質因子（FND-058）。
set -uo pipefail
ROOT=/nfs/home/nelson0314/WACV-s3
PY="$HOME/venvs/wacv/bin/python"
export PYTHONPATH="$ROOT" HF_HOME="$HOME/hf_cache" PYTHONIOENCODING=utf-8
cd "$ROOT"

OUT=runs/ip2p_dct_band_extend
mkdir -p "$OUT"

IMGS="task_attr_mod_color_11699 task_attr_mod_color_136767 task_attr_mod_color_184837 task_attr_mod_color_32648 task_attr_mod_color_6205 task_env_weather_112463 task_env_weather_246440 task_env_weather_63722 task_obj_add_40931 task_obj_remove_380621 task_obj_swap_joint_mask_276754 task_obj_swap_joint_mask_533428 task_obj_swap_rand_mask_417469"

# tag:condition:eps:device
POINTS="
dct_e18:dct_shield:1.8:6
dct_e22:dct_shield:2.2:6
dct_y_e10:dct_shield_y:1.0:7
dct_y_e14:dct_shield_y:1.4:7
"

for p in $POINTS; do
  IFS=: read -r tag cond eps dev <<< "$p"
  CUDA_VISIBLE_DEVICES="$dev" nohup "$PY" scripts/ip2p_run.py \
      --out "$OUT/$tag" --conditions "$cond" --eps "$eps" --q-alg 0.95 \
      --dct-steps 1000 --data data/omniedit150 --images $IMGS \
      > "$OUT/$tag.log" 2>&1 &
  echo "[dct] $tag cond=$cond eps=$eps dev=$dev pid=$!"
done
echo "[dct] 全部送出（$(date)）"

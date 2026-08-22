#!/usr/bin/env bash
# 加法項的價目分配：均勻 對 內容相依。
#
# 含加性下限那個主線設定的加法項，價目只看頻格、跨區塊是常數——**那正是
# DCT-Shield 的形狀**（逐係數 eps·Q(w)）。這一批問的是：把同樣的預算改成
# 內容相依的分配，防禦強度會不會掉。
#
#   uniform     現行（`b_pg_*`，由 axis_necessity.sh 跑）
#   complement  只花在乘法那一半可達量最少的區塊上
#   watson      亮度遮蔽 × 對比遮蔽（Watson 1993 / Podilchuk & Zeng 1998）
#
# 三者的**總預算相同**（價目表被正規化到同一個平均值），故差異是分配而不是
# 強度。半徑仍要掃，等失真的比較由 tradeoff_curve.py 內插。
set -uo pipefail
ROOT=/nfs/home/nelson0314/WACV-s3
PY="$HOME/venvs/wacv/bin/python"
export PYTHONPATH="$ROOT" HF_HOME="$HOME/hf_cache" PYTHONIOENCODING=utf-8
cd "$ROOT"

OUT=runs/ip2p_floor_gate
mkdir -p "$OUT"

IMGS="task_attr_mod_color_11699 task_attr_mod_color_136767 task_attr_mod_color_184837 task_attr_mod_color_32648 task_attr_mod_color_6205 task_env_weather_112463 task_env_weather_246440 task_env_weather_63722 task_obj_add_40931 task_obj_remove_380621 task_obj_swap_joint_mask_276754 task_obj_swap_joint_mask_533428 task_obj_swap_rand_mask_417469"

COMMON="--data data/omniedit150 --conditions phase_gain --loss latent_norm --steps 1000 --quantile 0 --freq-weight jpeg_luma --freq-weight-power 0.25 --hop 8 --gain-ratio 1.0 --spectral-floor 0.04"

# tag:floor_gate:radius:device
POINTS="
comp_r09:complement:0.9:0
comp_r15:complement:1.5:0
comp_r20:complement:2.0:1
comp_r24:complement:2.4:1
wat_r09:watson:0.9:2
wat_r15:watson:1.5:2
wat_r20:watson:2.0:3
wat_r24:watson:2.4:3
"

for p in $POINTS; do
  IFS=: read -r tag fg rad dev <<< "$p"
  CUDA_VISIBLE_DEVICES="$dev" nohup "$PY" scripts/ip2p_run.py \
      --out "$OUT/$tag" --floor-gate "$fg" --radius "$rad" \
      --images $IMGS $COMMON > "$OUT/$tag.log" 2>&1 &
  echo "[floor_gate] $tag gate=$fg radius=$rad dev=$dev pid=$!"
done
echo "[floor_gate] 全部送出（$(date)）"

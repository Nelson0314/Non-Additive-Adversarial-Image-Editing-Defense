#!/usr/bin/env bash
# 相位軸與幅度軸各自是否必要——等失真三方對照。
#
# 為什麼要這一批：可學幅度增益是在**舊設定**（encoder_target 損失、二值頻率
# 閘、100 步）上定案的，等失真 margin 只有 6.1%。現行設定（latent_norm、
# jpeg_luma 定價 0.25、hop 8、1000 步）下從來沒有 gain_ratio = 0 的對照，
# 也從來沒有「凍結相位、只學幅度」的對照。
#
#   phase       純相位（gain_max = 0，theta_max 封頂在 pi）
#   gain_only   凍結相位、只學幅度（上界不封頂）
#   phase_gain  兩者（現行主線）
#
# 每個條件掃三到四個半徑，之後由 scripts/tradeoff_curve.py 在等失真處內插。
# **不可在固定半徑下比**——半徑只是強度軸。
#
# 兩組：a 組是非加性主線（--spectral-floor 0），b 組是含加性下限主線
# （--spectral-floor 0.04；加法項的強度不隨半徑變，故半徑仍只驅動乘法那半）。
set -uo pipefail
ROOT=/nfs/home/nelson0314/WACV-s3
PY="$HOME/venvs/wacv/bin/python"
export PYTHONPATH="$ROOT" HF_HOME="$HOME/hf_cache" PYTHONIOENCODING=utf-8
cd "$ROOT"

OUT=runs/ip2p_axis_necessity
mkdir -p "$OUT"

IMGS="task_attr_mod_color_11699 task_attr_mod_color_136767 task_attr_mod_color_184837 task_attr_mod_color_32648 task_attr_mod_color_6205 task_env_weather_112463 task_env_weather_246440 task_env_weather_63722 task_obj_add_40931 task_obj_remove_380621 task_obj_swap_joint_mask_276754 task_obj_swap_joint_mask_533428 task_obj_swap_rand_mask_417469"

COMMON="--data data/omniedit150 --loss latent_norm --steps 1000 --quantile 0 --freq-weight jpeg_luma --freq-weight-power 0.25 --hop 8"

# tag:condition:radius:floor:device
# 純相位不用增益，故 --gain-ratio 傳 0（傳 1.0 會在 CSV 上記下一個沒有作用的值）。
POINTS_A="
a_ph_r12:phase:1.2:0:0
a_ph_r22:phase:2.2:0:0
a_ph_rpi:phase:3.1416:0:1
a_gn_r09:gain_only:0.9:0:1
a_gn_r16:gain_only:1.6:0:2
a_gn_r26:gain_only:2.6:0:2
a_pg_r11:phase_gain:1.1:0:3
a_pg_r18:phase_gain:1.8:0:3
a_pg_r25:phase_gain:2.5:0:4
a_pg_r30:phase_gain:3.0:0:4
b_ph_r08:phase:0.8:0.04:5
b_ph_r18:phase:1.8:0.04:5
b_ph_rpi:phase:3.1416:0.04:6
b_gn_r08:gain_only:0.8:0.04:6
b_gn_r15:gain_only:1.5:0.04:7
b_gn_r24:gain_only:2.4:0.04:7
"
POINTS_B="
b_pg_r09:phase_gain:0.9:0.04:4
b_pg_r15:phase_gain:1.5:0.04:4
b_pg_r20:phase_gain:2.0:0.04:5
b_pg_r24:phase_gain:2.4:0.04:5
"

case "${1:-}" in
  two) POINTS="$POINTS_A" ;;   # basic-2，八張卡各兩個
  one) POINTS="$POINTS_B" ;;   # basic-1，卡 4/5 各兩個
  *) echo "用法：$0 one|two"; exit 2 ;;
esac

for p in $POINTS; do
  IFS=: read -r tag cond rad floor dev <<< "$p"
  gr=1.0
  [ "$cond" = "phase" ] && gr=0
  CUDA_VISIBLE_DEVICES="$dev" nohup "$PY" scripts/ip2p_run.py \
      --out "$OUT/$tag" --conditions "$cond" --radius "$rad" \
      --spectral-floor "$floor" --gain-ratio "$gr" --images $IMGS $COMMON \
      > "$OUT/$tag.log" 2>&1 &
  echo "[axis] $tag cond=$cond radius=$rad floor=$floor gain_ratio=$gr dev=$dev pid=$!"
done
echo "[axis] $1 全部送出（$(date)）"

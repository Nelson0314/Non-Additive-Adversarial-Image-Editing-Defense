#!/usr/bin/env bash
# 只有加性下限：相位與幅度都不動。
#
# 這是整個加性裁決底下唯一沒跑過的對照。`DECISIONS.md` 撤回「頻譜加性項不做」
# 時，站得住的唯一理由是「非加性那一半買的是**感知代價**——等 DISTS 下純加性
# 擋下率一樣好甚至略好，但 PSNR 差 2.4–3.2 dB」，而那句話的證據是一次性探針裡
# `radius 0.1` 的近似（theta_max 與 gain_max 都還有 0.1，不是真的關掉），
# 程式已刪除、hop 還是舊的 16。
#
# 今日又量到：`--spectral-floor 0.04` 時加法項佔可用預算的 **67.6%**，而它把
# 逐區塊的空間集中度由 0.531 壓到 0.163（DCT-Shield 由構造為 0）。
# 「乘法那一半」在字面上是主體，在預算上不是。
#
# **強度旗鈕是 `--spectral-floor` 不是 radius**：radius 在這一格上完全沒有
# 作用。分析時要給 `matched_distortion_table.py --strength spectral_floor`。
#
# 用法：bash scripts/floor_only_sweep.sh "0 1 2"
set -uo pipefail
ROOT=/nfs/home/nelson0314/WACV-s3
PY="$HOME/venvs/wacv/bin/python"
export PYTHONPATH="$ROOT" HF_HOME="$HOME/hf_cache" PYTHONIOENCODING=utf-8
export DIFFPURE_CKPT="$HOME/thirdparty/diffpure/256x256_diffusion_uncond.pt"
export TOKENIZERS_PARALLELISM=false
cd "$ROOT"

OUT=runs/ip2p_floor_only
mkdir -p "$OUT"

IMGS="task_attr_mod_color_11699 task_attr_mod_color_136767 task_attr_mod_color_184837 task_attr_mod_color_32648 task_attr_mod_color_6205 task_env_weather_112463 task_env_weather_246440 task_env_weather_63722 task_obj_add_40931 task_obj_remove_380621 task_obj_swap_joint_mask_276754 task_obj_swap_joint_mask_533428 task_obj_swap_rand_mask_417469"

# radius 傳 2.0 只是為了讓 CLI 有值；它對 floor_only 沒有作用。
COMMON="--data data/omniedit150 --conditions floor_only --gain-ratio 0 --radius 2.0 --loss latent_norm --steps 1000 --quantile 0 --freq-weight jpeg_luma --freq-weight-power 0.25 --hop 8"

DEVS=(${1:-0 1 2})

# tag:spectral_floor
POINTS="
f0020:0.020
f0040:0.040
f0060:0.060
f0090:0.090
f0140:0.140
f0200:0.200
"

i=0
for p in $POINTS; do
  IFS=: read -r tag fl <<< "$p"
  dev=${DEVS[$(( i / 2 % ${#DEVS[@]} ))]}
  i=$(( i + 1 ))
  CUDA_VISIBLE_DEVICES="$dev" nohup "$PY" scripts/ip2p_run.py \
      --out "$OUT/$tag" --spectral-floor "$fl" \
      --images $IMGS $COMMON > "$OUT/$tag.log" 2>&1 &
  echo "[floor_only] $tag spectral_floor=$fl dev=$dev pid=$!"
done
echo "[floor_only] 全部送出（$(date)）"

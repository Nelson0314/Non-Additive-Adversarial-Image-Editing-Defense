#!/usr/bin/env bash
# 知覺定價的力道 `--freq-weight-power`：0.35 在帶內優於 0.25，等失真對照未做。
#
# `docs/RESULTS.md`「知覺加權在等失真下成立，但定價力道 0.25 不是最佳」：
# gamma 0.35 在幾乎相同的失真上拿到 8/13 而 0.25 是 6/13，該頁自己註明
# 「應改為 0.35，或先補一組等失真的 gamma 掃描再定」。這是後者。
#
# 掃在「純相位＋加性下限」上（目前帶內最好的工作點）。
# 用法：bash scripts/pricing_power_sweep.sh "3 4"
set -uo pipefail
ROOT=/nfs/home/nelson0314/WACV-s3
PY="$HOME/venvs/wacv/bin/python"
export PYTHONPATH="$ROOT" HF_HOME="$HOME/hf_cache" PYTHONIOENCODING=utf-8
export DIFFPURE_CKPT="$HOME/thirdparty/diffpure/256x256_diffusion_uncond.pt"
export TOKENIZERS_PARALLELISM=false
cd "$ROOT"

OUT=runs/ip2p_pricing_power
mkdir -p "$OUT"

IMGS="task_attr_mod_color_11699 task_attr_mod_color_136767 task_attr_mod_color_184837 task_attr_mod_color_32648 task_attr_mod_color_6205 task_env_weather_112463 task_env_weather_246440 task_env_weather_63722 task_obj_add_40931 task_obj_remove_380621 task_obj_swap_joint_mask_276754 task_obj_swap_joint_mask_533428 task_obj_swap_rand_mask_417469"

COMMON="--data data/omniedit150 --conditions phase --gain-ratio 0 --spectral-floor 0.04 --loss latent_norm --steps 1000 --quantile 0 --freq-weight jpeg_luma --hop 8"

DEVS=(${1:-3 4})

# 每卡最多兩個 process（`docs/OPERATIONS.md`：每個約 9 GB，3090 是 24 GB，
# 而卡是多人共用的）。**點數超過 卡數×2 時必須拒絕**，不可讓卡號公式繞回去
# 疊加——實測 `i/2 % 卡數` 在 6 點配 2 卡時把 4 個 process 塞到同一張卡上，
# 那一批六點掛掉四點，全部是 CUDA OOM。
require_slots() {
  local n_points="$1" n_devs="$2"
  if [ "$n_points" -gt $(( n_devs * 2 )) ]; then
    echo "錯誤：$n_points 個工作點需要至少 $(( (n_points + 1) / 2 )) 張卡，" >&2
    echo "      只給了 $n_devs 張。每卡最多 2 個 process。" >&2
    exit 2
  fi
}


# tag:freq_weight_power:radius
POINTS="
g015_rpi:0.15:3.1416
g035_r18:0.35:1.8
g035_rpi:0.35:3.1416
g050_rpi:0.50:3.1416
"

require_slots "$(echo "$POINTS" | grep -c .)" "${#DEVS[@]}"
i=0
for p in $POINTS; do
  IFS=: read -r tag gam rad <<< "$p"
  dev=${DEVS[$(( i / 2 % ${#DEVS[@]} ))]}
  i=$(( i + 1 ))
  CUDA_VISIBLE_DEVICES="$dev" nohup "$PY" scripts/ip2p_run.py \
      --out "$OUT/$tag" --freq-weight-power "$gam" --radius "$rad" \
      --images $IMGS $COMMON > "$OUT/$tag.log" 2>&1 &
  echo "[pricing] $tag gamma=$gam radius=$rad dev=$dev pid=$!"
done
echo "[pricing] 全部送出（$(date)）"

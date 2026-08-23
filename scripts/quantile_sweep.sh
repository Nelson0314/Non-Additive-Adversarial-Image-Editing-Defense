#!/usr/bin/env bash
# 紋理閘的能量分位數：空間選擇性本身值多少。
#
# `--quantile` 決定紋理閘裡「梯度能量 / 參考值」那個因子的參考分位數。
# 定案值 0 使該因子恆為 1，也就是**只剩邊緣因子**——空間選擇性目前幾乎沒有
# 被用上，而它正是本方法對 DCT-Shield 的主要構造差異（DCT-Shield 的預算是
# 逐係數的 eps·Q，跨區塊是常數；殘差指紋量到 block_gini 0.10 對本方法 0.43）。
# 這個旗標從未單獨量過。
#
# 掃在**含加性下限、純相位**那個設定上：加法項本來就負責平坦區，把乘法那半
# 的閘再收緊一點，兩條支路的分工會更乾淨。等失真的比較由
# matched_distortion_table.py 內插，對照組是同設定 quantile 0（b_ph_*）。
#
# 用法：bash scripts/quantile_sweep.sh "0 1 2"
set -uo pipefail
ROOT=/nfs/home/nelson0314/WACV-s3
PY="$HOME/venvs/wacv/bin/python"
export PYTHONPATH="$ROOT" HF_HOME="$HOME/hf_cache" PYTHONIOENCODING=utf-8
export DIFFPURE_CKPT="$HOME/thirdparty/diffpure/256x256_diffusion_uncond.pt"
export TOKENIZERS_PARALLELISM=false
cd "$ROOT"

OUT=runs/ip2p_quantile_sweep
mkdir -p "$OUT"

IMGS="task_attr_mod_color_11699 task_attr_mod_color_136767 task_attr_mod_color_184837 task_attr_mod_color_32648 task_attr_mod_color_6205 task_env_weather_112463 task_env_weather_246440 task_env_weather_63722 task_obj_add_40931 task_obj_remove_380621 task_obj_swap_joint_mask_276754 task_obj_swap_joint_mask_533428 task_obj_swap_rand_mask_417469"

COMMON="--data data/omniedit150 --conditions phase --gain-ratio 0 --spectral-floor 0.04 --loss latent_norm --steps 1000 --freq-weight jpeg_luma --freq-weight-power 0.25 --hop 8"

DEVS=(${1:-0 1 2})

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


# tag:quantile:radius
POINTS="
q30_r10:0.3:1.0
q30_r20:0.3:2.0
q30_rpi:0.3:3.1416
q60_r10:0.6:1.0
q60_r20:0.6:2.0
q60_rpi:0.6:3.1416
"

require_slots "$(echo "$POINTS" | grep -c .)" "${#DEVS[@]}"
i=0
for p in $POINTS; do
  IFS=: read -r tag q rad <<< "$p"
  dev=${DEVS[$(( i / 2 % ${#DEVS[@]} ))]}
  i=$(( i + 1 ))
  CUDA_VISIBLE_DEVICES="$dev" nohup "$PY" scripts/ip2p_run.py \
      --out "$OUT/$tag" --quantile "$q" --radius "$rad" \
      --images $IMGS $COMMON > "$OUT/$tag.log" 2>&1 &
  echo "[quantile] $tag quantile=$q radius=$rad dev=$dev pid=$!"
done
echo "[quantile] 全部送出（$(date)）"

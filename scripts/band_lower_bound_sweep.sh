#!/usr/bin/env bash
# 徑向頻率閘的下界 r_min：一個已記錄但從未被採用的效率增益。
#
# `docs/RESULTS.md` 的 FND-042 寫得很明確：**拉高頻率下限在四個指標上一致
# 更好**，而定案值保留 0.12 的唯一理由是「它是既有批次的基準」。那批既有
# 批次早就被換掉了（損失、頻率閘的知覺定價、hop、加性下限全部改過），這個
# 理由已經不成立。
#
# 另一個指向同一處的讀數是殘差指紋：DCT-Shield 把 95% 的殘差能量放在半
# Nyquist 以上，本方法只有 60%（`runs/ip2p_residual_signature`）。而它在等
# 失真上贏，正是因為高頻對 DISTS 便宜、對 VAE encoder 仍然有效
# （`encoder_frequency_response` 量到梯度能量到 r=0.88 都還有 0.08）。
#
# 掃在「純相位＋加性下限」上，因為那是目前帶內最好的工作點。
# 用法：bash scripts/band_lower_bound_sweep.sh "0 1 2"
set -uo pipefail
ROOT=/nfs/home/nelson0314/WACV-s3
PY="$HOME/venvs/wacv/bin/python"
export PYTHONPATH="$ROOT" HF_HOME="$HOME/hf_cache" PYTHONIOENCODING=utf-8
export DIFFPURE_CKPT="$HOME/thirdparty/diffpure/256x256_diffusion_uncond.pt"
export TOKENIZERS_PARALLELISM=false
cd "$ROOT"

OUT=runs/ip2p_band_lower_bound
mkdir -p "$OUT"

IMGS="task_attr_mod_color_11699 task_attr_mod_color_136767 task_attr_mod_color_184837 task_attr_mod_color_32648 task_attr_mod_color_6205 task_env_weather_112463 task_env_weather_246440 task_env_weather_63722 task_obj_add_40931 task_obj_remove_380621 task_obj_swap_joint_mask_276754 task_obj_swap_joint_mask_533428 task_obj_swap_rand_mask_417469"

COMMON="--data data/omniedit150 --conditions phase --gain-ratio 0 --spectral-floor 0.04 --loss latent_norm --steps 1000 --quantile 0 --freq-weight jpeg_luma --freq-weight-power 0.25 --hop 8"

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


# tag:r_min:radius
POINTS="
rm20_r18:0.20:1.8
rm20_rpi:0.20:3.1416
rm30_r18:0.30:1.8
rm30_rpi:0.30:3.1416
rm40_rpi:0.40:3.1416
"

# **別人的卡一律不碰**。這一道是**強制**的，不是提醒——列印了卻不擋等於沒擋
# （實測踩過兩次：一次只讀 nvidia-smi 的前三行，一次等待器印了空卡清單卻沒有
# 依它決定要不要送）。任何一張指定的卡上有別人的 process 就直接拒絕啟動。
bash scripts/free_cards.sh --assert "${DEVS[*]}" || exit 3

require_slots "$(echo "$POINTS" | grep -c .)" "${#DEVS[@]}"
i=0
for p in $POINTS; do
  IFS=: read -r tag rmin rad <<< "$p"
  dev=${DEVS[$(( i / 2 % ${#DEVS[@]} ))]}
  i=$(( i + 1 ))
  CUDA_VISIBLE_DEVICES="$dev" nohup "$PY" scripts/ip2p_run.py \
      --out "$OUT/$tag" --r-min "$rmin" --radius "$rad" \
      --images $IMGS $COMMON > "$OUT/$tag.log" 2>&1 &
  echo "[r_min] $tag r_min=$rmin radius=$rad dev=$dev pid=$!"
done
echo "[r_min] 全部送出（$(date)）"

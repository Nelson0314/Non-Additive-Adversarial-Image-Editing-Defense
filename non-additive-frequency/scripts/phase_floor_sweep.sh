#!/usr/bin/env bash
# 純相位 ＋ 加性下限：把強度旋鈕由半徑換成 floor。
#
# 為什麼：相位封頂在 pi，所以 `phase` 這一支的半徑掃到 3.1416 就到頂了，
# 失真也就停在那裡。加法項的強度**不隨半徑變**（`spectral_floor` 是獨立的
# 係數），故要把純相位那一支推到失真帶的上緣，只能推 floor。
#
# 這一支回答的是：加法下限開著的時候，可學幅度增益還剩多少貢獻。若純相位
# ＋下限在等失真上追得平「相位＋增益＋下限」，那麼增益可以拿掉，而拿掉之後
# 乘法那一半重新變回「幅度譜逐位保留」，新穎性主張也跟著收回一格。
set -uo pipefail
ROOT=/nfs/home/nelson0314/WACV-s3
PY="$HOME/venvs/wacv/bin/python"
export PYTHONPATH="$ROOT" HF_HOME="$HOME/hf_cache" PYTHONIOENCODING=utf-8
# 不 source ~/env.sh：它最後一行會把工作目錄換到舊的 ~/WACV（坑一）。
# 但 DIFFPURE_CKPT 只寫在那裡，少了它 gridpure／fdpure 會被判為
# 「相依不齊」而**靜默跳過**，報表上只剩一行提示。
export DIFFPURE_CKPT="$HOME/thirdparty/diffpure/256x256_diffusion_uncond.pt"
export TOKENIZERS_PARALLELISM=false
cd "$ROOT"

OUT=runs/ip2p_phase_floor
mkdir -p "$OUT"

IMGS="task_attr_mod_color_11699 task_attr_mod_color_136767 task_attr_mod_color_184837 task_attr_mod_color_32648 task_attr_mod_color_6205 task_env_weather_112463 task_env_weather_246440 task_env_weather_63722 task_obj_add_40931 task_obj_remove_380621 task_obj_swap_joint_mask_276754 task_obj_swap_joint_mask_533428 task_obj_swap_rand_mask_417469"

COMMON="--data data/omniedit150 --conditions phase --gain-ratio 0 --radius 3.1416 --loss latent_norm --steps 1000 --quantile 0 --freq-weight jpeg_luma --freq-weight-power 0.25 --hop 8"

DEVS=(${1:-0 1})

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


# tag:spectral_floor
POINTS="
f050:0.05
f060:0.06
f080:0.08
f100:0.10
"

# **別人的卡一律不碰**。這一道是**強制**的，不是提醒——列印了卻不擋等於沒擋
# （實測踩過兩次：一次只讀 nvidia-smi 的前三行，一次等待器印了空卡清單卻沒有
# 依它決定要不要送）。任何一張指定的卡上有別人的 process 就直接拒絕啟動。
bash scripts/free_cards.sh --assert "${DEVS[*]}" || exit 3

require_slots "$(echo "$POINTS" | grep -c .)" "${#DEVS[@]}"
i=0
for p in $POINTS; do
  IFS=: read -r tag fl <<< "$p"
  dev=${DEVS[$(( i / 2 % ${#DEVS[@]} ))]}
  i=$(( i + 1 ))
  CUDA_VISIBLE_DEVICES="$dev" nohup "$PY" scripts/ip2p_run.py \
      --out "$OUT/$tag" --spectral-floor "$fl" --images $IMGS $COMMON \
      > "$OUT/$tag.log" 2>&1 &
  echo "[phase_floor] $tag floor=$fl dev=$dev pid=$!"
done
echo "[phase_floor] 全部送出（$(date)）"

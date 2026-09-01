#!/usr/bin/env bash
# 把「只學幅度」那一支的掃描延伸到主線工作點所在的失真。
#
# 為什麼：`phase` 的失真有**構造上的**天花板（theta 封頂在 pi），掃到 3.1416
# 就到頂，所以它在 DISTS 0.1377 上回報「範圍外」是正確答案而不是掃太少。
# `gain_only` 沒有那條天花板——增益不是週期量——所以它在 0.1006 就停住只是
# 掃描不夠遠。不補這幾點，等失真的三方對照在主線工作點上少一格。
#
# 用法：bash scripts/gain_reach_extension.sh "4 5 6 7"
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

OUT=runs/ip2p_axis_necessity
mkdir -p "$OUT"

IMGS="task_attr_mod_color_11699 task_attr_mod_color_136767 task_attr_mod_color_184837 task_attr_mod_color_32648 task_attr_mod_color_6205 task_env_weather_112463 task_env_weather_246440 task_env_weather_63722 task_obj_add_40931 task_obj_remove_380621 task_obj_swap_joint_mask_276754 task_obj_swap_joint_mask_533428 task_obj_swap_rand_mask_417469"

COMMON="--data data/omniedit150 --conditions gain_only --gain-ratio 1.0 --loss latent_norm --steps 1000 --quantile 0 --freq-weight jpeg_luma --freq-weight-power 0.25 --hop 8"

DEVS=(${1:-0 1 2 3})

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


# tag:radius:floor
POINTS="
a_gn_r36:3.6:0
a_gn_r46:4.6:0
b_gn_r32:3.2:0.04
b_gn_r40:4.0:0.04
"

# **別人的卡一律不碰**。這一道是**強制**的，不是提醒——列印了卻不擋等於沒擋
# （實測踩過兩次：一次只讀 nvidia-smi 的前三行，一次等待器印了空卡清單卻沒有
# 依它決定要不要送）。任何一張指定的卡上有別人的 process 就直接拒絕啟動。
bash scripts/free_cards.sh --assert "${DEVS[*]}" || exit 3

require_slots "$(echo "$POINTS" | grep -c .)" "${#DEVS[@]}"
i=0
for p in $POINTS; do
  IFS=: read -r tag rad floor <<< "$p"
  dev=${DEVS[$(( i % ${#DEVS[@]} ))]}
  i=$(( i + 1 ))
  CUDA_VISIBLE_DEVICES="$dev" nohup "$PY" scripts/ip2p_run.py \
      --out "$OUT/$tag" --radius "$rad" --spectral-floor "$floor" \
      --images $IMGS $COMMON > "$OUT/$tag.log" 2>&1 &
  echo "[gain_reach] $tag radius=$rad floor=$floor dev=$dev pid=$!"
done
echo "[gain_reach] 全部送出（$(date)）"

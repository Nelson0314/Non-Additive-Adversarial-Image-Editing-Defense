#!/usr/bin/env bash
# 隨機化的幾何 EOT：唯一有機會改善裁切那一格的東西。
#
# 量測的依據（`runs/ip2p_residual_signature/band_transfer.csv`）：裁切縮放
# 留下 51–99% 的殘差能量，對**原網格**的餘弦是 0.000，對**算子自己搬過的
# 同一擾動**卻是 0.995–0.996。擾動原封不動地通過了，只是被搬到別的位置與
# 尺度上。所以問題是**對位**，不是能量，也不是頻帶——沒有任何一帶的方向
# 存活率高於 0.02，選頻帶救不了它。
#
# `--purify-aware eot_ops` 裡本來就有裁切，但那是**固定**的中心 0.10；固定的
# 變換會被 co-adapt，不產生對一族幾何的不變性。`eot_geometry` 每步抽比例
# **與位置**，評測用的 0.10 落在族內。
#
# 半徑取與 `runs/ip2p_axis_necessity/b_ph_*` 相同的三點，好做等失真對照。
#
# 用法：bash scripts/eot_geometry_sweep.sh "4 5"
set -uo pipefail
ROOT=/nfs/home/nelson0314/WACV-s3
PY="$HOME/venvs/wacv/bin/python"
export PYTHONPATH="$ROOT" HF_HOME="$HOME/hf_cache" PYTHONIOENCODING=utf-8
export DIFFPURE_CKPT="$HOME/thirdparty/diffpure/256x256_diffusion_uncond.pt"
export TOKENIZERS_PARALLELISM=false
cd "$ROOT"

OUT=runs/ip2p_eot_geometry
mkdir -p "$OUT"

IMGS="task_attr_mod_color_11699 task_attr_mod_color_136767 task_attr_mod_color_184837 task_attr_mod_color_32648 task_attr_mod_color_6205 task_env_weather_112463 task_env_weather_246440 task_env_weather_63722 task_obj_add_40931 task_obj_remove_380621 task_obj_swap_joint_mask_276754 task_obj_swap_joint_mask_533428 task_obj_swap_rand_mask_417469"

COMMON="--data data/omniedit150 --conditions phase --gain-ratio 0 --spectral-floor 0.04 --loss latent_norm --steps 1000 --quantile 0 --freq-weight jpeg_luma --freq-weight-power 0.25 --hop 8 --purify-aware eot_geometry"

DEVS=(${1:-4 5})

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


POINTS="g_r08:0.8 g_r18:1.8 g_rpi:3.1416"

# **別人的卡一律不碰**。這一道是**強制**的，不是提醒——列印了卻不擋等於沒擋
# （實測踩過兩次：一次只讀 nvidia-smi 的前三行，一次等待器印了空卡清單卻沒有
# 依它決定要不要送）。任何一張指定的卡上有別人的 process 就直接拒絕啟動。
bash scripts/free_cards.sh --assert "${DEVS[*]}" || exit 3

require_slots "$(echo "$POINTS" | grep -c .)" "${#DEVS[@]}"
i=0
for p in $POINTS; do
  IFS=: read -r tag rad <<< "$p"
  dev=${DEVS[$(( i % ${#DEVS[@]} ))]}
  i=$(( i + 1 ))
  CUDA_VISIBLE_DEVICES="$dev" nohup "$PY" scripts/ip2p_run.py \
      --out "$OUT/$tag" --radius "$rad" --images $IMGS $COMMON \
      > "$OUT/$tag.log" 2>&1 &
  echo "[eot_geometry] $tag radius=$rad dev=$dev pid=$!"
done
echo "[eot_geometry] 全部送出（$(date)）"

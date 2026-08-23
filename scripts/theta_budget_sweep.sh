#!/usr/bin/env bash
# 幅度相依的相位上限：文獻已有的解法，本專案從未實作。
#
# `docs/reference/SURVEY_PHASE_PRIORART.md` §2.2 把它列為「本輪查證最有操作
# 價值的一項」：Perturbing the Phase（arXiv:2602.06577）用一條閉式約束
#
#     |theta| <= 2·arcsin( eps / (2|X|) )
#
# 讓相位的可動範圍隨局部幅度反比縮放，於是像素域的位移被 eps 界住。它處理的
# 正是 FND-038——**固定的 theta 不等於固定的失真**，同一個 theta 在 24 張圖上
# PSNR 由 23.15 漂到 39.54。
#
# 掃法：半徑固定在 pi（讓上限而不是半徑當強度旗鈕），掃 eps。對照組是同設定
# 的 `runs/ip2p_axis_necessity/b_ph_*`（budget 關閉、半徑當旗鈕）。
#
# 用法：bash scripts/theta_budget_sweep.sh "0 1 2"
set -uo pipefail
ROOT=/nfs/home/nelson0314/WACV-s3
PY="$HOME/venvs/wacv/bin/python"
export PYTHONPATH="$ROOT" HF_HOME="$HOME/hf_cache" PYTHONIOENCODING=utf-8
export DIFFPURE_CKPT="$HOME/thirdparty/diffpure/256x256_diffusion_uncond.pt"
export TOKENIZERS_PARALLELISM=false
cd "$ROOT"

OUT=runs/ip2p_theta_budget
mkdir -p "$OUT"

IMGS="task_attr_mod_color_11699 task_attr_mod_color_136767 task_attr_mod_color_184837 task_attr_mod_color_32648 task_attr_mod_color_6205 task_env_weather_112463 task_env_weather_246440 task_env_weather_63722 task_obj_add_40931 task_obj_remove_380621 task_obj_swap_joint_mask_276754 task_obj_swap_joint_mask_533428 task_obj_swap_rand_mask_417469"

COMMON="--data data/omniedit150 --conditions phase --gain-ratio 0 --spectral-floor 0.04 --radius 3.1416 --loss latent_norm --steps 1000 --quantile 0 --freq-weight jpeg_luma --freq-weight-power 0.25 --hop 8"

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


# tag:theta_budget
POINTS="
e0010:0.010
e0020:0.020
e0040:0.040
e0080:0.080
e0160:0.160
"

require_slots "$(echo "$POINTS" | grep -c .)" "${#DEVS[@]}"
i=0
for p in $POINTS; do
  IFS=: read -r tag eps <<< "$p"
  dev=${DEVS[$(( i / 2 % ${#DEVS[@]} ))]}
  i=$(( i + 1 ))
  CUDA_VISIBLE_DEVICES="$dev" nohup "$PY" scripts/ip2p_run.py \
      --out "$OUT/$tag" --theta-budget "$eps" \
      --images $IMGS $COMMON > "$OUT/$tag.log" 2>&1 &
  echo "[theta_budget] $tag eps=$eps dev=$dev pid=$!"
done
echo "[theta_budget] 全部送出（$(date)）"

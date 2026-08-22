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
# 不 source ~/env.sh：它最後一行會把工作目錄換到舊的 ~/WACV（坑一）。
# 但 DIFFPURE_CKPT 只寫在那裡，少了它 gridpure／fdpure 會被判為
# 「相依不齊」而**靜默跳過**，報表上只剩一行提示。
export DIFFPURE_CKPT="$HOME/thirdparty/diffpure/256x256_diffusion_uncond.pt"
export TOKENIZERS_PARALLELISM=false
cd "$ROOT"

OUT=runs/ip2p_floor_gate
mkdir -p "$OUT"

IMGS="task_attr_mod_color_11699 task_attr_mod_color_136767 task_attr_mod_color_184837 task_attr_mod_color_32648 task_attr_mod_color_6205 task_env_weather_112463 task_env_weather_246440 task_env_weather_63722 task_obj_add_40931 task_obj_remove_380621 task_obj_swap_joint_mask_276754 task_obj_swap_joint_mask_533428 task_obj_swap_rand_mask_417469"

COMMON="--data data/omniedit150 --conditions phase_gain --loss latent_norm --steps 1000 --quantile 0 --freq-weight jpeg_luma --freq-weight-power 0.25 --hop 8 --gain-ratio 1.0 --spectral-floor 0.04"

# 卡號由參數給，**不寫死**：卡是多人共用的，寫死過一次就把八個 process 送到
# 別人正在用的四張卡上。用法 `bash scripts/floor_gate_sweep.sh "4 5 6 7"`，
# 每張卡放兩個。
DEVS=(${1:-0 1 2 3})
# 要掃哪些價目分配。第二個參數給空白分隔的名字，預設是最早的兩個變體。
GATES=(${2:-complement watson})
# 半徑固定這四點：uniform 的同四點由 axis_necessity.sh 的 b_pg_* 提供，
# 等失真的內插要有重疊的區間才成立。
RADII=(0.9 1.5 2.0 2.4)
# 名字的縮寫，讓輸出目錄一眼看得出是哪一個分配。
abbrev() {
  case "$1" in
    complement) echo comp ;; complement_rank) echo rank ;;
    watson) echo wat ;; uniform) echo unif ;; *) echo "$1" ;;
  esac
}

i=0
for fg in "${GATES[@]}"; do
 for rad in "${RADII[@]}"; do
  tag="$(abbrev "$fg")_r$(echo "$rad" | tr -d '.')"
  dev=${DEVS[$(( i / 2 % ${#DEVS[@]} ))]}
  i=$(( i + 1 ))
  CUDA_VISIBLE_DEVICES="$dev" nohup "$PY" scripts/ip2p_run.py \
      --out "$OUT/$tag" --floor-gate "$fg" --radius "$rad" \
      --images $IMGS $COMMON > "$OUT/$tag.log" 2>&1 &
  echo "[floor_gate] $tag gate=$fg radius=$rad dev=$dev pid=$!"
 done
done
echo "[floor_gate] 全部送出（$(date)）"

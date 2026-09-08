#!/usr/bin/env bash
# 高容量色彩網格的抗淨化：十個算子、同失真隨機直接對照。
#
# 為什麼是這一格
# `runs/ip2p_color_hunt/` 量到把雙邊網格由 8×8 放大到 16×16（其餘完全相同）
# 在**兩個軸上同時**贏過原本的工作點：DISTS 0.2379 換到位移 0.5441，而同失真
# 的隨機是 0.4183，倍率 **1.30**——先前這一族最好的是 1.08–1.19。
#
# 但未淨化欄推不出衰減形狀。色彩族唯一站得住的賣點就是「衰減形狀平」，
# 而那個形狀先前是在 8×8、`latent_norm` 的工作點上量的。容量放大之後
# 擾動的空間頻率上升，**抗模糊與抗 JPEG 那幾欄很可能跟著掉**，這一批就是
# 要看它掉多少。
#
# 同失真的隨機對照是 `grid_rand_r020`（DISTS 0.2309，對上 0.2379），
# **直接跑出來的不是內插的**。
#
# **只讀已存的防禦圖，不重跑防禦。**
#
# 用法：bash scripts/purify_color_capacity.sh "<卡號…>"
set -uo pipefail
ROOT=/nfs/home/nelson0314/WACV-s3
PY="$HOME/venvs/wacv/bin/python"
export PYTHONPATH="$ROOT" HF_HOME="$HOME/hf_cache" PYTHONIOENCODING=utf-8
export TOKENIZERS_PARALLELISM=false
cd "$ROOT" || { echo "錯誤：找不到 $ROOT" >&2; exit 2; }

DEVS=(${1:-})
[ ${#DEVS[@]} -lt 1 ] && { echo "用法：$0 \"<卡號…>\"" >&2; exit 2; }
bash scripts/free_cards.sh --assert "${DEVS[*]}" || exit 3

OUT=runs/ip2p_color_capacity_purify
mkdir -p "$OUT"

PLANT=task_attr_mod_color_11699
MARIO=task_attr_mod_color_6205
BOTH="$PLANT $MARIO"
PUR="identity jpeg75 jpeg30 blur1 blur2 crop_resize0.1 grayscale gray_world auto_levels0.01 clahe2"
COMMON="--data data/omniedit150 --attacker ip2p --seeds 3 --purifiers $PUR --codefense"

i=0
launch() {                     # $1 tag  $2 來源  $3 額外
  local tag="$1" run="$2" extra="${3:-}"
  local dev=${DEVS[$(( i % ${#DEVS[@]} ))]}; i=$((i + 1))
  if [ ! -f "$run/results.csv" ] || ! ls "$run"/*__def.png >/dev/null 2>&1; then
    echo "[skip] $run 缺防禦圖或 results.csv" >&2
    return
  fi
  CUDA_VISIBLE_DEVICES="$dev" setsid nohup "$PY" scripts/phase_retention.py \
      --run "$run" $COMMON --images $BOTH $extra \
      --out "$OUT/${tag}_all.csv" --gallery "$OUT/gallery_${tag}" \
      < /dev/null >> "$OUT/${tag}.log" 2>&1 &
  disown
  echo "[purify] $tag dev=$dev src=$run"
}

launch g16       runs/ip2p_color_hunt/cg_subj_r010_g16
launch rand_r020 runs/ip2p_color/grid_rand_r020
# 空白地板：淨化算子自己就會把編輯推開，不扣掉它「淨化後位移較大」無法解讀。
# 來源目錄只用來定位設定，且**必須含齊要量的每一張影像**（`--floor` 的格
# 由來源的 results.csv 建，加不進不在表裡的影像）。
launch floor     runs/ip2p_color_hunt/cg_subj_r010_g16 --floor

sleep 20
echo "啟動了 $(ps -u "$USER" -o cmd | grep -c '[p]hase_retention') 個 phase_retention（$(date)）"

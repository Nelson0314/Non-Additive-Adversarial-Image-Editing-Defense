#!/usr/bin/env bash
# `image_guidance` 工作點的抗淨化，兩張影像、十個算子。
#
# 為什麼要這一批
# ────────────────────────────────────────────────────────────────────
# 色彩族唯一站得住的賣點是**衰減形狀平**（`runs/ip2p_color_purify/`：
# JPEG 30 保留 47–56%，相位族是 11%）。那個形狀是在 `latent_norm` 的工作點上
# 量的，而 `latent_norm` 已被證實在這一族裡與位移反向（等失真對隨機 0.86），
# 改用 `image_guidance` 之後倍率變成 1.08–1.14。
#
# 換了損失之後**衰減形狀還在不在**，是這一族還剩多少價值的關鍵，
# 而它不能從未淨化欄推出來。
#
# 同失真的隨機對照一起送：`image_guidance` r=0.05 的失真是 DISTS 0.1297，
# 落在隨機曲線的 (0.0875, 0.2029) 與 (0.1526, 0.3169) 之間，
# 故對照取 `grid_rand_r0083`（由 `--radius 0.083` 產生）。
#
# **只讀已存的防禦圖，不重跑防禦。**
#
# 用法：bash scripts/purify_color_ig.sh "<卡號…>"
set -uo pipefail
ROOT=/nfs/home/nelson0314/WACV-s3
PY="$HOME/venvs/wacv/bin/python"
export PYTHONPATH="$ROOT" HF_HOME="$HOME/hf_cache" PYTHONIOENCODING=utf-8
export TOKENIZERS_PARALLELISM=false
cd "$ROOT" || { echo "錯誤：找不到 $ROOT" >&2; exit 2; }

DEVS=(${1:-})
[ ${#DEVS[@]} -lt 1 ] && { echo "用法：$0 \"<卡號…>\"" >&2; exit 2; }
bash scripts/free_cards.sh --assert "${DEVS[*]}" || exit 3

OUT=runs/ip2p_color_ig_purify
mkdir -p "$OUT"

PLANT=task_attr_mod_color_11699
MARIO=task_attr_mod_color_6205
PUR="identity jpeg75 jpeg30 blur1 blur2 crop_resize0.1 grayscale gray_world auto_levels0.01 clahe2"
COMMON="--data data/omniedit150 --attacker ip2p --seeds 3 --purifiers $PUR --codefense"

# tag:來源目錄:影像
CELLS="ig_r005_plant:runs/ip2p_color_ig/ig_r005_plant:$PLANT \
ig_r005_mario:runs/ip2p_color_ig/ig_r005_mario:$MARIO \
rand_matched:runs/ip2p_color/grid_rand_r0083:$PLANT,$MARIO"

i=0
launch() {                     # $1 tag  $2 來源  $3 影像（逗號分隔）  $4 額外
  local tag="$1" run="$2" imgs="$3" extra="${4:-}"
  local dev=${DEVS[$(( i % ${#DEVS[@]} ))]}; i=$((i + 1))
  CUDA_VISIBLE_DEVICES="$dev" setsid nohup "$PY" scripts/phase_retention.py \
      --run "$run" $COMMON --images ${imgs//,/ } $extra \
      --out "$OUT/${tag}_all.csv" --gallery "$OUT/gallery_${tag}" \
      < /dev/null >> "$OUT/${tag}.log" 2>&1 &
  disown
  echo "[purify] $tag dev=$dev src=$run"
}

for cell in $CELLS; do
  tag="${cell%%:*}"; rest="${cell#*:}"
  run="${rest%%:*}"; imgs="${rest##*:}"
  if ls "$run"/*__def.png >/dev/null 2>&1 && [ -f "$run/results.csv" ]; then
    launch "$tag" "$run" "$imgs"
  else
    echo "[skip] $run 缺防禦圖或 results.csv" >&2
  fi
done
# 地板：兩張影像一格，來源目錄只用來定位設定。
launch floor runs/ip2p_color_ig/ig_r005_plant "$PLANT,$MARIO" --floor

sleep 20
echo "啟動了 $(ps -u "$USER" -o cmd | grep -c '[p]hase_retention') 個 phase_retention（$(date)）"

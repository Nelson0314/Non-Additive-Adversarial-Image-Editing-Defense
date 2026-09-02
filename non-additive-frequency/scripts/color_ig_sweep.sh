#!/usr/bin/env bash
# 色彩網格族在 `image_guidance` 下的強度掃描：兩個半徑 × 兩張影像，一格一卡。
#
# 為什麼只掃這個損失
# ────────────────────────────────────────────────────────────────────
# `runs/latent_norm_probe/`：把各族已存的防禦圖丟進同一個編碼器重算
# `‖E(x_def)‖₂`（`latent_norm` 訓練時最小化的量），得到
#
#   相位 r0.9        0.594／0.526 × 原圖   位移 0.5418
#   網格 latent_norm 0.667／0.760 × 原圖   位移 0.2867
#   網格 隨機 r0.10  1.067／1.059 × 原圖   位移 0.3169   ← 把損失推**高**
#   網格 image_gui.  0.880／0.951 × 原圖   位移 0.4516   ← 幾乎沒動損失
#
# **在色彩族裡，`latent_norm` 與位移是反向的**：把它壓得最低的那一格位移最小，
# 幾乎沒動它的那一格位移最大。等失真對隨機的倍率也一致——`latent_norm` 收斂解
# 是 0.86（低於隨機），`image_guidance` 是 1.08（`runs/ip2p_color_converge/`）。
#
# 所以這一批不再掃 `latent_norm`。`image_guidance` 目前只有 r=0.10 一個點，
# 一個點畫不出取捨曲線、做不了等失真內插；這一批補 r=0.05 與 r=0.20。
#
# `--ig-zt diffuse_src` 沿用專案既有的全部 image_guidance 批次。
# 八千步：r=0.10 那一格跑滿八千步仍未觸發早停，這兩格照同一個預算跑，
# 停止原因逐列記在 CSV。
#
# 用法：bash scripts/color_ig_sweep.sh "<卡號…>" [只跑這些 tag 的前綴]
set -uo pipefail
ROOT=/nfs/home/nelson0314/WACV-s3
PY="$HOME/venvs/wacv/bin/python"
export PYTHONPATH="$ROOT" HF_HOME="$HOME/hf_cache" PYTHONIOENCODING=utf-8
export TOKENIZERS_PARALLELISM=false
cd "$ROOT" || { echo "錯誤：找不到 $ROOT" >&2; exit 2; }

DEVS=(${1:-})
[ ${#DEVS[@]} -lt 1 ] && { echo "用法：$0 \"<卡號…>\" [tag 前綴]" >&2; exit 2; }
bash scripts/free_cards.sh --assert "${DEVS[*]}" || exit 3

OUT=runs/ip2p_color_ig
mkdir -p "$OUT"

COMMON="--data data/omniedit150 --conditions color_grid --loss image_guidance --ig-zt diffuse_src --steps 8000 --eval-every 200 --patience 8 --save-weights"

# tag:radius:image
CELLS="ig_r005_plant:0.05:task_attr_mod_color_11699 \
ig_r005_mario:0.05:task_attr_mod_color_6205 \
ig_r020_plant:0.20:task_attr_mod_color_11699 \
ig_r020_mario:0.20:task_attr_mod_color_6205"

ONLY="${2:-}"

n=$(echo $CELLS | wc -w)
if [ "$n" -gt $(( ${#DEVS[@]} * 2 )) ]; then
  echo "錯誤：$n 格需要至少 $(( (n + 1) / 2 )) 張卡，只給了 ${#DEVS[@]} 張。" >&2
  exit 2
fi

i=0
for cell in $CELLS; do
  tag="${cell%%:*}"; rest="${cell#*:}"
  rad="${rest%%:*}"; img="${rest##*:}"
  if [ -n "$ONLY" ]; then
    case "$tag" in $ONLY*) ;; *) continue ;; esac
  fi
  dev=${DEVS[$(( i % ${#DEVS[@]} ))]}
  i=$(( i + 1 ))
  CUDA_VISIBLE_DEVICES="$dev" setsid nohup "$PY" scripts/ip2p_run.py \
      --out "$OUT/$tag" --radius "$rad" --images "$img" $COMMON \
      < /dev/null > "$OUT/$tag.log" 2>&1 &
  disown
  echo "[ig] $tag radius=$rad image=$img dev=$dev"
done

sleep 25
echo "啟動了 $(ps -u "$USER" -o cmd | grep '[i]p2p_run.py' | grep -c python) 行 ip2p_run（$(date)）"

#!/usr/bin/env bash
# 等乙的抗淨化讓出第五張卡，把 latent_norm 對照組送上去，跑完再做它的抗淨化。
# **總數維持五張**：丙的訓練佔四張，這一支只用一張。
set -uo pipefail
cd /nfs/home/nelson0314/WACV-s3
export PYTHONIOENCODING=utf-8
echo "$(date +%H:%M) 等乙的抗淨化收工"
while true; do
  n=$(ps -u "$USER" -o cmd | grep -c '[p]hase_retention')
  [ "$n" -eq 0 ] && break
  sleep 120
done
sleep 60
FREE=($(bash scripts/free_cards.sh))
echo "$(date +%H:%M) 空卡：${FREE[*]:-（沒有）}"
[ "${#FREE[@]}" -lt 1 ] && { echo "沒有空卡" >&2; exit 4; }
bash scripts/latent_norm_control.sh "${FREE[0]}" || exit 3
while true; do
  n=$(ps -u "$USER" -o cmd | grep -c '[i]p2p_run.py --out runs/ip2p_latent_norm_ctrl')
  [ "$n" -eq 0 ] && break
  sleep 120
done
echo "$(date +%H:%M) 對照組訓練完，跑它的抗淨化"
FREE=($(bash scripts/free_cards.sh))
[ "${#FREE[@]}" -lt 1 ] && { echo "沒有空卡" >&2; exit 4; }
SRC=runs/ip2p_latent_norm_ctrl OUT=runs/ip2p_latent_norm_purify TAGS="ln_fixed" \
bash scripts/purify_points.sh "${FREE[0]}"
echo "$(date +%H:%M) 對照組完成"

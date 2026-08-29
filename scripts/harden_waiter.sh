#!/usr/bin/env bash
# 等現行的 ig 批次收工，再把強化批次送出去。**卡最多五張。**
#
# 為什麼要一支等待器：現在四張卡在跑 `ip2p_ig_converge`、第五張在跑逐張串接
# 的淨化，五張已經滿了。強化批次要四張訓練＋一張淨化，必須等前一批把卡讓
# 出來。**等待器不自己判斷卡空不空**——它只負責等，判斷交給
# `ig_hardening.sh` 裡的 `free_cards.sh --assert`，那一道會 exit。
set -uo pipefail
ROOT=/nfs/home/nelson0314/WACV-s3
cd "$ROOT"
export PYTHONIOENCODING=utf-8

echo "$(date +%H:%M) 等待器啟動：等 ip2p_ig_converge 的四個 process 收工"
while true; do
  n=$(ps -u "$USER" -o cmd | grep -c '[i]p2p_run.py --out runs/ip2p_ig_converge')
  c=$(ps -u "$USER" -o cmd | grep -c '[i]ncremental_pipeline')
  [ "$n" -eq 0 ] && [ "$c" -eq 0 ] && break
  sleep 180
done
echo "$(date +%H:%M) 前一批收工（訓練 0、串接 0）"

# 前一批的 results.csv 被舊版 --skip-existing 截掉的列，用已存的 PNG 補回來。
# **不重跑最佳化**，只讀四張圖；訓練當下才知道的欄位留空。
for spec in "ig_d21 2.1 diffuse_src" "ig_d25 2.5 diffuse_src" \
            "ig_n30 3.0 noise" "ig_n35 3.5 noise"; do
  set -- $spec
  PYTHONPATH="$ROOT" "$HOME/venvs/wacv/bin/python" scripts/rebuild_rows.py \
      --run "runs/ip2p_ig_converge/$1" --condition phase_gain \
      --radius "$2" --loss image_guidance --ig-zt "$3" \
      --steps 8000 --step-size 0.01 2>&1 | tail -2
done

FREE=($(bash scripts/free_cards.sh))
echo "$(date +%H:%M) 空卡：${FREE[*]:-（沒有）}"
if [ "${#FREE[@]}" -lt 5 ]; then
  echo "空卡不足五張，不送。手動處理。" >&2
  exit 4
fi
TRAIN="${FREE[0]} ${FREE[1]} ${FREE[2]} ${FREE[3]}"
CHAIN="${FREE[4]}"
bash scripts/ig_hardening.sh "$TRAIN" || { echo "派工被守門擋下" >&2; exit 3; }

SRC=runs/ip2p_ig_harden OUT=runs/ip2p_harden_purify GAL=runs/gallery_harden \
TAGS="ih_l25 ih_l15 ih_eot ih_both" SHORT="l25 l15 eot both" COND=phase_gain \
TITLE="影像引導消除損失：L∞ 投影與寬 EOT" \
SUB="4 條件 · 9 算子 · 1 種子 · image_guidance diffuse_src r2.5 · 8000 步上限 · 固定步長 0.01" \
setsid nohup bash scripts/incremental_chain.sh "$CHAIN" \
    > /tmp/harden_chain.log 2>&1 < /dev/null &
disown
echo "$(date +%H:%M) 強化批次已送：訓練卡 $TRAIN、串接卡 $CHAIN"

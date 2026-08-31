#!/usr/bin/env bash
# 把兩個損失加起來。四個權重、兩張影像、8000 步。
#
# 為什麼值得試
# ────────────────────────────────────────────────────────────────────
# 同兩張影像、同設定、只換 `--loss` 的直接對照（`runs/ip2p_latent_norm_purify`
# 對 `runs/ip2p_ig_converge`，各自扣自己的地板）：
#
#   | | 未淨化 | JPEG 75 | JPEG 30 | 模糊 σ1 |
#   |---|---|---|---|---|
#   | latent_norm    | **0.6843** | 0.2691 | 0.1232（**低於地板 0.2547**） | 0.2154 |
#   | image_guidance | 0.5850 | **0.4209** | **0.2604** | 0.1787 |
#
# **舊的未淨化最強但一被壓縮就塌，新的未淨化較弱但每一級 JPEG 都撐得住。**
# 兩者是互補的，而從來沒有人把它們加在一起。
#
# `bl_w000` 是同批基準線（純影像引導），沒有它分不出改善來自加法還是別的。
# **權重的四個值沒有出處，是本專案指定的**；舊項已除以它在乾淨影像上的值，
# 故權重 1 真的是等權。
#
# 用法：bash scripts/blend_loss_round.sh "<四個卡號>"
set -uo pipefail
ROOT=/nfs/home/nelson0314/WACV-s3
PY="$HOME/venvs/wacv/bin/python"
export PYTHONPATH="$ROOT" HF_HOME="$HOME/hf_cache" PYTHONIOENCODING=utf-8
export TOKENIZERS_PARALLELISM=false
cd "$ROOT" || exit 2

DEVS=(${1:-})
[ ${#DEVS[@]} -ne 4 ] && { echo "用法：$0 \"<四個卡號>\"" >&2; exit 2; }
bash scripts/free_cards.sh --assert "${DEVS[*]}" || exit 3
grep -q -- "--latent-norm-weight" scripts/ip2p_run.py || {
  echo "錯誤：scripts/ip2p_run.py 不認得 --latent-norm-weight，先同步" >&2; exit 2; }

OUT=runs/ip2p_blend_loss
mkdir -p "$OUT"
IMGS="task_attr_mod_color_11699 task_obj_remove_380621"

BASE="--data data/omniedit150 --conditions phase_gain --quantile 0 \
--freq-weight jpeg_luma --freq-weight-power 0.25 --hop 8 --gain-ratio 1.0 \
--loss image_guidance --ig-zt diffuse_src --radius 2.5 \
--steps 8000 --step-size 0.01 --eval-every 100 --eval-draws 8 \
--patience 15 --min-delta 0.0002 --save-weights --skip-existing"

POINTS="
bl_w000:--latent-norm-weight~0
bl_w025:--latent-norm-weight~0.25
bl_w100:--latent-norm-weight~1.0
bl_w400:--latent-norm-weight~4.0
"

i=0
for p in $POINTS; do
  IFS=: read -r tag extra <<< "$p"
  dev=${DEVS[$i]}; i=$((i + 1))
  CUDA_VISIBLE_DEVICES="$dev" setsid nohup "$PY" scripts/ip2p_run.py \
      --out "$OUT/$tag" $BASE $(echo "$extra" | tr '~' ' ') \
      --images $IMGS < /dev/null >> "$OUT/$tag.log" 2>&1 &
  disown
  echo "[blend] $tag dev=$dev $(echo "$extra" | tr '~' ' ')"
done
sleep 20
echo "啟動了 $(ps -u "$USER" -o cmd | grep -c '[i]p2p_run') 個 ip2p_run process"

#!/usr/bin/env bash
# `--theta-budget`：把相位閘從「限角度」改成「限位移」。四個值、兩張影像。
#
# 它修的是什麼
# ────────────────────────────────────────────────────────────────────
# 現行的閘乘的是**角度**：`shift = theta × tex_gate[b] × freq_gate[w]`。而旋轉
# 一個係數造成的實際像素改變是 `|S(b,w)| · theta` 那一級——閘完全不看 `|S|`。
# 後果是 FND-038 記過的缺陷：**固定的 theta 不等於固定的失真**，同一個 theta
# 在 24 張影像上 PSNR 由 23.15 漂到 39.54。
#
# `--theta-budget eps` 由**原圖**幅度算每個（視窗, 頻格）的上限
#     theta_cap = 2·asin( eps / (2|S|) )，  |S| 很小時取 pi（相位自由）
# 也就是把每個係數的**位移量**限在 eps 之內，而不是把角度限住。上限與兩個閘
# 同型、不參與最佳化。
#
# **這一批的理由是量測會變準，不是效果會變好。** FND-038 汙染的是每一個等
# 失真比較，包含 `ip2p_ig_converge` 與 `ip2p_ig_harden` 的所有表。效果由
# 使用者判斷（`CLAUDE.md`）。
#
# eps 的四個值**是本專案指定的，沒有出處**。取法記下來以便日後檢查：`tb_off`
# 是關閉（即現行行為）當基準線；其餘三個跨一個數量級，用來看 `theta_cap` 何時
# 開始真的咬住——咬不住時逐位元等於基準線，咬太緊時等於整體降半徑。
#
# 其餘旗標與 `ig_d25` 完全相同，故可直接對照。
#
# 用法：bash scripts/theta_budget_round.sh "<四個卡號>"
set -uo pipefail
ROOT=/nfs/home/nelson0314/WACV-s3
PY="$HOME/venvs/wacv/bin/python"
export PYTHONPATH="$ROOT" HF_HOME="$HOME/hf_cache" PYTHONIOENCODING=utf-8
export TOKENIZERS_PARALLELISM=false
cd "$ROOT" || { echo "錯誤：找不到 $ROOT" >&2; exit 2; }

DEVS=(${1:-})
[ ${#DEVS[@]} -ne 4 ] && { echo "用法：$0 \"<四個卡號>\"" >&2; exit 2; }
bash scripts/free_cards.sh --assert "${DEVS[*]}" || exit 3

OUT=runs/ip2p_theta_budget
mkdir -p "$OUT"
IMGS="task_attr_mod_color_11699 task_obj_remove_380621"

BASE="--data data/omniedit150 --conditions phase_gain --quantile 0 \
--freq-weight jpeg_luma --freq-weight-power 0.25 --hop 8 --gain-ratio 1.0 \
--loss image_guidance --ig-zt diffuse_src --radius 2.5 \
--steps 8000 --step-size 0.01 --eval-every 100 --eval-draws 8 \
--patience 15 --min-delta 0.0002 --save-weights --skip-existing"

POINTS="
tb_off:--theta-budget~0
tb_003:--theta-budget~0.03
tb_010:--theta-budget~0.10
tb_030:--theta-budget~0.30
"

i=0
for p in $POINTS; do
  IFS=: read -r tag extra <<< "$p"
  dev=${DEVS[$i]}; i=$((i + 1))
  CUDA_VISIBLE_DEVICES="$dev" setsid nohup "$PY" scripts/ip2p_run.py \
      --out "$OUT/$tag" $BASE $(echo "$extra" | tr '~' ' ') \
      --images $IMGS < /dev/null >> "$OUT/$tag.log" 2>&1 &
  disown
  echo "[tbudget] $tag dev=$dev $(echo "$extra" | tr '~' ' ')"
done
sleep 20
echo "啟動了 $(ps -u "$USER" -o cmd | grep -c '[i]p2p_run') 個 ip2p_run process"

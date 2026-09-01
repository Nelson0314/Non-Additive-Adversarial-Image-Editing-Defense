#!/usr/bin/env bash
# 影像引導消除損失的兩項強化：交付前的像素域 L∞ 投影、與寬 EOT。
#
# **不設效果判準**（`CLAUDE.md`）。這一批只做三件事：跑到收斂、忠實記錄
# 數據、留下影像。有沒有效果由使用者判斷。
#
# 為什麼是這兩項
# ────────────────────────────────────────────────────────────────────
# 兩個症狀各有一個量到的成因：
#
# 1. **失真肉眼看很大。** `runs/ip2p_ig_converge` 在 DISTS 0.153–0.180 下
#    L∞ 是 0.793–0.927，而主線相位族在相近的 DISTS（0.145）只有 0.42。
#    同樣的 DISTS 代表殘差是稀疏高振幅尖峰。成因可以指名：最佳化只被約束在
#    θ 的半徑球上，**像素域沒有任何約束**。`--linf-deliver` 把 L∞ 由事後
#    才看到的欄位變成事前就守住的約束。
#
# 2. **只擋得住低壓縮 JPEG。** 扣地板的淨增益由 identity 0.544 緩降到
#    jpeg30 0.231，但 blur σ2 只有 0.077（地板 0.299）、crop 15% 只有 0.089
#    （地板 0.561）。既有的 `--purify-aware eot_ops` 裡模糊恆為 σ=1.0、裁切
#    恆為中心 0.10，**固定的變換會被 co-adapt**；`eot_broad` 先抽算子類別、
#    再在該類的族內抽參數。
#
# 四個工作點（一卡一個）
# ────────────────────────────────────────────────────────────────────
#   ih_l25   只有 L∞ 投影，eps = 0.25
#   ih_l15   只有 L∞ 投影，eps = 0.15
#   ih_eot   只有寬 EOT
#   ih_both  兩者都有（eps = 0.20）
#
# **eps 的兩個值是本專案指定的，沒有出處。** 取法寫下來以便日後檢查：主線
# 相位族在 DISTS 0.145 上的 L∞ 是 0.42，0.25 與 0.15 分別是它的 0.60 與
# 0.36 倍，兩者都明顯低於它，故是真的約束而不是空轉；`ig_d25`（同設定、
# 不給旗標，已在 `runs/ip2p_ig_converge`）即 eps = ∞ 的那一點，三點合起來
# 是一條取捨曲線。
#
# 其餘旗標與 `ig_d25` **完全相同**（diffuse_src、radius 2.5、固定步長 0.01、
# 8000 步上限、同樣的 early stop 設定），故可直接對照。
#
# 用法：bash scripts/ig_hardening.sh "<四個卡號>"
set -uo pipefail
ROOT=/nfs/home/nelson0314/WACV-s3
PY="$HOME/venvs/wacv/bin/python"
export PYTHONPATH="$ROOT" HF_HOME="$HOME/hf_cache" PYTHONIOENCODING=utf-8
export TOKENIZERS_PARALLELISM=false
cd "$ROOT" || { echo "錯誤：找不到 $ROOT" >&2; exit 2; }

DEVS=(${1:-})
[ ${#DEVS[@]} -ne 4 ] && { echo "用法：$0 \"<四個卡號>\"" >&2; exit 2; }

# 卡是多人共用的。**列印了卻不擋等於沒擋**，這一道會 exit。
bash scripts/free_cards.sh --assert "${DEVS[*]}" || exit 3

# 遠端驅動認不認得新旗標。漏同步的失效方式是「卡空轉半小時才發現被 argparse
# 擋下」（`docs/OPERATIONS.md`）。
for flag in --linf-deliver --eot-sigmas --eot-fractions; do
  grep -q -- "$flag" scripts/ip2p_run.py || {
    echo "錯誤：scripts/ip2p_run.py 不認得 $flag，先同步本機的改動" >&2; exit 2; }
done
grep -q "eot_broad" scripts/ip2p_run.py || {
  echo "錯誤：scripts/ip2p_run.py 沒有 eot_broad，先同步" >&2; exit 2; }

OUT=runs/ip2p_ig_harden
mkdir -p "$OUT"
IMGS=$(tr '\n' ' ' < runs/ip2p_fair_comparison/images10.txt)

BASE="--data data/omniedit150 --conditions phase_gain --quantile 0 \
--freq-weight jpeg_luma --freq-weight-power 0.25 --hop 8 --gain-ratio 1.0 \
--loss image_guidance --ig-zt diffuse_src --radius 2.5 \
--steps 8000 --step-size 0.01 --eval-every 100 --eval-draws 8 \
--patience 15 --min-delta 0.0002 --save-weights --skip-existing"

POINTS="
ih_l25:--linf-deliver~0.25
ih_l15:--linf-deliver~0.15
ih_eot:--purify-aware~eot_broad
ih_both:--purify-aware~eot_broad~--linf-deliver~0.20
"

i=0
for p in $POINTS; do
  IFS=: read -r tag extra <<< "$p"
  dev=${DEVS[$i]}; i=$((i + 1))
  CUDA_VISIBLE_DEVICES="$dev" setsid nohup "$PY" scripts/ip2p_run.py \
      --out "$OUT/$tag" $BASE $(echo "$extra" | tr '~' ' ') \
      --images $IMGS < /dev/null >> "$OUT/$tag.log" 2>&1 &
  disown
  echo "[harden] $tag dev=$dev $(echo "$extra" | tr '~' ' ')"
done
sleep 20
echo "啟動了 $(ps -u "$USER" -o cmd | grep -c '[i]p2p_run') 個 ip2p_run process"

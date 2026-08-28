#!/usr/bin/env bash
# 多品質集成損失 ＋ 單一格點交付。**抗淨化那一側唯一還沒試、又有文獻依據的機制。**
#
# 依據
# ────────────────────────────────────────────────────────────────────
# Shin & Song（`reference/SURVEY_FREQUENCY.md` §1.15）的核心結論是**單一品質會
# 過度特化**，而我們的 `--deliver-jpeg 0.85` 正是單一品質；症狀在資料上看得到
# ——jpeg75 的淨增益 0.4370 掉到 jpeg30 的 0.2087。MBRS（ACM MM 2021，
# `reference/SURVEY_NOISE_RESISTANCE.md` §1.3）用「每個 mini-batch 隨機選真實／
# 模擬／無失真」拿到同一個效果。
#
# **集成只能放在損失上，交付仍是單一格點。** 放到交付上就是已否決的
# `--purify-aware`——那三個變體把 JPEG 放進迴圈卻**交付未壓縮的圖**，量化格點
# 的性質完全沒拿到。本批的順序是「先自壓到 QD 的格點、再讓攻擊方以抽到的品質
# 壓一次」，兩件事都在。
#
# 兩個點的差別只有品質集合
# ────────────────────────────────────────────────────────────────────
#   ens_hi   (95, 75, 50)  文獻的預設值
#   ens_lo   (75, 50, 30)  對齊我方實際有優勢的那一段（jpeg30）
#
# 半徑與交付品質都與 `ours_ph_q` 逐字相同（DISTS 0.0928），所以比較接近等失真；
# 嚴格的等失真仍要走 `matched_distortion_table.py` 內插。
#
# 用法：bash scripts/ensemble_quality_sweep.sh "<卡號>" ["<tag> ..."]
set -uo pipefail
ROOT=/nfs/home/nelson0314/WACV-s3
PY="$HOME/venvs/wacv/bin/python"
export PYTHONPATH="$ROOT" HF_HOME="$HOME/hf_cache" PYTHONIOENCODING=utf-8
export TOKENIZERS_PARALLELISM=false
cd "$ROOT"

OUT=runs/ip2p_ensemble
mkdir -p "$OUT"
IMGS=$(tr '\n' ' ' < runs/ip2p_fair_comparison/images10.txt)

# 與 `mainline_defense.sh` 的 `ours_ph_q` 逐字相同，**唯一的變因是集成損失**。
BASE="--data data/omniedit150 --loss latent_norm --steps 1000 --quantile 0 \
--freq-weight jpeg_luma --freq-weight-power 0.25 --hop 8 --spectral-floor 0.04 \
--conditions phase --gain-ratio 0 --radius 0.9 --deliver-jpeg 0.85 \
--purify-aware eot_jpeg"

POINTS="
ens_hi:--eot-qualities~95~75~50
ens_lo:--eot-qualities~75~50~30
"

DEVS=(${1:-})
ONLY="${2:-}"
[ ${#DEVS[@]} -eq 0 ] || [ -z "${DEVS[0]}" ] && {
  echo "用法：$0 \"<卡號>\" [\"<tag> ...\"]" >&2; exit 2; }

# 派工前先確認遠端的驅動認得這些旗鈕（本地改完忘了同步已經發生過一次）。
grep -q -- "--eot-qualities" scripts/ip2p_run.py || {
  echo "錯誤：scripts/ip2p_run.py 不認得 --eot-qualities，先同步本機的改動" >&2
  exit 2; }

bash scripts/free_cards.sh --assert "${DEVS[*]}" || exit 3

selected() { [ -z "$ONLY" ] && return 0; for t in $ONLY; do [ "$t" = "$1" ] && return 0; done; return 1; }

n=0
for p in $POINTS; do IFS=: read -r tag _ <<< "$p"; selected "$tag" && n=$(( n + 1 )); done
[ "$n" -eq 0 ] && { echo "錯誤：選到 0 個 tag" >&2; exit 2; }
if [ "$n" -gt $(( ${#DEVS[@]} * 2 )) ]; then
  echo "錯誤：$n 個工作點需要至少 $(( (n + 1) / 2 )) 張卡，只給了 ${#DEVS[@]} 張" >&2
  exit 2
fi

i=0
for p in $POINTS; do
  IFS=: read -r tag extra <<< "$p"
  selected "$tag" || continue
  dev=${DEVS[$(( i / 2 % ${#DEVS[@]} ))]}
  i=$(( i + 1 ))
  CUDA_VISIBLE_DEVICES="$dev" nohup "$PY" scripts/ip2p_run.py \
      --out "$OUT/$tag" $BASE $(echo "$extra" | tr '~' ' ') \
      --images $IMGS > "$OUT/$tag.log" 2>&1 &
  echo "[ensemble] $tag dev=$dev pid=$!  $(echo "$extra" | tr '~' ' ')"
done
echo "[ensemble] 送出 $i 個（$(date)）"

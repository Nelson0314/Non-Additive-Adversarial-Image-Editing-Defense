#!/usr/bin/env bash
# 主線頭對頭的防禦圖：本方法（量化交付與否）對 DCT-Shield（原生與抗 JPEG）。
#
# 十張影像（`runs/ip2p_fair_comparison/images10.txt`）。由十三張移除三張
# ——龍、鐵門、松鼠——那三張在所有條件下都沒擋下過，留著只會稀釋每一個平均。
#
# 八個工作點分四組，每一組回答一件事：
#
#   ours_ph_q  / ours_pg_q    量化交付（--deliver-jpeg 0.85），r = 0.9
#   ours_pg_q20/ ours_ph_q20   量化交付的**強工作點**：r = 2.0（相位那支封頂在 pi）。
#                              r = 0.9 是為了與 DCT-Shield 等失真而選的，但它把本方法
#                              壓在遠低於自身能力的地方——實測 r=2.0 時 jpeg75 的淨增益
#                              是 0.4931 而 r=0.9 只有 0.4072，交叉點因此被推過去。
#                              兩者都要在表上，缺一個就看不出這是取捨還是失敗。
#   ours_ph_n  / ours_pg_n    **同半徑**不量化，唯一的變因就是量化交付
#   ours_pg_m                 **等失真**不量化（r = 2.0），與量化點的 DISTS 相近
#   dct_native                DCT-Shield §5.4 原生（q_alg 0.95）
#   dct_aj85 / dct_aj75       DCT-Shield §6.3 抗 JPEG（Y-only，兩個品質）
#
# **`ph` 是純相位、`pg` 是相位＋可學幅度增益**，兩者都開加性下限 0.04。
#
# 用法：bash scripts/mainline_defense.sh "<卡號>" ["<tag> ..."]
set -uo pipefail
ROOT=/nfs/home/nelson0314/WACV-s3
PY="$HOME/venvs/wacv/bin/python"
export PYTHONPATH="$ROOT" HF_HOME="$HOME/hf_cache" PYTHONIOENCODING=utf-8
export TOKENIZERS_PARALLELISM=false
cd "$ROOT"

OUT=runs/ip2p_mainline
mkdir -p "$OUT"
IMGS=$(tr '\n' ' ' < runs/ip2p_fair_comparison/images10.txt)

BASE="--data data/omniedit150 --loss latent_norm --steps 1000 --quantile 0 \
--freq-weight jpeg_luma --freq-weight-power 0.25 --hop 8 --spectral-floor 0.04"

# tag:額外旗標（用 ~ 代替空白，送出前換回）
POINTS="
ours_ph_q:--conditions~phase~--gain-ratio~0~--radius~0.9~--deliver-jpeg~0.85
ours_pg_q:--conditions~phase_gain~--gain-ratio~1.0~--radius~0.9~--deliver-jpeg~0.85
ours_pg_q20:--conditions~phase_gain~--gain-ratio~1.0~--radius~2.0~--deliver-jpeg~0.85
ours_ph_q20:--conditions~phase~--gain-ratio~0~--radius~3.1416~--deliver-jpeg~0.85
ours_ph_n:--conditions~phase~--gain-ratio~0~--radius~0.9
ours_pg_n:--conditions~phase_gain~--gain-ratio~1.0~--radius~0.9
ours_pg_m:--conditions~phase_gain~--gain-ratio~1.0~--radius~2.0
dct_native:--conditions~dct_shield~--q-alg~0.95~--eps~1.4~--dct-steps~1000
dct_aj85:--conditions~dct_shield_y~--q-alg~0.85~--eps~1.0~--dct-steps~1000
dct_aj75:--conditions~dct_shield_y~--q-alg~0.75~--eps~0.6~--dct-steps~1000
"

DEVS=(${1:-})
ONLY="${2:-}"
[ ${#DEVS[@]} -eq 0 ] && { echo "用法：$0 \"<卡號>\" [\"<tag> ...\"]" >&2; exit 2; }

# **別人的卡一律不碰**，這一道是強制的。
bash scripts/free_cards.sh --assert "${DEVS[*]}" || exit 3

selected() { [ -z "$ONLY" ] && return 0; for t in $ONLY; do [ "$t" = "$1" ] && return 0; done; return 1; }

n=0
for p in $POINTS; do IFS=: read -r tag _ <<< "$p"; selected "$tag" && n=$(( n + 1 )); done
[ "$n" -eq 0 ] && { echo "錯誤：選到 0 個 tag" >&2; exit 2; }
if [ "$n" -gt $(( ${#DEVS[@]} * 2 )) ]; then
  echo "錯誤：$n 個工作點需要至少 $(( (n + 1) / 2 )) 張卡，只給了 ${#DEVS[@]} 張" >&2; exit 2
fi

i=0
for p in $POINTS; do
  IFS=: read -r tag extra <<< "$p"
  selected "$tag" || continue
  dev=${DEVS[$(( i / 2 % ${#DEVS[@]} ))]}
  i=$(( i + 1 ))
  CUDA_VISIBLE_DEVICES="$dev" nohup "$PY" scripts/ip2p_run.py \
      --out "$OUT/$tag" $(echo "$extra" | tr '~' ' ') \
      --images $IMGS $BASE > "$OUT/$tag.log" 2>&1 &
  echo "[mainline] $tag dev=$dev pid=$!  $(echo "$extra" | tr '~' ' ')"
done
echo "[mainline] 送出 $i 個（$(date)）"

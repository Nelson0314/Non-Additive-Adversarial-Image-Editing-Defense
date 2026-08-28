#!/usr/bin/env bash
# 色散度那條軸的防禦圖，供報告頁使用。**全部是隨機、不最佳化的對照。**
#
# `scripts/dispersion_probe.py` 只寫 CSV、不存圖，而報告頁要的是防禦圖、
# 淨化圖與編輯圖，所以這一批走標準管線（`ip2p_run.py`）重新產一次。
#
# 五個條件覆蓋整條色散度軸：
#
#   d_k1      K=1，逐視窗一個位移套用到整個通帶 —— **古典位移場**
#   d_k4      K=4，四個八度帶各自一個位移 —— 色散變形
#   d_kfull   逐頻格獨立的隨機相位 —— 現行家族的隨機對照
#   w_smooth  像素域的單值平滑位移場（粗網格 16）
#   w_fold    像素域的單值位移場（粗網格 64），會折疊
#
# 半徑由 `runs/dispersion_probe/results.csv` 的曲線**內插**到平均
# DISTS ≈ 0.1286（失真帶下界），故五個條件近似等失真。逐圖的 DISTS 仍會漂，
# 報告頁照實把每一格的 DISTS 寫出來。
#
# 用法：bash scripts/dispersion_defense.sh "<卡號>" ["<tag> ..."]
set -uo pipefail
ROOT=/nfs/home/nelson0314/WACV-s3
PY="$HOME/venvs/wacv/bin/python"
export PYTHONPATH="$ROOT" HF_HOME="$HOME/hf_cache" PYTHONIOENCODING=utf-8
export TOKENIZERS_PARALLELISM=false
cd "$ROOT" || { echo "錯誤：找不到 $ROOT" >&2; exit 2; }

OUT=runs/ip2p_dispersion
mkdir -p "$OUT"
IMGS=$(tr '\n' ' ' < runs/ip2p_fair_comparison/images10.txt)

BASE="--data data/omniedit150 --steps 1 --quantile 0 --hop 8 --block 32"

# tag:條件與半徑（~ 代表空白）
POINTS="
d_k1:--conditions~disp_k1~--radius~6.6
d_k4:--conditions~disp_k4~--radius~5.0
d_kfull:--conditions~disp_kfull~--radius~2.38
w_smooth:--conditions~warp_rand~--radius~5.65~--warp-grid~16
w_fold:--conditions~warp_rand~--radius~1.74~--warp-grid~64
"

DEVS=(${1:-})
ONLY="${2:-}"
[ ${#DEVS[@]} -eq 0 ] || [ -z "${DEVS[0]}" ] && {
  echo "用法：$0 \"<卡號>\" [\"<tag> ...\"]" >&2; exit 2; }

grep -q "disp_k1" scripts/ip2p_run.py || {
  echo "錯誤：scripts/ip2p_run.py 不認得 disp_k1，先同步本機的改動" >&2; exit 2; }

selected() { [ -z "$ONLY" ] && return 0; for t in $ONLY; do [ "$t" = "$1" ] && return 0; done; return 1; }

n=0
for p in $POINTS; do IFS=: read -r tag _ <<< "$p"; selected "$tag" && n=$(( n + 1 )); done
[ "$n" -eq 0 ] && { echo "錯誤：選到 0 個 tag" >&2; exit 2; }
if [ "$n" -gt $(( ${#DEVS[@]} * 2 )) ]; then
  echo "錯誤：$n 個工作點需要至少 $(( (n + 1) / 2 )) 張卡，只給了 ${#DEVS[@]} 張" >&2
  exit 2
fi

bash scripts/free_cards.sh --assert "${DEVS[*]}" || exit 3

i=0
for p in $POINTS; do
  IFS=: read -r tag extra <<< "$p"
  selected "$tag" || continue
  dev=${DEVS[$(( i / 2 % ${#DEVS[@]} ))]}
  i=$(( i + 1 ))
  CUDA_VISIBLE_DEVICES="$dev" nohup "$PY" scripts/ip2p_run.py \
      --out "$OUT/$tag" $BASE $(echo "$extra" | tr '~' ' ') \
      --images $IMGS > "$OUT/$tag.log" 2>&1 &
  echo "[dispersion] $tag dev=$dev pid=$!  $(echo "$extra" | tr '~' ' ')"
done
echo "[dispersion] 送出 $i 個（$(date)）"

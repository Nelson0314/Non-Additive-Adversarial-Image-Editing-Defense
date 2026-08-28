#!/usr/bin/env bash
# 色散度探針。**不做最佳化、不跑編輯**，只有 VAE 前向，故是分鐘量級。
#
# 它問的是位移場的病灶是「位移」還是「單值」：單值場強迫所有空間頻率一起搬，
# 那正是它留在自然影像流形上、因而對編碼器無害的原因。K（獨立位移的頻帶數）
# 把古典位移場（K=1）與現行的紋理重相位（K=每格獨立）放在同一條軸上。
#
# 判準寫在 `scripts/dispersion_probe.py` 的檔頭（D1–D4），跑之前已定案。
#
# **這個排序不作結論**：探針量的是隨機方向，最佳化會自己挑方向。本專案兩次
# 踩過——隨機平面低估天花板 18.6 倍、定價取向高估 1.4 倍而最佳化後打平。
# 讀數只用來決定要不要上機。
#
# 用法：bash scripts/dispersion_probe.sh "<卡號>"
set -uo pipefail
ROOT=/nfs/home/nelson0314/WACV-s3
PY="$HOME/venvs/wacv/bin/python"
export PYTHONPATH="$ROOT" HF_HOME="$HOME/hf_cache" PYTHONIOENCODING=utf-8
export TOKENIZERS_PARALLELISM=false
cd "$ROOT" || { echo "錯誤：找不到 $ROOT" >&2; exit 2; }

OUT=runs/dispersion_probe
mkdir -p "$OUT"
IMGS=$(tr '\n' ' ' < runs/ip2p_fair_comparison/images10.txt)

DEVS=(${1:-})
[ ${#DEVS[@]} -eq 0 ] || [ -z "${DEVS[0]}" ] && {
  echo "用法：$0 \"<卡號>\"" >&2; exit 2; }

[ -f scripts/dispersion_probe.py ] || {
  echo "錯誤：找不到 scripts/dispersion_probe.py，先同步本機的改動" >&2; exit 2; }
[ -f src/defense/dispersion.py ] || {
  echo "錯誤：找不到 src/defense/dispersion.py，先同步本機的改動" >&2; exit 2; }

bash scripts/free_cards.sh --assert "${DEVS[0]}" || exit 3

CUDA_VISIBLE_DEVICES="${DEVS[0]}" nohup "$PY" scripts/dispersion_probe.py \
    --out "$OUT" --images $IMGS > "$OUT/probe.log" 2>&1 &
echo "[dispersion] dev=${DEVS[0]} pid=$!  → $OUT/results.csv、$OUT/summary.csv"

#!/usr/bin/env bash
# 色散變形：**可學 ＋ 接上兩個閘**，跑到收斂。不設效果判準（`CLAUDE.md`）。
#
# 與 `runs/ip2p_dispersion` 那一批的差別，三項缺一不可
# ────────────────────────────────────────────────────────────────────
# 那一批是**隨機、零可學參數、兩個閘全關**，所以殘差是滿幅度尖峰
# （L∞ 0.98、PSNR 18.1，同 DISTS 下現行家族是 0.59／26.2），而且從頭到尾沒有
# 被最佳化過。本批：
#
#   1. `u_k(b)` 逐（視窗, 頻帶）成為 PGD 參數，由**零**起步（零位移即恆等）
#   2. 接上紋理閘（結構張量）與**帶級**知覺定價 `jpeg_luma^0.25`
#      ——取帶內平均而不是逐格，這樣第 k 帶仍然是一個純平移
#   3. 固定評估 ＋ early stop，與影像引導那批同一套收斂判定
#
# 半徑
# ────────────────────────────────────────────────────────────────────
# r = 8 是 CPU 上校準過的：飽和時 DISTS 0.128（K=4）到 0.151（K=1），夾住
# 失真帶 0.1286–0.1447。
#
# 用法：bash scripts/dispersion_optimized.sh "<三個卡號>" ["<tag> ..."]
set -uo pipefail
ROOT=/nfs/home/nelson0314/WACV-s3
PY="$HOME/venvs/wacv/bin/python"
export PYTHONPATH="$ROOT" HF_HOME="$HOME/hf_cache" PYTHONIOENCODING=utf-8
export TOKENIZERS_PARALLELISM=false
cd "$ROOT" || { echo "錯誤：找不到 $ROOT" >&2; exit 2; }

OUT=runs/ip2p_dispersion_opt
mkdir -p "$OUT"
IMGS=$(tr '\n' ' ' < runs/ip2p_fair_comparison/images10.txt)

STEPS="${STEPS:-6000}"
RADIUS="${RADIUS:-8.0}"
BASE="--data data/omniedit150 --loss latent_norm --quantile 0 \
--freq-weight jpeg_luma --freq-weight-power 0.25 --hop 8 --block 32 \
--radius $RADIUS --steps $STEPS \
--eval-every 100 --eval-draws 8 --patience 12 --min-delta 0.002"

POINTS="d1_opt:disp_k1_opt d4_opt:disp_k4_opt d8_opt:disp_k8_opt"

DEVS=(${1:-})
ONLY="${2:-}"
[ ${#DEVS[@]} -eq 0 ] || [ -z "${DEVS[0]}" ] && {
  echo "用法：$0 \"<三個卡號>\" [\"<tag> ...\"]" >&2; exit 2; }

grep -q "disp_k4_opt" scripts/ip2p_run.py || {
  echo "錯誤：scripts/ip2p_run.py 不認得 disp_k4_opt，先同步本機的改動" >&2; exit 2; }

selected() { [ -z "$ONLY" ] && return 0; for t in $ONLY; do [ "$t" = "$1" ] && return 0; done; return 1; }

n=0
for p in $POINTS; do IFS=: read -r tag _ <<< "$p"; selected "$tag" && n=$(( n + 1 )); done
[ "$n" -eq 0 ] && { echo "錯誤：選到 0 個 tag" >&2; exit 2; }
[ "$n" -gt ${#DEVS[@]} ] && {
  echo "錯誤：$n 個工作點、一卡一個，需要 $n 張卡" >&2; exit 2; }

bash scripts/free_cards.sh --assert "${DEVS[*]}" || exit 3

i=0
for p in $POINTS; do
  IFS=: read -r tag cond <<< "$p"
  selected "$tag" || continue
  dev=${DEVS[$i]}; i=$(( i + 1 ))
  CUDA_VISIBLE_DEVICES="$dev" nohup "$PY" scripts/ip2p_run.py \
      --out "$OUT/$tag" $BASE --conditions "$cond" \
      --images $IMGS > "$OUT/$tag.log" 2>&1 &
  echo "[disp-opt] $tag dev=$dev pid=$!  cond=$cond r=$RADIUS steps=$STEPS"
done
echo "[disp-opt] 送出 $i 個（$(date)）"

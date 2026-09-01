#!/usr/bin/env bash
# 公平比較 · 階段 A2 · 放行低頻，把擾動從主體攤到平坦背景。
#
# 由來（使用者 2026-08-20 提出）：等 DISTS 錨點的對照圖顯示 DCT-Shield 把能量
# 鋪滿整張圖，我們則全部集中在主體上。本機量到的分布（單張，隨機 θ）：
# 平坦像素佔 68.9% 卻只吃到 25.8% 的改變能量，逐像素 |Δ| 主體是平坦區的 4.19 倍。
#
# 相位算子逐位保留幅度譜，平坦區的 AC 幾乎是零，所以「把擾動鋪到背景」只有
# 一條路——放行低頻，也就是調低 `r_min`。本機在**對齊 DISTS** 之後量到：
#
#   r_min=0.12（現行） θ*=1.200  LPIPS 0.0560  PSNR 32.43  主體/平坦 = 4.19
#   r_min=0.06        θ*=0.303  LPIPS 0.0291  PSNR 37.62  主體/平坦 = 2.89
#
# 同一個 DISTS 下 LPIPS 減半、PSNR 高 5.2 dB、主體的損傷降 46%。**但那只是
# 失真側，且是隨機 θ 的單張。** 防禦位移會不會跟著掉，只有 GPU 能回答，
# 這一輪就是去問那件事。
#
# r_min 只有三檔是真正不同的：block=32 的徑向格點是 2/32 的倍數，可用的最小
# 半徑是 0.0625 與 0.0884。r_min ∈ (0.0884, 0.125] 選到同一組格——本機實測
# 0.09 與 0.12 逐位相同（477 格）。0.08 放行 479 格、0.06 放行 480 格。
# **三格之差（0.55%）就是全部的差別**，因為那三格是能量最大的低頻。
set -u
source ~/env.sh >/dev/null 2>&1
cd /nfs/home/nelson0314/WACV-s3
export PYTHONPATH=/nfs/home/nelson0314/WACV-s3 TOKENIZERS_PARALLELISM=false
PY=$HOME/venvs/wacv/bin/python
ROOT=runs/fair0820/a2_rmin
mkdir -p $ROOT
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a $ROOT/log.txt; }

while ps -u $USER -o cmd | grep "[i]p2p_run" | grep -q fair0820/b/; do sleep 30; done
log "階段 A2 開始（r_min=0.12 那一檔沿用階段 A 的 p00_*，不重跑）"

IMGS=$(tr '\n' ' ' < runs/fair0820/images25.txt)
[ -n "$IMGS" ] || { log "影像清單是空的，中止"; exit 1; }

# tag  gpu  r_min  θ　　θ 網格由本機的 DISTS(θ) 量測選出，各自罩住 0.02–0.11
JOBS="
rm08_t040  0  0.08  0.40
rm08_t055  1  0.08  0.55
rm08_t070  2  0.08  0.70
rm08_t100  3  0.08  1.00
rm06_t020  4  0.06  0.20
rm06_t030  5  0.06  0.30
rm06_t045  6  0.06  0.45
rm06_t060  7  0.06  0.60
"
while read -r tag gpu rmin th; do
  [ -n "${tag:-}" ] || continue
  CUDA_VISIBLE_DEVICES=$gpu nohup $PY scripts/ip2p_run.py \
    --out "$ROOT/$tag" --data data/omniedit150 --images $IMGS \
    --conditions phase --radius "$th" --r-min "$rmin" --pixel-gate-sigma 0 \
    > "$ROOT/$tag.log" 2>&1 &
done <<< "$JOBS"
wait
for d in $ROOT/*/; do
  log "$(basename $d)：$(grep -c '^task_' $d/results.csv 2>/dev/null) 列"
done
log "階段 A2 完成"

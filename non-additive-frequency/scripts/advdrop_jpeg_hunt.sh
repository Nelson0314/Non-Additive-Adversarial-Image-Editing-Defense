#!/usr/bin/env bash
# JPEG-30 那一欄還是對不上（論文 0.826，我們 0.145–0.290），這一輪把兩個
# 未查證的實作自由度各自隔離出來。
#
# 已經排除的解釋：
#   成功率的定義   兩種定義都試過（label 與 defended），最高只到 0.290
#   攻擊強度不足   步長 6/8 之下無防禦已經是 1.000、Bit-6 0.990–1.000，
#                  兩欄完全對上論文，所以不是攻擊太弱
#
# 這一輪要試的兩個：
#   H1 色彩空間   官方程式碼在 RGB 上量化，但變數命名成 y/cb/cr、Figure 4 的
#                 管線圖也畫 YCbCr。JPEG 對色度做 4:2:0，擾動若平均分布在
#                 RGB，轉到 YCbCr 後有一部分落在會被下採樣抹掉的色度上
#   H2 下採樣     我們的 JPEG 用 PIL 預設（品質 30 時是 4:2:0）。論文沒寫。
#                 4:4:4 不動色度，擾動應該活得比較久
#
# 四格全跑：{rgb, ycbcr} × {4:2:0, 4:4:4}，eps=100、步長 8、200 張。
set -u
source ~/env.sh >/dev/null 2>&1
cd /nfs/home/nelson0314/WACV-s3
export PYTHONPATH=/nfs/home/nelson0314/WACV-s3 TOKENIZERS_PARALLELISM=false
PY=$HOME/venvs/wacv/bin/python
ROOT=runs/advdrop_repro
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a $ROOT/full_log.txt; }

while ps -u $USER -o cmd | grep -q "[a]dvdrop_repro.py"; do sleep 20; done
log "JPEG 追因開始"

g=0
for color in rgb ycbcr; do
  for sub in -1 0; do
    tag="hunt_${color}_sub${sub#-}"
    CUDA_VISIBLE_DEVICES=$g nohup $PY scripts/advdrop_repro.py \
      --out "$ROOT/$tag" --eps 100 --step-size 8 --color "$color" \
      --jpeg-subsampling "$sub" > "$ROOT/$tag.log" 2>&1 &
    g=$(((g+1) % 2))
  done
done
wait
for color in rgb ycbcr; do
  for sub in -1 0; do
    tag="hunt_${color}_sub${sub#-}"
    log "$tag：$(grep -hE '^eps=' $ROOT/$tag.log | tail -1)"
  done
done
log "JPEG 追因完成"

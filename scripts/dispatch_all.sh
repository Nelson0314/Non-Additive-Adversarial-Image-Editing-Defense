#!/usr/bin/env bash
# 一次把整批派工送出。**先跑 `nvidia-smi` 看卡，卡是多人共用的。**
#
# 兩台各 8 張 3090，每卡 2 個 process（hop 8 每個約 9 GB）。這一支把 28 個
# 工作點分到指定的卡號上，卡號由參數給——寫死過一次就把八個 process 送到
# 別人正在用的四張卡上。
#
# 用法：
#     bash scripts/dispatch_all.sh one "<卡號>"    # basic-1 那一半
#     bash scripts/dispatch_all.sh two "<卡號>"    # basic-2 那一半
#
# 分組的理由：
#   one  = 補完被斷線打斷的四批（純相位＋下限、gain 延伸、rank 分配）
#          ＋ 新的純加性對照。這幾件都是等在飛的結論。
#   two  = DCT-Shield 自己的抗 JPEG 設定（先前測錯的那一格）
#          ＋ 隨機化幾何 EOT（唯一有機會改善裁切的東西）。
set -uo pipefail
ROOT=/nfs/home/nelson0314/WACV-s3
cd "$ROOT"

WHICH="${1:-}"
DEVS="${2:-0 1 2 3 4 5 6 7}"
case "$WHICH" in
  one)
    set -- $DEVS
    bash scripts/phase_floor_sweep.sh    "${1:-0} ${2:-1}"
    bash scripts/gain_reach_extension.sh "${3:-2} ${4:-3}"
    bash scripts/floor_gate_sweep.sh     "${5:-4} ${6:-5}" "complement_rank"
    bash scripts/floor_only_sweep.sh     "${7:-6} ${8:-7}"
    ;;
  two)
    set -- $DEVS
    bash scripts/dct_antijpeg_configs.sh "${1:-0} ${2:-1} ${3:-2} ${4:-3}"
    bash scripts/eot_geometry_sweep.sh   "${5:-4} ${6:-5}"
    ;;
  *) echo "用法：$0 one|two \"<卡號>\""; exit 2 ;;
esac
echo "[dispatch] $WHICH 送出完畢（$(date)）"

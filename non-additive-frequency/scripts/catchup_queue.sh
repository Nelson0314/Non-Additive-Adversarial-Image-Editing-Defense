#!/usr/bin/env bash
# 斷線之後的補跑佇列。**兩段，依「以抗淨化為主」排序。**
#
# 斷線期間的實際進度（由遠端讀回）：
#   ip2p_deliver_jpeg_patch   55 份 results，三個 QD 臂齊
#   ip2p_patch_res            s08／s16 齊；s24／s32／s16_rand 從沒開始
#   佇列在「卡不是空的」那一步停住，抗淨化一格都沒量
#
# 故這一支只做兩件，其餘放掉（s24／s32／Voronoi／極座標都不擋現有主張）：
#
#   一　已訓練好的臂的十二算子抗淨化（QD 三臂 ＋ 帶限 s08／s16）
#       不需訓練，防禦圖已在遠端。這一段直接填滿目前整片空白的抗淨化表。
#   二　兩個新臂（`scripts/facelock_colour_round.sh`）
#       `fl_patch`  身分損失接在 VAE 的一次往返上，不經過 UNet
#       `col_full`  色彩濾鏡作用在**整張圖**上，不受支撐限制
#       使用者裁定：**不跑 s16_rand**，機時改給這兩個。
#       代價要記著：帶限批因此沒有自己的地板，`s08`／`s16` 的效果數字
#       只能與 `free` 臂的隨機對照並列，不是同參數化的地板。
#
# 用法：bash scripts/catchup_queue.sh "<卡號…>"
set -uo pipefail
ROOT=/nfs/home/nelson0314/WACV-s3
cd "$ROOT" || { echo "錯誤：找不到 $ROOT" >&2; exit 2; }
DEVS="${1:-0 1 3 4 5}"
LOG=runs/execution_logs/catchup.log
mkdir -p runs/execution_logs

echo "=== [$(date '+%F %T')] 補跑佇列啟動　卡 $DEVS ===" | tee -a "$LOG"

echo "=== [$(date '+%F %T')] 一之一　量化交付批的抗淨化 ===" | tee -a "$LOG"
bash scripts/jpeg_band_purify_round.sh "$DEVS" deliver_jpeg 2>&1 | tee -a "$LOG"
rc=${PIPESTATUS[0]}
[ "$rc" -ne 0 ] && echo "=== [$(date '+%F %T')] 一之一 中止 rc=$rc ===" | tee -a "$LOG"

echo "=== [$(date '+%F %T')] 一之二　帶限批 s08／s16 的抗淨化 ===" | tee -a "$LOG"
bash scripts/jpeg_band_purify_round.sh "$DEVS" patch_res "s08 s16" 2>&1 | tee -a "$LOG"
rc=${PIPESTATUS[0]}
[ "$rc" -ne 0 ] && echo "=== [$(date '+%F %T')] 一之二 中止 rc=$rc ===" | tee -a "$LOG"

echo "=== [$(date '+%F %T')] 二　facelock 損失 ＋ 全圖色彩濾鏡 ===" | tee -a "$LOG"
bash scripts/facelock_colour_round.sh "$DEVS" "fl_patch col_full" 2>&1 | tee -a "$LOG"
rc=${PIPESTATUS[0]}
[ "$rc" -ne 0 ] && echo "=== [$(date '+%F %T')] 二 中止 rc=$rc ===" | tee -a "$LOG"

echo "=== [$(date '+%F %T')] 補跑佇列收工 ===" | tee -a "$LOG"

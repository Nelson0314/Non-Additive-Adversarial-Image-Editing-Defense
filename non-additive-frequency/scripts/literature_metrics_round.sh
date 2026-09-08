#!/usr/bin/env bash
# 用文獻的指標把內容軸與損失軸的每一格重算一次。**CPU、不佔卡。**
#
# 為什麼要跑這一批
#   本專案現行的身分讀數只有一個辨識器（facenet 的 InceptionResnetV1）。
#   單張的燒測已經看到三個辨識器**幅度差很大**：`free` 在 facenet 上掉 1.07、
#   在 ArcFace 上掉 0.71、在 AdaFace 上只掉 0.20，而 `tint` 在 AdaFace 上
#   反而上升。單一辨識器的結論可能是那個網路的特性，不是防禦的性質。
#
#   同時補上 CLIP-S（FaceLock 的六個指標裡本專案唯一沒有的那個）。
#
# 臂與來源目錄的對應和 `runs/ip2p_content_constraint/summarise.py` 一致：
# `free` 與 `pretty` 在 `ip2p_face_defence/`（`*_plain_*`／`*_pretty_*`），
# 其餘在 `ip2p_content_constraint/`。
#
# 用法：bash scripts/literature_metrics_round.sh ["<臂…>"]
set -uo pipefail
ROOT=/nfs/home/nelson0314/WACV-s3
PY="$HOME/venvs/wacv/bin/python"
export PYTHONPATH="$ROOT" HF_HOME="$HOME/hf_cache" PYTHONIOENCODING=utf-8
export TOKENIZERS_PARALLELISM=false
# **不佔卡**：三個辨識器與 CLIP 全部走 CPU。空字串會讓 torch 看不到任何卡。
export CUDA_VISIBLE_DEVICES=""
# 這一批曾經 segfault；faulthandler 在**任何** python 呼叫之前就要開，
# 只在正式那一行之前 export 的話 preflight 掛掉時拿不到位置。
export PYTHONFAULTHANDLER=1
cd "$ROOT" || exit 2

ARMS="${1:-free tint lowproj chroma lowproj_tile rand pretty identity identity_layout}"
CATS="clothing accessory background"
OUT=runs/ip2p_content_constraint/literature_metrics.csv
LOG=runs/execution_logs/literature_metrics.log

dir_for() {  # $1=臂 $2=類別 $3=影像
  case "$1" in
    free)   echo "runs/ip2p_face_defence/$2_plain_$3" ;;
    pretty) echo "runs/ip2p_face_defence/$2_pretty_$3" ;;
    *)      echo "runs/ip2p_content_constraint/$1_$2_$3" ;;
  esac
}

# 影像清單由 `free` 臂實際跑過的格推導，與派工腳本同一個作法——目錄檔登記了
# 20 張而只跑了 10 張，用目錄檔會多出十格沒有對照的列。
# **`-type d`：只取目錄。** 派工把每一格的 stdout 存成 `<格名>.log`，就落在
# 格目錄旁邊，`ls -d ...task_*` 會把日誌檔也算成影像（十個目錄變二十個）。
IMGS=$(find runs/ip2p_face_defence -maxdepth 1 -type d -name 'clothing_plain_task_*' \
       2>/dev/null | sed 's|.*/clothing_plain_||' | sort)
[ -z "$IMGS" ] && { echo "錯誤：推導不出影像清單" >&2; exit 2; }
echo "影像 $(echo $IMGS | wc -w) 張、臂 $(echo $ARMS | wc -w) 個" | tee -a "$LOG"

# **清單走檔案不走 argv。** 270 個 `--entry` 會讓 `import onnxruntime`
# segfault（`scripts/literature_metrics.py` 的 `_read_entries` 有完整說明），
# 而症狀是 core dump 不是錯誤訊息。
LIST=runs/execution_logs/literature_entries.txt
: > "$LIST"
MISSING=0
for arm in $ARMS; do
  for cat in $CATS; do
    for img in $IMGS; do
      d=$(dir_for "$arm" "$cat" "$img")
      if [ -f "$d/results.csv" ]; then
        echo "${arm}_${cat}_${img}=$d" >> "$LIST"
      else
        MISSING=$((MISSING + 1))
      fi
    done
  done
done
N=$(wc -l < "$LIST")
echo "找到 $N 格，缺 $MISSING 格（那些臂還沒跑完，不是錯誤）" | tee -a "$LOG"
[ "$N" -eq 0 ] && { echo "錯誤：一格都沒有" >&2; exit 2; }

# preflight 走一次 parser 與全部檔案檢查，不載任何權重。
"$PY" scripts/literature_metrics.py --out "$OUT" \
  --entries-file "$LIST" --preflight > /dev/null \
  || { echo "錯誤：preflight 失敗" >&2; exit 2; }
echo "preflight 通過" | tee -a "$LOG"

# `nice` 降優先權：CPU 是共用的，這一批跑一小時，不該影響別人的互動工作。
# **`-u`（不緩衝）＋ `PYTHONFAULTHANDLER`**：這一批曾經 segfault，而
# `| tee` 讓 stdout 變成 block buffering，於是掛掉時最後幾行 print 還在
# buffer 裡沒寫出來——日誌上看起來像是「載完模型就死了」，實際位置在更後面。
nice -n 10 "$PY" -u scripts/literature_metrics.py --out "$OUT" \
  --entries-file "$LIST" 2>&1 | tee -a "$LOG"
echo "=== [$(date '+%F %T')] 寫入 $OUT ===" | tee -a "$LOG"

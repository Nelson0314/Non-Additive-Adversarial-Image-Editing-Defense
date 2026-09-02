#!/usr/bin/env bash
# 語意誘餌批次：把六張影像攤到指定的卡上，各自跑完整個誘餌目錄。
#
# 這一族的產物裡沒有噪聲，淨化在定義上是恆等（`scripts/semantic_decoy.py`
# 的模組 docstring 有完整說明）。成本很低：每張影像是
# 1（原圖的攻擊編輯）＋ 誘餌數 × 2（誘餌本身 ＋ 對誘餌的攻擊編輯）次編輯，
# 十個誘餌就是 21 次，約 6 分鐘。
#
# **它可以與別的批次共用卡**，只要那張卡上的 process 不超過兩個。
#
# 用法：bash scripts/decoy_sweep.sh "<卡號…>" [組…] [影像…]
set -uo pipefail
ROOT=/nfs/home/nelson0314/WACV-s3
PY="$HOME/venvs/wacv/bin/python"
export PYTHONPATH="$ROOT" HF_HOME="$HOME/hf_cache" PYTHONIOENCODING=utf-8
export TOKENIZERS_PARALLELISM=false
cd "$ROOT" || { echo "錯誤：找不到 $ROOT" >&2; exit 2; }

DEVS=(${1:-})
[ ${#DEVS[@]} -lt 1 ] && { echo "用法：$0 \"<卡號…>\" [組…]" >&2; exit 2; }
# 卡是多人共用的。**這個檢查會 exit，不是印出來就算。**
bash scripts/free_cards.sh --assert "${DEVS[*]}" || exit 3

OUT=runs/ip2p_decoy
mkdir -p "$OUT"
# **不可以叫 `GROUPS`。** 那是 bash 的內建陣列（當前使用者的群組 ID），
# 指派會被**靜默忽略**，`$GROUPS` 展開成主要群組 ID。實測踩過：三個 process
# 都以「目錄裡沒有這些組：['2068']」立刻死掉，而 2068 就是 `id -g`。
# 同一類的名字還有 `UID`／`RANDOM`／`SECONDS`／`LINENO`／`PWD`。
DECOY_GROUPS="${2:-background collision}"

# 預設兩張（盆栽人與瑪利歐）。第三個參數可以覆寫。
IMGS=(${3:-task_attr_mod_color_11699 task_attr_mod_color_6205})

# 每張卡分一段影像。分段而不是一卡一張，是因為每張影像自己的
# `編輯(原圖)` 只算一次，同一個 process 內可以重用。
n=${#DEVS[@]}
i=0
for dev in "${DEVS[@]}"; do
  chunk=""
  j=$i
  while [ $j -lt ${#IMGS[@]} ]; do
    chunk="$chunk ${IMGS[$j]}"
    j=$(( j + n ))
  done
  i=$(( i + 1 ))
  [ -z "$chunk" ] && continue
  # **每張卡一個獨立的輸出目錄**：每寫一列是整份重寫 CSV，兩個 process
  # 寫同一個目錄會互相蓋掉（`docs/OPERATIONS.md` 記過）。
  sub="$OUT/dev$dev"
  CUDA_VISIBLE_DEVICES="$dev" setsid nohup "$PY" scripts/semantic_decoy.py \
      --out "$sub" --images $chunk --groups $DECOY_GROUPS \
      < /dev/null > "$OUT/dev$dev.log" 2>&1 &
  disown
  echo "[decoy] dev=$dev 影像：$chunk"
done

sleep 20
echo "啟動了 $(ps -u "$USER" -o cmd | grep -c '[s]emantic_decoy') 個 semantic_decoy process（$(date)）"

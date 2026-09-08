#!/usr/bin/env bash
# 新前提下的第一批：臉是受保護主體，成功的定義是「編輯後認不出是誰」。
#
# 換掉了什麼
#   主體    ATR 的 Face+Hair（`--subject-source face`），不再是 CLIPSeg 依
#           登記名詞猜的物件。載體（衣服）與主體（臉）天然分離。
#   指令    三類有意義的編輯（`data/attack_prompts.yaml`）：改變衣著或顏色、
#           加上配件、變換背景與背景物品。兩類完全固定，只有 clothing 的
#           衣物名詞逐張變。
#   載體    吸附 → 挖掉臉框 → 內縮 → 往內羽化，四個旗標。
#   讀數    身分相似度由 `scripts/identity_probe.py` 事後算，不在這裡。
#
# 這一批固定 `--attack-category`，逐張分片。一張圖三類就是三次派工。
#
# 用法：bash scripts/face_defence_round.sh "<卡號…>" <clothing|accessory|background> [載體組]
#       載體組：plain（預設，精修但不美化）／tint（＋tint 3.0）／scatter（分散 6 塊）
set -uo pipefail
ROOT=/nfs/home/nelson0314/WACV-s3
PY="$HOME/venvs/wacv/bin/python"
export PYTHONPATH="$ROOT" HF_HOME="$HOME/hf_cache" PYTHONIOENCODING=utf-8
export TOKENIZERS_PARALLELISM=false
cd "$ROOT" || { echo "錯誤：找不到 $ROOT" >&2; exit 2; }

DEVS=(${1:-})
CAT="${2:-}"
GROUP="${3:-plain}"
[ ${#DEVS[@]} -lt 1 ] && { echo "用法：$0 \"<卡號…>\" <類別> [載體組]" >&2; exit 2; }
case "$CAT" in clothing|accessory|background) ;;
  *) echo "第二個參數必須是 clothing／accessory／background" >&2; exit 2 ;; esac
case "$GROUP" in
  plain)   EXTRA="" ;;
  tint)    EXTRA="--patch-tint 3.0" ;;
  scatter) EXTRA="--carrier-scatter 6" ;;
  # 美化組：小面積 ＋ 分散 ＋ 色調貼合衣服。存在理由是「鋪滿整件衣服時，
  # 改那件衣服的編輯本來就改不動」——覆蓋率不降下來，clothing 那一類的
  # 結果近乎恆真。支撐由全圖的 26% 降到 5%。
  pretty)  EXTRA="--carrier-scatter 6 --patch-tint 3.0" ;;
  *) echo "第三個參數必須是 plain／tint／scatter／pretty" >&2; exit 2 ;;
esac
bash scripts/free_cards.sh --assert "${DEVS[*]}" || exit 3

OUT=runs/ip2p_face_defence
mkdir -p "$OUT"

# 影像清單直接由目錄檔生成——寫死在腳本裡的話，目錄改了而清單沒改不會報錯。
IMGS=$("$PY" - <<'PYEOF'
import yaml
spec = yaml.safe_load(open("data/attack_prompts.yaml", encoding="utf-8"))
print(" ".join(sorted(spec["images"])))
PYEOF
)
NIMG=$(echo $IMGS | wc -w)
[ "$NIMG" -eq 0 ] && { echo "錯誤：目錄檔裡沒有影像。" >&2; exit 2; }

COMMON="--data data/omniedit150 --conditions patch \
--attack-prompts data/attack_prompts.yaml --attack-category $CAT \
--subject-source face \
--loss image_guidance --ig-zt diffuse_src \
--subject-mask data/attack_prompts.yaml \
--patch-placement complement --radius 0.05 --patch-carrier clothes \
--carrier-refine 4 --carrier-erode 3 --carrier-feather 8 \
--steps 6000 --step-size 0.01 --eval-every 200 --eval-draws 8 \
--patience 10 --min-delta 0.0002 --save-weights"

ONLY="${4:-}"
SEL=""
for img in $IMGS; do
  if [ -n "$ONLY" ]; then case "$img" in $ONLY*) ;; *) continue ;; esac; fi
  SEL="$SEL $img"
done
N=$(echo $SEL | wc -w)
[ "$N" -eq 0 ] && { echo "錯誤：過濾器 '$ONLY' 一張都沒對上。" >&2; exit 2; }
if [ "$N" -gt $(( ${#DEVS[@]} * 2 )) ]; then
  echo "錯誤：$N 張影像需要至少 $(( (N + 1) / 2 )) 張卡，只給了 ${#DEVS[@]} 張。" >&2
  exit 2
fi

i=0
for img in $SEL; do
  dev=${DEVS[$(( i % ${#DEVS[@]} ))]}
  i=$(( i + 1 ))
  CUDA_VISIBLE_DEVICES="$dev" setsid nohup "$PY" scripts/ip2p_run.py \
      --out "$OUT/${CAT}_${GROUP}_$img" --images "$img" $COMMON $EXTRA \
      < /dev/null > "$OUT/${CAT}_${GROUP}_$img.log" 2>&1 &
  disown
  echo "[face] $CAT/$GROUP $img dev=$dev"
done

sleep 25
echo "啟動了 $(ps -u "$USER" -o cmd | grep '[i]p2p_run.py' | grep -c python) 行 ip2p_run（$(date)）"

#!/usr/bin/env bash
# 把花紋放在**衣服**上：補丁的支撐再交集一層語意載體。
#
# 起點
# `runs/ip2p_patch_direct/` 的補集補丁在主體逐位元不動的前提下拿到主體內位移
# 0.57–0.60，但支撐是整個主體補集——含牆面與地板，於是花紋貼在牆上，讀起來
# 是一塊壞掉的區域。本輪把支撐交集 ATR 人體解析的衣物類別
# （`src/defense/carrier_mask.py`），花紋止於衣物輪廓。
#
# 只跑瑪利歐一張
# `task_attr_mod_color_11699`（盆栽人）**沒有合法載體**：`person` 是登記的
# 主體，衣服全部落在受保護核心內（本機實測 legal_clothes = 0.0000），
# `PatchParam.reset` 會拋錯。那不是缺陷是事實，故該張不在本輪。
#
# 瑪利歐上的三個載體面積（交集之前／之後）：
#   clothes 0.2315 / 0.0861      upper 0.0634 / 0.0178      pants 0.1682 / 0.0683
# 交集之後只剩左邊那尊——現行主體遮罩把右邊那尊整個算成主體。
#
# 六格全部與 `runs/ip2p_patch_direct/ig_comp_subj` 只差載體與內容旋鈕：
#
#   clothes_plain    載體 clothes，內容逐像素學（與 ig_comp_subj 只差載體）
#   clothes_tile     ＋ 64×64 平舖：規則重複讀起來是布料印花
#   clothes_tile_a   ＋ 不透明度 0.6：衣服的明暗從花紋底下透出來
#   upper_tile       只有上衣（面積 0.0178）
#   pants_tile       只有吊帶褲（面積 0.0683）
#   clothes_rand     同一個支撐、不最佳化。梯度買到多少由這一格減出來
#
# 用法：bash scripts/patch_carrier_round.sh "<卡號…>" [tag 前綴]
set -uo pipefail
ROOT=/nfs/home/nelson0314/WACV-s3
PY="$HOME/venvs/wacv/bin/python"
export PYTHONPATH="$ROOT" HF_HOME="$HOME/hf_cache" PYTHONIOENCODING=utf-8
export TOKENIZERS_PARALLELISM=false
cd "$ROOT" || { echo "錯誤：找不到 $ROOT" >&2; exit 2; }

DEVS=(${1:-})
[ ${#DEVS[@]} -lt 1 ] && { echo "用法：$0 \"<卡號…>\" [tag 前綴]" >&2; exit 2; }
bash scripts/free_cards.sh --assert "${DEVS[*]}" || exit 3

# 解析器的權重要先在**單一** process 裡抓下來。六個 process 同時去抓同一個
# repo 會寫壞 HF 的快取目錄，症狀是隨機幾個 process 死在讀不完整的檔案上。
"$PY" - <<'PYEOF' || exit 4
from transformers import AutoModelForSemanticSegmentation, SegformerImageProcessor
from src.defense.carrier_mask import CARRIER_REPO
SegformerImageProcessor.from_pretrained(CARRIER_REPO)
AutoModelForSemanticSegmentation.from_pretrained(CARRIER_REPO)
print("carrier weights cached:", CARRIER_REPO)
PYEOF

OUT=runs/ip2p_patch_carrier
mkdir -p "$OUT"
MARIO=task_attr_mod_color_6205

COMMON="--data data/omniedit150 --images $MARIO --conditions patch \
--loss image_guidance --ig-zt diffuse_src --ig-weight subject \
--subject-mask data/decoy_catalogue.yaml --patch-placement complement --radius 0.05 \
--steps 6000 --step-size 0.01 --eval-every 200 --eval-draws 8 \
--patience 10 --min-delta 0.0002 --save-weights"

POINTS="
clothes_plain:--patch-carrier~clothes
clothes_tile:--patch-carrier~clothes~--patch-tile~64
clothes_tile_a:--patch-carrier~clothes~--patch-tile~64~--patch-alpha~0.6
upper_tile:--patch-carrier~upper~--patch-tile~64
pants_tile:--patch-carrier~pants~--patch-tile~64
clothes_rand:--conditions~patch_rand~--patch-carrier~clothes
"

n=$(echo "$POINTS" | grep -c ':')
if [ "$n" -gt $(( ${#DEVS[@]} * 2 )) ]; then
  echo "錯誤：$n 個 process 需要至少 $(( (n + 1) / 2 )) 張卡，只給了 ${#DEVS[@]} 張。" >&2
  exit 2
fi

ONLY="${2:-}"
i=0
while IFS= read -r line; do
  [ -z "$line" ] && continue
  tag="${line%%:*}"; flags="${line#*:}"
  if [ -n "$ONLY" ]; then
    case "$tag" in $ONLY*) ;; *) continue ;; esac
  fi
  dev=${DEVS[$(( i % ${#DEVS[@]} ))]}
  i=$(( i + 1 ))
  args=$(echo "$flags" | tr '~' ' ')
  CUDA_VISIBLE_DEVICES="$dev" setsid nohup "$PY" scripts/ip2p_run.py \
      --out "$OUT/$tag" $COMMON $args \
      < /dev/null > "$OUT/$tag.log" 2>&1 &
  disown
  echo "[carrier] $tag dev=$dev  $args"
done <<< "$POINTS"

sleep 25
echo "啟動了 $(ps -u "$USER" -o cmd | grep '[i]p2p_run.py' | grep -c python) 行 ip2p_run（$(date)）"

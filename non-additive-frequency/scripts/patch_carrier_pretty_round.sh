#!/usr/bin/env bash
# 兩組獨立的問題，共用一次派工。
#
# A · 美化（載體 = ATR 衣服）
# `runs/ip2p_patch_carrier/` 已經量過平舖與半透明在衣服載體上的代價
# （0.2732 → 0.2342 → 0.2028）。缺的是**平滑化**：`--patch-tv` 把內容的
# 總變差壓下去，逐像素噪點變成柔和色塊。四格把三個旋鈕的組合補齊。
#
# B · 固定形狀浮水印（不管美觀與失真）
# 支撐是 `count` 塊固定的方塊，內容仍然學。擺放走 `crop_safe`：候選限制在
# **裁切後仍留存的中央方框**內（`--patch-crop-keep 0.8` 對上 `crop_resize0.1`
# 每邊裁 10%），並取最靠畫面中心者——裁切是繞中心的，離中心越近存活率越高。
# `--radius` 是**加起來**的面積，故 `sq1_a04` 與 `sq2_a04` 是同一個總預算下
# 「一塊」對「拆成兩塊」的直接對照。
#
# 只跑瑪利歐一張：盆栽人的 `person` 是登記主體，沒有合法衣物載體。
# B 組不受該限制，但為了與 A 組同一張圖上並列，這一輪一併只跑瑪利歐。
#
# 用法：bash scripts/patch_carrier_pretty_round.sh "<卡號…>" [tag 前綴]
set -uo pipefail
ROOT=/nfs/home/nelson0314/WACV-s3
PY="$HOME/venvs/wacv/bin/python"
export PYTHONPATH="$ROOT" HF_HOME="$HOME/hf_cache" PYTHONIOENCODING=utf-8
export TOKENIZERS_PARALLELISM=false
cd "$ROOT" || { echo "錯誤：找不到 $ROOT" >&2; exit 2; }

DEVS=(${1:-})
[ ${#DEVS[@]} -lt 1 ] && { echo "用法：$0 \"<卡號…>\" [tag 前綴]" >&2; exit 2; }
bash scripts/free_cards.sh --assert "${DEVS[*]}" || exit 3

OUT=runs/ip2p_patch_shape
mkdir -p "$OUT"
MARIO=task_attr_mod_color_6205

COMMON="--data data/omniedit150 --images $MARIO --conditions patch \
--loss image_guidance --ig-zt diffuse_src --ig-weight subject \
--subject-mask data/decoy_catalogue.yaml \
--steps 6000 --step-size 0.01 --eval-every 200 --eval-draws 8 \
--patience 10 --min-delta 0.0002 --save-weights"

CARRIER="--patch-placement complement --radius 0.05 --patch-carrier clothes"
SHAPE="--patch-placement crop_safe --patch-crop-keep 0.8"

POINTS="
tv03:$CARRIER~--patch-tv~0.3
tv10:$CARRIER~--patch-tv~1.0
tile_tv:$CARRIER~--patch-tile~64~--patch-tv~0.3
pretty:$CARRIER~--patch-tile~64~--patch-alpha~0.6~--patch-tv~0.3
sq1_a02:$SHAPE~--radius~0.02~--patch-count~1
sq1_a04:$SHAPE~--radius~0.04~--patch-count~1
sq2_a04:$SHAPE~--radius~0.04~--patch-count~2
sq2_a04_rand:$SHAPE~--radius~0.04~--patch-count~2~--conditions~patch_rand
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
  echo "[shape] $tag dev=$dev  $args"
done <<< "$POINTS"

sleep 25
echo "啟動了 $(ps -u "$USER" -o cmd | grep '[i]p2p_run.py' | grep -c python) 行 ip2p_run（$(date)）"

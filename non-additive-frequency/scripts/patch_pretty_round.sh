#!/usr/bin/env bash
# 「美化」與效果的取捨：把標記從雜訊推向刻意設計的樣子，看效果掉多少。
#
# 起點
# `runs/ip2p_patch_direct/` 的補集補丁在主體逐位元不動的前提下拿到主體內位移
# 0.57–0.60（相位族是 0.41–0.67），但**產物是彩色雜訊**——讀起來是「這塊
# 壞掉了」而不是「這裡有一個標記」。
#
# 三個正交的旋鈕，各自一格，全部與 `ig_comp_subj`（現行最好的那一格）只差
# 一個變因：
#
#   tile64   只學一塊 64×64 的磚，整片重複。規則重複是「刻意放上去」的視覺
#            訊號，參數量同時由 3·512² 降到 3·64²（64 倍）。
#   a050     不透明度 0.5：原圖從標記底下透出來，讀成疊上去的浮水印。
#   tv       內容的總變差懲罰：柔和色塊而不是逐像素噪點。
#   pretty   三個一起開——「最像設計品」的那一端。
#
# 全部用 `image_guidance` ＋ 主體加權：`runs/ip2p_patch_direct/` 量到那個組合
# 與新寫的 `edit_divergence` 打平甚至略勝，故取較簡單的那一個當基底。
#
# 用法：bash scripts/patch_pretty_round.sh "<卡號…>" [tag 前綴]
set -uo pipefail
ROOT=/nfs/home/nelson0314/WACV-s3
PY="$HOME/venvs/wacv/bin/python"
export PYTHONPATH="$ROOT" HF_HOME="$HOME/hf_cache" PYTHONIOENCODING=utf-8
export TOKENIZERS_PARALLELISM=false
cd "$ROOT" || { echo "錯誤：找不到 $ROOT" >&2; exit 2; }

DEVS=(${1:-})
[ ${#DEVS[@]} -lt 1 ] && { echo "用法：$0 \"<卡號…>\" [tag 前綴]" >&2; exit 2; }
bash scripts/free_cards.sh --assert "${DEVS[*]}" || exit 3

OUT=runs/ip2p_patch_pretty
mkdir -p "$OUT"
IMGS="task_attr_mod_color_11699 task_attr_mod_color_6205"

COMMON="--data data/omniedit150 --images $IMGS --conditions patch \
--loss image_guidance --ig-zt diffuse_src --ig-weight subject \
--subject-mask data/decoy_catalogue.yaml --patch-placement complement --radius 0.05 \
--steps 6000 --step-size 0.01 --eval-every 200 --eval-draws 8 \
--patience 10 --min-delta 0.0002 --save-weights"

POINTS="
tile64:--patch-tile~64
a050:--patch-alpha~0.5
tv03:--patch-tv~0.3
pretty:--patch-tile~64~--patch-alpha~0.5~--patch-tv~0.3
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
  echo "[pretty] $tag dev=$dev  $args"
done <<< "$POINTS"

sleep 25
echo "啟動了 $(ps -u "$USER" -o cmd | grep '[i]p2p_run.py' | grep -c python) 行 ip2p_run（$(date)）"

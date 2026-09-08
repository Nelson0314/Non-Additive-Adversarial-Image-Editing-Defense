#!/usr/bin/env bash
# 可見的**可學**補丁：主體之外一塊高幅度區域，內容由 PGD 學出來。
#
# 這一輪補的缺口
# `runs/patch_probe/` 已經量過**固定內容**的補丁（噪聲／中灰／照片）：把主體
# 以外的每一個像素換成白噪聲，主體內位移只有 0.111–0.149；角落一塊最大的
# 合法方塊只有 0.010–0.036。相位族在更低的失真下是 0.41–0.67。
#
# 那一批留下兩件沒做的事，這一輪就是那兩件：
#   1. 內容沒有最佳化過。
#   2. 損失從來沒有寫成「只對**主體**那些 latent token 負責」
#      （`--ig-weight subject`）。
#
# 面積的上限由遮罩決定，不是由我們挑：兩張影像與主體遮罩零重疊的最大邊長只有
# 120（盆栽人）與 166（瑪利歐），故 radius=0.05（邊長 114）是**兩張都放得下
# 的最大值**。瑪利歐另外單獨跑一格 radius=0.10（邊長 162）。
#
# `patch_rand` 是關鍵對照：**完全相同的幾何**、完全相同的面積與位置，
# 只差沒有梯度。兩者的差就是「最佳化買到了什麼」。
#
# 用法：bash scripts/patch_learned_round.sh "<卡號…>" [tag 前綴]
set -uo pipefail
ROOT=/nfs/home/nelson0314/WACV-s3
PY="$HOME/venvs/wacv/bin/python"
export PYTHONPATH="$ROOT" HF_HOME="$HOME/hf_cache" PYTHONIOENCODING=utf-8
export TOKENIZERS_PARALLELISM=false
cd "$ROOT" || { echo "錯誤：找不到 $ROOT" >&2; exit 2; }

DEVS=(${1:-})
[ ${#DEVS[@]} -lt 1 ] && { echo "用法：$0 \"<卡號…>\" [tag 前綴]" >&2; exit 2; }
bash scripts/free_cards.sh --assert "${DEVS[*]}" || exit 3

OUT=runs/ip2p_patch_learned
mkdir -p "$OUT"
BOTH="task_attr_mod_color_11699 task_attr_mod_color_6205"
MARIO=task_attr_mod_color_6205

# 主體遮罩是必要的：補丁要放在主體之外。形狀參數與其他批次相同。
COMMON="--data data/omniedit150 --loss image_guidance --ig-zt diffuse_src \
--ig-weight subject --subject-mask data/decoy_catalogue.yaml \
--steps 3000 --step-size 0.01 --eval-every 200 --eval-draws 8 \
--patience 10 --min-delta 0.0002 --save-weights"

POINTS="
a05_far:--conditions~patch~--radius~0.05~--patch-placement~far:$BOTH
a05_rand:--conditions~patch_rand~--radius~0.05~--patch-placement~far:$BOTH
a05_near:--conditions~patch~--radius~0.05~--patch-placement~near:$BOTH
a10_far_mario:--conditions~patch~--radius~0.10~--patch-placement~far:$MARIO
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
  tag="${line%%:*}"; rest="${line#*:}"
  flags="${rest%%:*}"; imgs="${rest#*:}"
  if [ -n "$ONLY" ]; then
    case "$tag" in $ONLY*) ;; *) continue ;; esac
  fi
  dev=${DEVS[$(( i % ${#DEVS[@]} ))]}
  i=$(( i + 1 ))
  args=$(echo "$flags" | tr '~' ' ')
  CUDA_VISIBLE_DEVICES="$dev" setsid nohup "$PY" scripts/ip2p_run.py \
      --out "$OUT/$tag" --images $imgs $COMMON $args \
      < /dev/null > "$OUT/$tag.log" 2>&1 &
  disown
  echo "[patch] $tag dev=$dev  $args"
done <<< "$POINTS"

sleep 25
echo "啟動了 $(ps -u "$USER" -o cmd | grep '[i]p2p_run.py' | grep -c python) 行 ip2p_run（$(date)）"

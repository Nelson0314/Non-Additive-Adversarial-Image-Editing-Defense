#!/usr/bin/env bash
# 可見補丁 ＋ **直接以防禦效果為目標**的損失，訓練到收斂。
#
# 損失
# `edit_divergence`（`src/defense/edit_divergence_loss.py`）：
#
#     L = − E_{t,ε} ‖ x̂₀(z_t ; c_I(x_def)) − x̂₀(z_t ; c_I(x)) ‖²
#
# 不指定要把影像條件推到哪裡，只要求**模型建出來的東西離原圖的結果遠**——
# 那正是位移。文字條件取空字串，因為威脅模型的前提是攻擊指令未知。
#
# 支撐
# `--patch-placement complement` 用**整個主體遮罩補集**，是「主體逐位元不動」
# 這個約束底下可用面積的最大值（兩張影像分別是 39.4% 與 37.3%）。
# 方形的那一格是對照：面積小很多，但形狀是論文那一族的樣子。
#
# 收斂
# 六千步、每兩百步以固定抽樣評估、連續十次無改善才停。停止原因逐列進 CSV。
#
# 用法：bash scripts/patch_direct_round.sh "<卡號…>" [tag 前綴]
set -uo pipefail
ROOT=/nfs/home/nelson0314/WACV-s3
PY="$HOME/venvs/wacv/bin/python"
export PYTHONPATH="$ROOT" HF_HOME="$HOME/hf_cache" PYTHONIOENCODING=utf-8
export TOKENIZERS_PARALLELISM=false
cd "$ROOT" || { echo "錯誤：找不到 $ROOT" >&2; exit 2; }

DEVS=(${1:-})
[ ${#DEVS[@]} -lt 1 ] && { echo "用法：$0 \"<卡號…>\" [tag 前綴]" >&2; exit 2; }
bash scripts/free_cards.sh --assert "${DEVS[*]}" || exit 3

OUT=runs/ip2p_patch_direct
mkdir -p "$OUT"
BOTH="task_attr_mod_color_11699 task_attr_mod_color_6205"
MARIO=task_attr_mod_color_6205

COMMON="--data data/omniedit150 --conditions patch --ig-zt diffuse_src \
--subject-mask data/decoy_catalogue.yaml --radius 0.05 \
--steps 6000 --step-size 0.01 --eval-every 200 --eval-draws 8 \
--patience 10 --min-delta 0.0002 --save-weights"

# tag:旗標:影像
POINTS="
div_comp_subj:--loss~edit_divergence~--patch-init~random~--ig-weight~subject~--patch-placement~complement:$BOTH
div_comp_uni:--loss~edit_divergence~--patch-init~random~--patch-placement~complement:$BOTH
ig_comp_subj:--loss~image_guidance~--ig-weight~subject~--patch-placement~complement:$BOTH
div_a05_subj:--loss~edit_divergence~--patch-init~random~--ig-weight~subject~--patch-placement~far:$BOTH
div_a10_mario:--loss~edit_divergence~--patch-init~random~--ig-weight~subject~--patch-placement~far~--radius~0.10:$MARIO
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
  echo "[direct] $tag dev=$dev  $args"
done <<< "$POINTS"

sleep 25
echo "啟動了 $(ps -u "$USER" -o cmd | grep '[i]p2p_run.py' | grep -c python) 行 ip2p_run（$(date)）"

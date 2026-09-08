#!/usr/bin/env bash
# 平滑化的甜蜜點：`--patch-tv` 掃描，以及 α 單獨用的代價。
#
# 為什麼掃這一條
# `runs/ip2p_patch_shape/` 量到三個美化旋鈕**強烈次可加**（主體區位移，
# 基準是 `clothes_plain` 的 0.2732）：
#
#     tv 1.0 單獨      0.2064   （76%）
#     平舖 單獨        0.2342   （86%，取自 ip2p_patch_carrier）
#     平舖 ＋ tv 0.3   0.0875   （32%）
#     三個全開         0.0518   （19%，隨機對照是 0.0265）
#
# 也就是說**單獨用時 tv 最便宜，組合起來卻塌得最兇**。色彩族當初的結論是
# 「中間沒有甜蜜點」（`OVERNIGHT_FINDINGS.md` §一），這一組問補丁族是不是
# 同樣沒有：tv ∈ {0.3, 0.5, 1.0, 2.0, 3.0} 連成一條曲線（0.3 與 1.0 已有）。
#
# `a06_only` 補上從未單獨跑過的 α——先前 α 一律與平舖綁在一起，兩者的貢獻
# 分不開。
#
# `sq1_a04_far` 是 `crop_safe` 的**同批次對照**：同樣 4% 一塊方塊，改放在
# 離主體最遠處（舊的預設）。沒有它就分不出「擺在中央」本身要不要付效果的
# 代價，而那正是 B 組唯一要回答的問題的另一半。
#
# 用法：bash scripts/patch_smooth_sweep.sh "<卡號…>" [tag 前綴]
set -uo pipefail
ROOT=/nfs/home/nelson0314/WACV-s3
PY="$HOME/venvs/wacv/bin/python"
export PYTHONPATH="$ROOT" HF_HOME="$HOME/hf_cache" PYTHONIOENCODING=utf-8
export TOKENIZERS_PARALLELISM=false
cd "$ROOT" || { echo "錯誤：找不到 $ROOT" >&2; exit 2; }

DEVS=(${1:-})
[ ${#DEVS[@]} -lt 1 ] && { echo "用法：$0 \"<卡號…>\" [tag 前綴]" >&2; exit 2; }
bash scripts/free_cards.sh --assert "${DEVS[*]}" || exit 3

OUT=runs/ip2p_patch_smooth
mkdir -p "$OUT"
MARIO=task_attr_mod_color_6205

COMMON="--data data/omniedit150 --images $MARIO --conditions patch \
--loss image_guidance --ig-zt diffuse_src --ig-weight subject \
--subject-mask data/decoy_catalogue.yaml \
--steps 6000 --step-size 0.01 --eval-every 200 --eval-draws 8 \
--patience 10 --min-delta 0.0002 --save-weights"

CARRIER="--patch-placement complement --radius 0.05 --patch-carrier clothes"

POINTS="
tv05:$CARRIER~--patch-tv~0.5
tv20:$CARRIER~--patch-tv~2.0
tv30:$CARRIER~--patch-tv~3.0
a06_only:$CARRIER~--patch-alpha~0.6
sq1_a04_far:--patch-placement~far~--radius~0.04~--patch-count~1
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
  echo "[smooth] $tag dev=$dev  $args"
done <<< "$POINTS"

sleep 25
echo "啟動了 $(ps -u "$USER" -o cmd | grep '[i]p2p_run.py' | grep -c python) 行 ip2p_run（$(date)）"

#!/usr/bin/env bash
# 兩個讓產物不再像雜訊的方向，先在三張影像上篩。
#
# B · `--patch-tint`：粗尺度上像原本那件衣服
#   λ · mean_over_support( (blur_σ(c) − blur_σ(x))² )。與 `--patch-tv` 針對
#   不同的東西：tv 罰掉所有局部變化、花紋也被壓平（實測平舖配 tv 0.3 只剩
#   基準的 32%），tint 只罰低頻、花紋的高頻結構完全保留。掃三個 λ。
#
# C · 色彩重映射施加在衣服上
#   `apply_where = 載體 ∧ 主體補集`，於是重映射只作用在衣服上而主體逐位元
#   不動（w=0 處恆等）。產物是既有像素的平滑重映射、不是貼上去的內容——
#   這一族先前被否決的理由是「高容量時產物不自然」，但那是**全圖**施加時量
#   的，限制在衣服上、且嚴重變色可接受時那個約束不成立。半徑放大到 0.30／
#   0.60，不加 TV。
#
# 三張影像：合法載體最大的、中間的、最小的，涵蓋範圍。
#
# 用法：bash scripts/patch_tint_colour_screen.sh "<卡號…>" [tag 前綴]
set -uo pipefail
ROOT=/nfs/home/nelson0314/WACV-s3
PY="$HOME/venvs/wacv/bin/python"
export PYTHONPATH="$ROOT" HF_HOME="$HOME/hf_cache" PYTHONIOENCODING=utf-8
export TOKENIZERS_PARALLELISM=false
cd "$ROOT" || { echo "錯誤：找不到 $ROOT" >&2; exit 2; }

DEVS=(${1:-})
[ ${#DEVS[@]} -lt 1 ] && { echo "用法：$0 \"<卡號…>\" [tag 前綴]" >&2; exit 2; }
bash scripts/free_cards.sh --assert "${DEVS[*]}" || exit 3

OUT=runs/ip2p_patch_tint_colour
mkdir -p "$OUT"

# 合法載體 0.3182 / 0.1235 / 0.0589
IMGS="task_attr_mod_color_123744 task_attr_mod_color_71985 task_attr_mod_color_42929"

BASE="--data data/omniedit150 --images $IMGS \
--subject-mask data/carrier_catalogue.yaml \
--ig-weight-catalogue data/carrier_catalogue.yaml \
--patch-carrier clothes \
--steps 6000 --step-size 0.01 --eval-every 200 --eval-draws 8 \
--patience 10 --min-delta 0.0002 --save-weights"

PATCH="--conditions patch --loss image_guidance --ig-zt diffuse_src \
--ig-weight subject --patch-placement complement --radius 0.05"
COLOUR="--conditions color_grid --loss image_guidance --ig-zt diffuse_src \
--ig-weight subject --color-grid 16 --color-luma-bins 16"

POINTS="
tint03:$PATCH~--patch-tint~0.3
tint10:$PATCH~--patch-tint~1.0
tint30:$PATCH~--patch-tint~3.0
tint10_tile:$PATCH~--patch-tint~1.0~--patch-tile~64
cgrid_r030:$COLOUR~--radius~0.30
cgrid_r060:$COLOUR~--radius~0.60
cgrid_rand:--conditions~color_grid_rand~--loss~image_guidance~--ig-zt~diffuse_src~--color-grid~16~--color-luma-bins~16~--radius~0.60
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
      --out "$OUT/$tag" $BASE $args \
      < /dev/null > "$OUT/$tag.log" 2>&1 &
  disown
  echo "[screen] $tag dev=$dev  $args"
done <<< "$POINTS"

sleep 25
echo "啟動了 $(ps -u "$USER" -o cmd | grep '[i]p2p_run.py' | grep -c python) 行 ip2p_run（$(date)）"

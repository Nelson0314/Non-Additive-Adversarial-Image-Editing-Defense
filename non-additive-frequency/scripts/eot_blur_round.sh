#!/usr/bin/env bash
# 丙：把 EOT 的模糊族擴大，並把已結案的裁切從取樣裡拿掉。
#
# 兩個改動都由量測驅動
# ────────────────────────────────────────────────────────────────────
# 1. **σ2 目前落在族的邊界上。** `eot_broad` 的 sigma 族是 {0.5,1.0,1.5,2.0}，
#    而評測用的正是 σ2——EOT 對族的邊界點本來就最弱（同樣的論證讓
#    `eot_geometry` 把評測用的 0.10 放在族內）。擴到 3.0 讓 σ2 變成內點。
# 2. **裁切那兩欄是結構性不可贏的**（`runs/ip2p_eot_geom_purify/README.md`：
#    唯一被指名可能有效的隨機化幾何 EOT 實測仍在地板的五分之一以下），而
#    每一步有四分之一的機率被花在它上面。`--eot-classes` 把那份取樣預算還給
#    JPEG 與模糊。
#
# 四個點把兩個改動拆開，才分得出各自的貢獻；**對照是同批的 `eb_r25`**
# （`runs/ip2p_eot_matched`，預設族、含裁切、同半徑、同兩張影像）。
#
# 用法：bash scripts/eot_blur_round.sh "<四個卡號>"
set -uo pipefail
ROOT=/nfs/home/nelson0314/WACV-s3
PY="$HOME/venvs/wacv/bin/python"
export PYTHONPATH="$ROOT" HF_HOME="$HOME/hf_cache" PYTHONIOENCODING=utf-8
export TOKENIZERS_PARALLELISM=false
cd "$ROOT" || { echo "錯誤：找不到 $ROOT" >&2; exit 2; }

DEVS=(${1:-})
[ ${#DEVS[@]} -ne 4 ] && { echo "用法：$0 \"<四個卡號>\"" >&2; exit 2; }
bash scripts/free_cards.sh --assert "${DEVS[*]}" || exit 3
grep -q -- "--eot-classes" scripts/ip2p_run.py || {
  echo "錯誤：scripts/ip2p_run.py 不認得 --eot-classes，先同步本機的改動" >&2; exit 2; }

OUT=runs/ip2p_eot_blur
mkdir -p "$OUT"
IMGS="task_attr_mod_color_11699 task_obj_remove_380621"
NOCROP="--eot-classes~identity~jpeg~blur"
WIDE="--eot-sigmas~0.5~1.0~2.0~3.0"

BASE="--data data/omniedit150 --conditions phase_gain --quantile 0 \
--freq-weight jpeg_luma --freq-weight-power 0.25 --hop 8 --gain-ratio 1.0 \
--loss image_guidance --ig-zt diffuse_src --radius 2.5 \
--purify-aware eot_broad \
--steps 8000 --step-size 0.01 --eval-every 100 --eval-draws 8 \
--patience 15 --min-delta 0.0002 --save-weights --skip-existing"

POINTS="
bl_wide:$WIDE
bl_nocrop:$NOCROP
bl_both:$WIDE~$NOCROP
bl_s40:--eot-sigmas~1.0~2.0~3.0~4.0~$NOCROP
"

i=0
for p in $POINTS; do
  IFS=: read -r tag extra <<< "$p"
  dev=${DEVS[$i]}; i=$((i + 1))
  CUDA_VISIBLE_DEVICES="$dev" setsid nohup "$PY" scripts/ip2p_run.py \
      --out "$OUT/$tag" $BASE $(echo "$extra" | tr '~' ' ') \
      --images $IMGS < /dev/null >> "$OUT/$tag.log" 2>&1 &
  disown
  echo "[eotblur] $tag dev=$dev $(echo "$extra" | tr '~' ' ')"
done
sleep 20
echo "啟動了 $(ps -u "$USER" -o cmd | grep -c '[i]p2p_run') 個 ip2p_run process"

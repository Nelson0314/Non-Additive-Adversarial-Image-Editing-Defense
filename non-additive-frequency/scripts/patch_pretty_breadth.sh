#!/usr/bin/env bash
# 十二張擴增影像加上美觀花紋。
#
# 為什麼是這兩組旋鈕
# `runs/ip2p_patch_smooth/` 與 `runs/ip2p_patch_shape/` 在瑪利歐上量過三個
# 美化旋鈕的代價（基準 `clothes_plain` 的主體區位移 0.2732）：
#
#     α 0.6 單獨      0.2459  90%       平舖 64 單獨   0.2342  86%
#     tv 1.0 單獨     0.2064  76%       α ＋ 平舖      0.2028  74%
#     平舖 ＋ tv 0.3  0.0875  32%       三個全開       0.0518  19%
#
# 取代價最低而外觀改善最明顯的兩組：`tile_a`（α ＋ 平舖，讀起來是布料印花）
# 與 `tv10`（柔和色塊）。**平舖與 tv 不併用**——那一組交互作用只剩預期的四成。
#
# 逐張分片，每個 process 一張影像一組旋鈕。
#
# 用法：bash scripts/patch_pretty_breadth.sh "<卡號…>" <組名> [影像名前綴]
#       組名：tile_a／tv10／tint10／tint30
set -uo pipefail
ROOT=/nfs/home/nelson0314/WACV-s3
PY="$HOME/venvs/wacv/bin/python"
export PYTHONPATH="$ROOT" HF_HOME="$HOME/hf_cache" PYTHONIOENCODING=utf-8
export TOKENIZERS_PARALLELISM=false
cd "$ROOT" || { echo "錯誤：找不到 $ROOT" >&2; exit 2; }

DEVS=(${1:-})
GROUP="${2:-}"
[ ${#DEVS[@]} -lt 1 ] && { echo "用法：$0 \"<卡號…>\" <tile_a|tv10>" >&2; exit 2; }
case "$GROUP" in
  tile_a) EXTRA="--patch-tile 64 --patch-alpha 0.6" ;;
  tv10)   EXTRA="--patch-tv 1.0" ;;
  tint10) EXTRA="--patch-tint 1.0" ;;
  tint30) EXTRA="--patch-tint 3.0" ;;
  *) echo "第二個參數必須是 tile_a／tv10／tint10／tint30，收到 '${GROUP}'" >&2; exit 2 ;;
esac
bash scripts/free_cards.sh --assert "${DEVS[*]}" || exit 3

OUT=runs/ip2p_patch_breadth_pretty
mkdir -p "$OUT"

IMGS="task_attr_mod_color_123744 task_attr_mod_color_158070 \
task_env_weather_193600 task_attr_mod_color_30214 \
task_obj_swap_rand_mask_401099 task_attr_mod_color_71985 \
task_obj_add_362298 task_attr_mod_color_202962 task_obj_remove_704852 \
task_obj_swap_rand_mask_417469 task_obj_remove_657585 \
task_attr_mod_color_42929"

COMMON="--data data/omniedit150 --conditions patch \
--loss image_guidance --ig-zt diffuse_src --ig-weight subject \
--subject-mask data/carrier_catalogue.yaml \
--ig-weight-catalogue data/carrier_catalogue.yaml \
--patch-placement complement --radius 0.05 --patch-carrier clothes \
--steps 6000 --step-size 0.01 --eval-every 200 --eval-draws 8 \
--patience 10 --min-delta 0.0002 --save-weights"

# **先套用影像過濾再數**。反過來寫的話，只想送其中幾張時仍會拿整份清單去
# 比對卡數而被自己的守門擋下——實測踩過：`$0 "0 1 2 3" tv10 task_attr` 明明
# 只有六張也被回「12 張影像需要至少 6 張卡」。
ONLY_IMG="${3:-}"
SEL=""
for img in $IMGS; do
  if [ -n "$ONLY_IMG" ]; then
    case "$img" in $ONLY_IMG*) ;; *) continue ;; esac
  fi
  SEL="$SEL $img"
done
NIMG=$(echo $SEL | wc -w)
[ "$NIMG" -eq 0 ] && { echo "錯誤：過濾器 '$ONLY_IMG' 一張都沒對上。" >&2; exit 2; }
if [ "$NIMG" -gt $(( ${#DEVS[@]} * 2 )) ]; then
  echo "錯誤：$NIMG 張影像需要至少 $(( (NIMG + 1) / 2 )) 張卡，只給了 ${#DEVS[@]} 張。" >&2
  exit 2
fi

i=0
for img in $SEL; do
  dev=${DEVS[$(( i % ${#DEVS[@]} ))]}
  i=$(( i + 1 ))
  CUDA_VISIBLE_DEVICES="$dev" setsid nohup "$PY" scripts/ip2p_run.py \
      --out "$OUT/${GROUP}_$img" --images "$img" $COMMON $EXTRA \
      < /dev/null > "$OUT/${GROUP}_$img.log" 2>&1 &
  disown
  echo "[pretty] ${GROUP} $img dev=$dev"
done

sleep 25
echo "啟動了 $(ps -u "$USER" -o cmd | grep '[i]p2p_run.py' | grep -c python) 行 ip2p_run（$(date)）"

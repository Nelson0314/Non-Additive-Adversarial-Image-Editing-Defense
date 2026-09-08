#!/usr/bin/env bash
# 衣物載體擴增到十二張影像。
#
# 為什麼是這十二張
# 先前只有瑪利歐一張，「只夠決定方向、不足以支撐結論」。候選由 ATR 人體
# 解析篩出（`face >= 0.01` 擋掉沒有人的影像、`clothes >= 0.10`），147 張裡
# 42 張過關；再逐張看圖挑出**受保護物件與穿衣服的人分開**的那些——這一族
# 結構上需要那個分離，人被登記成主體時合法載體是 0。
#
# 登記在 `data/carrier_catalogue.yaml`，逐張的合法載體面積 0.0589–0.3182
# （瑪利歐是 0.0861，多數比它大）。瑪利歐不在這裡：
# `runs/ip2p_patch_carrier/clothes_plain` 已用完全相同的設定量過它。
#
# 兩個條件，同一組影像、同一個支撐：
#   plain   最佳化（`clothes_plain` 的設定）
#   rand    同幾何不最佳化。梯度買到多少由這一格減出來
#
# **逐張分片**：一張影像 6000 步約 50 分鐘（獨佔）／85 分鐘（雙開），十二張
# 串在一個 process 裡跑不完。每個 process 一張，輸出目錄帶影像名——「每寫
# 一列是整份重寫 CSV」，目錄名不帶全部變因會互相蓋掉。
#
# 用法：
#   bash scripts/patch_carrier_breadth.sh "<卡號…>"          # 12 格 plain
#   bash scripts/patch_carrier_breadth.sh "<卡號…>" rand      # 隨機對照
#   bash scripts/patch_carrier_breadth.sh "<卡號…>" task_x    # 只送某幾張
#
# **rand 一定要單獨送**：與 12 格 plain 同時送會讓其中一張卡疊到三個
# process，而每卡上限是 2（疊到 3 個實測整批 OOM）。
set -uo pipefail
ROOT=/nfs/home/nelson0314/WACV-s3
PY="$HOME/venvs/wacv/bin/python"
export PYTHONPATH="$ROOT" HF_HOME="$HOME/hf_cache" PYTHONIOENCODING=utf-8
export TOKENIZERS_PARALLELISM=false
cd "$ROOT" || { echo "錯誤：找不到 $ROOT" >&2; exit 2; }

DEVS=(${1:-})
[ ${#DEVS[@]} -lt 1 ] && { echo "用法：$0 \"<卡號…>\" [rand|影像名前綴]" >&2; exit 2; }
bash scripts/free_cards.sh --assert "${DEVS[*]}" || exit 3

OUT=runs/ip2p_patch_breadth
mkdir -p "$OUT"

IMGS="task_attr_mod_color_123744 task_attr_mod_color_158070 \
task_env_weather_193600 task_attr_mod_color_30214 \
task_obj_swap_rand_mask_401099 task_attr_mod_color_71985 \
task_obj_add_362298 task_attr_mod_color_202962 task_obj_remove_704852 \
task_obj_swap_rand_mask_417469 task_obj_remove_657585 \
task_attr_mod_color_42929"

# `--ig-weight subject` 走的是**另一個**目錄旗標，預設仍指向
# decoy_catalogue.yaml。漏給它會在第一張影像上就拋錯（守門有擋，不會靜默）。
COMMON="--data data/omniedit150 --conditions patch \
--loss image_guidance --ig-zt diffuse_src --ig-weight subject \
--subject-mask data/carrier_catalogue.yaml \
--ig-weight-catalogue data/carrier_catalogue.yaml \
--patch-placement complement --radius 0.05 --patch-carrier clothes \
--steps 6000 --step-size 0.01 --eval-every 200 --eval-draws 8 \
--patience 10 --min-delta 0.0002 --save-weights"

ONLY="${2:-}"

if [ "$ONLY" = "rand" ]; then
  dev=${DEVS[0]}
  CUDA_VISIBLE_DEVICES="$dev" setsid nohup "$PY" scripts/ip2p_run.py \
      --out "$OUT/rand_all" --images $IMGS $COMMON --conditions patch_rand \
      < /dev/null > "$OUT/rand_all.log" 2>&1 &
  disown
  echo "[breadth] rand_all dev=$dev（十二張串跑，不最佳化）"
  sleep 20
  exit 0
fi

# 每卡最多 2 個 process。守門要擋在派工前面，不是印出來就算。
NIMG=$(echo $IMGS | wc -w)
if [ "$NIMG" -gt $(( ${#DEVS[@]} * 2 )) ]; then
  echo "錯誤：$NIMG 張影像需要至少 $(( (NIMG + 1) / 2 )) 張卡，只給了 ${#DEVS[@]} 張。" >&2
  exit 2
fi

i=0
for img in $IMGS; do
  if [ -n "$ONLY" ]; then
    case "$img" in $ONLY*) ;; *) continue ;; esac
  fi
  dev=${DEVS[$(( i % ${#DEVS[@]} ))]}
  i=$(( i + 1 ))
  CUDA_VISIBLE_DEVICES="$dev" setsid nohup "$PY" scripts/ip2p_run.py \
      --out "$OUT/plain_$img" --images "$img" $COMMON \
      < /dev/null > "$OUT/plain_$img.log" 2>&1 &
  disown
  echo "[breadth] plain $img dev=$dev"
done

sleep 25
echo "啟動了 $(ps -u "$USER" -o cmd | grep '[i]p2p_run.py' | grep -c python) 行 ip2p_run（$(date)）"

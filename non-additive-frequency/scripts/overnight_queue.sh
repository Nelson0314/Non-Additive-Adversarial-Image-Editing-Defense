#!/usr/bin/env bash
# 十二小時的自走佇列：六組 × 十張影像 = 60 格。
#
# 六組 = {plain, pretty} × {clothing, accessory, background}
#   plain   載體精修但不美化（吸附／挖臉框／內縮／往內羽化），支撐鋪滿載體
#   pretty  ＋ 分散 6 塊（支撐降到全圖 5%）＋ tint 3.0（色調拉向衣服）
#
# **為什麼要 pretty**：鋪滿整件衣服時，`clothing` 那一類攻擊的目標正好是被
# 蓋滿的區域，改不動幾乎是必然的，那個結果近乎恆真。分散之後支撐由 26% 降到
# 5%，clothing 才問得出東西。
#
# 排程方式
# 每一輪：等 ip2p_run 排空 → **重新取空卡**（前一輪跑完之後別人可能已經佔走）
# → 送下一批。每卡兩個 process。一格 6000 步，雙開約 85 分鐘。
#
# 已經跑過的格會被跳過（輸出目錄裡有 results.csv 就不重送），故中途斷掉再送
# 一次是安全的。
#
# 用法：bash scripts/overnight_queue.sh            # 直接跑，自己等卡
set -uo pipefail
ROOT=/nfs/home/nelson0314/WACV-s3
PY="$HOME/venvs/wacv/bin/python"
export PYTHONPATH="$ROOT" HF_HOME="$HOME/hf_cache" PYTHONIOENCODING=utf-8
export TOKENIZERS_PARALLELISM=false
cd "$ROOT" || { echo "錯誤：找不到 $ROOT" >&2; exit 2; }

OUT=runs/ip2p_face_defence
mkdir -p "$OUT"

# 十張影像：載體 0.114–0.343，涵蓋單人／多人、照片／插畫。
IMGS="task_obj_remove_257910 task_obj_remove_284852 task_obj_remove_410264 \
task_obj_remove_509477 task_obj_remove_720199 task_obj_swap_joint_mask_533428 \
task_env_weather_195273 task_attr_mod_color_123744 task_env_weather_70149 \
task_attr_mod_color_202962"

CATS="clothing accessory background"
# **不可以叫 GROUPS。** 那是 bash 的內建陣列（當前使用者的群組 ID），指派會被
# **靜默忽略**：$GROUPS 展開成 2068，於是六組靜默變成三組而不報錯。
# 這一條已經記在 docs/DEFECTS.md 裡，寫這支腳本時仍然踩到了。
DEF_GROUPS="plain pretty"

BASE="--data data/omniedit150 --conditions patch \
--attack-prompts data/attack_prompts.yaml --subject-source face \
--loss image_guidance --ig-zt diffuse_src \
--subject-mask data/attack_prompts.yaml \
--patch-placement complement --radius 0.05 --patch-carrier clothes \
--carrier-refine 4 --carrier-erode 3 --carrier-feather 8 \
--steps 6000 --step-size 0.01 --eval-every 200 --eval-draws 8 \
--patience 10 --min-delta 0.0002 --save-weights"

# 待跑清單：跳過已有 results.csv 的格。
QUEUE=""
for cat in $CATS; do
  for grp in $DEF_GROUPS; do
    for img in $IMGS; do
      QUEUE="$QUEUE ${cat}_${grp}_${img}"
    done
  done
done
TOTAL=$(echo $QUEUE | wc -w)
echo "佇列 $TOTAL 格（六組 × 十張）  $(date)"
echo "已有 results.csv 的格會在**送出的當下**跳過——建佇列時檢查的話，"
echo "此刻還在跑的格尚未寫出 CSV，會被排進去重跑。"
[ "$TOTAL" -eq 0 ] && { echo "全部跑完了。"; exit 0; }

round=0
pending=$QUEUE
while [ -n "$(echo $pending | tr -d ' ')" ]; do
  # 等前一輪排空。**不設上限**：卡被別人佔住時就是要等。
  while [ "$(ps -u "$USER" -o cmd | grep -c '[i]p2p_run.py')" -gt 0 ]; do
    sleep 120
  done
  DEVS=$(bash scripts/free_cards.sh)
  if [ -z "$DEVS" ]; then
    echo "[$(date +%H:%M)] 沒有空卡，等 10 分鐘再看"
    sleep 600
    continue
  fi
  NDEV=$(echo $DEVS | wc -w)
  SLOTS=$(( NDEV * 2 ))
  round=$(( round + 1 ))
  echo "[$(date +%H:%M)] 第 $round 輪：卡 $DEVS（$SLOTS 個 slot），剩 $(echo $pending | wc -w) 格"

  i=0
  next=""
  for tag in $pending; do
    # **在這裡跳過**，不是在建佇列時：前一輪剛跑完的格此刻才有 results.csv。
    if [ -f "$OUT/$tag/results.csv" ]; then
      echo "    $tag 已完成，跳過"
      continue
    fi
    if [ "$i" -ge "$SLOTS" ]; then next="$next $tag"; continue; fi
    cat="${tag%%_*}"; rest="${tag#*_}"
    grp="${rest%%_*}"; img="${rest#*_}"
    case "$grp" in
      plain)  EXTRA="" ;;
      pretty) EXTRA="--carrier-scatter 6 --patch-tint 3.0" ;;
    esac
    # 卡號輪流。DEVS 是空白分隔的字串，用 cut 取第 n 個最直接——
    # `eval` 加位置參數在這裡只會多一層引號問題。
    dev=$(echo $DEVS | cut -d' ' -f$(( i % NDEV + 1 )))
    i=$(( i + 1 ))
    CUDA_VISIBLE_DEVICES="$dev" setsid nohup "$PY" scripts/ip2p_run.py \
        --out "$OUT/$tag" --images "$img" $BASE --attack-category "$cat" $EXTRA \
        < /dev/null > "$OUT/$tag.log" 2>&1 &
    disown
    echo "    $tag → 卡 $dev"
  done
  pending="$next"
  sleep 60
  echo "    啟動了 $(ps -u "$USER" -o cmd | grep -c '[i]p2p_run.py') 行"
done

echo "[$(date +%H:%M)] 佇列跑完。"

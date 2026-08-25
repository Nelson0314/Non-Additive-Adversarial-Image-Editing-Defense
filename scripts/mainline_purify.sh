#!/usr/bin/env bash
# 主線頭對頭的抗淨化 ＋ 人眼比對圖。**只讀已存的防禦圖，不重跑防禦。**
#
# 九個淨化算子，三個家族各掃多個強度——一個點看不出斜率：
#
#   JPEG    90 / 75 / 50 / 30   本方法的量化交付與 DCT-Shield 的 `Q_alg` 都有
#                               一條**單向**的品質界線，四點才畫得出它落在哪裡
#   模糊    sigma 1 / 2
#   裁切    0.10 / 0.15
#   identity                    淨增益的分母，不可缺
#
# **空白地板不可省略**（`DECISIONS.md`）：淨化算子自己就會把編輯推開，不扣掉它，
# 「淨化後位移較大」無法排除「該算子本來就把編輯推得比較開」這個平庸解釋。
# 地板與條件無關，用 `floor` 這個 tag 跑一次共用。
#
# `--gallery` 一定要開：報告頁要的是「淨化之後的編輯長什麼樣」，而
# `phase_retention.py` 預設把那些影像用完即棄。編輯本來就算過，存圖不多花運算。
#
# 用法：bash scripts/mainline_purify.sh "<卡號>" "<tag> ..." ["<分片>"]
set -uo pipefail
ROOT=/nfs/home/nelson0314/WACV-s3
PY="$HOME/venvs/wacv/bin/python"
export PYTHONPATH="$ROOT" HF_HOME="$HOME/hf_cache" PYTHONIOENCODING=utf-8
export TOKENIZERS_PARALLELISM=false
cd "$ROOT"

SRC=runs/ip2p_mainline
OUT=runs/ip2p_mainline_purify
GAL=runs/gallery_mainline
mkdir -p "$OUT" "$GAL"

# 分片依任務族群切，不用流水號。十張：色五、景一、物四。
A="task_attr_mod_color_11699 task_attr_mod_color_136767 task_attr_mod_color_184837 task_attr_mod_color_32648 task_attr_mod_color_6205"
B="task_env_weather_112463 task_obj_remove_380621 task_obj_swap_joint_mask_276754 task_obj_swap_joint_mask_533428 task_obj_swap_rand_mask_417469"

# 可由環境變數覆寫（計畫零的加密品質格點用 `PUR="identity jpeg90 jpeg75
# jpeg60 jpeg50 jpeg40 jpeg30 jpeg20"`）。**預設不變**，既有批次的算子
# 集合因此逐字相同，跨批次可比。
PUR="${PUR:-identity jpeg90 jpeg75 jpeg50 jpeg30 blur1 blur2 crop_resize0.1 crop_resize0.15}"
COMMON="--data data/omniedit150 --attacker ip2p --seeds 3 --purifiers $PUR"

DEVS=(${1:-})
TAGS=(${2:-})
SHARDS=(${3:-color object})
[ ${#DEVS[@]} -eq 0 ] || [ ${#TAGS[@]} -eq 0 ] && {
  echo "用法：$0 \"<卡號>\" \"<tag> ...\" [\"<分片>\"]" >&2
  echo "可選 tag：$(ls -1 $SRC 2>/dev/null | grep -v '\.log$' | tr '\n' ' ')floor" >&2; exit 2; }
for s in "${SHARDS[@]}"; do case "$s" in color|object) ;; *)
  echo "錯誤：分片名只能是 color / object，收到 $s" >&2; exit 2 ;; esac; done

bash scripts/free_cards.sh --assert "${DEVS[*]}" || exit 3

n=$(( ${#TAGS[@]} * ${#SHARDS[@]} ))
if [ "$n" -gt $(( ${#DEVS[@]} * 2 )) ]; then
  echo "錯誤：$n 個 process 需要至少 $(( (n + 1) / 2 )) 張卡，只給了 ${#DEVS[@]} 張" >&2; exit 2
fi

# 地板的防禦圖就是原圖，故借任何一個已完成的目錄當影像來源即可。
FLOOR_SRC=""
for d in "$SRC"/*/; do
  [ "$(ls -1 "$d"/*__def.png 2>/dev/null | wc -l)" -eq 10 ] && FLOOR_SRC="${d%/}" && break
done

i=0
for tag in "${TAGS[@]}"; do
  if [ "$tag" = floor ]; then
    run="$FLOOR_SRC"; extra="--floor"
    [ -z "$run" ] && { echo "錯誤：地板需要一個已完成的防禦目錄當影像來源" >&2; exit 3; }
  else
    run="$SRC/$tag"; extra=""
    n_def=$(ls -1 "$run"/*__def.png 2>/dev/null | wc -l)
    [ "$n_def" -ne 10 ] && { echo "錯誤：$tag 只有 $n_def/10 張防禦圖" >&2; exit 3; }
  fi
  for sh in "${SHARDS[@]}"; do
    case $sh in color) IM="$A" ;; object) IM="$B" ;; esac
    dev=${DEVS[$(( i / 2 % ${#DEVS[@]} ))]}
    i=$(( i + 1 ))
    CUDA_VISIBLE_DEVICES="$dev" nohup "$PY" scripts/phase_retention.py \
        --run "$run" --images $IM $COMMON $extra --gallery "$GAL/$tag" \
        --out "$OUT/${tag}_${sh}.csv" > "$OUT/${tag}_${sh}.log" 2>&1 &
    echo "[purify] $tag/$sh run=$run dev=$dev pid=$!"
  done
done
echo "[purify] 送出 $i 個（$(date)）"

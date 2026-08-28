#!/usr/bin/env bash
# 色散度那條軸的抗淨化 ＋ 人眼比對圖。**只讀已存的防禦圖，不重跑防禦。**
#
# 算子集合與主線逐字相同（`scripts/mainline_purify.sh`），跨批次因此可比：
#
#   JPEG 90 / 75 / 50 / 30、模糊 sigma 1 / 2、裁切 0.10 / 0.15、identity
#
# **空白地板不可省略**（`DECISIONS.md`）：淨化算子自己就會把編輯推開。
# 地板與條件無關，用 `floor` 這個 tag 跑一次共用。
#
# `--gallery` 一定要開：報告頁要的是「淨化之後的編輯長什麼樣」，而
# `phase_retention.py` 預設把那些影像用完即棄。
#
# **種子數預設 1 而不是主線的 3**：這一批是給報告頁看的，五個條件 × 九個算子
# × 十張已經是 450 次編輯；三顆種子會變成 1350 次。種子數逐列寫在 CSV 裡，
# 與主線的列並排時要看那一欄。
#
# 用法：bash scripts/dispersion_purify.sh "<卡號>" "<tag> ..." ["<分片>"]
set -uo pipefail
ROOT=/nfs/home/nelson0314/WACV-s3
PY="$HOME/venvs/wacv/bin/python"
export PYTHONPATH="$ROOT" HF_HOME="$HOME/hf_cache" PYTHONIOENCODING=utf-8
export TOKENIZERS_PARALLELISM=false
cd "$ROOT" || { echo "錯誤：找不到 $ROOT" >&2; exit 2; }

# 三者可由環境變數覆寫，供 `ip2p_dispersion_opt`（可學＋接閘那一批）重用。
SRC="${SRC:-runs/ip2p_dispersion}"
OUT="${OUT:-runs/ip2p_dispersion_purify}"
GAL="${GAL:-runs/gallery_dispersion}"
mkdir -p "$OUT" "$GAL"

# 分片依任務族群切，不用流水號。十張：色五、景一、物四。
A="task_attr_mod_color_11699 task_attr_mod_color_136767 task_attr_mod_color_184837 task_attr_mod_color_32648 task_attr_mod_color_6205"
B="task_env_weather_112463 task_obj_remove_380621 task_obj_swap_joint_mask_276754 task_obj_swap_joint_mask_533428 task_obj_swap_rand_mask_417469"

PUR="${PUR:-identity jpeg90 jpeg75 jpeg50 jpeg30 blur1 blur2 crop_resize0.1 crop_resize0.15}"
SEEDS="${SEEDS:-1}"
COMMON="--data data/omniedit150 --attacker ip2p --seeds $SEEDS --purifiers $PUR"

DEVS=(${1:-})
TAGS=(${2:-})
SHARDS=(${3:-color object})
[ ${#DEVS[@]} -eq 0 ] || [ ${#TAGS[@]} -eq 0 ] && {
  echo "用法：$0 \"<卡號>\" \"<tag> ...\" [\"<分片>\"]" >&2; exit 2; }

n=$(( ${#TAGS[@]} * ${#SHARDS[@]} ))
if [ "$n" -gt $(( ${#DEVS[@]} * 2 )) ]; then
  echo "錯誤：$n 個工作點需要至少 $(( (n + 1) / 2 )) 張卡，只給了 ${#DEVS[@]} 張" >&2
  exit 2
fi

bash scripts/free_cards.sh --assert "${DEVS[*]}" || exit 3

# 地板的影像來源取任何一個已完成的防禦目錄——地板量的是**乾淨影像**過算子
# 之後的編輯位移，與用哪個條件的目錄無關。
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

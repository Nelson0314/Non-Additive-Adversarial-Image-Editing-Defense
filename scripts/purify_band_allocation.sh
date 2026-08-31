#!/usr/bin/env bash
# `runs/ip2p_band_allocation` 那一批的抗淨化那一半。
#
# **只讀已存的防禦圖，不重跑防禦**（`scripts/phase_retention.py`）。
# 空白地板不可省略：淨化算子自己就會把編輯推開，不扣掉它，「淨化後位移較大」
# 無法排除「該算子本來就把編輯推得比較開」這個平庸解釋。
#
# 算子取六個，對準這一輪要回答的問題（JPEG、模糊、裁切）：
#     identity jpeg75 jpeg30 blur1 blur2 crop_resize0.1
# 刻意不含 jpeg90／jpeg50（同一條 JPEG 曲線上的內點）與 crop_resize0.15
# （`runs/ip2p_eot_geom_purify` 量到它的地板 0.581，可讀範圍只剩 0.19）。
#
# `--gallery` 一律開：**擋下與否要用眼睛判「重畫」對「劣化」**
# （`docs/GOAL.md`；SigLIP 代理已實測會把「人還在、只是被蓋上紋理」標成
# blocked）。沒有圖就沒有辦法判。
#
# 分片
# ────────────────────────────────────────────────────────────────────
# 十張切兩片，讓同一個條件同時佔兩張卡，牆鐘時間減半。分片名只能是
# `retention_table.py` 的 `SHARDS`（color／scene／object／all），因為
# `tag_of()` 由檔名後綴還原條件標籤。
#
#     color   五張 task_attr_mod_color_*
#     object  其餘五張。**其中 task_env_weather_112463 是天氣類**，
#             放在這一片只是為了湊成 5+5 的等分，名字是分區標籤不是任務族。
#
# 兩個 process 不可寫同一個輸出目錄：每一片各自寫 `<tag>_<片>.csv` 與
# `gallery_<tag>_<片>/`。
#
# 用法：bash scripts/purify_band_allocation.sh "<五個卡號>" "<條件標籤...>"
set -uo pipefail
ROOT=/nfs/home/nelson0314/WACV-s3
PY="$HOME/venvs/wacv/bin/python"
export PYTHONPATH="$ROOT" HF_HOME="$HOME/hf_cache" PYTHONIOENCODING=utf-8
export TOKENIZERS_PARALLELISM=false
cd "$ROOT" || { echo "錯誤：找不到 $ROOT" >&2; exit 2; }

DEVS=(${1:-})
TAGS=(${2:-})
[ ${#DEVS[@]} -lt 1 ] && { echo "用法：$0 \"<卡號>\" \"<條件標籤...>\"" >&2; exit 2; }
[ ${#TAGS[@]} -lt 1 ] && { echo "用法：$0 \"<卡號>\" \"<條件標籤...>\"" >&2; exit 2; }
bash scripts/free_cards.sh --assert "${DEVS[*]}" || exit 3

SRC=runs/ip2p_band_allocation
OUT="$SRC/purify"
mkdir -p "$OUT"

COLOR="task_attr_mod_color_11699 task_attr_mod_color_136767 task_attr_mod_color_184837 task_attr_mod_color_32648 task_attr_mod_color_6205"
OBJECT="task_env_weather_112463 task_obj_remove_380621 task_obj_swap_joint_mask_276754 task_obj_swap_joint_mask_533428 task_obj_swap_rand_mask_417469"
PUR="identity jpeg75 jpeg30 blur1 blur2 crop_resize0.1"
COMMON="--data data/omniedit150 --attacker ip2p --seeds 3 --purifiers $PUR"

# 只送有防禦圖的條件。缺的一律印出來，不靜默略過——靜默略過會讓出表時
# 少一列而沒有人發現。地板用第一個條件的目錄當來源（`--floor` 把原圖當
# 防禦圖，來源目錄只是用來列出影像清單）。
GOOD=""
for t in "${TAGS[@]}"; do
  if ls "$SRC/$t"/*__def.png >/dev/null 2>&1; then
    GOOD="$GOOD $t"
  else
    echo "[skip] $SRC/$t 沒有防禦圖" >&2
  fi
done
[ -z "$GOOD" ] && { echo "錯誤：沒有任何條件有防禦圖" >&2; exit 2; }
FIRST=$(echo $GOOD | awk '{print $1}')

n=$(( ($(echo $GOOD | wc -w) + 1) * 2 ))       # （條件數 ＋ 地板）× 兩片
# 每卡最多 2 個 process（`docs/OPERATIONS.md`）。疊到 3 個實測整批 OOM。
if [ "$n" -gt $(( ${#DEVS[@]} * 2 )) ]; then
  echo "錯誤：$n 個 process 需要至少 $(( (n + 1) / 2 )) 張卡，只給了 ${#DEVS[@]} 張。" >&2
  exit 2
fi

i=0
launch() {              # $1 輸出標籤  $2 片名  $3 影像  $4 防禦圖來源  $5 額外旗標
  local tag="$1" shard="$2" imgs="$3" run="$4" extra="${5:-}"
  local dev=${DEVS[$(( i % ${#DEVS[@]} ))]}; i=$((i + 1))
  CUDA_VISIBLE_DEVICES="$dev" setsid nohup "$PY" scripts/phase_retention.py \
      --run "$SRC/$run" $COMMON --images $imgs $extra \
      --out "$OUT/${tag}_${shard}.csv" --gallery "$OUT/gallery_${tag}_${shard}" \
      < /dev/null >> "$OUT/${tag}_${shard}.log" 2>&1 &
  disown
  echo "[purify] $tag $shard dev=$dev $extra"
}

for t in $GOOD; do
  launch "$t" color  "$COLOR"  "$t"
  launch "$t" object "$OBJECT" "$t"
done
# 地板的 `--run` 指到第一個條件的目錄：`--floor` 把**原圖**當防禦圖，
# 來源目錄只用來定位這一批的設定，不會讀到它的防禦圖。
launch floor color  "$COLOR"  "$FIRST" --floor
launch floor object "$OBJECT" "$FIRST" --floor

sleep 25
echo "啟動了 $(ps -u "$USER" -o cmd | grep -c '[p]hase_retention') 個 phase_retention process"

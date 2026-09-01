#!/usr/bin/env bash
# 低頻乘性明暗場的抗淨化：`runs/ip2p_shading/README.md` 留下的未答問題。
#
# 那一批只量了未淨化的位移（等失真下勝過同失真隨機 1.62–1.65 倍，但只有
# 相位法的 22%），結語寫的是：
#
#     「它能不能提高抗淨化，尤其是模糊與裁切那兩格，要另外跑一批才知道，
#       本檔沒有答案。」
#
# 這一批就是那一批。問的是一個構造上的問題：**一個能量全落在 `f_n ≲ 0.03`
# 的低頻場，在模糊與裁切那兩欄活不活得下來。** 相位法在那兩欄的失效機制
# 分別是「模糊抽掉高頻載體」（σ=1 存活 0.169）與「裁切造成同步失效」
# （能量存活 51–99%、方向餘弦 0.995 卻沒人對回去），兩者都不作用在低頻場上。
#
# **只讀已存的防禦圖，不重跑防禦**（`scripts/phase_retention.py`）。
# 四個條件的 `*__def.png` 都在 `runs/ip2p_shading/<tag>/`。
#
# 同失真隨機對照（`rand_*`）與最佳化的（`opt_*`）一起送，因為低頻低自由度的
# 參數化特別容易死在「與同失真隨機無法區分」（FND-004 的死法）。
#
# 第四個參數可以覆寫影像清單，供補跑用（分批的結果放不同目錄，分析時把兩個
# 目錄一起餵給 retention_table.py／purify_matched_table.py 即可；影像集合不相交
# 就不會有重複的格）。
#
# 用法：bash scripts/purify_shading_field.sh "<卡號…>" [條件…] [nofloor] [影像…]
set -uo pipefail
ROOT=/nfs/home/nelson0314/WACV-s3
PY="$HOME/venvs/wacv/bin/python"
export PYTHONPATH="$ROOT" HF_HOME="$HOME/hf_cache" PYTHONIOENCODING=utf-8
export TOKENIZERS_PARALLELISM=false
cd "$ROOT" || { echo "錯誤：找不到 $ROOT" >&2; exit 2; }

DEVS=(${1:-})
[ ${#DEVS[@]} -lt 1 ] && { echo "用法：$0 \"<卡號…>\" [條件…] [nofloor]" >&2; exit 2; }
# 卡是多人共用的。**這個檢查會 exit，不是印出來就算。**
bash scripts/free_cards.sh --assert "${DEVS[*]}" || exit 3

SRC=runs/ip2p_shading
OUT=${OUT_DIR:-runs/ip2p_shading_purify}
mkdir -p "$OUT"

# `runs/ip2p_shading` 的十三張裡**還在資料集裡的那十張**。
# 三張（`task_env_weather_246440`、`task_env_weather_63722`、
# `task_obj_add_40931`）的防禦圖還在，但已經不在 `data/omniedit150`
# （現為 147 張）。防禦圖不入版控、資料集入版控，兩者會分岔；
# `phase_retention.py` 現在會在開工前就擋下這種清單。
IMGS="task_attr_mod_color_11699 task_attr_mod_color_136767 task_attr_mod_color_184837 task_attr_mod_color_32648 task_attr_mod_color_6205 task_env_weather_112463 task_obj_remove_380621 task_obj_swap_joint_mask_276754 task_obj_swap_joint_mask_533428 task_obj_swap_rand_mask_417469"
IMGS="${4:-$IMGS}"
# 現行六算子協定，與 runs/ip2p_matched_headtohead 同一組。
PUR="identity jpeg75 jpeg30 blur1 blur2 crop_resize0.1"
COMMON="--data data/omniedit150 --attacker ip2p --seeds 3 --purifiers $PUR"

TAGS="${2:-opt_r010 opt_r020 rand_r010 rand_r020}"
WANT_FLOOR=1
[ "${3:-}" = "nofloor" ] && WANT_FLOOR=0

GOOD=""
for t in $TAGS; do
  if ls "$SRC/$t"/*__def.png >/dev/null 2>&1 && [ -f "$SRC/$t/results.csv" ]; then
    GOOD="$GOOD $t"
  else
    echo "[skip] $SRC/$t 缺防禦圖或 results.csv" >&2
  fi
done
[ -z "$GOOD" ] && { echo "錯誤：沒有任何條件可送" >&2; exit 2; }
FIRST=$(echo $GOOD | awk '{print $1}')

n=$(( $(echo $GOOD | wc -w) + WANT_FLOOR ))
# 每卡最多 2 個 process。疊到 3 個實測整批 OOM。
if [ "$n" -gt $(( ${#DEVS[@]} * 2 )) ]; then
  echo "錯誤：$n 個 process 需要至少 $(( (n + 1) / 2 )) 張卡，只給了 ${#DEVS[@]} 張。" >&2
  exit 2
fi

i=0
launch() {
  local tag="$1" run="$2" extra="${3:-}"
  local dev=${DEVS[$(( i % ${#DEVS[@]} ))]}; i=$((i + 1))
  CUDA_VISIBLE_DEVICES="$dev" setsid nohup "$PY" scripts/phase_retention.py \
      --run "$SRC/$run" $COMMON --images $IMGS $extra \
      --out "$OUT/${tag}_all.csv" --gallery "$OUT/gallery_${tag}" \
      < /dev/null >> "$OUT/${tag}.log" 2>&1 &
  disown
  echo "[purify] $tag dev=$dev $extra"
}

for t in $GOOD; do launch "$t" "$t"; done
# 地板的 `--run` 只用來定位設定，`--floor` 把原圖當防禦圖。
[ "$WANT_FLOOR" -eq 1 ] && launch floor "$FIRST" --floor

sleep 20
echo "啟動了 $(ps -u "$USER" -o cmd | grep -c '[p]hase_retention') 個 phase_retention process"

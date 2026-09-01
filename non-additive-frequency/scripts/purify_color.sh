#!/usr/bin/env bash
# 色彩重映射的抗淨化，**十個算子**：現行六個 ＋ 四個色彩類。
#
# 為什麼要加那四個
# ────────────────────────────────────────────────────────────────────
# 色彩防禦繞開的是空間性的失效機制（模糊砍高頻、裁切造成同步失效），代價是
# 多開了一個攻擊面。AdvCF（arXiv:2011.06690）圖 10 量到色彩攻擊在
# JPEG q30／中值濾波／resize&pad 上存活 75–82%，**但灰階轉換只剩約 18%**。
# 不測灰階，色彩方向的主張就不成立。
#
# 四個算子涵蓋攻擊方在**沒有乾淨參照**時能做的全部色彩正規化：丟掉色度
# （grayscale）、對齊白平衡（gray_world）、對齊逐通道動態範圍（auto_levels）、
# 對齊局部對比（clahe）。有參照的手段（直方圖匹配回原圖）不在威脅模型內。
#
# 共防禦參照
# ────────────────────────────────────────────────────────────────────
# `--codefense` 另外量 `LPIPS(編輯(p(D(x))), p(D(編輯(x))))`，與現行參照並列
# 寫進同一列。`D` 由 `runs/ip2p_color/<tag>/` 的 `__w.pt`（最佳化的）或
# `defense_seed`（隨機的）重建。右側不需要額外的編輯，故成本近似為零。
#
# **只讀已存的防禦圖，不重跑防禦。**
#
# 用法：bash scripts/purify_color.sh "<卡號…>" [條件…] [nofloor]
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

SRC=runs/ip2p_color
OUT=runs/ip2p_color_purify
mkdir -p "$OUT"

IMGS="task_attr_mod_color_11699 task_attr_mod_color_6205 task_env_weather_112463 task_obj_remove_380621 task_obj_swap_joint_mask_276754 task_obj_add_13726"
# 現行六個 ＋ 四個色彩類。前六個與 runs/ip2p_matched_headtohead、
# runs/ip2p_shading_purify 同一組，故那三批的六欄可以並列。
PUR="identity jpeg75 jpeg30 blur1 blur2 crop_resize0.1 grayscale gray_world auto_levels0.01 clahe2"
COMMON="--data data/omniedit150 --attacker ip2p --seeds 3 --purifiers $PUR --codefense"

# 五格，剛好一卡一個：網格族兩個強度、曲線族一個、一個同族的隨機對照、
# 一個空白地板。隨機對照不可省——低自由度的參數化特別容易死在
# 「與同失真隨機無法區分」（FND-004 的死法）。
TAGS="${2:-grid_r005 grid_r010 curve_r3 grid_rand_r005}"
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
# 地板的 `--run` 只用來定位設定，`--floor` 把原圖當防禦圖；那一格的 `D` 是
# 恆等，於是共防禦參照在 identity 欄恰為 0。
[ "$WANT_FLOOR" -eq 1 ] && launch floor "$FIRST" --floor

sleep 20
echo "啟動了 $(ps -u "$USER" -o cmd | grep -c '[p]hase_retention') 個 phase_retention process（$(date)）"

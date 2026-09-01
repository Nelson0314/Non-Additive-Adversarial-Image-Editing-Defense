#!/usr/bin/env bash
# `runs/ip2p_ig_lowdist`的抗淨化那一半。
#
# **只讀已存的防禦圖，不重跑防禦**（`scripts/phase_retention.py`）。
# 空白地板不可省略（`docs/GOAL.md`）：裁切 10% 的地板實測 0.520，不扣掉它，
# 「淨化後位移較大」無法排除「該算子本來就把編輯推得比較開」這個平庸解釋。
#
# 影像只有兩張，切不動分片，故一個條件一個 process。八個條件 ＋ 地板 = 九個，
# 五張卡每卡兩個放得下；空卡不足時用第二個參數分波送。
#
# 算子六個，對準這一輪要回答的問題：
#     identity jpeg75 jpeg30 blur1 blur2 crop_resize0.1
#
# `--gallery` 一律開：**擋下與否要用眼睛判「重畫」對「劣化」**（SigLIP 代理
# 已實測會把「人還在、只是被蓋上紋理」標成 blocked）。沒有圖就沒辦法判。
# 這一批要判的正是「擾動有沒有從主體挪開」，圖比數字更直接。
#
# 用法：bash scripts/purify_ig_lowdist.sh "<五個卡號>"
set -uo pipefail
ROOT=/nfs/home/nelson0314/WACV-s3
PY="$HOME/venvs/wacv/bin/python"
export PYTHONPATH="$ROOT" HF_HOME="$HOME/hf_cache" PYTHONIOENCODING=utf-8
export TOKENIZERS_PARALLELISM=false
cd "$ROOT" || { echo "錯誤：找不到 $ROOT" >&2; exit 2; }

DEVS=(${1:-})
[ ${#DEVS[@]} -lt 1 ] && { echo "用法：$0 \"<卡號>\" [\"<條件標籤...>\"] [nofloor]" >&2; exit 2; }
bash scripts/free_cards.sh --assert "${DEVS[*]}" || exit 3

SRC=runs/ip2p_ig_lowdist
OUT="$SRC/purify"
mkdir -p "$OUT"
IMGS="task_attr_mod_color_11699 task_attr_mod_color_6205"
PUR="identity jpeg75 jpeg30 blur1 blur2 crop_resize0.1"
COMMON="--data data/omniedit150 --attacker ip2p --seeds 3 --purifiers $PUR"
# 卡是多人共用的，空卡數會變。第二個參數給條件子集就能分波送；
# 第三個參數給 `nofloor` 表示地板已經跑過，不要重跑。
TAGS="${2:-r25_f04 r25_f02 r18_f08 r18_f04 r12_f08 r12_f04 r25_f08_comp}"
WANT_FLOOR=1
[ "${3:-}" = "nofloor" ] && WANT_FLOOR=0

# 只送有防禦圖的條件。缺的一律印出來，不靜默略過。
GOOD=""
for t in $TAGS; do
  if ls "$SRC/$t"/*__def.png >/dev/null 2>&1; then GOOD="$GOOD $t"
  else echo "[skip] $SRC/$t 沒有防禦圖" >&2; fi
done
[ -z "$GOOD" ] && { echo "錯誤：沒有任何條件有防禦圖" >&2; exit 2; }
FIRST=$(echo $GOOD | awk '{print $1}')

n=$(( $(echo $GOOD | wc -w) + WANT_FLOOR ))
# 每卡最多 2 個 process（`docs/OPERATIONS.md`）。疊到 3 個實測整批 OOM。
if [ "$n" -gt $(( ${#DEVS[@]} * 2 )) ]; then
  echo "錯誤：$n 個 process 需要至少 $(( (n + 1) / 2 )) 張卡，只給了 ${#DEVS[@]} 張。" >&2
  exit 2
fi

i=0
launch() {              # $1 輸出標籤  $2 防禦圖來源  $3 額外旗標
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
# 地板的 `--run` 指到第一個條件的目錄：`--floor` 把**原圖**當防禦圖，
# 來源目錄只用來定位這一批的設定，不會讀到它的防禦圖。
[ "$WANT_FLOOR" -eq 1 ] && launch floor "$FIRST" --floor

sleep 25
echo "啟動了 $(ps -u "$USER" -o cmd | grep -c '[p]hase_retention') 個 phase_retention process"

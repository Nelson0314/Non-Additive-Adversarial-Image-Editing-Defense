#!/usr/bin/env bash
# `runs/ip2p_band_allocation` 那一批的抗淨化那一半。
#
# **只讀已存的防禦圖，不重跑防禦**（`scripts/phase_retention.py`）。
# 空白地板不可省略：淨化算子自己就會把編輯推開，不扣掉它，「淨化後位移較大」
# 無法排除「該算子本來就把編輯推得比較開」這個平庸解釋。
#
# 算子取六個，對準這一輪要回答的問題（JPEG、模糊、裁切）：
#     identity jpeg75 jpeg30 blur1 blur2 crop_resize0.1
# 刻意不含 jpeg90／jpeg50（同一條 JPEG 曲線上的內點，不改變結論）與
# crop_resize0.15（`runs/ip2p_eot_geom_purify` 已量到它的地板 0.581，
# 可讀範圍只剩 0.19，訊噪比最差的一格）。
#
# `--gallery` 一律開：抗淨化是主主張，而**擋下與否要用眼睛判「重畫」對
# 「劣化」**（`docs/GOAL.md`；SigLIP 代理已實測會把「人還在、只是被蓋上
# 紋理」標成 blocked）。沒有圖就沒有辦法判。
#
# 用法：bash scripts/purify_band_allocation.sh "<五個卡號>" [all|conditions|floor]
set -uo pipefail
ROOT=/nfs/home/nelson0314/WACV-s3
PY="$HOME/venvs/wacv/bin/python"
export PYTHONPATH="$ROOT" HF_HOME="$HOME/hf_cache" PYTHONIOENCODING=utf-8
export TOKENIZERS_PARALLELISM=false
cd "$ROOT" || { echo "錯誤：找不到 $ROOT" >&2; exit 2; }

DEVS=(${1:-})
[ ${#DEVS[@]} -lt 5 ] && { echo "用法：$0 \"<五個卡號>\" [all|conditions|floor]" >&2; exit 2; }
bash scripts/free_cards.sh --assert "${DEVS[*]}" || exit 3

PARTS="${2:-all}"
case "$PARTS" in all|conditions|floor) ;; *)
  echo "用法：$0 \"<卡號>\" [all|conditions|floor]" >&2; exit 2 ;;
esac

SRC=runs/ip2p_band_allocation
OUT="$SRC/purify"
mkdir -p "$OUT"
IMGS=$(cat runs/ip2p_fair_comparison/images10.txt | tr '\n' ' ')
PUR="identity jpeg75 jpeg30 blur1 blur2 crop_resize0.1"
COMMON="--data data/omniedit150 --attacker ip2p --seeds 3 --purifiers $PUR"
TAGS="hi_r25 hi_r40 mid_r25 mid_r40 lo_r25 lo_r40 surv_r25 surv_r40"

# 只送有防禦圖的條件。缺的那些一律印出來，不靜默略過——靜默略過會讓
# 出表時少一列而沒有人發現。
POINTS=""
for t in $TAGS; do
  if [ -d "$SRC/$t" ] && ls "$SRC/$t"/*__def.png >/dev/null 2>&1; then
    POINTS="$POINTS $t"
  else
    echo "[skip] $SRC/$t 沒有防禦圖" >&2
  fi
done
n=$(echo $POINTS | wc -w)
[ "$PARTS" = "floor" ] && n=0
[ "$PARTS" != "conditions" ] && n=$((n + 1))     # 地板一個 process

# 每卡最多 2 個 process（`docs/OPERATIONS.md`）。疊到 3 個實測整批 OOM。
if [ "$n" -gt $(( ${#DEVS[@]} * 2 )) ]; then
  echo "錯誤：$n 個 process 需要至少 $(( (n + 1) / 2 )) 張卡，只給了 ${#DEVS[@]} 張。" >&2
  exit 2
fi

i=0
if [ "$PARTS" != "floor" ]; then
  for t in $POINTS; do
    dev=${DEVS[$(( i % ${#DEVS[@]} ))]}; i=$((i + 1))
    CUDA_VISIBLE_DEVICES="$dev" setsid nohup "$PY" scripts/phase_retention.py \
        --run "$SRC/$t" $COMMON --images $IMGS \
        --out "$OUT/$t.csv" --gallery "$OUT/gallery_$t" \
        < /dev/null >> "$OUT/$t.log" 2>&1 &
    disown
    echo "[purify] $t dev=$dev"
  done
fi
if [ "$PARTS" != "conditions" ]; then
  dev=${DEVS[$(( i % ${#DEVS[@]} ))]}
  CUDA_VISIBLE_DEVICES="$dev" setsid nohup "$PY" scripts/phase_retention.py \
      --run "$SRC/hi_r25" $COMMON --images $IMGS --floor \
      --out "$OUT/floor.csv" --gallery "$OUT/gallery_floor" \
      < /dev/null >> "$OUT/floor.log" 2>&1 &
  disown
  echo "[purify] floor dev=$dev"
fi
sleep 25
echo "啟動了 $(ps -u "$USER" -o cmd | grep -c '[p]hase_retention') 個 phase_retention process"

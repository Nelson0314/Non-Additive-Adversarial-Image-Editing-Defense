#!/usr/bin/env bash
# 等失真頭對頭的抗淨化：本方法與 DCT-Shield 在**同一個算子集合**上並排。
#
# 為什麼要這一批
# ────────────────────────────────────────────────────────────────────
# `runs/ip2p_mainline_purify` 只跑了 JPEG 軸五個算子，而且沒有留 `--gallery`。
# 於是兩件事做不到：
#   1. 等失真下的**模糊與裁切**欄沒有 DCT 對照。目前只能引用
#      `runs/ip2p_purify_headtohead`，但那一批本方法的失真比 `dct_e18`
#      高出 22% 與 41%，不是等失真。
#   2. 淨化之後的圖沒有 DCT 對照，無法並排看。
# 本批用現行六算子協定把兩件一起補掉。
#
# **只讀已存的防禦圖，不重跑防禦**（`scripts/phase_retention.py`）。
# `runs/ip2p_mainline/<條件>/` 底下的 `__def.png` 都在，故不需要重新訓練。
#
# 三組等失真配對（DISTS 差距都在 3% 以內，取自 `runs/ip2p_mainline`，
# 本方法與 DCT-Shield 在同一批裡跑）：
#     低失真    ours_pg_n 0.0829   對  dct_native       0.0804
#     中失真    ours_ph_q 0.1062   對  dct_aj50_eps0.22 0.1066
#     中高失真  ours_pg_q 0.1332   對  dct_aj85         0.1264
#
# **空白地板要自己跑，不可沿用舊的。** 舊地板在裁切欄是舊參照量的
# （幾何類現行取「同一個算子淨化過的原圖」為參照、地板由構造為 0），
# 沿用會讓幾何欄的基準混掉。
#
# `--gallery` 一律開：擋下與否要用眼睛判「重畫」對「劣化」，SigLIP 代理
# 已實測會把「人還在、只是被蓋上紋理」標成 blocked。
#
# 用法：bash scripts/purify_matched_headtohead.sh "<卡號…>" [條件…] [nofloor]
set -uo pipefail
ROOT=/nfs/home/nelson0314/WACV-s3
PY="$HOME/venvs/wacv/bin/python"
export PYTHONPATH="$ROOT" HF_HOME="$HOME/hf_cache" PYTHONIOENCODING=utf-8
export TOKENIZERS_PARALLELISM=false
cd "$ROOT" || { echo "錯誤：找不到 $ROOT" >&2; exit 2; }

DEVS=(${1:-})
[ ${#DEVS[@]} -lt 1 ] && {
  echo "用法：$0 \"<卡號…>\" [條件…] [nofloor]" >&2; exit 2; }
# 卡是多人共用的。**這個檢查會 exit，不是印出來就算。**
bash scripts/free_cards.sh --assert "${DEVS[*]}" || exit 3

SRC=runs/ip2p_mainline
OUT=runs/ip2p_matched_headtohead
mkdir -p "$OUT"
IMGS="task_attr_mod_color_11699 task_attr_mod_color_6205"
# 現行六算子協定，與 runs/ip2p_split_band、runs/ip2p_ig_loss 同一組，
# 於是新數字與那幾批可以並列。
PUR="identity jpeg75 jpeg30 blur1 blur2 crop_resize0.1"
COMMON="--data data/omniedit150 --attacker ip2p --seeds 3 --purifiers $PUR"

TAGS="${2:-ours_pg_n dct_native ours_ph_q dct_aj50_eps0.22 ours_pg_q dct_aj85}"
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

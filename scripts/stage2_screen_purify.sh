#!/usr/bin/env bash
# 分階段訓練篩選批的**抗淨化讀數**。只讀 `scripts/stage2_screen.sh` 存下來的
# 防禦圖，不重跑防禦。
#
# 算子只留模糊與裁切兩族
# ────────────────────────────────────────────────────────────────────
# 這一批要問的就是那兩欄（JPEG 那一欄已經有一個實測有效的手段——量化交付，
# 而它與階段二是兩個獨立的旗鈕，混在同一批會分不出誰的功勞）。`identity`
# 不可省：它是 `phase_retention.py` 算保留率的分母，也是「未淨化那一側賠了
# 多少」的讀數，而那正是判準 S4 的分母。
#
# 空白地板不重跑
# ────────────────────────────────────────────────────────────────────
# 地板那一格的「防禦圖」就是原圖，量到的是算子自己造成的位移，只與
# （影像, 算子, 種子）有關。本批的三張是十張的子集，
# `runs/ip2p_mainline_purify/floor_{color,object}.csv` 已經有它們在同一組
# 算子上的地板，**重跑是白花 35 分鐘**。出表時把兩個目錄一起餵給
# `retention_table.py`（它的 `--src` 收多個路徑）。
#
# **出表時要注意 n 欄**：本批的條件是 3 張，主線的條件是 10 張，
# `docs/PENDING.md` 已記過不同 n 的列不可並列。S2／S3 的比較對象是
# `ours_pg_m` **restricted 到同樣這三張**，由
# `runs/ip2p_mainline_purify/ours_pg_m_*.csv` 的逐圖列取子集重算，不是用它的
# 十張平均。
#
# 用法：bash scripts/stage2_screen_purify.sh "<卡號>" ["<tag> ..."]
set -uo pipefail
ROOT=/nfs/home/nelson0314/WACV-s3
PY="$HOME/venvs/wacv/bin/python"
export PYTHONPATH="$ROOT" HF_HOME="$HOME/hf_cache" PYTHONIOENCODING=utf-8
export TOKENIZERS_PARALLELISM=false
cd "$ROOT"

SRC=runs/ip2p_stage2
OUT=runs/ip2p_stage2/purify
GAL=runs/gallery_stage2
mkdir -p "$OUT" "$GAL"

LIST="$SRC/images3.txt"
[ -f "$LIST" ] || { echo "錯誤：找不到 $LIST" >&2; exit 2; }
IMGS=$(tr '\n' ' ' < "$LIST")
N_IMG=$(wc -l < "$LIST")

PUR="${PUR:-identity blur1 blur2 crop_resize0.1 crop_resize0.15}"
COMMON="--data data/omniedit150 --attacker ip2p --seeds 3 --purifiers $PUR"

DEVS=(${1:-})
TAGS=(${2:-s2_tight s2_hard s2_null})
[ ${#DEVS[@]} -eq 0 ] || [ -z "${DEVS[0]}" ] && {
  echo "用法：$0 \"<卡號>\" [\"<tag> ...\"]" >&2; exit 2; }

bash scripts/free_cards.sh --assert "${DEVS[*]}" || exit 3

n=${#TAGS[@]}
if [ "$n" -gt $(( ${#DEVS[@]} * 2 )) ]; then
  echo "錯誤：$n 個 process 需要至少 $(( (n + 1) / 2 )) 張卡，只給了 ${#DEVS[@]} 張" >&2
  echo "      每卡 3 個實測會 OOM（docs/OPERATIONS.md）" >&2
  exit 2
fi

i=0
for tag in "${TAGS[@]}"; do
  run="$SRC/$tag"
  n_def=$(ls -1 "$run"/*__def.png 2>/dev/null | wc -l)
  [ "$n_def" -ne "$N_IMG" ] && {
    echo "錯誤：$tag 只有 $n_def/$N_IMG 張防禦圖" >&2; exit 3; }
  dev=${DEVS[$(( i / 2 % ${#DEVS[@]} ))]}
  i=$(( i + 1 ))
  CUDA_VISIBLE_DEVICES="$dev" nohup "$PY" scripts/phase_retention.py \
      --run "$run" --images $IMGS $COMMON --gallery "$GAL/$tag" \
      --out "$OUT/${tag}.csv" > "$OUT/${tag}.log" 2>&1 &
  echo "[stage2-purify] $tag dev=$dev pid=$!"
done
echo "[stage2-purify] 送出 $i 個（$(date)）"

#!/usr/bin/env bash
# 影像引導消除損失：跑到收斂。**不設效果判準**（`CLAUDE.md`）。
#
# 這一批只做三件事：跑到收斂、忠實記錄數據、留下影像。有沒有效果由使用者判斷。
#
# 收斂怎麼判
# ────────────────────────────────────────────────────────────────────
# 訓練用的損失每一步重抽 `(t, eps)`，逐步值本來就會抖（實測 0.16–0.61），那是
# 取樣變異不是參數在漂。判收斂一律看 `trace.csv` 的 `eval` 欄——它用一組
# **固定**的 8 抽樣（`--eval-seed` 與訓練的 `--seed` 分開）每 100 步評估一次。
# `--patience 15 --min-delta 0.0002` 表示連續 12 次評估（＝1200 步）都沒有比
# 歷史最佳再低 0.2% 就停。結果 CSV 的 `stop_reason` 分得出「跑滿」與「早停」。
#
# 五個工作點，一卡一個
# ────────────────────────────────────────────────────────────────────
# 實測單步成本：一卡一個 0.595 s/步、一卡兩個 1.06 s/步。一卡一個較慢但每個
# process 佔約 10 GiB，兩個就吃掉 20/24 GiB——卡是共用的，上次別人插進來
# 3.9 GiB 就 OOM 了。
#
#   ig_d21 / ig_d25   diffuse_src，r = 2.1 / 2.5
#   ig_n30 / ig_n35   noise，r = 3.0 / 3.5
#   ln_long           latent_norm，**同樣的固定步長**——對照組必須拿到同一套
#                     收斂處理，否則會變成用收斂的方法比未收斂的對照
#
# 用法：bash scripts/image_guidance_converge.sh "<五個卡號>" ["<tag> ..."]
set -uo pipefail
ROOT=/nfs/home/nelson0314/WACV-s3
PY="$HOME/venvs/wacv/bin/python"
export PYTHONPATH="$ROOT" HF_HOME="$HOME/hf_cache" PYTHONIOENCODING=utf-8
export TOKENIZERS_PARALLELISM=false
cd "$ROOT" || { echo "錯誤：找不到 $ROOT" >&2; exit 2; }

OUT=runs/ip2p_ig_converge
mkdir -p "$OUT"
IMGS=$(tr '\n' ' ' < runs/ip2p_fair_comparison/images10.txt)

STEPS="${STEPS:-8000}"
# **步長固定，不隨步數縮小。** 預設公式是 α = radius/(steps·0.25)，於是把
# --steps 由 1000 拉到 12000 會讓 α 變成 1/12——「跑更多步」同時「每步走更
# 小」，兩者互相抵銷。實測（`runs/ip2p_ig_stepsize/`，同影像、同工作點、
# 同半徑）：固定 α=0.01 在第 1000 步就到 eval 0.0103，而綁定 α=0.00083 跑到
# 第 12000 步才 0.0133；結果列是 DISTS 0.1480／位移 0.5624 對 0.0769／0.3428。
# 0.01 正是 1000 步公式在 radius 2.5 下的值，屬於已知可用的量級。
STEP_SIZE="${STEP_SIZE:-0.01}"
BASE="--data data/omniedit150 --conditions phase_gain --quantile 0 \
--freq-weight jpeg_luma --freq-weight-power 0.25 --hop 8 --gain-ratio 1.0 \
--steps $STEPS --step-size $STEP_SIZE \
--eval-every 100 --eval-draws 8 --patience 15 --min-delta 0.0002"

POINTS="
ig_d21:--loss~image_guidance~--ig-zt~diffuse_src~--radius~2.1
ig_d25:--loss~image_guidance~--ig-zt~diffuse_src~--radius~2.5
ig_n30:--loss~image_guidance~--ig-zt~noise~--radius~3.0
ig_n35:--loss~image_guidance~--ig-zt~noise~--radius~3.5
ln_long:--loss~latent_norm~--radius~2.5
"

DEVS=(${1:-})
ONLY="${2:-}"
[ ${#DEVS[@]} -eq 0 ] || [ -z "${DEVS[0]}" ] && {
  echo "用法：$0 \"<五個卡號>\" [\"<tag> ...\"]" >&2; exit 2; }

for flag in --eval-every --patience --min-delta --ig-zt; do
  grep -q -- "$flag" scripts/ip2p_run.py || {
    echo "錯誤：scripts/ip2p_run.py 不認得 $flag，先同步本機的改動" >&2; exit 2; }
done

selected() { [ -z "$ONLY" ] && return 0; for t in $ONLY; do [ "$t" = "$1" ] && return 0; done; return 1; }

n=0
for p in $POINTS; do IFS=: read -r tag _ <<< "$p"; selected "$tag" && n=$(( n + 1 )); done
[ "$n" -eq 0 ] && { echo "錯誤：選到 0 個 tag" >&2; exit 2; }
if [ "$n" -gt ${#DEVS[@]} ]; then
  echo "錯誤：$n 個工作點、一卡一個，需要 $n 張卡，只給了 ${#DEVS[@]} 張" >&2
  exit 2
fi

bash scripts/free_cards.sh --assert "${DEVS[*]}" || exit 3

i=0
for p in $POINTS; do
  IFS=: read -r tag extra <<< "$p"
  selected "$tag" || continue
  dev=${DEVS[$i]}
  i=$(( i + 1 ))
  CUDA_VISIBLE_DEVICES="$dev" nohup "$PY" scripts/ip2p_run.py \
      --out "$OUT/$tag" $BASE $(echo "$extra" | tr '~' ' ') \
      --images $IMGS > "$OUT/$tag.log" 2>&1 &
  echo "[converge] $tag dev=$dev pid=$!  steps=$STEPS $(echo "$extra" | tr '~' ' ')"
done
echo "[converge] 送出 $i 個（$(date)）"

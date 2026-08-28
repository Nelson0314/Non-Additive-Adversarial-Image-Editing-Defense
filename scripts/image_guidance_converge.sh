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
#   ln_long           latent_norm 同步數，**當數據不當判準**
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

STEPS="${STEPS:-12000}"
BASE="--data data/omniedit150 --conditions phase_gain --quantile 0 \
--freq-weight jpeg_luma --freq-weight-power 0.25 --hop 8 --gain-ratio 1.0 \
--steps $STEPS --eval-every 100 --eval-draws 8 --patience 15 --min-delta 0.0002"

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

#!/usr/bin/env bash
# 失真匹配組：把 DCT-Shield 抗 JPEG 版（Y-only, Q_alg 0.85）的預算 eps 拉高，
# 直到它的防禦圖失真追上本方法的工作點，才有等失真的頭對頭。
#
# 為什麼是把對方拉上來而不是把自己壓下去
# ────────────────────────────────────────────────────────────────────
# 現況（十張、`fid_dists`）：
#
#     dct_aj85  eps 1.0  0.1118      ours_pg_q    r0.9 量化  0.1283
#                                    ours_pg_m    r2.0 無量化 0.1453
#                                    ours_ph_q20  r=pi 量化   0.1828
#                                    ours_pg_q20  r2.0 量化   0.1947
#
# 也就是本方法所有能擋下的工作點都比 DCT-Shield 的論文設定更花失真預算。
# 把本方法壓到 0.1118 會同時削掉它的強度，看起來像本方法輸；把 DCT-Shield
# 抬到 0.19 則不動本方法，且順便回答「它加預算會不會跟著變強」。兩者都要有
# 才不是挑對自己有利的那一邊，但先做後者——它不需要重調本方法的半徑。
#
# **這是預算消融，不是論文設定。** §6.3 的 eps 是 1.0；這裡的 eps > 1 一律
# 只能標成「budget-matched」，不可以拿去代表 DCT-Shield 的論文結果。
#
# 用法：bash scripts/mainline_matched.sh "<卡號>" ["<tag> ..."]
set -uo pipefail
ROOT=/nfs/home/nelson0314/WACV-s3
PY="$HOME/venvs/wacv/bin/python"
export PYTHONPATH="$ROOT" HF_HOME="$HOME/hf_cache" PYTHONIOENCODING=utf-8
export TOKENIZERS_PARALLELISM=false
cd "$ROOT"

OUT=runs/ip2p_mainline
mkdir -p "$OUT"
IMGS=$(tr '\n' ' ' < runs/ip2p_fair_comparison/images10.txt)

POINTS="
dct_aj85_eps1.5:--conditions~dct_shield_y~--q-alg~0.85~--eps~1.5~--dct-steps~1000
dct_aj85_eps2.2:--conditions~dct_shield_y~--q-alg~0.85~--eps~2.2~--dct-steps~1000
dct_aj85_eps3.2:--conditions~dct_shield_y~--q-alg~0.85~--eps~3.2~--dct-steps~1000
dct_aj85_eps4.5:--conditions~dct_shield_y~--q-alg~0.85~--eps~4.5~--dct-steps~1000
"

DEVS=(${1:-})
ONLY="${2:-}"
[ ${#DEVS[@]} -eq 0 ] && { echo "用法：$0 \"<卡號>\" [\"<tag> ...\"]" >&2; exit 2; }

bash scripts/free_cards.sh --assert "${DEVS[*]}" || exit 3

selected() { [ -z "$ONLY" ] && return 0; for t in $ONLY; do [ "$t" = "$1" ] && return 0; done; return 1; }

n=0
for p in $POINTS; do IFS=: read -r tag _ <<< "$p"; selected "$tag" && n=$(( n + 1 )); done
[ "$n" -eq 0 ] && { echo "錯誤：選到 0 個 tag" >&2; exit 2; }
if [ "$n" -gt $(( ${#DEVS[@]} * 2 )) ]; then
  echo "錯誤：$n 個工作點需要至少 $(( (n + 1) / 2 )) 張卡，只給了 ${#DEVS[@]} 張" >&2; exit 2
fi

i=0
for p in $POINTS; do
  IFS=: read -r tag extra <<< "$p"
  selected "$tag" || continue
  dev=${DEVS[$(( i / 2 % ${#DEVS[@]} ))]}
  i=$(( i + 1 ))
  CUDA_VISIBLE_DEVICES="$dev" nohup "$PY" scripts/ip2p_run.py \
      --out "$OUT/$tag" $(echo "$extra" | tr '~' ' ') \
      --images $IMGS --data data/omniedit150 > "$OUT/$tag.log" 2>&1 &
  echo "[matched] $tag dev=$dev pid=$!  $(echo "$extra" | tr '~' ' ')"
done
echo "[matched] 送出 $i 個（$(date)）"

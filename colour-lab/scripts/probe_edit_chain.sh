#!/usr/bin/env bash
# 探針批的 ip2p 編輯鏈。**整條鏈跑在同一張卡上**：分母、16 個取樣點、
# curve_control 三者的位移要互相比較，卡不同就多了一個變因。
#
# 順序：分母 → 等探針的防禦圖寫完 → 16 個取樣點 → 等 curve_control 的解
# → curve_control → 位移彙整。
set -u
L=/nfs/home/nelson0314/WACV-colour-lab
cd "$L" || exit 1
export PYTHONPATH=$L
export HF_HOME=/var/cache/huggingface
PY=/nfs/home/nelson0314/venvs/wacv/bin/python
LOG=$L/logs
EDITS=$L/runs/colour_probe_edits
PRE=$L/runs/colour_probe_preflight
POINTS=$L/runs/colour_probe/points
SOLVE=$L/runs/curve_control
mkdir -p "$LOG" "$EDITS"

stamp() { date -Is; }

run() {
  name=$1; shift
  echo "[START] $name $(stamp)"
  "$@" > "$LOG/edit_${name}.log" 2>&1
  rc=$?
  echo "[END] $name rc=$rc $(stamp)"
  return $rc
}

edit_one() {
  # $1 = 條件名, $2 = 防禦圖目錄（空字串＝未防禦的分母）
  if [ -z "$2" ]; then
    run "$1" $PY scripts/edit_preflight.py --data data/portraits \
      --scenarios ip2p --suffix _si18 --s-i 1.8 --out "$PRE"
  else
    run "$1" $PY scripts/edit_preflight.py --data data/portraits \
      --scenarios ip2p --suffix _si18 --s-i 1.8 \
      --defended "$2" --out "$EDITS/$1"
  fi
}

echo "[CHAIN] start $(stamp) card=${CUDA_VISIBLE_DEVICES:-unset}"

edit_one denominator ""
rc=$?
if [ "$rc" -ne 0 ]; then echo "[ABORT] 分母失敗 rc=$rc"; exit 1; fi

# 探針是 CPU 工作，可能還沒寫完。16 個點 × 8 張 = 128 張防禦圖。
i=0
while [ "$i" -lt 180 ]; do
  n=$(ls -1 "$POINTS" 2>/dev/null | wc -l)
  m=$(find "$POINTS" -name '*__def.png' 2>/dev/null | wc -l)
  if [ "$n" -eq 16 ] && [ "$m" -eq 128 ]; then break; fi
  i=$((i + 1))
  sleep 20
done
n=$(ls -1 "$POINTS" 2>/dev/null | wc -l)
m=$(find "$POINTS" -name '*__def.png' 2>/dev/null | wc -l)
echo "[CHAIN] 取樣點 $n 個、防禦圖 $m 張 $(stamp)"
if [ "$n" -ne 16 ] || [ "$m" -ne 128 ]; then
  echo "[ABORT] 探針的產物不完整，不用半套的輸入往下跑"; exit 1
fi

for d in $(ls -1 "$POINTS" | sort); do
  edit_one "$d" "$POINTS/$d"
  rc=$?
  if [ "$rc" -ne 0 ]; then echo "[FAIL] $d rc=$rc"; fi
done

# curve_control 的四個分片各解兩張，湊滿八張才動。
i=0
while [ "$i" -lt 900 ]; do
  k=$(find "$SOLVE" -name '*__curve_control__defended.png' 2>/dev/null | wc -l)
  if [ "$k" -eq 8 ]; then break; fi
  i=$((i + 1))
  sleep 20
done
k=$(find "$SOLVE" -name '*__curve_control__defended.png' 2>/dev/null | wc -l)
echo "[CHAIN] curve_control 防禦圖 $k / 8 $(stamp)"
if [ "$k" -ne 8 ]; then
  echo "[ABORT] curve_control 的解不完整"; exit 1
fi

# `paper_baseline.py` 寫的是 `__defended.png`，`edit_preflight.py` 找的是
# `__def.png`。改名而不是改腳本：兩支都是主 repo 的唯讀複製。
mkdir -p "$SOLVE/defended"
for f in $(find "$SOLVE" -name '*__curve_control__defended.png' | sort); do
  b=$(basename "$f" __defended.png)
  cp "$f" "$SOLVE/defended/${b}__def.png"
done
echo "[CHAIN] 改名後 $(ls -1 "$SOLVE/defended" | wc -l) 張 $(stamp)"

edit_one curve_control "$SOLVE/defended"
rc=$?
echo "[CHAIN] curve_control 編輯 rc=$rc $(stamp)"

run displacement $PY scripts/edit_displacement.py \
  --defended-root "$EDITS" --preflight "$PRE" --data data/portraits \
  --ip2p-arm ip2p_si18 --out "$EDITS/displacement.csv"
rc=$?
echo "[CHAIN] done rc=$rc $(stamp)"

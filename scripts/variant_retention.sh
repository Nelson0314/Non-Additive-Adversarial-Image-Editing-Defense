#!/usr/bin/env bash
# 改動一與改動三的**抗淨化**讀數，5 張（看方向）。
#
# 為什麼非做不可：改動三（把淨化算子放進最佳化迴圈）在未淨化的位移上是輸的
# ——等 DISTS 錨點上 fixed75 只有基準的 0.750 倍、eot_ops 0.900 倍。那正是
# 它應該付出的代價，**它的價值只可能出現在淨化之後的讀數上**。只報未淨化的
# 位移等於用錯的尺去量它。
#
# 四個條件的失真刻意取在同一段（DISTS 0.060–0.064），比較才有意義：
#   基準 theta=1.4          0.0605
#   改動一 phase_gain r=1.2 0.0635
#   改動三 eot_ops r=1.27   0.0612
#   改動三 fixed75 r=1.27   0.0633
# 另加空白地板（防禦圖就是原圖），否則「淨化後位移比較大」無法排除
# 「該算子本來就把編輯推得比較開」。
set -u
source ~/env.sh >/dev/null 2>&1
cd /nfs/home/nelson0314/WACV-s3
export PYTHONPATH=/nfs/home/nelson0314/WACV-s3 TOKENIZERS_PARALLELISM=false
PY=$HOME/venvs/wacv/bin/python
ROOT=runs/fair0820/a7_ret
mkdir -p $ROOT
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a $ROOT/log.txt; }

# 五類任務各一張，取自校準用的 25 張
mapfile -t IMGS < <($PY - <<'PY'
import pathlib
names = (pathlib.Path("runs/fair0820/images25.txt")
         .read_text(encoding="utf-8").split())
seen, out = set(), []
for t in ("task_attr_mod", "task_env", "task_obj_add", "task_obj_remove",
          "task_obj_swap"):
    for n in names:
        if n.startswith(t) and t not in seen:
            out.append(n); seen.add(t); break
print("\n".join(out))
PY
)
[ "${#IMGS[@]}" -eq 5 ] || { log "只挑到 ${#IMGS[@]} 張，中止"; exit 1; }
log "抗淨化（5 張）開始：${IMGS[*]}"

PUR="identity blur1 jpeg75 crop_resize0.1"

# 來源目錄各自就是單一批次（一個 process 跑完整 25 張），不需要合併
declare -A SRC=(
  [base]="runs/fair0820/a_phase/p00_t14"
  [gain]="runs/fair0820/a5_gain/pg_r12"
  [eot]="runs/fair0820/a6_purify/eot_t127"
  [f75]="runs/fair0820/a6_purify/f75_t127"
)
for k in "${!SRC[@]}"; do
  [ -f "${SRC[$k]}/results.csv" ] || { log "缺 ${SRC[$k]}/results.csv，中止"; exit 1; }
done

g=0
for k in base gain eot f75; do
  for img in "${IMGS[@]}"; do
    CUDA_VISIBLE_DEVICES=$g nohup $PY scripts/phase_retention.py \
      --run "${SRC[$k]}" --data data/omniedit150 --attacker ip2p \
      --images "$img" --purifiers $PUR --seeds 3 \
      --out "$ROOT/${k}_${img}.csv" > "$ROOT/${k}_${img}.log" 2>&1 &
    g=$(((g+1) % 8))
  done
done
# 空白地板：與條件無關，只需一次
for img in "${IMGS[@]}"; do
  CUDA_VISIBLE_DEVICES=$g nohup $PY scripts/phase_retention.py \
    --run "${SRC[base]}" --data data/omniedit150 --attacker ip2p --floor \
    --images "$img" --purifiers $PUR --seeds 3 \
    --out "$ROOT/floor_${img}.csv" > "$ROOT/floor_${img}.log" 2>&1 &
  g=$(((g+1) % 8))
done
wait
log "完成，$(cat $ROOT/*.csv 2>/dev/null | grep -c '^task_') 列"

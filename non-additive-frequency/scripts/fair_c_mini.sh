#!/usr/bin/env bash
# 階段 C 的縮小版：每個方法 5 張，只看趨勢（使用者 2026-08-21 裁定）。
#
# 全量版（75 張）已於 00:40 停止，DCT-Shield 那一格當時已完成 69 張——**那些
# 列保留**，本輪只補相位與隨機相位，以及五張的空白地板。五張分屬五個任務類別，
# 各取一張，避免趨勢被單一類別支配。
#
# 一個 process 只負責一格（一張圖 × 一個條件），共 15 個，分 8 張卡。
# 全量版是 24 個 process 擠 8 張卡、一格 868 秒；這裡密度減半，一格會更快。
set -u
source ~/env.sh >/dev/null 2>&1
cd /nfs/home/nelson0314/WACV-s3
export PYTHONPATH=/nfs/home/nelson0314/WACV-s3 TOKENIZERS_PARALLELISM=false
PY=$HOME/venvs/wacv/bin/python
ROOT=runs/fair0820/c
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a $ROOT/log.txt; }

RUN=runs/fair0820/b_merged
[ -f "$RUN/results.csv" ] || { log "$RUN 沒有 results.csv，中止"; exit 1; }

# 五類各取一張，且必須是 DCT 那一格已經跑完的（才有得比）
mapfile -t IMGS < <($PY - <<'PY'
import csv, glob, collections
done = collections.defaultdict(set)
for f in glob.glob("runs/fair0820/c/ret_*.csv"):
    with open(f, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r.get("condition") == "dct_shield":
                done[r["image"]].add(r["purifier"])
need = {"identity", "blur1", "jpeg75", "crop_resize0.1"}
ok = sorted(i for i, p in done.items() if need <= p)
picked, seen = [], set()
for t in ("task_attr_mod", "task_env", "task_obj_add", "task_obj_remove",
          "task_obj_swap"):
    for i in ok:
        if i.startswith(t) and t not in seen:
            picked.append(i); seen.add(t); break
print("\n".join(picked))
PY
)
[ "${#IMGS[@]}" -eq 5 ] || { log "只挑到 ${#IMGS[@]} 張，不是 5，中止"; exit 1; }
log "縮小版開始，5 張：${IMGS[*]}"

PUR="identity blur1 jpeg75 crop_resize0.1"
g=0
for img in "${IMGS[@]}"; do
  for cond in phase phase_rand; do
    CUDA_VISIBLE_DEVICES=$g nohup $PY scripts/phase_retention.py \
      --run "$RUN" --data data/omniedit150 --attacker ip2p \
      --images "$img" --conditions "$cond" --purifiers $PUR --seeds 3 \
      --out "$ROOT/mini_${cond}_${img}.csv" \
      > "$ROOT/mini_${cond}_${img}.log" 2>&1 &
    g=$(((g+1) % 8))
  done
  CUDA_VISIBLE_DEVICES=$g nohup $PY scripts/phase_retention.py \
    --run "$RUN" --data data/omniedit150 --attacker ip2p --floor \
    --images "$img" --purifiers $PUR --seeds 3 \
    --out "$ROOT/mini_floor_${img}.csv" \
    > "$ROOT/mini_floor_${img}.log" 2>&1 &
  g=$(((g+1) % 8))
done
wait
log "縮小版完成，$(cat $ROOT/mini_*.csv 2>/dev/null | grep -c '^task_') 列"

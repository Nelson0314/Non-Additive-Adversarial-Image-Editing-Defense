#!/usr/bin/env bash
# 人眼預算批次（hb5）：五張圖、五個類別、七個條件的抗編輯＋抗淨化。
#
# 為什麼要分片：`write_csv` 每次呼叫都整份覆寫 results.csv，兩個行程寫同一個
# 目錄會互相蓋掉。故每個工作有自己的 --out，最後由 merge_runs.py 併起來。
#
# 用法：  bash scripts/run_hb5.sh <job> <gpu>
# 工作表見 JOBS 區塊；波次與相依見 docs 的批次說明。

set -euo pipefail

JOB="${1:?用法：run_hb5.sh <job> <gpu>}"
GPU="${2:?用法：run_hb5.sh <job> <gpu>}"

ROOT=/nfs/home/nelson0314/WACV-s3
PY="$HOME/venvs/wacv/bin/python"

# env.sh 的 PYTHONPATH 指向 $HOME/WACV（另一個舊 repo），這裡必須自己明給。
export PYTHONPATH="$ROOT"
export HF_HOME="$HOME/hf_cache"
export PYTHONIOENCODING=utf-8
export CUDA_VISIBLE_DEVICES="$GPU"

cd "$ROOT"

DATA=data/lo_aligned
IMAGES=(man_02 woman_02 dog_03 horse_03 cat_01)
B=runs/hb5

mkdir -p "$B"

run() { echo "[hb5] $JOB gpu=$GPU :: $*"; "$@"; }

case "$JOB" in

  # ---------- 波次一：攻擊＋抗編輯評測 ----------
  # photoguard_c 是 6486 s/圖（實測），四個工作分掉五張圖。
  s1_pgc_a)  run "$PY" scripts/apa_baseline.py --out "$B/g0" --data "$DATA" \
                 --conditions photoguard_c --images man_02 woman_02 ;;
  s1_pgc_b)  run "$PY" scripts/apa_baseline.py --out "$B/g1" --data "$DATA" \
                 --conditions photoguard_c --images dog_03 horse_03 ;;
  s1_pgc_c)  run "$PY" scripts/apa_baseline.py --out "$B/g2p" --data "$DATA" \
                 --conditions photoguard_c --images cat_01 ;;
  # 弱 baseline 與另外兩個加性方法，全部走各自論文的原生預算（使用者
  # 2026-08-14 裁定不對齊）。跑完接 s1_pgc_c。
  s1_cheap)  run "$PY" scripts/apa_baseline.py --out "$B/g2" --data "$DATA" \
                 --conditions apa_weak mist dia_r --images "${IMAGES[@]}" ;;

  # A 臂三條件不必重跑：runs/phaseA_human 已在人眼半徑上跑完 24 張。
  # 故本工作在波次一就能開始，不等任何東西。
  s2_arm)    run "$PY" scripts/phase_retention.py --run runs/phaseA_human \
                 --data "$DATA" --seeds 3 --images "${IMAGES[@]}" \
                 --conditions add phase phase_rand \
                 --out "$B/retention_arm.csv" ;;

  # ---------- 併批（CPU，不占卡） ----------
  merge1)    run "$PY" scripts/merge_runs.py --out "$B" \
                 --src "$B/g2" --src-pick runs/phaseA_human \
                 --images "${IMAGES[@]}" ;;
  merge2)    run "$PY" scripts/merge_runs.py --out runs/hb5_pgc \
                 --src "$B/g0" "$B/g1" "$B/g2p" ;;

  # ---------- 波次二：抗淨化 retention ----------
  # 十個算子（含 C&R 串接）× 3 個編輯 seed，實測 659 s/cell。
  s2_base_1) run "$PY" scripts/phase_retention.py --run "$B" --data "$DATA" \
                 --seeds 3 --images man_02 woman_02 dog_03 \
                 --conditions apa_weak mist dia_r --out "$B/retention_b1.csv" ;;
  s2_base_2) run "$PY" scripts/phase_retention.py --run "$B" --data "$DATA" \
                 --seeds 3 --images horse_03 cat_01 \
                 --conditions apa_weak mist dia_r --out "$B/retention_b2.csv" ;;
  s2_pgc_1)  run "$PY" scripts/phase_retention.py --run runs/hb5_pgc --data "$DATA" \
                 --seeds 3 --images man_02 woman_02 --out "$B/retention_p1.csv" ;;
  s2_pgc_2)  run "$PY" scripts/phase_retention.py --run runs/hb5_pgc --data "$DATA" \
                 --seeds 3 --images dog_03 horse_03 --out "$B/retention_p2.csv" ;;
  s2_pgc_3)  run "$PY" scripts/phase_retention.py --run runs/hb5_pgc --data "$DATA" \
                 --seeds 3 --images cat_01 --out "$B/retention_p3.csv" ;;

  *) echo "未知的 job：$JOB" >&2; exit 2 ;;
esac

echo "[hb5] $JOB 完成"

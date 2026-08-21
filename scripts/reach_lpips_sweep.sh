#!/usr/bin/env bash
# 強度延伸掃描：把本方法推到 DCT-Shield 的 LPIPS 水準，讓等 LPIPS 錨點第一次
# 算得出來。basic-1 的 8 張卡。
#
# 為什麼要這一批
# ────────────────────────────────────────────────────────────────────
# `runs/ip2p_fair_comparison/curve_fid_lpips_*_anchors.csv` 目前**四個錨點全部
# out_of_range**：DCT-Shield ε=1 落在 LPIPS 0.4584，而本方法最強的掃描點只到
# 0.2617（phase_gain r=1.5 gr=0.25）。兩條曲線在 LPIPS 軸上**完全不重疊**，
# 於是協定禁止內插，「誰比較強」在這個軸上根本沒有讀數。
#
# `runs/distortion_axis_analysis/axis_fits.csv` 量到：在 LPIPS 軸上各方法的
# 原點斜率是 1.20–1.52（變異係數 0.108），在 DISTS 軸上是 2.08–7.08（0.459）。
# 也就是說 LPIPS 對四個方法開的是同一個價、DISTS 不是。故主軸取 LPIPS，
# 而缺的是**可達性**：本方法能不能付到那麼多失真。
#
# 純相位付不到——theta 封頂在 pi，實測 LPIPS 飽和在 0.219。唯一能往上走的是
# 幅度增益（`gain_max = radius x gain_ratio`，不封頂）。故本批掃 radius 與
# gain_ratio 的二維網格。
#
# 判準：至少一個工作點的 fid_lpips >= 0.43（DCT-Shield ε=1 的水準），
# 且該處的 edit_lpips 落在 0.55 以上。掃不到就是構造上的可達性上限。
set -u
source ~/env.sh >/dev/null 2>&1
cd /nfs/home/nelson0314/WACV-s3
export PYTHONPATH=/nfs/home/nelson0314/WACV-s3 TOKENIZERS_PARALLELISM=false
PY=$HOME/venvs/wacv/bin/python
ROOT=runs/ip2p_reach_lpips
mkdir -p $ROOT
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a $ROOT/log.txt; }

IMGS=$(tr '\n' ' ' < runs/ip2p_fair_comparison/images25.txt)
[ -n "$IMGS" ] || { log "影像清單是空的，中止"; exit 1; }
log "強度延伸掃描開始，$(echo $IMGS | wc -w) 張"

# tag        gpu 條件        radius gain_ratio
# gr=0.25 是既有 a5_gain 那批的比例，往上延伸；gr=0.5／1.0 讓同一個 radius
# 換到更多增益。gain_only 那四點回答「純幅度自己能不能走到那裡」。
JOBS="
pg025_r20  0  phase_gain  2.0  0.25
pg025_r25  0  phase_gain  2.5  0.25
pg025_r30  1  phase_gain  3.0  0.25
pg025_r35  1  phase_gain  3.5  0.25
pg050_r15  2  phase_gain  1.5  0.5
pg050_r20  2  phase_gain  2.0  0.5
pg050_r25  3  phase_gain  2.5  0.5
pg050_r30  3  phase_gain  3.0  0.5
pg100_r10  4  phase_gain  1.0  1.0
pg100_r15  4  phase_gain  1.5  1.0
pg100_r20  5  phase_gain  2.0  1.0
pg100_r25  5  phase_gain  2.5  1.0
go100_r16  6  gain_only   1.6  1.0
go100_r20  6  gain_only   2.0  1.0
go100_r24  7  gain_only   2.4  1.0
go100_r30  7  gain_only   3.0  1.0
"
while read -r tag gpu cond rad gr; do
  [ -n "${tag:-}" ] || continue
  CUDA_VISIBLE_DEVICES=$gpu nohup $PY scripts/ip2p_run.py \
    --out "$ROOT/$tag" --data data/omniedit150 --images $IMGS \
    --conditions "$cond" --radius "$rad" --gain-ratio "$gr" \
    > "$ROOT/$tag.log" 2>&1 &
done <<< "$JOBS"
wait
for d in $ROOT/*/; do
  log "$(basename $d)：$(grep -c '^task_' $d/results.csv 2>/dev/null) 列"
done
log "強度延伸掃描完成"

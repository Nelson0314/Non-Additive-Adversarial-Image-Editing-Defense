#!/usr/bin/env bash
# 強度延伸掃描的續段：把 gain_max 推到能碰到 LPIPS 0.43 的量級。basic-2，
# 接在步數對照之後。
#
# 為什麼需要續段
# ────────────────────────────────────────────────────────────────────
# 第一批（`runs/ip2p_reach_lpips`）的網格頂點是 phase_gain radius=2.5
# gain_ratio=1.0，即 `gain_max = 2.5`，只到 LPIPS 0.297。DCT-Shield ε=1 在
# 0.4286，仍然摸不到，故等 LPIPS 錨點依舊算不出來。
#
# 由第一批的斜率取網格：gain_max 由 1.0 走到 2.5 時 LPIPS 由 0.187 走到
# 0.297，約 0.073／單位。要到 0.43 需要 gain_max 約 4.3，故本批掃 3.0–6.6
# 並在兩側各留一點。**這是外插出來的網格，不是外插出來的結論**——結論仍由
# 實測的曲線內插得出。
#
# radius 固定在 3.0（theta_max 封頂在 pi，radius 再高也不會改變相位那一半），
# 只動 gain_ratio。gain_only 那三點回答「純幅度自己能不能走到那裡」。
set -u
source ~/env.sh >/dev/null 2>&1
cd /nfs/home/nelson0314/WACV-s3
export PYTHONPATH=/nfs/home/nelson0314/WACV-s3 TOKENIZERS_PARALLELISM=false
PY=$HOME/venvs/wacv/bin/python
ROOT=runs/ip2p_reach_lpips_ext
mkdir -p $ROOT
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a $ROOT/log.txt; }

# 等步數對照把卡讓出來。中括號寫法使樣式字串本身不匹配 ssh 連線。
while ps -u $USER -o cmd | grep -q "[i]p2p_run.py --out runs/ip2p_pgd_steps"; do
  sleep 30
done

IMGS=$(tr '\n' ' ' < runs/ip2p_fair_comparison/images25.txt)
[ -n "$IMGS" ] || { log "影像清單是空的，中止"; exit 1; }
log "強度延伸續段開始，$(echo $IMGS | wc -w) 張"

# tag        gpu 條件        radius gain_ratio  → gain_max
JOBS="
pgx_g30    0  phase_gain  3.0  1.0
pgx_g42    1  phase_gain  3.0  1.4
pgx_g54    2  phase_gain  3.0  1.8
pgx_g66    3  phase_gain  3.0  2.2
pgx_g78    4  phase_gain  3.0  2.6
gox_g42    5  gain_only   3.0  1.4
gox_g54    6  gain_only   3.0  1.8
gox_g66    7  gain_only   3.0  2.2
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
log "強度延伸續段完成"

#!/usr/bin/env bash
# 抗淨化，比在**擋下率對齊**的工作點上。basic-2 八張卡。
#
# 為什麼要重排這個比較
# ────────────────────────────────────────────────────────────────────
# 既有的抗淨化表（runs/ip2p_fair_comparison/c）比的是**等 DISTS** 上的位移
# 淨增益，而今晚的視覺稽核顯示：在那個失真帶上，本方法的視覺擋下率是 0/13，
# DCT-Shield 是 8/13。兩個都擋不下的東西之間比「淨化後還剩多少位移」，比到的
# 是殘餘擾動的量，不是防禦。
#
# 本批改成先對齊**擋下率**再比抗淨化。三個工作點的 SigLIP 擋下數接近：
#
#   dct_s0100     DISTS 0.0435   8/13
#   add_ln_e04    DISTS 0.0893   8/13
#   pgx_g78       DISTS 0.1896   7/12
#
# 這三個的失真差 4.4 倍，那本身就是結論的一部分（要達到同樣的擋下率，本方法
# 付的失真是 DCT-Shield 的 4.4 倍）。但抗淨化是獨立的一軸：付得多的那個
# 未必守得住，付得少的那個未必守不住，而那正是本專案的主主張要回答的。
#
# 只跑**未防禦編輯確實執行了指令**的 13 張。其餘 12 張沒有攻擊可擋，
# 淨化之後量到的位移是取樣雜訊。
#
# 每一格同時存下「淨化後的防禦圖」與「淨化後的編輯」（`--gallery`）——
# 主主張至今沒有任何影像可看。
set -u
source ~/env.sh >/dev/null 2>&1
cd /nfs/home/nelson0314/WACV-s3
export PYTHONPATH=/nfs/home/nelson0314/WACV-s3 TOKENIZERS_PARALLELISM=false
PY=$HOME/venvs/wacv/bin/python
ROOT=runs/ip2p_retention_matched
mkdir -p $ROOT/gallery
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a $ROOT/log.txt; }

OBEYED=$(awk -F, 'NR>1 && $4=="obeyed" {print $2}' \
         runs/obedience_audit/undefended_obedience.csv | tr '\n' ' ')
[ -n "$OBEYED" ] || { log "服從清單是空的，中止"; exit 1; }
log "抗淨化（擋下率對齊）開始，$(echo $OBEYED | wc -w) 張"

# 四個算子。jpeg_then_resize 是 C&R 串接，本專案針對性最強但從未對現行條件
# 測過；identity 不可省，它是保留率的分母。
OPS="identity blur1 jpeg75 crop_resize0.1 jpeg_then_resize75 adverse_cleaner"

run() {
  local tag=$1 gpu=$2 rundir=$3 cond=$4; shift 4
  CUDA_VISIBLE_DEVICES=$gpu nohup $PY scripts/phase_retention.py \
    --run "$rundir" --data data/omniedit150 --attacker ip2p \
    --conditions "$cond" --images $OBEYED --purifiers $OPS \
    --gallery "$ROOT/gallery" --out "$ROOT/$tag.csv" "$@" \
    > "$ROOT/$tag.log" 2>&1 &
}
# 空白地板：淨化算子自己造成的位移。不可省略，所有讀數都要扣掉它。
# **不可傳 `--conditions`**：`--floor` 產生的列 condition 恆為 `none`
# （地板與條件無關，每張影像只需要一格），傳了會把自己的列全部濾掉。
CUDA_VISIBLE_DEVICES=0 nohup $PY scripts/phase_retention.py \
  --run runs/ip2p_pgd_steps/phase_s0100 --data data/omniedit150 --attacker ip2p \
  --images $OBEYED --purifiers $OPS --floor \
  --gallery "$ROOT/gallery" --out "$ROOT/floor.csv" \
  > "$ROOT/floor.log" 2>&1 &
run dct   1 runs/ip2p_pgd_steps/dct_s0100 dct_shield
run add   2 runs/ip2p_parameterization_control/add_ln_e04 add
run phase 3 runs/ip2p_reach_lpips_ext/pgx_g78 phase_gain
# 定案設定也跑，作為「擋不下的東西抗淨化再好也沒用」的對照
run settled 4 runs/ip2p_pgd_steps/phase_s0100 phase
wait
for f in $ROOT/*.csv; do
  log "$(basename $f)：$(( $(wc -l < $f) - 1 )) 列"
done
log "抗淨化（擋下率對齊）完成"

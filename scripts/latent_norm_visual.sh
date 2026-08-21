#!/usr/bin/env bash
# 把本方法的損失換成 DCT-Shield 用的 ‖E(x')‖₂，並在**視覺擋下率**上驗收。
# basic-1 八張卡。
#
# 為什麼是這個實驗
# ────────────────────────────────────────────────────────────────────
# 25 張的視覺稽核（runs/obedience_audit）量到：未防禦編輯只有 13/25 真的執行
# 了指令；在那 13 張上，本方法的定案設定擋下 **0**，DCT-Shield base 擋下 5、
# Y-only 擋下 4，而且三者的失真都在 DISTS 0.028–0.056 這個窄帶裡。
#
# 逐張看圖時，它擋下的每一格都是同一種失效：**IP2P 整個重畫出另一個場景**
# （太空人變成粉紅團塊上的小卡通、藥水瓶變成一堆橘子、要求移除冰山卻輸出
# 滿是冰山）。那不是溫和劣化，是 latent 被推離自然影像分布之後模型重新幻想。
#
# 兩者的損失剛好就差在這裡：DCT-Shield §4.2 最大化 ‖E(x')‖₂（推遠），本專案
# 走 `encoder_target`＝推向灰圖 latent。`--loss latent_norm` 早已實作，
# RESULTS 也記過它在等 LPIPS 上是 1.149 倍，但**從未用眼睛驗過**——而位移
# 這個讀數今晚已證明不量防禦。
#
# 判準：在 DISTS ≤ 0.06（DCT-Shield base 是 0.0435）的工作點上，視覺擋下率
# 是否從 0/13 起來。位移只作參考，不作判準。
set -u
source ~/env.sh >/dev/null 2>&1
cd /nfs/home/nelson0314/WACV-s3
export PYTHONPATH=/nfs/home/nelson0314/WACV-s3 TOKENIZERS_PARALLELISM=false
PY=$HOME/venvs/wacv/bin/python
ROOT=runs/ip2p_latent_norm_visual
mkdir -p $ROOT
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a $ROOT/log.txt; }

IMGS=$(tr '\n' ' ' < runs/ip2p_fair_comparison/images25.txt)
[ -n "$IMGS" ] || { log "影像清單是空的，中止"; exit 1; }
log "latent_norm 視覺驗收開始，$(echo $IMGS | wc -w) 張"

run() {
  local tag=$1 gpu=$2; shift 2
  CUDA_VISIBLE_DEVICES=$gpu nohup $PY scripts/ip2p_run.py \
    --out "$ROOT/$tag" --data data/omniedit150 --images $IMGS \
    --loss latent_norm "$@" > "$ROOT/$tag.log" 2>&1 &
}
# 純相位，四個強度罩住 DISTS 0.03–0.09
run ln_t10_s100  0 --conditions phase --radius 1.0
run ln_t15_s100  1 --conditions phase --radius 1.5
run ln_t20_s100  2 --conditions phase --radius 2.0
run ln_t26_s100  3 --conditions phase --radius 2.6
# 1000 步：今晚量到步數同時改善位移／LPIPS 與 L/D，但絕對失真下降，
# 故要配更大的 radius 才兌現
run ln_t15_s1000 4 --conditions phase --radius 1.5 --steps 1000
run ln_t26_s1000 5 --conditions phase --radius 2.6 --steps 1000
# 加上幅度增益的兩點，維持定案閘
run ln_g_r15     6 --conditions phase_gain --radius 1.5 --gain-ratio 0.25
run ln_g_r20     7 --conditions phase_gain --radius 2.0 --gain-ratio 0.25
wait
for d in $ROOT/*/; do
  log "$(basename $d)：$(grep -c '^task_' $d/results.csv 2>/dev/null) 列"
done
log "latent_norm 視覺驗收完成"

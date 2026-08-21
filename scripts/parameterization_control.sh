#!/usr/bin/env bash
# 在同一個失真帶上，把「參數化」與「損失」兩個變因分開。basic-2 八張卡。
#
# 今晚的視覺稽核留下一個未解的歸因問題
# ────────────────────────────────────────────────────────────────────
# DCT-Shield 在 DISTS 0.0435 上擋下 5/13，本方法在 DISTS 0.0555 上擋下 0/13。
# 兩者同時差在**兩件事**：參數化（DCT 係數加性 vs 加窗區塊相位旋轉）與損失
# （‖E(x')‖₂ vs 推向灰圖 latent）。單看那組數字無法歸因。
#
# 本批把兩個變因交叉：`add`（像素加性 δ，本專案既有的內部對照組）配兩種損失，
# 對上相位配兩種損失。若加性＋latent_norm 就能擋下，問題在參數化；若兩種
# 參數化在同一個損失下都擋得下，問題在損失。
#
# 加性那一臂用 `--conditions add`，半徑是 L∞ 上界，掃四檔罩住 DISTS 0.03–0.08。
#
# 另有一個獨立的量測：本方法在等 LPIPS 下的殘差 RMS 是 DCT-Shield 的 3.1 倍
# （0.0972 對 0.0311，見 runs/obedience_audit/residual_periodicity.csv）。
# 加性那一臂會直接告訴我們，同樣的 RMS 預算交給最簡單的參數化能換到什麼。
set -u
source ~/env.sh >/dev/null 2>&1
cd /nfs/home/nelson0314/WACV-s3
export PYTHONPATH=/nfs/home/nelson0314/WACV-s3 TOKENIZERS_PARALLELISM=false
PY=$HOME/venvs/wacv/bin/python
ROOT=runs/ip2p_parameterization_control
mkdir -p $ROOT
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a $ROOT/log.txt; }

IMGS=$(tr '\n' ' ' < runs/ip2p_fair_comparison/images25.txt)
[ -n "$IMGS" ] || { log "影像清單是空的，中止"; exit 1; }
log "參數化×損失 交叉開始，$(echo $IMGS | wc -w) 張"

run() {
  local tag=$1 gpu=$2; shift 2
  CUDA_VISIBLE_DEVICES=$gpu nohup $PY scripts/ip2p_run.py \
    --out "$ROOT/$tag" --data data/omniedit150 --images $IMGS "$@" \
    > "$ROOT/$tag.log" 2>&1 &
}
# 加性 × latent_norm（＝把 DCT-Shield 的損失搬到最簡單的參數化上）
run add_ln_e02 0 --conditions add --radius 0.02 --loss latent_norm
run add_ln_e04 1 --conditions add --radius 0.04 --loss latent_norm
run add_ln_e06 2 --conditions add --radius 0.06 --loss latent_norm
run add_ln_e10 3 --conditions add --radius 0.10 --loss latent_norm
# 加性 × encoder_target（＝把本專案的損失搬到最簡單的參數化上）
run add_et_e02 4 --conditions add --radius 0.02
run add_et_e04 5 --conditions add --radius 0.04
run add_et_e06 6 --conditions add --radius 0.06
run add_et_e10 7 --conditions add --radius 0.10
wait
for d in $ROOT/*/; do
  log "$(basename $d)：$(grep -c '^task_' $d/results.csv 2>/dev/null) 列"
done
log "參數化×損失 交叉完成"

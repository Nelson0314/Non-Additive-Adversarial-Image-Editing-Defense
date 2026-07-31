cd /work/nelson0314/WACV
source env.sh
set -x
python scripts/run_defense.py --sites PF,P --ranks 16 --limit 6 --steps 25 --lr 0.008 --tau_lpips 0.02 --eval_strengths 0.3,0.5,0.7 --out runs/e8_rank_tau0.02
python scripts/run_defense.py --sites PF,P --ranks 16 --limit 6 --steps 25 --lr 0.008 --tau_lpips 0.05 --eval_strengths 0.3,0.5,0.7 --out runs/e8_rank_tau0.05
python scripts/run_defense.py --sites PF,P --ranks 16 --limit 6 --steps 25 --lr 0.008 --tau_lpips 0.10 --eval_strengths 0.3,0.5,0.7 --out runs/e8_rank_tau0.10
echo EXPA_DONE

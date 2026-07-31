cd /work/nelson0314/WACV
source env.sh
set -x
python scripts/run_defense.py --sites W --ranks 16 --limit 1 --k_inv 20 --t_max 500 --align_steps 30 --align_lr 0.008 --steps 1 --no_eval --out runs/e11_wlr_0.008
python scripts/run_defense.py --sites W --ranks 16 --limit 1 --k_inv 20 --t_max 500 --align_steps 30 --align_lr 0.001 --steps 1 --no_eval --out runs/e11_wlr_0.001
echo STAGE0_DONE

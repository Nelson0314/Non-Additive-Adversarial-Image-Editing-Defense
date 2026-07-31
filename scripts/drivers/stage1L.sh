cd /work/nelson0314/WACV
source env.sh
set -x
python scripts/run_defense.py --sites L --ranks 16 --limit 2 --k_inv 20 --t_max 500 --align_steps 200 --align_lr 0.008 --steps 1 --no_eval --out runs/e12_align_L_fixed
echo STAGE1_DONE

cd /work/nelson0314/WACV
source env.sh
set -x
python scripts/run_defense.py --sites W --ranks 4 --limit 2 --k_inv 20 --t_max 500 --align_steps 200 --align_lr 0.001 --steps 1 --no_eval --out runs/e12_align_W_r4
python scripts/run_defense.py --sites W --ranks 16 --limit 2 --k_inv 20 --t_max 500 --align_steps 200 --align_lr 0.001 --steps 1 --no_eval --out runs/e12_align_W_r16
echo STAGE2_DONE

cd /work/nelson0314/WACV
source env.sh
set -x
python scripts/run_defense.py --sites LA --ranks 4 --k_inv 20 --t_max 500 --scales 10 --size 512 --lr 0.008 --n_edit 10 --strength 0.5 --seed 20260728 --limit 6 --steps 20 --out runs/e7_stepsLA_20
python scripts/run_defense.py --sites LA --ranks 4 --k_inv 20 --t_max 500 --scales 10 --size 512 --lr 0.008 --n_edit 10 --strength 0.5 --seed 20260728 --limit 6 --steps 100 --out runs/e7_stepsLA_100
echo STAGE2_DONE

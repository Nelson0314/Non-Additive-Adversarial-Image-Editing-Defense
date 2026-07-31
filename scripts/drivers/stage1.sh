cd /work/nelson0314/WACV
source env.sh
set -x
python scripts/run_defense.py --sites P --ranks 16 --size 512 --lr 0.008 --n_edit 10 --strength 0.5 --seed 20260728 --limit 6 --steps 20 --out runs/e7_stepsP_20
python scripts/run_defense.py --sites P --ranks 16 --size 512 --lr 0.008 --n_edit 10 --strength 0.5 --seed 20260728 --limit 6 --steps 100 --out runs/e7_stepsP_100
echo STAGE1_DONE

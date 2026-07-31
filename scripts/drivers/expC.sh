cd /work/nelson0314/WACV
source env.sh
set -x
python scripts/run_defense.py --sites P --ranks 16 --limit 6 --steps 25 --lr 0.008 --purify_mode all --eval_strengths 0.3,0.5,0.7 --out runs/e10_eot_all
echo EXPC_DONE

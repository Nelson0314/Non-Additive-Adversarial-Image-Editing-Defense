#!/bin/bash
# 本機煙霧測試：用 tiny-SD 把「跑一格 → 綁定者診斷」整條鏈走一遍。
#
# 用途不是產生結果，而是在花掉雲端機時之前確認 `e29_calibration.sh` /
# `e30_grid.sh` 的參數組合不會在管線上出錯——參數名拼錯、某道 hinge 在某個
# site 上崩潰、env.json 缺欄位導致診斷讀不到，這些都要在本機 30 秒內就知道。
#
# 模型是 `hf-internal-testing/tiny-stable-diffusion-pipe`，隨機初始化、64²、
# 8 步。**其上的收斂行為不能代表真實 SD**（與 `tests/test_pipeline.py` 的
# 同一項限制）。可以拿來看的只有「哪一道約束會啟動」這種結構性質。
#
# 輸出寫到 $OUT（預設系統暫存目錄），不進 `runs/`：那裡只放證據級的資料。
set -euo pipefail

PY="${PY:-C:/Users/nelso/miniconda3/envs/wacv/python.exe}"
OUT="${OUT:-${TMPDIR:-/tmp}/wacv_smoke}"
cd "$(dirname "$0")/../.."
export PYTHONIOENCODING=utf-8

COMMON="--model hf-internal-testing/tiny-stable-diffusion-pipe \
  --size 64 --steps 8 --k_inv 2 --n_edit 2 --limit 1 --no_eval \
  --guidance_scale 7.5 --beta_linf 0 --tau_lpips 0.05 --margin 1.0 --alpha_lpips 0"

"$PY" scripts/run_defense.py --sites C --ranks 32 --lr 0.3 $COMMON \
  --color_max_dev 2.0 --out "$OUT/smoke_C"
"$PY" scripts/run_defense.py --sites P --ranks 16 --lr 0.03 $COMMON \
  --out "$OUT/smoke_P"

"$PY" scripts/e27_binding_check.py "$OUT/smoke_C" "$OUT/smoke_P"

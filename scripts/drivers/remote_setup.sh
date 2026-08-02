#!/bin/bash
# 新雲端機器的環境準備。跑完 E29／E30 之前先跑這一支。
#
# 各步驟的來源：
#   - 套件清單取自 `environment.yml` 的 pip 區塊（不釘版本，程式碼不綁定
#     特定 torch/CUDA）。
#   - `PY` 用絕對路徑：Lightning AI 的背景腳本不是 login shell，`python3`
#     會取到系統直譯器而缺 numpy（E27 踩過）。
#   - `HF_HOME` 指到持久儲存：容器會被刪除，權重每次重下要 4 GB。
#   - 預先把 SD v1.4 抓下來，而不是等第一格開跑才下載——下載失敗要在
#     幾分鐘內知道，不是在兩小時的網格中途。
#
# 這一支尚未在本輪的遠端機器上跑過（2026-08-02 本機只能做到語法檢查）。
# 連上遠端後它是第一件要執行並看到成功輸出的事。
#
# 用法：
#   bash scripts/drivers/remote_setup.sh
#   PY=/path/to/python HF_HOME=/path/to/cache bash scripts/drivers/remote_setup.sh
set -euo pipefail

PY="${PY:-/home/zeus/miniconda3/envs/cloudspace/bin/python3}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
cd "$(dirname "$0")/../.."

echo "=== 直譯器 ==="
"$PY" -V
echo "PY=$PY"
echo "HF_HOME=$HF_HOME"

echo "=== 套件 ==="
"$PY" -m pip install --quiet --upgrade pip
"$PY" -m pip install --quiet torch torchvision diffusers transformers accelerate \
  piq opencv-contrib-python peft pytest pyyaml psutil matplotlib

echo "=== GPU ==="
"$PY" - <<'PYEOF'
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(f"{p.name}  {p.total_memory/2**30:.0f} GiB  sm_{p.major}{p.minor}")
# TF32 的狀態要在匯入 src.utils.device 之後才問。該模組在模組層級把兩個
# allow_tf32 設成 False（E28 §4：開著會讓 BDIA 精確反演由「好 5 個數量級」
# 退化成「好 37 倍」），只 import torch 讀到的是 PyTorch 自己的預設值
# `cudnn.allow_tf32 = True`——與實際跑實驗時的設定相反，看了會誤判。
from src.utils.device import tf32_enabled
print("TF32", "開啟（WACV_ALLOW_TF32=1）" if tf32_enabled() else "關閉",
      f"  matmul={torch.backends.cuda.matmul.allow_tf32}"
      f" cudnn={torch.backends.cudnn.allow_tf32}")
PYEOF

echo "=== 測試（基準 253 passed / 1 skipped）==="
"$PY" -m pytest -q

echo "=== 預抓 SD v1.4 權重 ==="
"$PY" - <<'PYEOF'
from diffusers import StableDiffusionPipeline
StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4",
                                        safety_checker=None)
print("SD v1.4 就緒")
PYEOF

echo "=== 準備完成。下一步：bash scripts/drivers/e29_calibration.sh ==="

#!/bin/bash
# Google Colab 的環境準備。跑 E29／E30 之前先跑這一支。
#
# 與 `remote_setup.sh` 的差別只有一項，但那一項是關鍵：
#
#   **不安裝 torch 與 torchvision。** Colab 的映像檔已裝好與該 runtime 驅動
#   相符的 torch；`pip install torch` 會從 PyPI 拉一個預設 CUDA 版本覆蓋掉它，
#   結果是 `torch.cuda.is_available()` 變成 False 或在第一次 kernel 呼叫時
#   當掉。這不是理論風險：diffusers、peft、piq 都把 torch 列為相依，pip 在
#   解相依時可能自行升級它。防法是把現有版本寫成 constraint 檔傳給 pip，
#   讓 pip 無法動它；裝完再核對一次版本與 CUDA，不符就中止而不是繼續往下跑。
#
# 其餘與 remote_setup.sh 相同：HF_HOME 指到可持久的位置、預先抓 SD v1.4
# （下載失敗要在幾分鐘內知道，不是在網格中途）、跑一次測試。
#
# 用法（在 repo 根目錄）：
#   bash scripts/drivers/colab_setup.sh
#   HF_HOME=/content/drive/MyDrive/hf_cache bash scripts/drivers/colab_setup.sh
set -euo pipefail

PY="${PY:-python3}"
# Colab 的 /content 在 runtime 結束時清空。權重 4 GB，同一天內重連多次的話
# 指到掛載的雲端硬碟可省下重抓；沒掛載時退回 /content。
export HF_HOME="${HF_HOME:-/content/hf_cache}"
cd "$(dirname "$0")/../.."
mkdir -p runs/logs

echo "=== 直譯器 ==="
"$PY" -V
echo "PY=$PY"
echo "HF_HOME=$HF_HOME"

echo "=== 記錄現有的 torch 版本 ==="
TORCH_BEFORE="$("$PY" -c 'import torch; print(torch.__version__)')"
TV_BEFORE="$("$PY" -c 'import torchvision; print(torchvision.__version__)')"
echo "torch $TORCH_BEFORE   torchvision $TV_BEFORE"

CONSTRAINTS="$(mktemp)"
printf 'torch==%s\ntorchvision==%s\n' "$TORCH_BEFORE" "$TV_BEFORE" > "$CONSTRAINTS"

echo "=== 套件（torch／torchvision 以 constraint 釘住）==="
# 清單取自 environment.yml 的 pip 區塊，扣掉 Colab 已內建的
# （numpy、matplotlib、pyyaml、psutil、pytest、transformers、accelerate）
# 與不可覆蓋的 torch／torchvision。仍列出 transformers 與 accelerate：
# 版本太舊時 diffusers 會在 import 期就失敗，交給 pip 判斷是否需要動。
"$PY" -m pip install --quiet -c "$CONSTRAINTS" \
  diffusers transformers accelerate \
  piq opencv-contrib-python peft pytest pyyaml psutil scipy

# pyiqa 單獨以 --no-deps 安裝。它提供 NIQE，而 MetricSuite.full() 對每張圖
# 各算一次 NIQE，所以這是 run_defense.py 評測路徑的執行期相依——E29 帶
# --no_eval 碰不到，E30 會在第一格評測時才當掉。
#
# 為什麼 --no-deps：pyiqa 宣告的相依有三十餘項，含 torch、torchvision、
# datasets、bitsandbytes、facexlib、tensorboard、pre-commit、ruff。上面的
# constraint 檔擋得住 torch 被換版，擋不住其餘二十幾個無關套件被裝進來，
# 而它們有各自的版本要求，可能連帶動到 Colab 既有的 numpy 或 transformers。
# 實測本專案用到的指標（niqe / gmsd / nlpd / lpips / dists / stlpips /
# ms_ssim / ssim / musiq）在執行期只載入 numpy、scipy、opencv、pillow、
# PyYAML、requests、tqdm、rich、huggingface_hub 與 torch —— 全部已由上一行
# 或 Colab 映像檔提供。haarpsi 與 vif_p 出自 piq 而非 pyiqa。
"$PY" -m pip install --quiet --no-deps pyiqa

echo "=== 核對 pyiqa 在 --no-deps 下匯入無誤 ==="
"$PY" - <<'PYEOF'
import torch, pyiqa
# 只建 NIQE：它是評測路徑真正會用到的那一個，且參數內建、不需下載權重。
# 匯入成功但建構失敗的話，缺的是相依而非套件本身，要在這裡就知道。
m = pyiqa.create_metric("niqe", device="cpu")
print("pyiqa", pyiqa.__version__, "NIQE =", float(m(torch.rand(1, 3, 256, 256))))
PYEOF

echo "=== 核對 torch 未被換掉 ==="
"$PY" - <<PYEOF
import sys, torch, torchvision
before, tv_before = "$TORCH_BEFORE", "$TV_BEFORE"
if torch.__version__ != before or torchvision.__version__ != tv_before:
    sys.exit(f"torch 被換掉了：{before} -> {torch.__version__}，"
             f"torchvision {tv_before} -> {torchvision.__version__}。"
             f"constraint 沒擋住，查是哪個套件強制升級的，不要略過。")
if not torch.cuda.is_available():
    sys.exit("torch.cuda.is_available() 是 False。runtime 沒配到 GPU，"
             "或安裝過程換掉了 CUDA 版本。")
print("torch", torch.__version__, "未變動，CUDA 可用")
PYEOF

echo "=== GPU ==="
"$PY" - <<'PYEOF'
import torch
p = torch.cuda.get_device_properties(0)
print(f"{p.name}  {p.total_memory / 2**30:.0f} GiB  sm_{p.major}{p.minor}")
# E27 在 H100 上量到的峰值是 10.3 GB（runs/e27d_C_lr0.3/summary.csv 的
# peak_mb）。低於此值的顯示記憶體放不下 512² 的訓練。
need = 10320.2 / 1024
have = p.total_memory / 2**30
print(f"E27 實測峰值 {need:.1f} GiB；本機 {have:.1f} GiB "
      f"({'足夠' if have > need * 1.15 else '餘裕不足，先跑 colab_probe.py 確認'})")
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

echo "=== 準備完成。下一步：python scripts/colab_probe.py ==="
echo "    Colab 配到的 GPU 每次不同，時間必須實測；探測完再決定要不要開 E30。"

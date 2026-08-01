"""裝置選擇與數值精度。禁用硬編碼 .cuda()，一律經此處取得裝置。

TF32 預設關閉（2026-08-01，E28）。PyTorch 的 `cudnn.allow_tf32` 預設為
`True`，Ampere 以上的卷積因此以約 10 bit 尾數執行。本專案不能接受這個預設，
理由有二：

1. 同一份程式在不同機器上會給出不同精度的數字。V100（Volta）沒有 TF32，
   E15–E23 全部是真 fp32；H100 與 RTX 2050（Ampere 以上）預設會走 TF32。
   跨機器比較的前提是數值定義相同。
2. BDIA 精確反演的整個存在理由就是數值精確性。實測（tiny-SD，本機
   RTX 2050）：TF32 開啟時 BDIA 的來回誤差只比 DDIM 好 37 倍，關閉後回到
   文件記載的 5 個數量級。`tests/test_pipeline.py::test_BDIA來回誤差遠小於DDIM`
   在 TF32 下失敗，正是被這個預設值弄壞的。

已知的既有資料受此影響的範圍：E27 在 H100 上的校準（`runs/e27*`）是在
TF32 開啟（當時的預設）下跑的。該批資料量的是 LPIPS、色度偏壓與編輯偏移，
對 1e-3 量級的相對誤差不敏感，且未使用 `--exact_inversion`（site C 與 site P
都不走反演），故結論不受影響。但這件事必須記著：若日後在 Ampere 以上重跑
site L/E/W 並開啟精確反演，而沒有關掉 TF32，反演會退化。

`WACV_ALLOW_TF32=1` 可換回 TF32 以換取速度（卷積約快 1.5–2×）。那是明確的
取捨，不是預設——需要它的場合應該在報告中寫明。
"""

import os

import torch

# 匯入本模組即生效。放在模組層級而非 get_device() 內，是因為有些呼叫端
# （例如評測腳本）直接用 torch 而不經過 get_device()，若靠函式呼叫來設定，
# 就會出現「跑了哪條路徑決定用什麼精度」的隱性差異。
_ALLOW_TF32 = os.environ.get("WACV_ALLOW_TF32", "0") == "1"
torch.backends.cudnn.allow_tf32 = _ALLOW_TF32
torch.backends.cuda.matmul.allow_tf32 = _ALLOW_TF32


def tf32_enabled() -> bool:
    """目前是否允許 TF32。供 `env.json` 記錄，使每批資料的精度有據可查。"""
    return bool(torch.backends.cudnn.allow_tf32)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def peak_memory_mb() -> float:
    """回傳目前裝置的 peak GPU 記憶體（MB）。CPU 上回傳 0。"""
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / (1024**2)


def reset_peak_memory() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

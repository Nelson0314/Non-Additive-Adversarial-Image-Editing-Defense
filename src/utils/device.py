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


def bf16_supported(device=None) -> bool:
    """目前裝置是否支援 bf16。

    V100（sm_70）沒有 bf16 硬體支援，`torch.cuda.is_bf16_supported()` 回傳
    False；RTX 5090（sm_120）回傳 True。CPU 上 torch 可以模擬 bf16 運算，
    故回傳 True 使結構性測試能在無 GPU 環境執行。
    """
    if device is not None and torch.device(device).type == "cpu":
        return True
    if not torch.cuda.is_available():
        return True
    return bool(torch.cuda.is_bf16_supported())


def resolve_precision(compute_dtype: torch.dtype) -> "tuple[torch.dtype, torch.dtype]":
    """由指定的計算精度推出 (骨幹 dtype, VAE dtype)。

    回傳的第一項給 UNet 與兩個 text encoder，第二項給 VAE。**這是一條規則，
    不是註解**：呼叫端不得自行決定 VAE 的 dtype，`SDWrapper` 一律經此函式取得。

    | compute_dtype | 骨幹 | VAE | 理由 |
    |---|---|---|---|
    | fp32 | fp32 | fp32 | 基準，E15–E23 的既有數字全部是這一格 |
    | fp16 | fp16 | **fp32** | SDXL 的 VAE 在 fp16 下的中間激活會超出 fp16 的最大值 65504 而變成 inf，解碼結果是全黑圖 |
    | bf16 | bf16 | bf16 | bf16 的指數位寬與 fp32 相同（8 bit），動態範圍一致，不會溢位 |

    fp16 那一格的處置只有兩種：把 VAE 留在 fp32，或換一份重新縮放過的
    VAE 權重（社群的 `sdxl-vae-fp16-fix`）。**本專案只能選前者**——威脅模型
    要求攻擊方使用 stock SDXL，換權重就是換模型（`docs/ARCH_2026-08-05.md`
    §7.1 已撤回該項）。故本函式只回傳 dtype，永遠不回傳權重來源，程式中
    也不存在任何載入替代 VAE 的路徑。

    未列入表中的 dtype 一律拋出。靜默落回 fp32 會讓「這批資料是什麼精度」
    無從查證，而精度正是跨卡比較時唯一的差異來源。
    """
    if compute_dtype == torch.float32:
        return torch.float32, torch.float32
    if compute_dtype == torch.float16:
        return torch.float16, torch.float32
    if compute_dtype == torch.bfloat16:
        return torch.bfloat16, torch.bfloat16
    raise ValueError(
        f"不支援的計算精度 {compute_dtype}。可用的只有 float32、float16、"
        "bfloat16 三種；靜默落回預設會讓每批資料的精度無從查證"
    )


def peak_memory_mb() -> float:
    """回傳目前裝置的 peak GPU 記憶體（MB）。CPU 上回傳 0。"""
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / (1024**2)


def reset_peak_memory() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

"""裝置選擇與數值精度。禁用硬編碼 .cuda()，一律經此處取得裝置。"""

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

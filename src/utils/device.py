"""裝置選擇。禁用硬編碼 .cuda()，一律經此處取得裝置。"""

import torch


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

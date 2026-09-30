"""RGB 張量的 PNG 存檔。

輸入為 (N,3,H,W) 或 (3,H,W)、[0,1]；四維輸入保留既有取首張的行為。
轉 fp32、clamp、round 至 uint8，建立父目錄並覆寫指定影像。
不包含歷史實驗專用的繪圖與 trace 輸出。
"""

from pathlib import Path

import torch


def _to_uint8(x: torch.Tensor):
    """張量轉存檔用的 uint8。**先轉 fp32**，因為 numpy 沒有 bfloat16。

    2026-08-06 修正。before：`a.permute(1, 2, 0).numpy()`，即直接對計算精度的
    張量呼叫 `.numpy()`。以 bf16 執行時 `torch.Tensor.numpy()` 以
    `TypeError: Got unsupported ScalarType BFloat16` 中止。

    症狀為何拖到現在才出現：走這條路徑的張量多半已在別處轉過 fp32
    （評測的 `pairwise` 需要，指標套件不做隱式轉型），只有
    `_train_nonadditive` 的 `save_image(result.x_base, …)` 是直接把
    `gen.generate` 的輸出交過來。那一行只有 N3（唯一有階段一的條件）會走到，
    而段 1 是本專案第一次在 GPU 上跑 N3。整格因此在**訓練已經跑完之後**
    才以 `failed` 收場，逐格紀錄裡的 `stage_reports` 全部遺失。

    轉型放在這裡而不是呼叫端：存檔的輸入沒有理由要求特定計算精度，而
    `save_image`／`save_residual` 兩個入口都經過此處。
    """
    import numpy as np

    a = x.detach().float().clamp(0, 1).cpu()
    if a.dim() == 4:
        a = a[0]
    return (a.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)


def save_image(x: torch.Tensor, path: Path) -> None:
    """建立父目錄並存入量化 RGB 影像；四維輸入取首張，目的檔案可被覆寫。"""
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(_to_uint8(x)).save(path)

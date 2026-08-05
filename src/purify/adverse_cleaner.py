"""Adverse Cleaner 淨化算子。

出處與參數查證見 `docs/_audit_purify.md` §1。摘要：

- DIA（ICCV 2025）參考文獻 [33] 指向 `lllyasviel/AdverseCleaner`，該 repo 已 404。
  演算法由兩個獨立鏡像（`shidoto/AdverseCleaner` 的 `clean.py`、
  `gogodr/AdverseCleanerExtension` 的 UI 預設值）交叉驗證，兩者七個參數一致。
- DIA 補充材料 §B.2 對此算子未指定任何參數，故採上游預設值。

原始 16 行程式（`clean.py`，未經改寫）：

    img = cv2.imread('input.png').astype(np.float32)
    y = img.copy()
    for _ in range(64):
        y = cv2.bilateralFilter(y, 5, 8, 8)
    for _ in range(4):
        y = guidedFilter(img, y, 4, 16)
    cv2.imwrite('output.png', y.clip(0, 255).astype(np.uint8))

本模組逐行對應上式，未加入任何額外處理。三項因介面轉換而必須明寫的細節：

1. **值域**。上游在 `[0, 255]` 上運作，`sigmaColor=8` 是該尺度上的量。本專案的
   張量在 `[0, 1]`，故先乘 255 再濾波，最後除回，**不改動 sigma**。
2. **通道順序**。`cv2.imread` 給的是 BGR，故先 RGB→BGR、算完再轉回。
   （bilateral 與 guided filter 對通道置換等變，此步不影響數值，但保持與原碼同形。）
3. **uint8 量化**。上游輸入來自 `cv2.imread`（uint8 PNG），輸出經
   `.astype(np.uint8)`。故輸入先四捨五入到 uint8 網格（對應影像以 PNG 存檔的行為），
   輸出沿用原碼的 `.astype`（**截斷**而非四捨五入）。

不可微：全部運算在 OpenCV C++ 內完成，無 autograd 圖。訓練時由
`src/purify/ops.py` 的 `straight_through` 包裝，不在本模組另寫近似版本。
"""

from __future__ import annotations

import torch

# 以下七個常數全部來自 `clean.py`，不得調整（見 `_audit_purify.md` §1.3 表）
BILATERAL_STEPS = 64
BILATERAL_D = 5
BILATERAL_SIGMA_COLOR = 8
BILATERAL_SIGMA_SPACE = 8
GUIDED_STEPS = 4
GUIDED_RADIUS = 4
GUIDED_EPS = 16


def _load_guided_filter():
    """取得 `cv2.ximgproc.guidedFilter`，缺套件時拋出並寫明缺什麼。

    `guidedFilter` 只存在於 opencv-contrib（原 repo 的 `environment.yaml` 即列
    `opencv-contrib-python`）。此處**不提供自寫的近似 guided filter**：那會變成
    我方重新設計別人的方法。
    """
    try:
        from cv2.ximgproc import guidedFilter
    except (ImportError, AttributeError) as e:
        raise NotImplementedError(
            "Adverse Cleaner 需要 cv2.ximgproc.guidedFilter，該模組屬 "
            "opencv-contrib-python（原 repo environment.yaml 所列）。"
            "目前環境的 cv2 沒有 ximgproc。請安裝 `opencv-contrib-python` 後重試；"
            "不得以自寫的近似 guided filter 代替。"
        ) from e
    return guidedFilter


def has_guided_filter() -> bool:
    """環境是否具備執行本算子的條件。供 `Purifier.available` 查詢。"""
    try:
        _load_guided_filter()
    except NotImplementedError:
        return False
    return True


def adverse_cleaner_real(x01: torch.Tensor) -> torch.Tensor:
    """真實 Adverse Cleaner。輸入輸出皆為 `(B, 3, H, W)`、RGB、`[0, 1]`。

    不可微（OpenCV 實作）。
    """
    import numpy as np
    import cv2

    guided_filter = _load_guided_filter()
    if x01.dim() != 4 or x01.shape[1] != 3:
        raise ValueError(f"Adverse Cleaner 需要 (B,3,H,W) 的 RGB 張量，收到 {tuple(x01.shape)}")

    out = []
    for i in range(x01.shape[0]):
        rgb = x01[i].detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy()
        # 對應 cv2.imread：uint8 網格上的 [0,255] float32，通道為 BGR
        img = np.round(rgb * 255.0).astype(np.float32)[:, :, ::-1].copy()
        y = img.copy()
        for _ in range(BILATERAL_STEPS):
            y = cv2.bilateralFilter(y, BILATERAL_D, BILATERAL_SIGMA_COLOR, BILATERAL_SIGMA_SPACE)
        for _ in range(GUIDED_STEPS):
            # guide 固定為原圖 img，非上一次迭代的結果
            y = guided_filter(img, y, GUIDED_RADIUS, GUIDED_EPS)
        # clean.py 最後一行：clip 後 astype(uint8)，即截斷
        u8 = y.clip(0, 255).astype(np.uint8)
        dec = u8[:, :, ::-1].astype(np.float32) / 255.0  # BGR → RGB
        out.append(torch.from_numpy(np.ascontiguousarray(dec)).permute(2, 0, 1))
    return torch.stack(out).to(x01.device, x01.dtype)

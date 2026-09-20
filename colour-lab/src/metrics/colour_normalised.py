"""把「防禦的色彩映射原封不動穿過編輯」那一段從位移裡扣掉。

為什麼要有這一欄
────────────────────────────────────────────────────────────────────
這條線的載體是**純顏色全域映射**，所以防禦圖與原圖的差有一部分會原封不動
穿過編輯，落在兩張編輯結果之間，被 LPIPS 算進位移。那一部分，攻擊方只要
對輸出做一次全域色彩校正就還得回去。

本模組先對兩張編輯結果做**逐通道 256 格 CDF 直方圖匹配**——匹配本身也是一個
單調的全域顏色映射，是攻擊方做得到的最強全域色彩校正——再量一次 LPIPS。
匹配之後還留著的位移，是色彩校正還不回去的那一部分。

方向是對稱的：`defended → undefended` 與 `undefended → defended` 各量一次，
兩個數都回報，不取單一方向，免得匹配方向本身變成一個藏起來的選擇。

**這不是判準。** 原位移與匹配後的位移兩個數都照報，哪個算數由使用者決定。
"""

from __future__ import annotations

from typing import Dict

import torch

BINS = 256


def _cdf(values: torch.Tensor) -> torch.Tensor:
    """(P,) 值域 [0,1] 的一個通道 → (BINS,) 累積分布，最後一格為 1。"""
    counts = torch.histc(values, bins=BINS, min=0.0, max=1.0)
    total = counts.sum()
    if float(total) <= 0:
        raise ValueError("通道是空的，算不出 CDF")
    return counts.cumsum(0) / total


def match_histogram(source: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """把 `source` 的逐通道直方圖匹配到 `reference`。

    兩者都是 (1,3,H,W)、值域 [0,1]。回傳與 `source` 同形狀的張量。

    每個通道求一張 256 格的查表：`LUT[k] = ref 的分位數 at src_cdf[k]`。
    查表是單調不減的（兩個 CDF 都單調），所以這一步本身是一個合法的全域
    色彩映射，不會憑空造出結構。
    """
    if source.shape != reference.shape:
        raise ValueError(f"形狀不符：{tuple(source.shape)} 對 {tuple(reference.shape)}")
    if source.dim() != 4 or source.shape[1] != 3:
        raise ValueError(f"要 (1,3,H,W)，收到 {tuple(source.shape)}")
    out = torch.empty_like(source)
    centres = (torch.arange(BINS, device=source.device, dtype=source.dtype) + 0.5) / BINS
    for c in range(3):
        src = source[:, c].reshape(-1).clamp(0.0, 1.0)
        ref = reference[:, c].reshape(-1).clamp(0.0, 1.0)
        src_cdf = _cdf(src)
        ref_cdf = _cdf(ref)
        # 對每一個來源格的 CDF 值，找參考 CDF 第一個 ≥ 它的格。
        idx = torch.searchsorted(ref_cdf.contiguous(), src_cdf.contiguous())
        lut = centres[idx.clamp(max=BINS - 1)]
        bins = (src * BINS).floor().clamp(0, BINS - 1).long()
        out[:, c] = lut[bins].reshape(source[:, c].shape)
    return out


def colour_normalised_pair(lpips_module, a: torch.Tensor, b: torch.Tensor
                           ) -> Dict[str, float]:
    """`a` = 編輯(原圖)、`b` = 編輯(防禦圖)。

    回傳四個數：原位移、兩個方向各自匹配後的位移、以及兩個方向的平均。
    """
    with torch.no_grad():
        raw = float(lpips_module(a, b))
        b_to_a = float(lpips_module(a, match_histogram(b, a)))
        a_to_b = float(lpips_module(match_histogram(a, b), b))
    return {
        "disp_lpips_raw": raw,
        "disp_lpips_matched_defended_to_plain": b_to_a,
        "disp_lpips_matched_plain_to_defended": a_to_b,
        "disp_lpips_matched_mean": 0.5 * (b_to_a + a_to_b),
    }

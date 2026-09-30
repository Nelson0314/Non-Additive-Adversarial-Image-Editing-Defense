"""把位移拆成「主體內」與「主體外」兩塊的 LPIPS。

用途
────────────────────────────────────────────────────────────────────
現行的位移讀數是**全圖** `LPIPS(編輯(原圖), 編輯(防禦圖))`。對色彩族與補丁族
這個讀數會被灌水：防禦端改掉的顏色（或貼上的補丁）會原封不動穿過編輯，
被 LPIPS 算成「編輯被推開了」，而受保護的主體可能完全沒有被保住。
原 `runs/ip2p_color_masked/`（不在 repo 內）的遮罩 r=0.20 就是這個疑慮的實例——它的
`image_guidance` 固定評估只由 0.158 降到 0.064（機制幾乎沒被碰到），
全圖位移卻有 0.415。

分區之後可以直接問：**主體那一塊的編輯有沒有被改變**。

構造：`piq.LPIPS` 的空間加權平均
────────────────────────────────────────────────────────────────────
`piq/perceptual.py:183`（0.8.0，已逐行核對）的 `ContentLoss.forward` 是

    loss = cat([(d_l * w_l).mean(dim=[2, 3]) for l], dim=1).sum(dim=1)

`d_l` 是第 l 層逐通道的平方差圖 (N, C_l, H_l, W_l)，`w_l` 是 LPIPS 學到的
逐通道線性權重。本模組**只把 `mean(dim=[2,3])` 換成以遮罩為權重的平均**：

    masked_l = sum_hw( m_l * d_l * w_l ) / sum_hw( m_l )

`m_l` 由輸入解析度的遮罩以 `adaptive_avg_pool2d` 降到 (H_l, W_l)，因此格子
的值是該格落在區域內的**面積比例**，不是硬性的 0/1。

遮罩全為 1 時逐位元退回 `piq.LPIPS()`（`tests/test_regional_lpips.py` 釘住）。
這一點是本模組唯一的正確性錨——分區讀數與全圖讀數必須是同一個度量，
否則兩者不可並列。

**不要另外 `piq.LPIPS()` 一份**：呼叫端應把 `MetricSuite.lpips_module`
傳進來，量測與分區用的是同一份權重、同一份顯存。
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn.functional as F


class RegionalLPIPS:
    """以遮罩為空間權重的 LPIPS。

    `lpips_module` 必須是 `piq.LPIPS` 的實例（取自
    `MetricSuite.lpips_module`）。本類別不建立自己的權重。
    """

    def __init__(self, lpips_module):
        import piq

        if not isinstance(lpips_module, piq.LPIPS):
            raise TypeError(
                "RegionalLPIPS 需要 piq.LPIPS 實例（請傳 "
                f"MetricSuite.lpips_module），收到 {type(lpips_module)!r}")
        self.m = lpips_module

    @torch.no_grad()
    def __call__(self, x: torch.Tensor, y: torch.Tensor,
                 mask: Optional[torch.Tensor] = None) -> float:
        """`x`、`y` 為 (N,3,H,W)、[0,1]；`mask` 為 (N,1,H,W)、[0,1]。

        `mask` 為 None 時等同全 1，回傳值與 `piq.LPIPS()(x, y)` 相同。
        遮罩總和為零時拋錯——那代表呼叫端拿了一個空區域，回傳 0 會被讀成
        「這一塊完全沒動」，是靜默失效。
        """
        m = self.m
        m.model.to(x)
        fx = m.get_features(x)
        fy = m.get_features(y)
        dists = m.compute_distance(fx, fy)

        total = None
        for d, w in zip(dists, m.weights):
            dw = d * w.to(d)
            if mask is None:
                term = dw.mean(dim=[2, 3])
            else:
                ml = F.adaptive_avg_pool2d(mask.to(dw), dw.shape[-2:])
                denom = ml.sum(dim=[2, 3])
                if float(denom.min()) <= 0.0:
                    raise ValueError(
                        f"遮罩在第 {tuple(dw.shape[-2:])} 層降取樣後總和為 0："
                        "區域太小或全為零。空區域的分區讀數沒有定義。")
                term = (dw * ml).sum(dim=[2, 3]) / denom
            total = term if total is None else torch.cat([total, term], dim=1)
        return float(total.sum(dim=1).mean())


def split_displacement(regional: RegionalLPIPS,
                       edit_orig: torch.Tensor,
                       edit_def: torch.Tensor,
                       mask: torch.Tensor) -> Dict[str, float]:
    """一次算出全圖／主體內／主體外三個位移。

    `mask` 是**主體**遮罩（1 = 主體）。補集用 `1 - mask`，故兩塊的權重和
    逐像素為 1。三個值都用同一份 LPIPS 權重。
    """
    return {
        "lpips_full": regional(edit_orig, edit_def, None),
        "lpips_subject": regional(edit_orig, edit_def, mask),
        "lpips_background": regional(edit_orig, edit_def, 1.0 - mask),
    }

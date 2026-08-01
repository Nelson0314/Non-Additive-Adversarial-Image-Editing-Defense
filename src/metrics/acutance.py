"""銳利度保留率 —— 量化「鈍化」。

為什麼需要這一項。LPIPS 對模糊相對不敏感，DISTS 因為建構在紋理相似度上
而較敏感，兩者在 VAE 重建這類失真上會給出方向相反的判斷（E18 實測：六張影像
LPIPS 全部改善、DISTS 三張退步）。人眼在 E18 的比對中報告的現象是「不差，
但比較鈍」——那是高頻能量流失，兩個感知指標都沒有直接量它。

E15 曾以同一個量抓到 site S 的低通性質（保留 78.9%，加性的 site P 為 103.2%），
但當時是臨時計算，沒有留下實作。本檔補上，使該數字可重現。

定義。以 Sobel 取梯度，能量為梯度平方和；保留率 = 重建的能量 ÷ 原圖的能量。

- 1.0 表示銳利度與原圖相同
- < 1.0 表示鈍化（低通、模糊）
- > 1.0 表示比原圖更銳（雜訊或加性擾動會造成，非必然是好事）

刻意不做正規化或裁切：能量比的絕對值本身就是要報的量，壓到 [0,1] 會把
「比原圖更銳」這個有意義的情形折疊掉。
"""

from typing import Dict

import torch
import torch.nn.functional as F

# Sobel。分開兩個方向，最後取平方和，等價於梯度模長平方。
_KX = torch.tensor([[-1.0, 0.0, 1.0],
                    [-2.0, 0.0, 2.0],
                    [-1.0, 0.0, 1.0]])
_KY = _KX.t().contiguous()


def _luma(x: torch.Tensor) -> torch.Tensor:
    """Rec.601 亮度。梯度取在亮度上而非各通道相加，避免色差主導。"""
    r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
    return 0.299 * r + 0.587 * g + 0.114 * b


def gradient_energy(x: torch.Tensor) -> torch.Tensor:
    """(N,3,H,W)、[0,1] 影像的梯度能量，逐張回傳純量。"""
    y = _luma(x)
    kx = _KX.to(y.device, y.dtype).view(1, 1, 3, 3)
    ky = _KY.to(y.device, y.dtype).view(1, 1, 3, 3)
    # replicate padding：zero padding 會在邊界造出不存在的強梯度
    y = F.pad(y, (1, 1, 1, 1), mode="replicate")
    gx = F.conv2d(y, kx)
    gy = F.conv2d(y, ky)
    return (gx.pow(2) + gy.pow(2)).flatten(1).sum(1)


@torch.no_grad()
def acutance(orig: torch.Tensor, rec: torch.Tensor) -> Dict[str, float]:
    """rec 相對 orig 的銳利度保留率。orig 能量為零時回傳 NaN 而非除以零。"""
    eo = gradient_energy(orig)
    er = gradient_energy(rec)
    ratio = torch.where(eo > 0, er / eo,
                        torch.full_like(eo, float("nan")))
    return {
        "acutance_ratio": float(ratio.mean()),
        "grad_energy_orig": float(eo.mean()),
        "grad_energy_rec": float(er.mean()),
    }

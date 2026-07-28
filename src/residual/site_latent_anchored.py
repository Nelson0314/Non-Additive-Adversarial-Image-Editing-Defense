"""site LA — latent 注入、輸出錨定於原圖。**本檔為分支上的提案，未進 main。**

## 動機

E0c 與 pilot 實測顯示 site L 有一個結構性問題：φ=0 時

    x_def = decode(denoise(inversion(encode(x))))

已經不等於 x。實測（n=6、t_max=500、k_inv=20）該重建為 PSNR 26.56 dB、
LPIPS 0.194、L∞ 1.0000，而 site P 在 φ=0 時是**逐元素相等**。

後果有二：

1. **E2 的前緣比較不對等**。P 的起點在保真軸的無窮遠端、L 的起點已在
   26.56 dB，兩條曲線的原點不同。spec §7.4 要用 P 與 L 的對比切割「秩」
   與「非加性」，但兩者還差了「有沒有重建誤差」這第三個因素，切割不乾淨。
2. **保真項被 φ 無法改善的常數佔據**。這是 spec §5.2 的 L∞ hinge 改為
   相對 `x_base` 的原因（見 objective.py 的修訂說明）。該修訂處理了損失
   函數的症狀，沒有處理架構上的成因。

## 本變體的作法

    x_def = clamp( x + [ G(x; φ) − G(x; 0) ], 0, 1 )

其中 G 為「inversion → 去噪（注入殘差）→ 解碼」。`G(x; 0)` 對 φ 為常數，
每張影像算一次即可快取。

於是 φ=0 時 `x_def = x` **逐元素相等**，與 site P 的起點一致，第 1、2 點
同時消除。

## 這是否犧牲了非加性？

**部分犧牲，必須在論文中明確區分，不得混為一談。** 三個 site 的性質：

| site | 殘差如何產生 | 殘差如何施加 |
|---|---|---|
| P  | 直接參數化（線性） | 加性 |
| L  | 經去噪鏈與 VAE（非線性） | 非加性（整張圖重新生成） |
| LA | 經去噪鏈與 VAE（非線性） | **加性** |

LA 的殘差仍是 x 與 φ 的高度非線性函數（它穿過 k_inv 次 UNet 與一次 VAE
解碼），但**施加方式**是加性的。因此 LA 不能替代 L 去支撐「非加性」的主張；
它的價值在於構成第三個對照點，把「殘差的產生方式」與「殘差的施加方式」
這兩個因素分離：

- L 耐淨化而 LA 不耐 ⟹ 關鍵在**施加方式**為非加性
- L 與 LA 皆耐而 P 不耐 ⟹ 關鍵在**產生方式**（經過擴散軌跡的結構）
- 三者皆耐 ⟹ 關鍵在低秩本身

這使原本的兩點對比變成三點對比，spec §7.4 的因果切割因此更緊。

**未驗證**：以上為設計論證，尚無實驗數據支持。
"""

from typing import Optional

import torch

from src.residual.site_latent import LatentResidual


class AnchoredLatentResidual(LatentResidual):
    """site L 的錨定變體。介面與 LatentResidual 相同，多一個 baseline 快取。

    `baseline` 由呼叫端在 φ=0 時計算一次後設入；未設入時 `anchor()` 會
    直接報錯而不是默默退化成 site L —— 兩者的語意不同，不可混淆。
    """

    site = "LA"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._baseline: Optional[torch.Tensor] = None

    def set_baseline(self, x_base: torch.Tensor) -> None:
        """設定 G(x; φ=0)。必須是 detach 過的常數。"""
        if x_base.requires_grad:
            raise ValueError(
                "baseline 必須對 φ 為常數，請先 detach；"
                "否則梯度會經由 baseline 回流並抵銷錨定的效果"
            )
        self._baseline = x_base

    @property
    def has_baseline(self) -> bool:
        return self._baseline is not None

    def anchor(self, x01: torch.Tensor, x_gen: torch.Tensor) -> torch.Tensor:
        """把生成結果錨定回原圖：x + (G(x;φ) − G(x;0))。"""
        if self._baseline is None:
            raise RuntimeError(
                "尚未設定 baseline。site LA 需要 G(x; φ=0) 才能錨定，"
                "缺少時不可退化為 site L，兩者是不同的方法"
            )
        return (x01 + (x_gen - self._baseline)).clamp(0.0, 1.0)

    def raw_residual(self) -> Optional[torch.Tensor]:
        """LA 的像素殘差需要一次完整前向才能得到，無法由參數直接算出。

        回傳 None 而非近似值：site P 的 raw_residual 是 clamp 前的 Δ，
        語意是「架構保證秩的那個量」；LA 的對應量在像素空間並不存在。
        """
        return None

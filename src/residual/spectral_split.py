"""幅度／相位交叉互換 — Zhou et al., ICML 2023（PAD）第 3 節的核心運算。

出處
────────────────────────────────────────────────────────────────────
Phase-aware Adversarial Defense for Improving Adversarial Robustness,
PMLR v202（ICML 2023）, https://proceedings.mlr.press/v202/zhou23m/zhou23m.pdf
官方程式碼 https://github.com/dwDavidxd/PAD 。

該篇圖 2 與表 1 的量化研究：把自然樣本 `x` 與對抗樣本 `x'` 各做一次二維
DFT，交叉互換幅度譜與相位譜再逆轉換，得到兩張新圖——

    只有幅度被攻擊  x_amp = F⁻¹( ξ_{x'} , φ_x  )
    只有相位被攻擊  x_pha = F⁻¹( ξ_x   , φ_{x'} )

該篇在 CIFAR-10／ResNet-18 上量到（表 1，PGD 15 步）：

    | | 完整對抗樣本 | 只有幅度 | 只有相位 |
    |---|---|---|---|
    | 準確率 | 0.01% | 24.32% | **6.12%** |
    | L2 雜訊 | 1.3860 | 0.9931 | **0.9385** |

即**相位那一半用更小的雜訊換到更大的破壞**。本專案把同一個分解搬到擴散
編輯防護的讀數上，用來回答 FND-040 欠的問題：任何保護擾動裡真正在起作用
的，是不是它的相位那一半。

與紋理重相位的關係
────────────────────────────────────────────────────────────────────
**這不是紋理重相位。** 兩者都動相位，但：

| | 本檔（PAD 分解） | `texture_rephase.py` |
|---|---|---|
| 變換 | **全域** DFT，整張圖一次 | 32×32 重疊區塊、Hann 加窗 |
| 相位怎麼來 | 從另一張圖**整份搬過來** | 由 PGD **最佳化**出旋轉角 θ |
| 用途 | 分析既有擾動 | 產生防禦擾動 |

本檔是**分析工具**，不是防禦方法，不實作 `ResidualModule` 的任何能力。

實作細節
────────────────────────────────────────────────────────────────────
* 用 `torch.fft.fft2`（**完整**複數 FFT，不是 `rfft2`）。理由：PAD 的操作
  是「把另一張圖的相位整份搬過來」，而 `rfft2` 只存半平面、隱含地假設
  輸入是實數；搬過來的半平面相位配上原本的另一半幅度時，共軛對稱不再成立，
  `irfft2` 會靜默地把虛部丟掉而不報錯。用完整 FFT 則兩張實數圖的譜各自
  共軛對稱，交叉互換後仍然對稱，逆轉換的虛部是浮點誤差量級（實測 < 1e-12），
  可以驗證而不是假設。
* 逐通道獨立做。PAD 的官方程式碼同此。
* **不夾取**。夾到 [0,1] 會破壞「幅度逐位保留」這個可驗證的性質；要夾的
  呼叫端自己夾，並自行承擔量測上的後果。
"""

from __future__ import annotations

from typing import Tuple

import torch


def to_spectrum(x: torch.Tensor) -> torch.Tensor:
    """(N,C,H,W) 實數影像 → 同形狀的複數頻譜。逐通道二維 DFT。"""
    if x.dim() != 4:
        raise ValueError(f"需要 (N,C,H,W)，收到 {tuple(x.shape)}")
    return torch.fft.fft2(x, dim=(-2, -1))


def split(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """回傳 (幅度譜, 相位譜)。相位單位是弧度，值域 (−π, π]。"""
    z = to_spectrum(x)
    return z.abs(), z.angle()


def recombine(amplitude: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
    """由幅度與相位組回實數影像：`F⁻¹( A · e^{iφ} )`。

    回傳的虛部應為浮點誤差量級。**該量由 `imag_residual` 對外提供供查核**，
    本函式直接取實部——若某個呼叫端把不對稱的譜餵進來，靜默取實部會掩蓋
    錯誤，故 `imag_residual` 存在且測試會釘住它。
    """
    z = amplitude * torch.exp(1j * phase.to(amplitude.dtype))
    return torch.fft.ifft2(z, dim=(-2, -1)).real


def imag_residual(amplitude: torch.Tensor, phase: torch.Tensor) -> float:
    """`recombine` 丟掉的虛部的最大絕對值。實數影像交叉互換後應 < 1e-9。"""
    z = amplitude * torch.exp(1j * phase.to(amplitude.dtype))
    return float(torch.fft.ifft2(z, dim=(-2, -1)).imag.abs().max())


def amplitude_only(x_ref: torch.Tensor, x_adv: torch.Tensor) -> torch.Tensor:
    """**只有幅度被擾動**的版本：`F⁻¹( ξ_{x_adv} , φ_{x_ref} )`。

    `x_ref` 是原圖、`x_adv` 是防禦圖。輸出保留原圖的全部相位（結構、輪廓、
    位置），只換上防禦圖的幅度（強度分布、質感）。
    """
    amp_adv, _ = split(x_adv)
    _, pha_ref = split(x_ref)
    return recombine(amp_adv, pha_ref)


def phase_only(x_ref: torch.Tensor, x_adv: torch.Tensor) -> torch.Tensor:
    """**只有相位被擾動**的版本：`F⁻¹( ξ_{x_ref} , φ_{x_adv} )`。

    與 `amplitude_only` 互補：兩者合起來用掉 `x_adv` 的全部頻譜資訊。
    """
    _, pha_adv = split(x_adv)
    amp_ref, _ = split(x_ref)
    return recombine(amp_ref, pha_adv)


def decompose(x_ref: torch.Tensor, x_adv: torch.Tensor
              ) -> dict:
    """一次給出兩個分解版本與查核量。

    回傳鍵：`amp_only`、`pha_only`、`imag_max`（兩者虛部殘量的較大者）。
    """
    amp_ref, pha_ref = split(x_ref)
    amp_adv, pha_adv = split(x_adv)
    return {
        "amp_only": recombine(amp_adv, pha_ref),
        "pha_only": recombine(amp_ref, pha_adv),
        "imag_max": max(imag_residual(amp_adv, pha_ref),
                        imag_residual(amp_ref, pha_adv)),
    }


def amplitude_deviation(x_ref: torch.Tensor, x_adv: torch.Tensor) -> float:
    """`x_adv` 的幅度譜相對 `x_ref` 的相對偏移，`‖ξ_adv − ξ_ref‖ / ‖ξ_ref‖`。

    紋理重相位由構造保證此值為 0（區塊譜上），本函式量的是**全域**譜，
    兩者不可直接互相比較——`texture_rephase.amplitude_deviation` 量的是
    加窗區塊譜，overlap-add 之後全域譜不再逐位保留。此處提供全域版本，
    是為了讓所有條件（含加性 baseline）落在同一把尺上。
    """
    amp_ref, _ = split(x_ref)
    amp_adv, _ = split(x_adv)
    return float((amp_adv - amp_ref).norm() / amp_ref.norm())

"""交付前把像素域殘差夾到 L∞ 球內。**改的是交出去的圖，不是損失。**

為什麼需要它
────────────────────────────────────────────────────────────────────
最佳化只被約束在參數 `θ` 的半徑球上，**像素域沒有任何約束**。實測
（`runs/ip2p_ig_converge`，影像引導消除損失四個工作點）在 DISTS 0.153–0.180
下 L∞ 是 **0.793–0.927**，而主線相位族在相近的 DISTS（0.145）只有 **0.42**。
同樣的 DISTS 代表殘差是**稀疏高振幅尖峰**：人眼對孤立尖峰極敏感、對彌散
紋理不敏感，而 DISTS 與 LPIPS 幾乎看不到這個差別。使用者看到「醜」而指標
說「一樣」，缺的就是這個軸。

`GOAL.md` 已經要求兩個失真軸都報，但**錨點**只走 DISTS，於是等失真比較會
系統性地偏袒尖峰型的解。這一支是把 L∞ 由「事後才看到的欄位」變成「事前
就守住的約束」。

為什麼是交付投影而不是損失懲罰
────────────────────────────────────────────────────────────────────
懲罰項要一個權重，而權重要掃；投影不必，而且它保證**交出去的圖真的滿足
界限**。這與 `--deliver-jpeg` 是同一個型態：迴圈的前向套直通估計版、交付
與存檔套真正的投影，於是「最佳化的對象」與「交出去的對象」是同一張圖。

直通估計（STE）的理由
────────────────────────────────────────────────────────────────────
硬夾取在球外梯度為零，被夾住的座標拿不到任何訊號、再也回不來，最佳化會停在
第一次撞界的地方。STE 讓反傳看見恆等映射，前向仍然輸出夾過的值——與
`src/baselines/jpeg_codec.py` 的 `jpeg_roundtrip_ste` 同一個作法。

**兩支函式的差別只有可微性**，其餘逐位元相同；報表上要分得出前向用的是哪
一支。
"""

from __future__ import annotations

import torch


def clamp_residual(x_def: torch.Tensor, x01: torch.Tensor,
                   eps: float) -> torch.Tensor:
    """把 `x_def - x01` 夾到 `[-eps, eps]`，再夾回合法值域 `[0, 1]`。

    值域夾取不可省：`x01` 已經貼在 0 或 1 的位置上時，殘差就算落在 L∞ 球內
    也可能把像素推出值域，存成 PNG 時會被無聲截斷，於是**存檔的圖不滿足
    界限**，而 CSV 上的 `fid_linf` 是從存檔的圖量的——兩者對不上。
    """
    if eps <= 0:
        raise ValueError(f"eps 必須為正，收到 {eps}")
    if x_def.shape != x01.shape:
        raise ValueError(f"形狀不合：{tuple(x_def.shape)} 對 {tuple(x01.shape)}")
    return (x01 + (x_def - x01).clamp(-eps, eps)).clamp(0.0, 1.0)


def clamp_residual_ste(x_def: torch.Tensor, x01: torch.Tensor,
                       eps: float) -> torch.Tensor:
    """`clamp_residual` 的直通估計版：前向相同，反傳當成恆等映射。"""
    hard = clamp_residual(x_def, x01, eps)
    return x_def + (hard - x_def).detach()

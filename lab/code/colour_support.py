"""顏色定義的支撐與彩度上界。

兩件事共用一個前提：**本專案的色偏只有兩種情形難看**——高飽和，以及與原膚色
的色差過大。兩者都是**全域上界**，不是遮罩；遮罩式的膚色保護已經量過，
帶窄出斑塊、帶寬出臉霧，而且臉框的色差量不到那個症狀。

所以這一檔提供的是：

| 名字 | 是什麼 | 擋什麼 |
|---|---|---|
| `skin_colour_support` | 原圖裡與膚色**同色**的像素（不限位置） | 「與原膚色色差過大」 |
| `chroma_p95` | 輸出彩度的 95 百分位 | 「高飽和」 |
"""

from __future__ import annotations

import torch

from src.defense.ncf_param import rgb_to_lab


def skin_colour_support(x01: torch.Tensor, face: torch.Tensor,
                        radius: float = 12.0) -> torch.Tensor:
    """原圖裡與膚色**同色**的像素，不限位置。回傳 (1,1,H,W) 的 0／1 遮罩。

    群心取臉框內 `(a, b)` 的**中位數**——平均會被頭髮與背景的離群值拉走。

    這一塊與臉框是兩個不同的支撐：臉框是位置定義的，這一塊是顏色定義的。
    全域映射動的是顏色，所以顏色定義的那一塊才是它真正的作用範圍：背景裡
    與膚色同色的東西會跟著臉一起被保護，而臉框上限完全看不到那些像素。
    """
    ab = rgb_to_lab(x01.clamp(0, 1).float())[:, 1:]
    sel = ab[0].permute(1, 2, 0)[face[0, 0] > 0.5]
    if sel.numel() == 0:
        raise ValueError("臉框內沒有像素，膚色群心定不出來")
    centre = sel.median(0).values.view(1, 2, 1, 1)
    dist = (ab - centre).pow(2).sum(1, keepdim=True).clamp_min(1e-12).sqrt()
    return (dist <= radius).to(ab.dtype)


def chroma_p95(x01: torch.Tensor, q: float = 0.95) -> torch.Tensor:
    """輸出彩度（Lab 的 C*）的 95 百分位。

    `sqrt` 先 `clamp_min`：`sqrt` 在 0 的導數是無限大，而純灰（`a = b = 0`）
    在人像上到處都是——白色天空整片就是。不夾的話第 0 步就拿到 NaN 梯度，
    而 `delta_e_torch` 那一份早就因為同一個理由夾過了。
    """
    lab = rgb_to_lab(x01.clamp(0, 1).float())
    c = lab[:, 1:].pow(2).sum(1).clamp_min(1e-12).sqrt().reshape(-1)
    return torch.quantile(c, q)

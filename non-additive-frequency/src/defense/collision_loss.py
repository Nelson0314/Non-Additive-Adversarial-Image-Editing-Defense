"""同色碰撞：把指令要改的區域在顏色統計上與周邊合流。

為什麼不再往「把 latent 推遠」的方向走
────────────────────────────────────────────────────────────────────
顏色映射對擴散編輯是語意上空的變換——可逆、不移除任何關於主體的資訊、也不與
指令衝突，所以模型照樣完成指令，只是輸出換個色調。`runs/objective_pilot/` 的
整圖載體上，把載體推到可達集合邊界（`‖E(x')‖₂` 由 152.6 升到 1045.1，由反三角
不等式 latent 位移至少 892）之後，位移只有 0.3626、主體身分降幅 +0.0172
（兩者都取整圖載體那 30 列的中位數；全部 45 列的中位數是 +0.0077，兩個母體
不同，不可並列），盲評十五格裡十四格判為指令完成。最佳化過的那一臂位移是 0.3098，**比隨機邊界
還低**。

所以這一支換機制：不推 latent，改破壞**定位**。區域與其周邊的顏色統計一致時，
「把襯衫改成紅色」沒有可指認的襯衫。

可否證的預測
────────────────────────────────────────────────────────────────────
屬性改色類指令的完成率下降，頭部配件與背景類不變。三類一起掉的話，機制解釋
就不是「定位」，這個目標函數的理由不成立。

為什麼是前兩階矩
────────────────────────────────────────────────────────────────────
一階與二階已經決定了 Monge–Kantorovitch 轉移（`ncf_param.mk_matrix`），與本
專案既有的色彩統計是同一組量；且不需要可微的排序，成本與完整分布距離差一個
數量級。**不需要擴散模型的梯度**，比 `latent_norm` 便宜一個數量級。
"""
from __future__ import annotations

from typing import Callable

import torch

from src.purify.ops import gaussian_blur

from .ncf_param import rgb_to_lab


def ring_of(region: torch.Tensor, width: int) -> torch.Tensor:
    """區域外的一圈環。膨脹減自己，故與區域不重疊。

    環而不是「區域的補集」：指令的定位是局部的比較，全圖的統計會被遠處的
    大面積背景支配。
    """
    if width < 1:
        raise ValueError('width 必須至少為 1')
    if region.ndim != 4 or region.shape[1] != 1:
        raise ValueError('region 必須是 (1,1,H,W)')
    sigma = float(width) / 2.0
    grown = (gaussian_blur(region, sigma) > 1e-3).to(region.dtype)
    return (grown - region).clamp(0., 1.)


def _moments(lab: torch.Tensor, mask: torch.Tensor, what: str):
    w = mask.to(device=lab.device, dtype=lab.dtype)
    total = w.sum()
    if float(total) < 2.:
        raise ValueError(f'{what} 的權重不足兩個像素，統計沒有意義')
    mean = (lab * w).sum(dim=(0, 2, 3)) / total
    var = (((lab - mean[None, :, None, None]) ** 2) * w).sum(dim=(0, 2, 3)) / total
    return mean, var.clamp_min(1e-12).sqrt()


def make_collision_loss(region: torch.Tensor, ring: torch.Tensor
                        ) -> Callable[[torch.Tensor], torch.Tensor]:
    """回傳 `loss(x01) -> 純量`，**要最小化**。

    `L = ‖mu_R − mu_E‖² + ‖sigma_R − sigma_E‖²`，兩項都在 Lab 單位下，尺度相同，
    所以不另設權重——加權會多一個沒有理由的超參數。
    """
    def loss(x01: torch.Tensor) -> torch.Tensor:
        lab = rgb_to_lab(x01)
        mu_r, sd_r = _moments(lab, region, 'region')
        mu_e, sd_e = _moments(lab, ring, 'ring')
        return (mu_r - mu_e).pow(2).sum() + (sd_r - sd_e).pow(2).sum()
    return loss

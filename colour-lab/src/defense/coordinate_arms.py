"""三個座標臂：原參數空間、正則化白化、白化＋硬性非暗化子空間。

問的是什麼
────────────────────────────────────────────────────────────────────
`runs/bernstein_reachability/` 量到的是**一階**的事：梯度能量幾乎全落在
非暗化方向上，可達增量卻仍是正向暗化最大。一階的落差有兩種來源，這個模組
把它們分開：

1. **座標尺度。** `∇_θJ` 是歐氏梯度，而域的形狀由 `G = F_xᵀF_x` 決定，兩者
   的單位不同；沿 `∇_θJ` 走因此不是沿「每單位擾動能量推得最多」的方向走。
   換座標就能修掉的話，這是尺度問題。
2. **可行幾何。** 非暗化方向上的半徑本來就短，換什麼座標都一樣短。

臂 A 與 B 的差別只有座標，域、起點、步數、接受準則全部相同；B 與 C 的差別
只有「增量要不要 `G`-正交於暗化」。

三個臂
────────────────────────────────────────────────────────────────────
| 臂 | 優化變數 `z` → 位移方向 `v` | 額外限制 |
|---|---|---|
| `theta` (A) | `v = z` | 無 |
| `whitened` (B) | `v = M^{−1/2} z` | 無 |
| `whitened_nondark` (C) | `v = M^{−1/2} z` | `UᵀG M^{−1/2} Δz = 0` |

`v` 不是 θ 本身，是送進**既有前向參數化**的方向向量：

    θ(v) = θ_c + (1 − m)·tanh(‖v‖)·r_max(v̂)·v̂

與 `bernstein_colour.BernsteinColourParam.theta()` 逐字同一條式子。域因此
仍然是精確的，七條硬限制全程成立，沒有任何「更新後鉗回」；三個臂換掉的
只有「優化變數怎麼對應到 `v`」。

`M` 為什麼不是 `G`
────────────────────────────────────────────────────────────────────
`G` 在這八張影像上的有效維度只有 30–50（`geometry.csv` 的 `eff_dim`），
其餘 200 多個方向的特徵值小到 `G^{−1/2}` 會把它們放大到數值上沒有意義。
`reachability.Whitening` 的作法是**截掉**那些方向；那對「方向的分類」是
對的，對「換座標之後還能不能優化」卻不是——被截掉的方向仍在域裡、仍然
改變 θ，截掉等於偷偷降維，會把幾何瓶頸偽裝成座標問題。這裡改成把小特徵
值抬到地板：

    G = QΛQᵀ,   M = Q·diag(max(λ_i, τ))·Qᵀ,   τ = 1e−4·λ_max

`M` 滿秩、`M^{−1/2}` 是 250×250 的可逆映射，條件數上限 `√(λ_max/τ) = 100`。
**沒有任何維度被丟掉。**

C 的限制
────────────────────────────────────────────────────────────────────
`U = {DARK}`（`a ≡ 常數`、`b ≡ 0`），與 `reachability.GProjector` 同一個
定義：`Δv = M^{−1/2}Δz` 要滿足 `DARKᵀ G Δv = 0`，也就是它產生的影像擾動
與整體暗化產生的擾動正交。在 `z` 空間這是一條線性限制 `nᵀΔz = 0`，
`n = M^{−1/2} G · DARK`（兩個矩陣都對稱）。實作是把**每一步的更新量**投影
到 `n` 的零空間：起點固定，於是 `Δz = z_k − z_0` 逐項落在零空間裡，限制是
恆等成立而不是懲罰。
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch

from src.defense import bernstein_colour as bc
from src.defense.reachability import DARK, symmetrise

#: 小特徵值的相對地板。**這是規格，不是可調參數。**
TAU_REL = 1e-4

#: 三個臂的名字與對應的標籤。標籤只為了與交付表的 A／B／C 對得上。
ARM_LABELS = {'theta': 'A', 'whitened': 'B', 'whitened_nondark': 'C'}


class RegularisedWhitening:
    """`M = Q·diag(max(λ, τ))·Qᵀ` 與它的平方根，`τ = tau_rel·λ_max`。

    與 `reachability.Whitening` 的唯一差別是**不截斷**（理由見模組說明）。
    `sqrt` 與 `inv_sqrt` 都再對稱化一次：`Q diag(·) Qᵀ` 在 float64 下兩半
    會差到 `1e−16` 相對量級，而 `to_z(to_v(z))` 要逐位還原得回來。
    """

    def __init__(self, gram, tau_rel: float = TAU_REL):
        g = symmetrise(gram)
        lam, vec = np.linalg.eigh(g)
        lam = np.clip(lam, 0.0, None)
        lam_max = float(lam.max())
        if lam_max <= 0:
            raise ValueError('G 是零矩陣：這張影像激發不出任何係數')
        self.dim = int(g.shape[0])
        self.gram = g
        self.tau_rel = float(tau_rel)
        self.tau = float(tau_rel) * lam_max
        self.lam_raw = lam
        self.lam = np.maximum(lam, self.tau)
        self.basis = np.ascontiguousarray(vec)
        self.floored = int((lam < self.tau).sum())
        self.lam_max = lam_max
        self.lam_min_raw = float(lam.min())
        inv = (vec * (self.lam ** -0.5)) @ vec.T
        fwd = (vec * (self.lam ** 0.5)) @ vec.T
        self.inv_sqrt = 0.5 * (inv + inv.T)
        self.sqrt = 0.5 * (fwd + fwd.T)

    def report(self) -> Dict[str, float]:
        return {
            'dim': self.dim,
            'tau_rel': self.tau_rel,
            'tau': self.tau,
            'lam_max': self.lam_max,
            'lam_min_raw': self.lam_min_raw,
            'floored_directions': self.floored,
            'condition_inv_sqrt': float(np.sqrt(self.lam.max() / self.lam.min())),
        }


class CoordinateArm:
    """一個臂：`z → v` 的線性映射，加上（可選的）更新量投影。"""

    def __init__(self, name: str, matrix: Optional[np.ndarray] = None,
                 inverse: Optional[np.ndarray] = None,
                 normal: Optional[np.ndarray] = None):
        if name not in ARM_LABELS:
            raise ValueError(f'未知的臂 {name}')
        self.name = name
        self.label = ARM_LABELS[name]
        self.matrix = (None if matrix is None
                       else torch.as_tensor(matrix, dtype=torch.float64))
        self.inverse = (None if inverse is None
                        else torch.as_tensor(inverse, dtype=torch.float64))
        if normal is None:
            self.normal = None
        else:
            n = torch.as_tensor(normal, dtype=torch.float64)
            nn = float(n @ n)
            if nn <= 0:
                raise ValueError('要投影掉的法向量長度為零')
            self.normal = n / float(np.sqrt(nn))

    # ---- 座標 ----

    def to_v(self, z: torch.Tensor) -> torch.Tensor:
        """優化變數 → 送進徑向映射的方向向量。可微。"""
        return z if self.matrix is None else self.matrix @ z

    def to_z(self, v: torch.Tensor) -> torch.Tensor:
        """反向，只在決定起點時用到。"""
        return v if self.inverse is None else self.inverse @ v

    def project(self, u: torch.Tensor) -> torch.Tensor:
        """更新量投影到限制的零空間。臂 A／B 是恆等。"""
        if self.normal is None:
            return u
        return u - self.normal * (self.normal @ u)

    def violation(self, dz: torch.Tensor) -> float:
        """`|nᵀΔz| / ‖Δz‖`，C 的限制殘量。沒有限制的臂回 0。"""
        if self.normal is None:
            return 0.0
        norm = float(dz.norm())
        if norm <= 0:
            return 0.0
        return abs(float(self.normal @ dz)) / norm


def build_arms(whitening: RegularisedWhitening) -> Dict[str, CoordinateArm]:
    """三個臂，共用同一份 `M^{±1/2}`。"""
    normal = whitening.inv_sqrt @ (whitening.gram @ DARK)
    return {
        'theta': CoordinateArm('theta'),
        'whitened': CoordinateArm('whitened', whitening.inv_sqrt,
                                  whitening.sqrt),
        'whitened_nondark': CoordinateArm('whitened_nondark',
                                          whitening.inv_sqrt, whitening.sqrt,
                                          normal=normal),
    }


class RadialTheta:
    """既有的前向參數化 `v → θ`，半徑在 `no_grad` 下求。

    與 `BernsteinColourParam.theta()` 同一條式子，差別只在 `v` 由臂給出而
    不是優化變數本身。`radius_calls` 記下二分求根被呼叫幾次，成本要報。
    """

    def __init__(self, domain: bc.BernsteinDomain, theta0: torch.Tensor,
                 iters: int = 40):
        self.domain = domain
        self.theta0 = theta0.detach().double()
        self.iters = int(iters)
        self.radius_calls = 0

    def __call__(self, v: torch.Tensor) -> torch.Tensor:
        norm = v.norm()
        with torch.no_grad():
            unit = (v / norm.clamp_min(1e-12)).detach()
            radius = self.domain.max_radius(self.theta0, unit,
                                            iters=self.iters)
            self.radius_calls += 1
        scale = (radius * (1.0 - bc.INTERIOR_MARGIN) * torch.tanh(norm)
                 / norm.clamp_min(1e-12))
        return self.theta0 + scale * v

    def radius(self, v: torch.Tensor) -> float:
        norm = v.detach().norm().clamp_min(1e-12)
        self.radius_calls += 1
        return self.domain.max_radius(self.theta0, (v.detach() / norm),
                                      iters=self.iters)


def slack_report(domain: bc.BernsteinDomain, theta: torch.Tensor
                 ) -> Dict[str, float]:
    """七條限制的相對餘裕，外加最小值。**量的是實際送進渲染的 float32 θ。**

    `apply_filter` 會把 θ 轉成影像的 dtype，所以可行性要在轉換後的值上判，
    不是在 float64 的中間量上判。
    """
    slacks = domain.relative_slacks(theta.detach().float())
    return {**slacks, 'min_slack': float(min(slacks.values()))}

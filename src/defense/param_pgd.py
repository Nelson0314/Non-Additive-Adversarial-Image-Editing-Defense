"""參數化 PGD：把「加性 δ」與「非加性算子的參數 φ」放在同一條迴圈上。

存在理由
────────────────────────────────────────────────────────────────────
本專案至今每一次「加性 vs 非加性」的比較都同時換掉四五個東西（階段一、
更新規則、約束形式、reward），於是差異無法歸因（FND-028／029）。這裡把
損失、更新規則、步數、種子、預算對齊程序**全部固定**，唯一的變因是
`Parameterization` 這一個介面的實作。

    for i in 0..steps-1:
        x_def ← param.render(x)
        g     ← ∇_φ L(x_def)
        φ     ← φ − α · sign(g)          # 兩邊同一條更新式
        φ     ← param.project()          # 各自的約束形式

更新規則取 sign 是為了對齊三個加性 baseline 中的四篇（`pgd.py` 的
`UPDATE_RULES`），不是因為它最好——FND-029 已記載 sign 丟掉梯度大小會讓
可見失真變差。此處要的是可歸因，不是最佳。

預算對齊
────────────────────────────────────────────────────────────────────
`fit_to_budget` 對**半徑**二分搜尋，使**最終迭代**落在目標 DISTS 上。
不沿迭代軌跡挑一格：那樣拿到的是一個還沒收斂的 φ，兩個條件各自停在不同
的最佳化進度上，比較的就不只是參數化。sign 更新會把半徑用滿
（FND-028 實測 `linf` 逐迭代恰為 `µ × iter`），故「半徑 → 最終失真」
是單調的，二分搜尋有意義。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Protocol

import torch

from src.residual.texture_rephase import PhaseResidual


class Parameterization(Protocol):
    """φ 的容器。實作者只需回答四件事：怎麼畫、參數是誰、怎麼投影、半徑多大。"""

    name: str

    def render(self, x01: torch.Tensor) -> torch.Tensor: ...
    def params(self) -> List[torch.Tensor]: ...
    def project(self) -> None: ...
    def set_radius(self, r: float) -> None: ...
    def reset(self, x01: torch.Tensor, seed: int) -> None: ...


class AdditiveParam:
    """φ = δ，`x_def = clamp(x + δ, 0, 1)`，L∞ 投影。加性對照組。

    `keep` (1,1,H,W) 或 (1,3,H,W) 給定時，δ 先乘上它再加到影像上。
    inpainting 下傳 `1 - mask`：重畫區的擾動被 `mask_latents` 歸零，
    留在那裡只是白付失真。等價於 PhotoGuard-c 原始碼的 `grad * (1 - cur_mask)`。
    """

    name = "add"

    def __init__(self, radius: float = 16.0 / 255.0,
                 keep: Optional[torch.Tensor] = None):
        self.radius = radius
        self.keep = keep
        self.delta: Optional[torch.Tensor] = None

    def reset(self, x01: torch.Tensor, seed: int) -> None:
        self.delta = torch.zeros_like(x01, requires_grad=True)

    def render(self, x01: torch.Tensor) -> torch.Tensor:
        d = self.delta if self.keep is None else self.delta * self.keep.to(self.delta)
        return (x01 + d).clamp(0.0, 1.0)

    def params(self) -> List[torch.Tensor]:
        return [self.delta]

    @torch.no_grad()
    def project(self) -> None:
        self.delta.clamp_(-self.radius, self.radius)

    def set_radius(self, r: float) -> None:
        self.radius = r


class PhaseParam:
    """φ = θ，`x_def = clamp(PhaseResidual(x), 0, 1)`，逐元素夾到 θ_max。

    半徑就是 `theta_max`，其上界是 π——相位是週期量，**本參數化的失真有
    構造上的天花板**，與加性的 ε 可以無限放大不同。天花板在哪由 `r_min`
    決定（實測：r_min=0.25 對應 DISTS 約 0.055，0.12 對應約 0.10）。
    達不到的預算點要標為 unreachable，不是 failed（與 FND-001 同型）。

    `gl_iters > 0` 時每次 render 多跑幾輪 Griffin-Lim 迭代投影，把 STFT
    一致性投影誤差壓下去。這是 FND-040 的判別實驗：效果若隨 `amp_dev` 一起
    塌掉，代表它來自新造的能量而非相位重排。
    """

    name = "phase"

    def __init__(self, size: int = 512, block: int = 32, r_min: float = 0.12,
                 radius: float = math.pi, energy_quantile: float = 0.5,
                 keep: Optional[torch.Tensor] = None, gl_iters: int = 0):
        self.size, self.block, self.r_min = size, block, r_min
        self.energy_quantile = energy_quantile
        self.radius = min(radius, math.pi)
        self.keep = keep
        self.gl_iters = gl_iters
        self.module: Optional[PhaseResidual] = None

    def reset(self, x01: torch.Tensor, seed: int) -> None:
        self.module = PhaseResidual(
            size=self.size, block=self.block, r_min=self.r_min,
            theta_max=self.radius, energy_quantile=self.energy_quantile,
            gl_iters=self.gl_iters,
        ).to(device=x01.device, dtype=x01.dtype)
        self.module.prepare_gates(x01, keep=self.keep)

    def render(self, x01: torch.Tensor) -> torch.Tensor:
        return self.module.pixel_residual(x01).clamp(0.0, 1.0)

    def params(self) -> List[torch.Tensor]:
        return [self.module.theta]

    @torch.no_grad()
    def project(self) -> None:
        self.module.theta.clamp_(-self.radius, self.radius)

    def set_radius(self, r: float) -> None:
        self.radius = min(r, math.pi)
        if self.module is not None:
            self.module.theta_max = self.radius


class RandomPhaseParam(PhaseParam):
    """同幅度的隨機相位，即 RPN 本身。**不最佳化**，只在 reset 時抽一次。

    FND-004 與 FND-018 兩次都是被「贏不過同失真隨機」擋下來的。此對照組
    自第一天存在，不是事後補的。
    """

    name = "phase_rand"

    def reset(self, x01: torch.Tensor, seed: int) -> None:
        super().reset(x01, seed)
        gen = torch.Generator(device="cpu").manual_seed(seed)
        init = torch.randn(self.module.theta.shape, generator=gen) * self.radius
        with torch.no_grad():
            self.module.theta.copy_(
                init.clamp(-self.radius, self.radius).to(
                    device=x01.device, dtype=x01.dtype)
            )

    def params(self) -> List[torch.Tensor]:
        return []


@dataclass
class ParamPGDResult:
    x_def: torch.Tensor
    radius: float
    history: List[Dict] = field(default_factory=list)


def run_param_pgd(
    x01: torch.Tensor,
    param: Parameterization,
    loss_fn: Callable[[torch.Tensor], torch.Tensor],
    *,
    steps: int = 100,
    saturate_at: float = 0.25,
    seed: int = 0,
    log_every: int = 0,
    transform: Optional[Callable[[torch.Tensor, int], torch.Tensor]] = None,
) -> ParamPGDResult:
    """共用迴圈。`loss_fn(x_def)` 回傳要**最小化**的純量。

    `saturate_at` 決定步長：`α = radius / (steps · saturate_at)`，即前四分之一
    的迭代就能走滿半徑，其餘用來在球面上調方向。兩個參數化共用同一個比例，
    故「誰先撞到約束」不會變成一個隱藏的變因。

    `transform` 給定時，損失改在 `transform(x_def, step)` 上計算——用來把
    可微分的淨化算子放進最佳化迴圈（`src/defense/purify_aware.py`）。
    **回傳的 `x_def` 仍然是未經 transform 的防禦圖**：transform 是攻擊方會
    做的事，不是我們交出去的東西。預設 `None`，行為與加入此參數之前逐位元
    相同。
    """
    param.reset(x01, seed)
    ps = param.params()
    alpha = param.radius / max(1.0, steps * saturate_at)
    history: List[Dict] = []

    if not ps:                                   # phase_rand：不最佳化
        with torch.no_grad():
            return ParamPGDResult(param.render(x01).detach(), param.radius, history)

    for i in range(steps):
        x_def = param.render(x01)
        loss = loss_fn(x_def if transform is None else transform(x_def, i))
        grads = torch.autograd.grad(loss, ps)
        with torch.no_grad():
            for p, g in zip(ps, grads):
                p.sub_(alpha * torch.sign(g))
        param.project()
        if log_every and (i % log_every == 0 or i == steps - 1):
            history.append({"step": i, "loss": float(loss.detach())})
            print(f"    [{param.name}] step {i:3d} loss {float(loss.detach()):.6f}",
                  flush=True)

    with torch.no_grad():
        x_def = param.render(x01).detach()
    return ParamPGDResult(x_def, param.radius, history)


def fit_to_budget(
    x01: torch.Tensor,
    param: Parameterization,
    loss_fn: Callable[[torch.Tensor], torch.Tensor],
    distortion_fn: Callable[[torch.Tensor, torch.Tensor], float],
    target: float,
    *,
    lo: float,
    hi: float,
    steps: int = 100,
    seed: int = 0,
    rounds: int = 8,
    tol: float = 0.002,
    transform: Optional[Callable[[torch.Tensor, int], torch.Tensor]] = None,
) -> ParamPGDResult:
    """對半徑二分搜尋，使最終迭代的失真落在 `target ± tol`。

    先在 `hi` 上跑一次確認目標可達。不可達時回傳 `hi` 的結果並把達到的
    失真寫在 `history` 裡，由呼叫端標成 unreachable——**不要**靜默回傳一個
    離目標很遠的結果，那會讓表格上的預算欄變成謊話。
    """
    param.set_radius(hi)
    top = run_param_pgd(x01, param, loss_fn, steps=steps, seed=seed,
                        transform=transform)
    d_top = distortion_fn(top.x_def, x01)
    if d_top < target - tol:
        top.history.append({"unreachable": True, "target": target,
                            "reached": d_top, "radius": hi})
        return top

    best, best_err = top, abs(d_top - target)
    for _ in range(rounds):
        mid = 0.5 * (lo + hi)
        param.set_radius(mid)
        res = run_param_pgd(x01, param, loss_fn, steps=steps, seed=seed,
                            transform=transform)
        d = distortion_fn(res.x_def, x01)
        if abs(d - target) < best_err:
            best, best_err = res, abs(d - target)
        if best_err <= tol:
            break
        if d < target:
            lo = mid
        else:
            hi = mid
    best.history.append({"unreachable": False, "target": target,
                         "reached": distortion_fn(best.x_def, x01),
                         "radius": best.radius})
    return best

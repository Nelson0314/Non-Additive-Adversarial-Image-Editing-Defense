"""四個自由度的色彩載體：保色相的明度曲線 ＋ 單向去飽和。

這個族是什麼
────────────────────────────────────────────────────────────────────
對正規化 RGB `c`，令 `m = max(R,G,B)`、`Y = 0.2126R + 0.7152G + 0.0722B`：

    f_θ(m) = (1 + t0)·m + t1·m(1 − m) + t2·m(1 − m)(2m − 1)
    T_θ(c) = [ f_θ(m)/m − s ]·c + s·Y·1,      θ = (t0, t1, t2, s)

三個通道乘的是**同一個**比例 `f_θ(m)/m − s`，加的是**同一個**灰量 `s·Y`，
所以整個族只有四個自由度，與影像大小無關。

`f_θ(m)/m` 不除法
────────────────────────────────────────────────────────────────────
`f_θ(m)` 的三項都帶一個 `m` 的因子，約掉之後

    g_θ(m) ≔ f_θ(m)/m = (1 + t0) + t1(1 − m) + t2(1 − m)(2m − 1)

是一個多項式，`m = 0` 處的值是 `d0`（見下），有限。實作直接算 `g_θ`，
**程式裡沒有除法**，因此 `m = 0` 的像素不需要特例、也不需要 clamp 繞過：
該處 `c = 0`、`Y = 0`，輸出逐位元為 0。

可行域
────────────────────────────────────────────────────────────────────
`f_θ` 的導數寫成 Bernstein 基底：

    f'(m) = d0(1 − m)² + 2·d1·m(1 − m) + d2·m²
    d0 = 1 + t0 + t1 − t2,   d1 = 1 + t0 + 2t2,   d2 = 1 + t0 − t1 − t2

Bernstein 係數落在一個區間裡，曲線的斜率就落在同一個區間裡（凸包性質），
所以可行域取

    0.75 ≤ d0, d1, d2 ≤ 1,        0 ≤ s ≤ 0.1

`(t0, t1, t2) ↔ (d0, d1, d2)` 是可逆線性映射（`coeffs_to_theta`），可行域
因此是 θ 空間裡的一個平行六面體 × 一段線段。

四條結構性質（`tests/test_tone_desat_param.py` 各釘一條）
────────────────────────────────────────────────────────────────────
| 性質 | 為什麼成立 |
|---|---|
| 色相不變 | `T(c) = a·c + b·1`，`a > 0`；HSV 色相只取決於 `(c_i − c_min)/(c_max − c_min)`，同乘再同加不改變它 |
| 飽和度不增 | 新飽和度 `aΔ/(aM + b)`，`b = sY ≥ 0`，故 ≤ `Δ/M` |
| 不會變亮 | 最大通道變成 `f(m) − s·m + s·Y`；`f(m) ≤ m`（因 `f(0)=0`、`f' ≤ 1`）且 `Y ≤ m`，故 ≤ `m` |
| 無急轉、無平坦段 | `0.75 ≤ f'(m) ≤ 1`，色帶（斜率暴衝）與色調壓平（斜率 0）都不可達 |

輸出不需要 clamp：最大通道 ≤ `m ≤ 1`，最小通道 = `(g − s)c_min + sY ≥ 0`
（`g − s ≥ 0.75 − 0.1 > 0`）。

載體介面
────────────────────────────────────────────────────────────────────
`render` / `params` / `project` / `state_dict` / `load_state_dict` /
`stages` / `set_amplitude` 與 `src/defense/color_param.py::ColorCurveParam`
同一組，所以 `src/defense/immunise.py::optimise_carrier` 接得上。
`project` 是**在 `(d0,d1,d2,s)` 座標下**對盒子做夾取再映射回 θ，這是精確的
投影（映射可逆），不是逐次逼近。
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import torch

#: 斜率（Bernstein 係數）的上下界。
SLOPE_LO = 0.75
SLOPE_HI = 1.0
#: 去飽和量的上下界。`s ≥ 0` 是「只能去飽和、不能加飽和」那條性質的來源。
S_LO = 0.0
S_HI = 0.1

#: 可行域內點，八張圖在它附近都有餘量。
THETA_CENTRE: Tuple[float, float, float, float] = (-0.15, 0.0, 0.0, 0.01)

#: Rec.709 亮度權重。`Y` 只用在加回去的灰量上。
LUMA = (0.2126, 0.7152, 0.0722)


def theta_to_coeffs(theta: Sequence[float]) -> Tuple[float, float, float]:
    """θ → `(d0, d1, d2)`，即 `f'` 的三個 Bernstein 係數。"""
    t0, t1, t2 = float(theta[0]), float(theta[1]), float(theta[2])
    return (1.0 + t0 + t1 - t2, 1.0 + t0 + 2.0 * t2, 1.0 + t0 - t1 - t2)


def coeffs_to_theta(d0: float, d1: float, d2: float
                    ) -> Tuple[float, float, float]:
    """`(d0, d1, d2)` → `(t0, t1, t2)`，`theta_to_coeffs` 的反函數。

    由三條式子直接解出：`t0 = (d0+d1+d2)/3 − 1`、`t1 = (d0−d2)/2`、
    `t2 = (2d1 − d0 − d2)/6`。可逆是 `project` 能一步到位的理由。
    """
    return ((d0 + d1 + d2) / 3.0 - 1.0,
            (d0 - d2) / 2.0,
            (2.0 * d1 - d0 - d2) / 6.0)


def slope(theta: Sequence[float], m):
    """`f'_θ(m)`。`m` 可以是 float 或 tensor。"""
    d0, d1, d2 = theta_to_coeffs(theta)
    one = 1.0 - m
    return d0 * one * one + 2.0 * d1 * m * one + d2 * m * m


def gain(theta: Sequence[float], m):
    """`g_θ(m) = f_θ(m)/m` 的多項式形式。不做除法。"""
    t0, t1, t2 = float(theta[0]), float(theta[1]), float(theta[2])
    return (1.0 + t0) + t1 * (1.0 - m) + t2 * (1.0 - m) * (2.0 * m - 1.0)


def tone(theta: Sequence[float], m):
    """`f_θ(m)`。量測與測試用；`render` 走的是 `gain`。"""
    return gain(theta, m) * m


def violations(theta: Sequence[float], tol: float = 1e-9) -> Dict[str, float]:
    """回傳每一道違反的量（未違反的不列）。空 dict 即可行。"""
    d = theta_to_coeffs(theta)
    s = float(theta[3])
    out: Dict[str, float] = {}
    for name, v in zip(("d0", "d1", "d2"), d):
        if v < SLOPE_LO - tol:
            out[f"{name}_below"] = SLOPE_LO - v
        if v > SLOPE_HI + tol:
            out[f"{name}_above"] = v - SLOPE_HI
    if s < S_LO - tol:
        out["s_below"] = S_LO - s
    if s > S_HI + tol:
        out["s_above"] = s - S_HI
    return out


def feasible(theta: Sequence[float], tol: float = 1e-9) -> bool:
    return not violations(theta, tol)


def project_theta(theta: Sequence[float]) -> Tuple[float, float, float, float]:
    """把 θ 投影回可行域：在 `(d, s)` 座標下夾取，再映射回 θ。"""
    d0, d1, d2 = theta_to_coeffs(theta)
    d0 = min(max(d0, SLOPE_LO), SLOPE_HI)
    d1 = min(max(d1, SLOPE_LO), SLOPE_HI)
    d2 = min(max(d2, SLOPE_LO), SLOPE_HI)
    t0, t1, t2 = coeffs_to_theta(d0, d1, d2)
    return (t0, t1, t2, min(max(float(theta[3]), S_LO), S_HI))


def max_radius(theta: Sequence[float], direction: Sequence[float]) -> float:
    """沿 `direction` 從可行的 `theta` 出發，走到可行域邊界的步長。

    四道限制全是 θ 的仿射函數，所以邊界距離有封閉解，不需要二分。
    `theta` 不可行時拋錯——不可行的起點算出來的半徑沒有意義。
    """
    if not feasible(theta, tol=1e-6):
        raise ValueError(f"起點不可行：{violations(theta)}")
    base = list(theta_to_coeffs(theta)) + [float(theta[3])]
    # `theta_to_coeffs` 帶常數 1，要的是線性部分，故減掉零向量的像。
    origin = theta_to_coeffs((0.0, 0.0, 0.0))
    image = theta_to_coeffs(direction[:3])
    rate = [image[i] - origin[i] for i in range(3)] + [float(direction[3])]
    lo = [SLOPE_LO] * 3 + [S_LO]
    hi = [SLOPE_HI] * 3 + [S_HI]
    best = float("inf")
    for v, w, a, b in zip(base, rate, lo, hi):
        if w > 1e-12:
            best = min(best, (b - v) / w)
        elif w < -1e-12:
            best = min(best, (a - v) / w)
    if best == float("inf"):
        raise ValueError("這個方向不受任何限制約束；方向應為非零向量")
    return best


def apply_filter(x01: torch.Tensor, theta) -> torch.Tensor:
    """`T_θ` 本體。`x01` 為 (N,3,H,W)、[0,1]；`theta` 為長度 4 的序列或 tensor。

    `theta` 是 tensor 時梯度接得回去，所以同一條路徑既是量測也是求解。
    """
    if isinstance(theta, torch.Tensor):
        th = theta.to(device=x01.device, dtype=x01.dtype)
        t0, t1, t2, s = th[0], th[1], th[2], th[3]
    else:
        t0, t1, t2, s = (torch.as_tensor(float(v), device=x01.device,
                                         dtype=x01.dtype) for v in theta)
    m = x01.amax(dim=1, keepdim=True)
    w = torch.tensor(LUMA, device=x01.device, dtype=x01.dtype).view(1, 3, 1, 1)
    y = (x01 * w).sum(dim=1, keepdim=True)
    g = (1.0 + t0) + t1 * (1.0 - m) + t2 * (1.0 - m) * (2.0 * m - 1.0)
    return (g - s) * x01 + s * y


class ToneDesatParam:
    """`T_θ` 的載體封裝。參數是長度 4 的 tensor `(t0, t1, t2, s)`。"""

    def __init__(self, theta: Sequence[float] = THETA_CENTRE,
                 device=None, dtype=torch.float32):
        self.amplitude = 1.0
        self.theta = torch.tensor([float(v) for v in theta], device=device,
                                  dtype=dtype).requires_grad_(True)

    # ---- 載體介面 ----

    def reset(self, x01: torch.Tensor, seed: int) -> None:
        """起點不隨 `seed` 變：這個族的起點由 `theta` 明給。

        參數只有四個，重啟能換到的盆很少；要掃的是取樣點，不是隨機起點。
        `x01` 只用來對齊 device 與 dtype。
        """
        self.theta = self.theta.detach().to(device=x01.device).requires_grad_(True)

    def render(self, x01: torch.Tensor) -> torch.Tensor:
        return apply_filter(x01, self.theta)

    def params(self) -> List[torch.Tensor]:
        return [self.theta]

    def state_dict(self):
        return {"theta": self.theta.detach().clone()}

    def load_state_dict(self, state) -> None:
        self.theta = state["theta"].detach().clone().requires_grad_(True)

    @property
    def stages(self):
        return [self]

    def set_amplitude(self, a) -> None:
        """這個族沒有可縮放的幅度段：θ 本身就是幅度。

        `optimise_carrier` 的退路（`fit_caps` 整體縮放）會呼叫它。縮放 θ 會
        把解移出可行域的定義，故不提供；靜默忽略會讓退路無聲失效，所以拋錯。
        """
        raise NotImplementedError(
            "ToneDesatParam 沒有幅度段；要縮小解請沿 θ − θ_centre 取半徑")

    @torch.no_grad()
    def project(self) -> None:
        self.theta.copy_(torch.tensor(project_theta(self.theta.tolist()),
                                      device=self.theta.device,
                                      dtype=self.theta.dtype))

    # ---- 量測 ----

    def coeffs(self) -> Tuple[float, float, float]:
        return theta_to_coeffs(self.theta.detach().tolist())

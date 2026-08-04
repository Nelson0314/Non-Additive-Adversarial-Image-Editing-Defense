"""把一個擾動方向縮放到指定的感知失真。

三個腳本各自寫過一份幾乎相同的二分搜尋（`gate_suppress.random_control`、
`random_baseline_curve.scale_to_lpips`、`ray_curve.solve_scale`），
2026-08-04 合併於此。

## 為什麼對 LPIPS 二分，而不是對振幅或 RMS

三個尺度互不等價。`runs/p17_kappa_visibility/` 在同一張影像上量到：

| 擾動 | LPIPS |
|---|---|
| PGD 解 | 0.2935 |
| 同振幅的隨機 sign | 0.1497 |
| 同 RMS 的高斯 | 0.1084 |

即**同振幅時，最佳化解的可辨失真是隨機的 2–3 倍**（LEDGER 2.19）。
要匹配的是感知失真本身，故直接對 LPIPS 搜尋。這也是本專案與 PhotoGuard
的隨機對照的差別——後者匹配的是振幅（1.24）。

## 到不了就拋出

`clip` 到 [0,1] 之後 LPIPS 有飽和上限，某些 (影像, 方向) 組合就是達不到
高預算。**那是關於該方向的結果，不是錯誤**，故明確拋出讓呼叫端記下來，
不靜默取最接近的值——「達不到 0.5」與「達到了 0.5」在下游完全不同。
"""

from typing import Callable, Tuple

import torch


def solve_k(
    lpips_fn: Callable[[torch.Tensor], float],
    build: Callable[[float], torch.Tensor],
    target: float,
    tol: float = 0.005,
    iters: int = 28,
    k_max: float = 64.0,
) -> Tuple[torch.Tensor, float, float]:
    """對縮放係數 k 二分，使 `lpips_fn(build(k))` ≈ target。

    回傳 (影像, 實際 LPIPS, k)。**`build` 由呼叫端提供**，因為「把解放大
    k 倍」對不同的參數化不是同一件事：

    - 加性位置：`build(k) = clip(x + k·δ)`，精確。
    - 空間變形：正確的作法是把**位移場**乘上 k 再重新取樣。影像空間的
      線性內插是交叉淡入，只在小位移下是它的一階近似
      （`x_warp(k·f) ≈ x + k·(x_warp(f) − x)`）。

    把 `build` 外提就是為了讓這個差別由呼叫端負責，而不是在此假設一種。

    到不了 target 時拋出 `ValueError`。**那是關於該方向的結果不是錯誤**，
    故不靜默取最接近的值——「達不到 0.5」與「達到了 0.5」在下游完全不同。
    """
    if target <= 0:
        raise ValueError(f"target 必須為正，收到 {target}")
    lo, hi = 0.0, 1.0
    while lpips_fn(build(hi)) < target:
        hi *= 2.0
        if hi > k_max:
            raise ValueError(
                f"縮放到 {k_max:g} 倍仍只到 LPIPS "
                f"{lpips_fn(build(k_max)):.4f}，達不到目標 {target}")
    k = hi
    for _ in range(iters):
        k = 0.5 * (lo + hi)
        x = build(k)
        got = lpips_fn(x)
        if abs(got - target) < tol:
            return x, got, k
        lo, hi = (k, hi) if got < target else (lo, k)
    x = build(k)
    return x, lpips_fn(x), k


def scale_to_lpips(
    lpips_fn: Callable[[torch.Tensor], float],
    x01: torch.Tensor,
    direction: torch.Tensor,
    target: float,
    **kw,
) -> Tuple[torch.Tensor, float, float]:
    """`solve_k` 的加性用法：`build(k) = clip(x01 + k·direction)`。

    `direction` 不需要正規化：k 會把它縮放到目標。
    """
    return solve_k(lpips_fn, lambda k: (x01 + k * direction).clamp(0, 1),
                   target, **kw)


def gaussian_control(
    lpips_fn: Callable[[torch.Tensor], float],
    x01: torch.Tensor,
    target: float,
    seed: int,
    generator_device: str = "cpu",
    **kw,
) -> Tuple[torch.Tensor, float, float]:
    """同一個失真量、但沒有經過最佳化的高斯擾動。

    **這個條件是必要的，不是加分項。** 本專案最大的一次假陽性正是缺了它：
    E2 判定 site L 有防禦效果，`e2_phi0` 的 φ=0 對照顯示數字完全相同——
    那是 VAE 來回而不是防禦（LEDGER 7.1）。實測在 LPIPS 0.14 與 0.53 上，
    隨機方向分別取得最佳化解 60% 與 74% 的語意失效（1.18、1.20）。

    種子由呼叫端給，且應隨失真等級而變：同一組雜訊縮放到不同振幅會讓
    各級之間相關，那不是獨立的對照。
    """
    g = torch.Generator(device=generator_device).manual_seed(seed)
    noise = torch.randn(x01.shape, generator=g).to(x01.device)
    return scale_to_lpips(lpips_fn, x01, noise, target, **kw)


def lpips_against(suite, x01: torch.Tensor) -> Callable[[torch.Tensor], float]:
    """把 `MetricSuite` 包成本模組要的 `lpips_fn`。"""
    def fn(x: torch.Tensor) -> float:
        with torch.no_grad():
            return float(suite.pairwise(x01, x)["lpips"])
    return fn

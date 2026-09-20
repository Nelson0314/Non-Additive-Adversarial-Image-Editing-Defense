"""把整個色彩預算結構上鎖在 HSV 的明度上：色相與飽和度逐位元不變。

為什麼要有這一族
────────────────────────────────────────────────────────────────────
AdvCF 的三條逐通道曲線互相獨立，振幅一大，三條曲線的差就是色相偏移——
預算花在哪個方向由求解端決定，而求解端沒有、也不該有自然度項（自然要由
結構保證，不是由損失保證）。本族把方向這件事從求解端拿掉：

    m = max(R,G,B)，  T(c) = c · f(m) / m

`f` 與 AdvCF 用同一條 64 段單調分段線性曲線（同一個 `advcf` 界、同一個
`radius`），所以參數量、斜率上下限、固定 0 與 1 兩端這些性質都一樣。

四個由構造保證的性質
────────────────────────────────────────────────────────────────────
1. **色相不變**：三個通道乘同一個比例。
2. **HSV 飽和度不變**：`S = (m − min)/m`，分子分母同乘一個比例後不變。
3. **不會溢出**：`c ≤ m` 且 `f(m) ≤ 1`，故 `T(c) ≤ f(m) ≤ 1`，不需要鉗回。
4. **明度單調**：`f` 單調不減，所以原本比較亮的顏色不會變得比較暗。

程式裡沒有除法
────────────────────────────────────────────────────────────────────
`f(m)/m` 逐段展開後分母消得掉：第 0 段 `f(m) = θ₀·m·K/s`，比例是常數
`θ₀·K/s`，與 `m` 無關；第 k ≥ 1 段的 `m ≥ k/K > 0`。故 `m = 0`（純黑）
不需要特例，也不會產生 0/0。
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch


class ValueCurveParam:
    """θ 為 (1,1,K)：一條吃 `max(R,G,B)` 的單調分段線性曲線，全域套用。"""

    name = "value_curve"

    def __init__(self, radius: float = 5.0, pieces: int = 64,
                 bound_mode: str = "advcf",
                 apply_where: Optional[torch.Tensor] = None,
                 init_jitter: float = 0.0):
        if pieces < 2:
            raise ValueError(f"pieces 必須至少為 2，收到 {pieces}")
        if bound_mode not in ("symmetric", "advcf"):
            raise ValueError(
                f"bound_mode 只能是 'symmetric' 或 'advcf'，收到 {bound_mode!r}")
        if not 0.0 <= float(init_jitter) <= 1.0:
            raise ValueError(f"init_jitter 要落在 [0,1]，收到 {init_jitter}")
        self.radius = radius
        self.pieces = pieces
        self.bound_mode = bound_mode
        self.apply_where = apply_where
        self.init_jitter = float(init_jitter)
        self.theta: Optional[torch.Tensor] = None

    # ---- 介面（與 ColorCurveParam 相同，optimise_carrier 直接吃）----

    def reset(self, x01: torch.Tensor, seed: int) -> None:
        k = self.pieces
        flat = torch.full((1, 1, k), 1.0 / k,
                          device=x01.device, dtype=x01.dtype)
        if self.init_jitter > 0:
            lo, hi = self.bounds()
            generator = torch.Generator(device="cpu").manual_seed(int(seed))
            u = torch.rand((1, 1, k), generator=generator).to(x01.device, x01.dtype)
            flat = (1.0 - self.init_jitter) * flat \
                + self.init_jitter * (lo + (hi - lo) * u)
        self.theta = flat.clone().requires_grad_(True)

    def render(self, x01: torch.Tensor) -> torch.Tensor:
        out = x01 * self._ratio(x01, self.theta)
        if self.apply_where is None:
            return out
        w = self.apply_where.to(device=out.device, dtype=out.dtype)
        return w * out + (1.0 - w) * x01

    def params(self) -> List[torch.Tensor]:
        return [self.theta]

    def state_dict(self):
        return {"theta": self.theta.detach().clone()}

    def load_state_dict(self, state):
        self.theta = state["theta"].detach().clone().requires_grad_(True)

    @property
    def stages(self):
        return [self]

    def set_amplitude(self, a):
        if isinstance(a, (list, tuple)):
            if len(a) != 1:
                raise ValueError("這個載體只有一段，逐段指定時長度要是 1")
            a = a[0]
        self.amplitude = float(a)

    @torch.no_grad()
    def project(self) -> None:
        lo, hi = self.bounds()
        self.theta.clamp_(lo, hi)

    def set_radius(self, r: float) -> None:
        self.radius = r

    def step_scale(self) -> float:
        lo, hi = self.bounds()
        return hi - lo

    # ---- 構造 ----

    def bounds(self) -> Tuple[float, float]:
        k = float(self.pieces)
        span = 1.0 + max(0.0, float(self.radius))
        if self.bound_mode == "advcf":
            return 1.0 / k, span / k
        return 1.0 / (k * span), span / k

    def _ratio(self, x01: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        """(N,3,H,W) → (N,1,H,W) 的比例場 `f(m)/m`，不含除法。

        第 k 段： f(m) = [C_k/K + (m − k/K)·θ_k]·K/s，其中 C_k = Σ_{i<k} θ_i。
        改寫成  f(m)/m = [(C_k/K − k·θ_k/K)/m + θ_k]·K/s。
        括號裡的 `1/m` 只在 `C_k − k·θ_k ≠ 0` 時出現，而 k = 0 時
        `C_0 = 0`、`k·θ_0 = 0`，整項為 0，所以第 0 段的比例是常數 `θ_0·K/s`。
        k ≥ 1 的段上 `m ≥ k/K`，分母有正下界。實作上直接用
        `m_safe = max(m, k/K)`：在各段上這與 `m` 相同（邊界 `m = k/K` 取等），
        只有第 0 段被夾成 `1/K`，而第 0 段的係數恰為 0，夾了也不影響值。
        """
        k = self.pieces
        th = theta.to(x01.dtype)[0, 0]                     # (K,)
        m = x01.max(dim=1, keepdim=True).values            # (N,1,H,W)
        idx = (m * k).floor().clamp_(0, k - 1).long()      # (N,1,H,W)
        s = th.sum()
        cum = torch.cat([th.new_zeros(1), th.cumsum(0)[:-1]])   # C_k
        steps = torch.arange(k, device=th.device, dtype=th.dtype)
        offset = (cum - steps * th) / k                    # (K,) 各段的截距
        off = offset[idx]
        slope = th[idx]
        floor_m = idx.to(x01.dtype) / k
        m_safe = torch.maximum(m, torch.clamp(floor_m, min=1.0 / k))
        return (off / m_safe + slope) * (k / s)

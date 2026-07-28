"""淨化算子 𝒫 — spec §5.1、§9.2。

spec §5.1 要求淨化寫進訓練目標而非事後量測，這需要淨化在訓練時可微。
但 JPEG 量化與 GrIDPure 都不可微，故每個算子分成兩個實作：

- `forward`      訓練用，可微（真實實作或其可微代理）
- `evaluate`     評測用，真實實作，不要求可微

**代理與真實實作的差距必須在報告中明列，不得省略**（spec §5.1 末段）。
`Purifier.proxy_gap` 提供直接量測此差距的方法，使該聲明有數字支撐而非
只是免責聲明。

本階段納入的算子與其可微性：

| 算子 | 訓練 | 評測 | 代理方式 |
|---|---|---|---|
| identity | 可微 | 同 | 無需代理（𝒫 必須含恆等算子，見 spec §5.1） |
| gaussian_blur | 可微 | 同 | 無需代理 |
| gaussian_noise | 可微 | 同 | 無需代理 |
| jpeg | 不可微 | 真實 | 直通估計（straight-through） |
| quantize | 不可微 | 真實 | 直通估計 |

GrIDPure 需要額外的擴散模型推論，成本遠高於上列各項，列為後續工作，
不在本階段的 𝒫 內。此為範圍限制，須在報告中載明。
"""

import io
import math
from typing import Dict, List

import torch
import torch.nn.functional as F


def _gaussian_kernel1d(sigma: float, device, dtype) -> torch.Tensor:
    radius = max(1, int(math.ceil(3.0 * sigma)))
    xs = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    k = torch.exp(-(xs**2) / (2.0 * sigma**2))
    return k / k.sum()


def gaussian_blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """可分離高斯模糊。sigma ≤ 0 時回傳原張量。"""
    if sigma <= 0:
        return x
    k = _gaussian_kernel1d(sigma, x.device, x.dtype)
    c = x.shape[1]
    pad = (k.numel() - 1) // 2
    xh = F.conv2d(
        F.pad(x, (pad, pad, 0, 0), mode="reflect"),
        k.view(1, 1, 1, -1).expand(c, 1, 1, -1), groups=c,
    )
    return F.conv2d(
        F.pad(xh, (0, 0, pad, pad), mode="reflect"),
        k.view(1, 1, -1, 1).expand(c, 1, -1, 1), groups=c,
    )


def gaussian_noise(x: torch.Tensor, sigma: float, seed: int = None) -> torch.Tensor:
    """加性高斯噪聲。seed 固定時同一強度可重現，供淨化強度掃描使用。"""
    if sigma <= 0:
        return x
    g = None if seed is None else torch.Generator(x.device).manual_seed(seed)
    n = torch.randn(x.shape, generator=g, device=x.device, dtype=x.dtype)
    return (x + sigma * n).clamp(0, 1)


def quantize_real(x: torch.Tensor, levels: int) -> torch.Tensor:
    """真實量化，不可微（round 的導數幾乎處處為零）。"""
    q = float(levels - 1)
    return torch.round(x.clamp(0, 1) * q) / q


def straight_through(x: torch.Tensor, hard: torch.Tensor) -> torch.Tensor:
    """前向取 `hard`、反向視為對 `x` 的恆等映射。

    寫法必須是 `hard.detach() + (x - x.detach())` 而不是常見的
    `x + (hard - x).detach()`。兩者在實數上等價，在 float32 上不等價：
    後者要算 `x + hard - x`，兩次捨入後不保證等於 `hard`，實測 JPEG 上
    確實出現逐元素差異。前者的 `x - x.detach()` 逐元素精確為 0，故前向
    位元等同於 `hard`。
    """
    return hard.detach() + (x - x.detach())


def quantize_proxy(x: torch.Tensor, levels: int) -> torch.Tensor:
    """量化的直通估計：前向為真實量化，反向視為恆等。

    這是代理與真實實作**唯一**的差異來源：前向數值位元等同，只有梯度不同。
    故此代理不引入前向誤差，`proxy_gap` 對 quantize 必為 0。
    """
    return straight_through(x, quantize_real(x, levels))


def jpeg_real(x: torch.Tensor, quality: int) -> torch.Tensor:
    """真實 JPEG 編解碼。經 PIL，故不可微且必須離開計算圖。"""
    from PIL import Image
    import numpy as np

    out = []
    for i in range(x.shape[0]):
        arr = (x[i].detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(
            np.uint8
        )
        buf = io.BytesIO()
        Image.fromarray(arr).save(buf, format="JPEG", quality=int(quality))
        buf.seek(0)
        dec = np.asarray(Image.open(buf).convert("RGB")).astype(np.float32) / 255.0
        out.append(torch.from_numpy(dec).permute(2, 0, 1))
    return torch.stack(out).to(x.device, x.dtype)


def jpeg_proxy(x: torch.Tensor, quality: int) -> torch.Tensor:
    """JPEG 的直通估計：前向呼叫真實編解碼，反向視為恆等。

    **此代理的梯度是錯的**，而非近似的：真實 JPEG 的區塊 DCT 量化在梯度
    上與恆等映射毫無關係。採用它的理由是前向數值完全正確，優化過程看到的
    是真實的 JPEG 輸出；代價是梯度方向只反映「淨化後的圖長什麼樣」而非
    「淨化本身如何反應擾動」。此限制須在報告中明列。
    """
    return straight_through(x, jpeg_real(x, quality))


class Purifier:
    """單一淨化設定。`forward` 供訓練、`evaluate` 供評測。"""

    def __init__(self, kind: str, strength: float = 0.0, seed: int = None):
        self.kind = kind
        self.strength = strength
        self.seed = seed
        if kind not in ("identity", "blur", "noise", "jpeg", "quantize"):
            raise ValueError(f"未知的淨化算子 {kind!r}")

    @property
    def differentiable(self) -> bool:
        """代理是否提供了真實梯度。jpeg 與 quantize 為直通估計，故為 False。"""
        return self.kind in ("identity", "blur", "noise")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.kind == "identity":
            return x
        if self.kind == "blur":
            return gaussian_blur(x, self.strength)
        if self.kind == "noise":
            return gaussian_noise(x, self.strength, self.seed)
        if self.kind == "jpeg":
            return jpeg_proxy(x, int(self.strength))
        return quantize_proxy(x, int(self.strength))

    @torch.no_grad()
    def evaluate(self, x: torch.Tensor) -> torch.Tensor:
        if self.kind == "identity":
            return x
        if self.kind == "blur":
            return gaussian_blur(x, self.strength)
        if self.kind == "noise":
            return gaussian_noise(x, self.strength, self.seed)
        if self.kind == "jpeg":
            return jpeg_real(x, int(self.strength))
        return quantize_real(x, int(self.strength))

    def proxy_gap(self, x: torch.Tensor) -> float:
        """代理與真實實作的前向最大絕對差，供報告引用。"""
        with torch.no_grad():
            return float((self.forward(x) - self.evaluate(x)).abs().max())

    def __repr__(self) -> str:
        return f"Purifier({self.kind}, strength={self.strength})"


def default_train_set() -> List[Purifier]:
    """訓練期的 𝒫。**必須包含恆等算子**（spec §5.1）。

    強度取各算子的中等值：訓練目標是耐受一般淨化，不是耐受某個極端設定。
    強度掃描留給 E3 的評測階段。
    """
    return [
        Purifier("identity"),
        Purifier("blur", 1.0),
        Purifier("jpeg", 75),
    ]


def eval_sweep() -> Dict[str, List[Purifier]]:
    """E3 的淨化強度掃描。每個算子由弱到強，含強度 0 的對照。"""
    return {
        "blur": [Purifier("blur", s) for s in (0.0, 0.5, 1.0, 1.5, 2.0, 3.0)],
        "jpeg": [Purifier("jpeg", q) for q in (95, 85, 75, 60, 45, 30)],
        "noise": [Purifier("noise", s, seed=0) for s in (0.0, 0.01, 0.02, 0.04, 0.08)],
        "quantize": [Purifier("quantize", n) for n in (256, 64, 32, 16, 8)],
    }

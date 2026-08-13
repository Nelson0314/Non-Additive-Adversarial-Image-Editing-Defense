"""site F — 紋理重相位 —— 非加性、停在像素空間、由構造保證恆等。

    x_def = OLA( irfft2( rfft2(w*P_b) * exp(i * g_b * m_w * theta_b) ) * w ) / OLA(w^2)

`P_b` 是以 hop = block/2 取出的重疊區塊，`w` 是 2D 週期 Hann 窗。相位偏移
`theta_b` 是唯一的可學參數；`g_b`（紋理度）與 `m_w`（徑向頻率）是由原圖與
網格算出的固定閘，不參與最佳化。

為什麼是相位
────────────────────────────────────────────────────────────────────
Random Phase Noise（Galerne, Gousseau & Morel, IEEE TIP 2011）證明：把影像的
傅立葉相位隨機化，微紋理的知覺外觀不變——紋理由幅度譜刻畫，相位只決定紋素
的擺放。本模塊把「隨機」換成「最佳化」。

其動機是三筆既有結果的共同機制（FND-004／013／026）：加性擾動把能量放在
高頻，VAE encoder 敏感而人眼不敏感；既有的非加性形式（warp、latent 球）
產生自然的低頻結構改動，VAE 忠實編碼它，於是每單位可見失真換到的位移少。
相位擾動要押的落差是——DISTS 的設計目標明文包含 tolerant to texture
resampling（Ding et al. 2020），人眼判紋理亦靠統計量，而 VAE encoder 的
卷積特徵對相位敏感。

三個由構造而非懲罰項保證的性質
────────────────────────────────────────────────────────────────────
1. **theta = 0 時逐位等於原圖。** 分析窗與合成窗各乘一次、再除以 `OLA(w^2)`,
   theta=0 時分子恰為 `OLA(w^2 * x)`，與分母逐位相消。不依賴 COLA 條件，故
   換窗型或換 hop 都不會靜默破壞恆等——這一點由測試實測。
2. **幅度譜逐位保留。** 係數乘上單位模的複數 `exp(i*theta)` 而非走
   `abs`／`angle` 再重組：後者在零幅度處梯度未定義，前者處處可微且
   `|X * e^{i*theta}| = |X|` 是代數恆等式。
3. **輸出為實數。** `rfft2`／`irfft2` 只存半平面，實數性由型別保證。
   但半平面裡 `fx = 0` 與 `fx = N/2` 兩行自身必須對 `fy` 共軛，對它們逐格
   施加獨立相位會破壞該關係（輸出仍是實數，只是那兩行的幅度不再保留）。
   故 `m_w` 在這兩行取 0——寧可少 2/17 的自由度，也不要一個只在兩行上
   失效、且不會有任何症狀的近似。

已知的不精確之處（要量，不要假設）
────────────────────────────────────────────────────────────────────
相鄰區塊各自改相位後重疊相加會部分抵消，故「局部幅度譜保留」在整張圖的
層級是近似而非恆等。`amplitude_deviation()` 回報該偏差，供報告列出。構造上
它應該接近 0；實測偏大即代表本模塊在造新能量而不是重排相位，那會讓它退化
成被紋理遮蔽的加性高頻噪聲（規格 §6 風險一）。
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.residual.base import ResidualModule


def hann2d(n: int, device, dtype) -> torch.Tensor:
    """(n, n) 的 2D 週期 Hann 窗，可分離構造。

    取週期（periodic）而非對稱版本：週期 Hann 在 hop = n/2 下相鄰窗和恰為 1,
    對稱版本不是。本模塊的恆等性由 `OLA(w^2)` 正規化保證而不依賴此性質，
    但保留週期版本使 `OLA(w^2)` 不出現接近零的格。
    """
    k = torch.arange(n, device=device, dtype=dtype)
    w = 0.5 - 0.5 * torch.cos(2.0 * math.pi * k / n)
    return torch.outer(w, w)


def radial_gate(block: int, r_min: float, device, dtype) -> torch.Tensor:
    """(block, block//2+1) 的徑向頻率閘。

    低頻格帶著區塊的位置與結構，動它會在重疊相加後產生接縫，故歸一化半徑
    低於 `r_min` 的格取 0。`fx = 0` 與 `fx = block//2` 兩行一律取 0，理由見
    模組 docstring 第 3 點。
    """
    fy = torch.fft.fftfreq(block, device=device, dtype=dtype) * 2.0   # [-1, 1)
    fx = torch.fft.rfftfreq(block, device=device, dtype=dtype) * 2.0  # [0, 1]
    r = torch.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)
    m = (r >= r_min).to(dtype)
    m[:, 0] = 0.0
    m[:, -1] = 0.0
    return m


def texture_gate(
    x01: torch.Tensor,
    block: int,
    hop: int,
    energy_quantile: float = 0.5,
    eps: float = 1e-8,
) -> torch.Tensor:
    """逐區塊的紋理度 g in [0,1]，形狀 (B, L)。固定不可學。

        g = (1 - coherence^2) * clamp(energy / energy_ref, 0, 1)

    `coherence = (l1 - l2)/(l1 + l2)` 取自結構張量。邊緣的梯度方向一致,
    coherence 接近 1；紋理的梯度方向雜亂，接近 0。平坦區兩者都小，靠第二個
    因子壓掉——那裡任何改動都直接可見，而 coherence 在該處是零除的雜訊。

    `energy_ref` 取該影像自己的分位數而非固定常數：梯度能量的絕對值隨影像
    內容差好幾個數量級，固定常數會讓不同影像拿到不同的有效閘，而那個差異
    不會有症狀（與 DEC-021 取第 0 次迭代作正規化常數同型）。
    """
    lum = (0.299 * x01[:, 0] + 0.587 * x01[:, 1] + 0.114 * x01[:, 2]).unsqueeze(1)
    kx = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        device=x01.device,
        dtype=x01.dtype,
    ).view(1, 1, 3, 3) / 8.0
    ky = kx.transpose(-1, -2)
    gx = F.conv2d(F.pad(lum, (1, 1, 1, 1), mode="reflect"), kx)
    gy = F.conv2d(F.pad(lum, (1, 1, 1, 1), mode="reflect"), ky)

    pad = block // 2
    comps = []
    for t in (gx * gx, gx * gy, gy * gy):
        t = F.pad(t, (pad, pad, pad, pad), mode="reflect")
        # 區塊內平均：unfold 後對區塊維取平均，與相位區塊完全同一組位置
        comps.append(F.unfold(t, kernel_size=block, stride=hop).mean(dim=1))
    jxx, jxy, jyy = comps

    tr = jxx + jyy
    disc = torch.sqrt(torch.clamp(((jxx - jyy) * 0.5) ** 2 + jxy ** 2, min=0.0))
    coh = (2.0 * disc) / (tr + eps)
    ref = torch.quantile(tr, energy_quantile, dim=1, keepdim=True).clamp_min(eps)
    return (1.0 - coh ** 2) * torch.clamp(tr / ref, 0.0, 1.0)


def rephase_blocks(blocks: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
    """對區塊做相位旋轉。`blocks` (..., n, n)，`shift` 可廣播到 (..., n, n//2+1)。

    乘上單位模的複數而非拆 `abs`／`angle`：`|X * e^{i*theta}| = |X|` 是代數
    恆等式，且 `angle` 在原點的梯度未定義。
    """
    spec = torch.fft.rfft2(blocks, norm="ortho")
    rot = torch.complex(torch.cos(shift), torch.sin(shift))
    return torch.fft.irfft2(spec * rot, s=blocks.shape[-2:], norm="ortho")


class PhaseResidual(ResidualModule):
    """site F。phi = theta，形狀 (1, L, block, block//2+1)，RGB 三通道共用。

    通道共用相位是依 2026-08-13 的顏色結論：等亮度色度擾動的位移比 RGB 獨立
    低 31%，而 RGB 獨立那組的效果全部來自它順帶改變的亮度。共用相位使擾動
    落在結構而非色度上，不白付色偏的代價。
    """

    site = "F"

    def __init__(
        self,
        size: int = 512,
        block: int = 32,
        hop: Optional[int] = None,
        r_min: float = 0.25,
        theta_max: float = math.pi,
        energy_quantile: float = 0.5,
        init_std: float = 0.0,
        seed: Optional[int] = None,
    ):
        super().__init__()
        if block % 2 != 0:
            raise ValueError(f"block 必須是偶數，收到 {block}")
        hop = block // 2 if hop is None else hop
        if hop > block:
            raise ValueError(f"hop ({hop}) 大於 block ({block}) 會留下未覆蓋的像素")
        if not (0.0 < theta_max <= math.pi):
            raise ValueError(f"theta_max 必須落在 (0, pi]，收到 {theta_max}")

        self.size = size
        self.block = block
        self.hop = hop
        self.pad = block // 2
        self.r_min = r_min
        self.theta_max = theta_max
        self.energy_quantile = energy_quantile

        padded = size + 2 * self.pad
        if (padded - block) % hop != 0:
            raise ValueError(
                f"size={size}、block={block}、hop={hop} 下 unfold 無法整除，"
                f"邊緣會被靜默丟棄"
            )
        side = (padded - block) // hop + 1
        self.n_blocks = side * side
        self.padded = padded

        nbins = (block, block // 2 + 1)
        if init_std > 0:
            gen = torch.Generator(device="cpu")
            if seed is not None:
                gen.manual_seed(seed)
            init = torch.randn(1, self.n_blocks, *nbins, generator=gen) * init_std
            init = init.clamp(-theta_max, theta_max)
        else:
            init = torch.zeros(1, self.n_blocks, *nbins)
        self.theta = nn.Parameter(init)

        self.register_buffer("window", torch.zeros(block, block), persistent=False)
        self.register_buffer("freq_gate", torch.zeros(*nbins), persistent=False)
        self.register_buffer("tex_gate", torch.zeros(1, self.n_blocks), persistent=False)
        self._gates_ready = False

    # ---- 閘 ----

    def prepare_gates(self, x01: torch.Tensor) -> None:
        """由原圖算出兩個固定閘。必須在第一次前向之前呼叫一次。

        閘取自**原圖**而非當前的防禦圖：閘若隨 phi 移動，`g_b` 會變成優化目標
        的一部分（把擾動搬到閘自己放寬的地方），那不是本模塊要量的東西。
        """
        device, dtype = x01.device, x01.dtype
        self.window = hann2d(self.block, device, dtype)
        self.freq_gate = radial_gate(self.block, self.r_min, device, dtype)
        self.tex_gate = texture_gate(
            x01, self.block, self.hop, self.energy_quantile
        ).detach()
        self._gates_ready = True

    def gate(self) -> torch.Tensor:
        """(1, L, 1, 1) 與 (block, nbins) 廣播後的合成閘，供診斷與前向共用。"""
        return self.tex_gate[..., None, None] * self.freq_gate

    def active_fraction(self) -> float:
        """紋理閘的有效面積佔比（規格 §6 風險三的前置量測）。"""
        return float(self.tex_gate.mean())

    # ---- 前向 ----

    def _fold_norm(self, batch: int) -> torch.Tensor:
        w2 = (self.window ** 2).reshape(1, -1, 1).expand(batch, -1, self.n_blocks)
        out = F.fold(
            w2,
            output_size=(self.padded, self.padded),
            kernel_size=self.block,
            stride=self.hop,
        )
        return out[..., self.pad:self.pad + self.size, self.pad:self.pad + self.size]

    def pixel_residual(self, x01: torch.Tensor) -> Optional[torch.Tensor]:
        if not self.enabled:
            return x01
        if not self._gates_ready:
            raise RuntimeError(
                "prepare_gates() 未呼叫。閘必須由原圖算出並固定，"
                "延後到第一次前向會讓它跟著防禦圖漂移"
            )
        b, c, h, w = x01.shape
        if h != self.size or w != self.size:
            raise ValueError(f"影像尺寸 {h}x{w} 與建構時的 size={self.size} 不符")

        xp = F.pad(x01, (self.pad,) * 4, mode="reflect")
        blocks = F.unfold(xp, kernel_size=self.block, stride=self.hop)
        blocks = blocks.view(b, c, self.block, self.block, self.n_blocks)
        blocks = blocks.permute(0, 1, 4, 2, 3)                       # (B,C,L,n,n)

        win = self.window
        shift = torch.clamp(self.theta, -self.theta_max, self.theta_max)
        shift = (shift * self.gate()).unsqueeze(1)                   # (1,1,L,n,nb)

        out = rephase_blocks(blocks * win, shift) * win
        out = out.permute(0, 1, 3, 4, 2).reshape(
            b, c * self.block * self.block, self.n_blocks
        )
        folded = F.fold(
            out,
            output_size=(self.padded, self.padded),
            kernel_size=self.block,
            stride=self.hop,
        )
        folded = folded[..., self.pad:self.pad + self.size,
                        self.pad:self.pad + self.size]
        norm = self._fold_norm(b).clamp_min(1e-8)
        return folded / norm

    # ---- 診斷 ----

    @torch.no_grad()
    def amplitude_deviation(self, x01: torch.Tensor) -> float:
        """整張圖層級的局部幅度譜相對偏差。構造上應接近 0（模組 docstring）。"""
        x_def = self.pixel_residual(x01)

        def spec(t):
            tp = F.pad(t, (self.pad,) * 4, mode="reflect")
            bl = F.unfold(tp, kernel_size=self.block, stride=self.hop)
            bl = bl.view(
                t.shape[0], t.shape[1], self.block, self.block, self.n_blocks
            ).permute(0, 1, 4, 2, 3)
            return torch.fft.rfft2(bl * self.window, norm="ortho").abs()

        a, b = spec(x01), spec(x_def)
        return float((a - b).norm() / a.norm().clamp_min(1e-12))

    def linf(self) -> float:
        return float(self.theta.detach().abs().max())

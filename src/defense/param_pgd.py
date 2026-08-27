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
                 hop: Optional[int] = None,
                 r_max: float = float("inf"),
                 radius: float = math.pi, energy_quantile: float = 0.5,
                 keep: Optional[torch.Tensor] = None, gl_iters: int = 0,
                 pixel_gate_sigma: float = 0.0, gain_ratio: float = 0.0,
                 phase_on: bool = True, gate_edge_power: float = 1.0,
                 freq_weight: str = "binary",
                 freq_weight_power: float = 1.0,
                 gain_weight: str = "shared",
                 channels: str = "rgb",
                 spectral_floor: float = 0.0,
                 floor_gate: str = "uniform",
                 theta_budget: float = 0.0,
                 coarsen: int = 1):
        self.size, self.block, self.r_min = size, block, r_min
        # None = block//2，逐位元等於加這個參數之前。更小的 hop 讓每個
        # 像素被更多區塊覆蓋，相鄰區塊獨立旋轉留下的接縫因此被平均掉。
        self.hop = hop
        self.r_max = r_max
        self.energy_quantile = energy_quantile
        # 紋理閘壓制邊緣那個因子的指數，預設 1.0 逐位元等於加它之前。
        # 理由見 `PhaseResidual.__init__`。
        if gate_edge_power < 0:
            raise ValueError(
                f"gate_edge_power 不可為負，收到 {gate_edge_power}")
        self.gate_edge_power = gate_edge_power
        # 頻率閘的知覺權重。"binary" = 二值帶通，逐位元等於加它之前。
        # 名字的合法性由 `PhaseResidual` 檢查，這裡只轉交。
        self.freq_weight = freq_weight
        self.freq_weight_power = freq_weight_power
        self.gain_weight = gain_weight
        self.channels = channels
        self.spectral_floor = spectral_floor
        # 加法項的價目表要不要隨區塊變。合法性由 `PhaseResidual` 檢查，
        # 這裡只轉交。`uniform` 逐位元等於加這個旋鈕之前。
        self.floor_gate = floor_gate
        # 幅度相依的相位上限。合法性由 `PhaseResidual` 檢查，這裡只轉交。
        # 0 = 關閉，逐位元等於加這個旗標之前。
        self.theta_budget = theta_budget
        # 三個空間場的視窗網格解析度。1 = 逐位元等於加這個旋鈕之前。
        # 合法性由 `PhaseResidual` 檢查，這裡只轉交。
        self.coarsen = coarsen
        # **radius 本身不封頂**，封頂只發生在傳給 `theta_max` 的那一刻。
        # 2026-08-21 之前這裡是 `min(radius, pi)`，於是 `--radius 3.5` 與
        # `--radius 4.5` 其實跑的是同一個 theta_max = pi——sigma 掃描看到的
        # 「theta >= 3 之後 DISTS 卡住」有一部分是這個夾取造成的，不全是相位
        # 的週期性。增益沒有週期性，它的上界必須跟著 radius 走，故分開處理。
        self.radius = radius
        self.keep = keep
        self.gl_iters = gl_iters
        # > 0 時在重疊相加之後再乘一層逐像素紋理遮罩（2026-08-20）。
        # 預設 0 = 關閉，此時逐位元與加這個選項之前相同。
        self.pixel_gate_sigma = pixel_gate_sigma
        # 幅度也可學（2026-08-21）。`gain_ratio` 把單一的強度旋鈕綁到兩個
        # 參數上：`gain_max = radius * gain_ratio`。綁在一起是為了讓既有的
        # 掃描與二分搜尋機制（`fit_to_budget`）不必改成二維搜尋。
        # `phase_on = False` 時 theta 凍結在 0，即「純幅度」變體。
        if gain_ratio < 0:
            raise ValueError(f"gain_ratio 不可為負，收到 {gain_ratio}")
        self.gain_ratio = gain_ratio
        self.phase_on = phase_on
        # 三個自由度：theta（相位）、gain（幅度）、floor（頻譜加性下限）。
        # 三個全關才是真的沒有東西可學。**這道檢查原本只看前兩個**，是在加性
        # 下限存在之前寫的，於是「相位與幅度都不動、只留下限」這一格被擋住
        # ——而那正是加性裁決底下唯一沒跑過的對照。
        if not phase_on and gain_ratio <= 0 and spectral_floor <= 0:
            raise ValueError(
                "phase_on=False、gain_ratio=0 且 spectral_floor=0 時"
                "沒有任何自由度")
        self.module: Optional[PhaseResidual] = None

    def reset(self, x01: torch.Tensor, seed: int) -> None:
        self.module = PhaseResidual(
            size=self.size, block=self.block, hop=self.hop,
            r_min=self.r_min,
            r_max=self.r_max,
            theta_max=min(self.radius, math.pi),
            energy_quantile=self.energy_quantile,
            gl_iters=self.gl_iters, pixel_gate_sigma=self.pixel_gate_sigma,
            gain_max=self.radius * self.gain_ratio,
            gate_edge_power=self.gate_edge_power,
            freq_weight=self.freq_weight,
            freq_weight_power=self.freq_weight_power,
            gain_weight=self.gain_weight,
            channels=self.channels,
            spectral_floor=self.spectral_floor,
            floor_gate=self.floor_gate,
            theta_budget=self.theta_budget,
            coarsen=self.coarsen,
        ).to(device=x01.device, dtype=x01.dtype)
        self.module.prepare_gates(x01, keep=self.keep)

    def render(self, x01: torch.Tensor) -> torch.Tensor:
        return self.module.pixel_residual(x01).clamp(0.0, 1.0)

    def params(self) -> List[torch.Tensor]:
        out = [self.module.theta] if self.phase_on else []
        if self.gain_ratio > 0:
            out.append(self.module.gain)
        if self.spectral_floor > 0:
            out.append(self.module.floor)
        return out

    @torch.no_grad()
    def project(self) -> None:
        if self.phase_on:
            t = min(self.radius, math.pi)
            self.module.theta.clamp_(-t, t)
            if self.module.theta_cap is not None:
                # 幅度相依的上限也是可行集的一部分。**投影到可行集而不是只在
                # 前向夾**：只在前向夾的話，被夾住的座標梯度是零，PGD 會把
                # 參數推到界外再也回不來，而報表上看不出來。
                cap = self.module.theta_cap
                torch.minimum(self.module.theta, cap, out=self.module.theta)
                torch.maximum(self.module.theta, -cap, out=self.module.theta)
        if self.gain_ratio > 0:
            g = self.radius * self.gain_ratio
            self.module.gain.clamp_(-g, g)
        if self.spectral_floor > 0:
            # 係數夾在 [-1, 1]，實際加上去的量是它乘價目表再乘
            # `spectral_floor`。負值等於相位翻轉 pi，不需要另設相位參數。
            self.module.floor.clamp_(-1.0, 1.0)

    def set_radius(self, r: float) -> None:
        # **相位封頂在 pi，增益不封頂**——相位是週期量，增益不是。
        self.radius = r
        if self.module is not None:
            self.module.theta_max = min(r, math.pi)
            self.module.gain_max = r * self.gain_ratio


class RandomPhaseParam(PhaseParam):
    """同幅度的隨機相位，即 RPN 本身。**不最佳化**，只在 reset 時抽一次。

    FND-004 與 FND-018 兩次都是被「贏不過同失真隨機」擋下來的。此對照組
    自第一天存在，不是事後補的。
    """

    name = "phase_rand"

    def reset(self, x01: torch.Tensor, seed: int) -> None:
        super().reset(x01, seed)
        gen = torch.Generator(device="cpu").manual_seed(seed)
        r = min(self.radius, math.pi)
        init = torch.randn(self.module.theta.shape, generator=gen) * r
        with torch.no_grad():
            self.module.theta.copy_(
                init.clamp(-r, r).to(
                    device=x01.device, dtype=x01.dtype)
            )

    def params(self) -> List[torch.Tensor]:
        return []


class ShadingParam:
    """φ = m（粗網格），`x_def = clamp(x · exp(upsample(m)), 0, 1)`。

    `docs/reference/SURVEY_ARCHITECTURE.md` 候選二：一個**極低頻的乘性增益
    場**，是「明暗／照明」而不是「雜訊」。16×16 的係數雙三次上採樣到 512²，
    帶寬 `f_n ≲ 0.03`——實測場的能量加權頻率半徑是 0.024–0.026
    （`runs/shading_field_cost/`）。

    它同時對準兩個失效機制，是三個候選裡唯一打兩個的：

    - **模糊拿不走它的能量**，因為它的能量本來就不在模糊拿得走的地方
      （σ=1 的高斯模糊對 `f_n < 0.03` 幾乎是恆等）。
    - **裁切放大搬不動它**：`f_n ≲ 0.02` 是 1.2488× 放大之後唯一還與自己
      正相關的帶（`runs/ip2p_residual_signature/crop_lowband.csv`）。

    三個構造上的選擇，各有理由：

    - **零初始化即恆等輸出**（`exp(0) = 1`），與相位 `θ = 0` 同性質。
    - **單通道（消色差）**。逐通道的彩色明暗場會撞上已否決的 `顏色通道`；
      彩色版只能當消融，且要主動引用該否決。
    - **半徑夾在 `m` 本身**，不是夾在上採樣之後——雙三次會過衝，夾在粗網格上
      預算才有定義；過衝造成的超出由輸出的 `clamp` 收掉。

    參數量 256（16×16），比現行的 59 萬少三個數量級，sign PGD 直接適用。
    """

    name = "shading"

    def __init__(self, radius: float = 0.10, grid: int = 16):
        if grid < 2:
            raise ValueError(f"grid 必須至少為 2，收到 {grid}")
        self.radius = radius
        self.grid = grid
        self.m: Optional[torch.Tensor] = None

    def reset(self, x01: torch.Tensor, seed: int) -> None:
        self.m = torch.zeros(1, 1, self.grid, self.grid,
                             device=x01.device, dtype=x01.dtype,
                             requires_grad=True)

    def render(self, x01: torch.Tensor) -> torch.Tensor:
        m = torch.nn.functional.interpolate(
            self.m, size=x01.shape[-2:], mode="bicubic", align_corners=False)
        return (x01 * torch.exp(m)).clamp(0.0, 1.0)

    def params(self) -> List[torch.Tensor]:
        return [self.m]

    @torch.no_grad()
    def project(self) -> None:
        self.m.clamp_(-self.radius, self.radius)

    def set_radius(self, r: float) -> None:
        self.radius = r


class ShadingRandomParam(ShadingParam):
    """同失真的隨機明暗場，**不最佳化**，只在 reset 時抽一次。

    候選二第 6 點步驟 3 標為**必做**：`位移場`（FND-004）的死法正是
    「與同失真隨機對照無法區分」，而低頻、低自由度的參數化特別容易重蹈覆轍。
    這一格與 `phase_rand` 同性質，理由也相同。
    """

    name = "shading_rand"

    def reset(self, x01: torch.Tensor, seed: int) -> None:
        super().reset(x01, seed)
        gen = torch.Generator(device="cpu").manual_seed(seed)
        init = torch.randn(self.m.shape, generator=gen) * self.radius
        with torch.no_grad():
            self.m.copy_(init.clamp(-self.radius, self.radius).to(
                device=x01.device, dtype=x01.dtype))

    def params(self) -> List[torch.Tensor]:
        return []


class WarpParam:
    """φ = c（粗網格上的位移量，單位是像素），`x_def = x ∘ (id + upsample(c))`。

    **這是 WaNet 式三元對照的最佳化那一格**（`runs/ip2p_warp/`）。它問的是
    `RESULTS.md` 的 FND-004（位移場與同失真隨機對照無法區分）是這一族方法的
    通病，還是舊實作的問題。機制上的懷疑來自 WaNet（ICLR 2021,
    arXiv:2102.10369）：沒有被專門訓練去區分位移場的網路，反應的不是位移場的
    身份，而是**重取樣內插產生的像素級 artifact**；WaNet 必須額外加入 noise
    mode（隨機 warp → 標正確類別）才逼得網路學會區分特定的場，而 IP2P 從未
    受過這種訓練。三元對照（`warp` / `warp_rand` / `warp_roundtrip`）把
    「幾何」與「內插 artifact」拆開量。

    **與 stAdv（arXiv:1801.02612）刻意的偏離，必須知道**
    ────────────────────────────────────────────────────────────────
    stAdv 的位移場是**逐像素稠密**的，並以流場的總變差（TV）正則化維持
    平滑。本類別改用 **16×16 粗網格＋雙三次上採樣**，沒有 TV 項——平滑性
    由上採樣核直接保證，自由度由 512 降到 2·16²。

    偏離的理由是**感知預算是在這個構造上量的**：本專案在本機量過同一構造
    （6 張、16×16 粗網格上採樣）的失真—位移對照表，

        最大位移 2 px → LPIPS 0.0229 / DISTS 0.0094 / PSNR 30.77
                 4 px → 0.0550 / 0.0239 / 25.74
                 8 px → 0.1205 / 0.0533 / 21.77
                16 px → 0.2323 / 0.1091 / 18.79
                24 px → 0.3191 / 0.1602 / 17.37

    換成稠密場＋TV 之後這張表全部作廢，而本批要在既有的失真帶
    （DISTS 0.1286 附近）上比較。**本類別因此不是 stAdv 的移植**，報表上
    不可寫成 stAdv 的結果。

    三個構造上的選擇，各有理由：

    - **半徑的單位是「最大位移像素數」，且夾在粗網格的係數上**，不是夾在
      上採樣之後。雙三次會過衝，夾在上採樣之後預算就沒有定義；與
      `ShadingParam` 同一條理由。過衝造成的超出由 `padding_mode="border"`
      與取樣本身收掉。
    - **零位移逐位等於原圖**。用像素中心的基準網格配 `align_corners=False`：
      `grid_sample` 的反正規化是 `((c + 1) · W − 1) / 2`，基準取
      `c_i = (2i + 1)/W − 1` 時，邊長為 2 的冪（本專案是 512）的每一步在
      float32 下都是精確的二進位分數，反正規化**恰好**回到整數 `i`，
      雙線性權重是 0/1。實測 H = 64 與 512 逐位相等，H = 511／300 只到
      1.5e−5／2.9e−5。**用 `align_corners=True` 配 `linspace` 做不到**
      （512 上誤差 5.8e−5），那是一筆無償的失真地板。
    - **`padding_mode="border"`**：邊界外取邊界值。用 `zeros` 會在邊緣造出
      一圈黑框，那是位移場以外的東西。

    `roundtrip=True` 時先施加 `+f` 再施加 `−f`。幾何幾乎回到原點，剩下的
    **只有兩次重取樣的內插 artifact**——這一格就是 WaNet 的機制解釋在本專案
    上的直接檢定。注意 `−f` 只是 `f` 的一階逆，殘餘幾何是 O(‖∇f‖·‖f‖)，
    不是零；它在同一個半徑上的失真遠低於單次 warp，故等失真比較時要掃到
    更大的半徑。
    """

    name = "warp"

    def __init__(self, radius: float = 8.0, grid: int = 16,
                 roundtrip: bool = False, init_std: float = 0.0):
        if grid < 2:
            raise ValueError(f"grid 必須至少為 2，收到 {grid}")
        self.radius = radius
        self.grid = grid
        self.roundtrip = roundtrip
        self.init_std = float(init_std)
        self.c: Optional[torch.Tensor] = None

    def reset(self, x01: torch.Tensor, seed: int) -> None:
        """通道 0 = x 方向位移，通道 1 = y 方向位移，單位都是像素。

        **`init_std = 0` 時是全零起點，逐位元等於加這個參數之前。**

        非零時用 `seed` 抽一個高斯起點。存在的理由是實測的病灶：
        `runs/ip2p_warp/step_probe_latent_norm.csv` 量到 `latent_norm` 在
        **零位移處有一個帶折點的局部極小**——梯度完全正常（absmean 3.4e−2、
        零元素比例 0.0000）但每走一步損失都上升（105.95 → 110.07），於是
        sign PGD 在 0 與 ±α 之間形成週期 2 的振盪、`|c|` 恆等於 α。
        而 `reset` 的 `seed` 參數先前**收下了卻沒有使用**，所以沒有離開
        折點的辦法。要問「這個損失對位移場行不行」就必須從折點以外起步，
        否則量到的是 sign PGD 的性質而不是損失的性質。
        """
        g = torch.Generator(device="cpu").manual_seed(seed)
        if self.init_std > 0:
            init = torch.randn(1, 2, self.grid, self.grid, generator=g) * self.init_std
            init = init.clamp(-self.radius, self.radius)
        else:
            init = torch.zeros(1, 2, self.grid, self.grid)
        self.c = init.to(device=x01.device, dtype=x01.dtype).requires_grad_(True)

    @staticmethod
    def _base_grid(h: int, w: int, device, dtype) -> torch.Tensor:
        """像素中心的基準取樣網格。邊長為 2 的冪時零位移逐位等於恆等。"""
        ys = (2 * torch.arange(h, device=device, dtype=dtype) + 1) / h - 1
        xs = (2 * torch.arange(w, device=device, dtype=dtype) + 1) / w - 1
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        return torch.stack((gx, gy), dim=-1).unsqueeze(0)

    def _displacement(self, x01: torch.Tensor) -> torch.Tensor:
        """粗網格係數雙三次上採樣到影像尺寸，(1, 2, H, W)，單位是像素。"""
        return torch.nn.functional.interpolate(
            self.c, size=x01.shape[-2:], mode="bicubic", align_corners=False)

    def _sample(self, x01: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        h, w = x01.shape[-2:]
        base = self._base_grid(h, w, x01.device, x01.dtype)
        # 像素 → 正規化座標：align_corners=False 下一個像素是 2/W。
        off = torch.stack((d[:, 0] * (2.0 / w), d[:, 1] * (2.0 / h)), dim=-1)
        return torch.nn.functional.grid_sample(
            x01, base + off, mode="bilinear", padding_mode="border",
            align_corners=False)

    def render(self, x01: torch.Tensor) -> torch.Tensor:
        d = self._displacement(x01)
        y = self._sample(x01, d)
        if self.roundtrip:
            y = self._sample(y, -d)
        return y.clamp(0.0, 1.0)

    @torch.no_grad()
    def effective_displacement(self, x01: torch.Tensor) -> torch.Tensor:
        """整條 render 走完之後，每個像素**實際**被搬了多遠（像素，(1,2,H,W)）。

        做法是把座標圖當成一張兩通道的影像，餵進**同一組**取樣算子，再減回
        原座標。單次 warp 時這恆等於 `upsample(c)`；往返時它量的是殘餘幾何，
        而那正是 `warp_roundtrip` 這一格唯一的汙染源——`−f` 只是 `f` 的一階
        逆，殘餘量的量級是 `‖∇f‖·‖f‖ ≈ radius² / 粗網格間距`，radius 大時
        不可忽略。**報表上必須把這個量寫出來**，否則「只剩內插 artifact」
        是一句沒有證據的話。

        邊界處 `padding_mode="border"` 會讓座標圖被夾住，那一圈的讀數偏小。
        """
        h, w = x01.shape[-2:]
        ys = torch.arange(h, device=x01.device, dtype=x01.dtype)
        xs = torch.arange(w, device=x01.device, dtype=x01.dtype)
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        coords = torch.stack((gx, gy)).unsqueeze(0)
        d = self._displacement(x01)
        out = self._sample(coords, d)
        if self.roundtrip:
            out = self._sample(out, -d)
        return out - coords

    def params(self) -> List[torch.Tensor]:
        return [self.c]

    @torch.no_grad()
    def project(self) -> None:
        # **夾的是粗網格上的係數**，單位是像素。上採樣會過衝，夾在上採樣
        # 之後預算就沒有定義。
        self.c.clamp_(-self.radius, self.radius)

    def set_radius(self, r: float) -> None:
        self.radius = r


class WarpRandomParam(WarpParam):
    """同半徑的隨機位移場，**不最佳化**，只在 reset 時抽一次。

    三元對照的第二格。與 `phase_rand`／`shading_rand` 同性質、同抽法
    （`randn × radius` 再夾回 ±radius）：sign PGD 會把 L∞ 球用滿，故隨機
    對照也取用滿的分布，否則「同半徑」在實質上不同。
    """

    name = "warp_rand"

    def reset(self, x01: torch.Tensor, seed: int) -> None:
        super().reset(x01, seed)
        gen = torch.Generator(device="cpu").manual_seed(seed)
        init = torch.randn(self.c.shape, generator=gen) * self.radius
        with torch.no_grad():
            self.c.copy_(init.clamp(-self.radius, self.radius).to(
                device=x01.device, dtype=x01.dtype))

    def params(self) -> List[torch.Tensor]:
        return []


class WarpRoundTripParam(WarpRandomParam):
    """**與 `warp_rand` 用同一個隨機場**，先施加 `f` 再施加 `−f`。

    三元對照的第三格。幾何幾乎回到原點，留下的主要是兩次雙線性重取樣的
    內插 artifact。它與 `warp_rand` 的差就是「幾何本身有沒有貢獻」；
    同種子同半徑下兩者抽到的 `c` 逐位相同（由 `tests/test_warp_param.py`
    釘住），故這個差不含抽樣變異。
    """

    name = "warp_roundtrip"

    def __init__(self, radius: float = 8.0, grid: int = 16):
        super().__init__(radius=radius, grid=grid, roundtrip=True)

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
    update: str = "sign",
    step_size: Optional[float] = None,
) -> ParamPGDResult:
    """共用迴圈。`loss_fn(x_def)` 回傳要**最小化**的純量。

    `saturate_at` 決定步長：`α = radius / (steps · saturate_at)`，即前四分之一
    的迭代就能走滿半徑，其餘用來在球面上調方向。兩個參數化共用同一個比例，
    故「誰先撞到約束」不會變成一個隱藏的變因。

    **`step_size` 明給時直接取代那條公式。** 存在的理由是一個實測的混淆：
    步長綁在半徑上，於是「放寬預算」同時「放大步長」，放寬預算不等於走得更遠，
    它同時讓振盪的幅度變大（`runs/ip2p_warp/DIAGNOSIS.md`：隨機遊走比值隨半徑
    由 0.84 掉到 0.40）。要問「不設預算會走到哪裡」就必須把兩者解耦。

    **`update` 選 `adam` 時改用 Adam，`alpha` 當成 lr。** sign PGD 在**帶折點
    的局部極小**上會形成週期 2 的振盪（同上文件第四節），那是更新規則的性質
    不是損失的性質；要分辨兩者就需要一個不靠梯度符號的更新規則。
    **`docs/GOAL.md` 把「Adam 更新規則」列在更早期已否決的方向裡，但那批的
    證據已刪除、且是在相位參數化上做的**，與這裡的位移場不是同一件事；
    使用者已就位移場的探索批次明確授權。報表上必須標明這一點，
    **不可以拿本旗標的結果去推翻或恢復相位臂上的那個否決**。

    `history` 逐筆多記一個 `param_absmean`（參數的絕對值平均），用來判定
    「有沒有真的在走」——只看損失下降分不出走了多遠。

    `transform` 給定時，損失改在 `transform(x_def, step)` 上計算——用來把
    可微分的淨化算子放進最佳化迴圈（`src/defense/purify_aware.py`）。
    **回傳的 `x_def` 仍然是未經 transform 的防禦圖**：transform 是攻擊方會
    做的事，不是我們交出去的東西。預設 `None`，行為與加入此參數之前逐位元
    相同。
    """
    if update not in ("sign", "adam"):
        raise ValueError(f"未知的更新規則 {update!r}，可用的是 sign／adam")

    param.reset(x01, seed)
    ps = param.params()
    alpha = (param.radius / max(1.0, steps * saturate_at)
             if step_size is None else float(step_size))
    history: List[Dict] = []

    if not ps:                                   # phase_rand：不最佳化
        with torch.no_grad():
            return ParamPGDResult(param.render(x01).detach(), param.radius, history)

    opt = torch.optim.Adam(ps, lr=alpha) if update == "adam" else None

    for i in range(steps):
        x_def = param.render(x01)
        loss = loss_fn(x_def if transform is None else transform(x_def, i))
        if opt is None:
            grads = torch.autograd.grad(loss, ps)
            with torch.no_grad():
                for p, g in zip(ps, grads):
                    p.sub_(alpha * torch.sign(g))
        else:
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        param.project()
        if log_every and (i % log_every == 0 or i == steps - 1):
            mag = float(sum(float(p.detach().abs().sum()) for p in ps)
                        / max(1, sum(p.numel() for p in ps)))
            history.append({"step": i, "loss": float(loss.detach()),
                            "param_absmean": mag})
            print(f"    [{param.name}] step {i:4d} loss {float(loss.detach()):.6f}"
                  f" |p| {mag:.5f}", flush=True)

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

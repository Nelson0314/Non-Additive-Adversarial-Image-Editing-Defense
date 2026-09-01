"""DCT 域的保長相位旋轉，`param_pgd.Parameterization` 的實作。

設計與判準寫在 `runs/dct_phase_design/README.md`，這裡只記構造與必須知道的坑。

要點
────────────────────────────────────────────────────────────────────
DCT 係數是實數，沒有現成的相位。文獻上 DFT 的相位項在 DCT 上對應的是**係數
的正負號**（Ito & Kiya 2007，DCT sign-only correlation）。正負號是二值的、
梯度過不去、失真固定為 `2|c|` 不可調，所以本模組把它**連續化**：把同一個
8×8 區塊內的兩格係數 `(c_i, c_j)` 當成平面向量乘上旋轉矩陣

    c' = R(θ) c,    R(θ) = [[cos θ, −sin θ], [sin θ, cos θ]]

`θ = π` 恰好就是聯合翻號（`c' = −c`），`θ = 0` 是恆等，中間是它的連續化。
保長 `‖c'‖ = ‖c‖`；8×8 DCT 正交歸一且區塊不重疊，故保長在**像素域**也成立。
失真有封閉式 `‖Δ‖₂ = 2‖c‖₂·sin(θ/2)`。

**保長的層級不可含混。** 本方法（`texture_rephase`）保的是逐（區塊, 頻格）的
幅度；這裡保的是**一對係數的平方和**，嚴格較弱——單一係數的幅度會變，只有
配對的合成長度不變。任何「保幅度」的說法都必須加上「逐配對」這個限定。

配對規則
────────────────────────────────────────────────────────────────────
`transpose`：`(u,v) ↔ (v,u)`，`u < v`。兩格的徑向頻率**相同**、JPEG 亮度
量化表上的階距幾乎相同（`base[0][1]=11` 對 `base[1][0]=12`），旋轉交換的是
「橫紋 vs 直紋」的方向，不是能量的大小或所在的頻帶——**保長只有在兩軸價錢
相同時才有感知意義**，這是配對規則的唯一判準。

`zigzag`：zigzag 序上相鄰兩格。兩格的頻率與價錢都不同，保長在感知上不成立。
**它是對照組，用來讓上面那句話變成可判定的**，不是備案。

DC 一律排除：DC 是區塊的平均亮度，動它等於整塊變亮，`src/baselines/dct_shield.py`
已記載那正是 DCT-Shield 平坦區可見方格的來源。

交付的到底是什麼動作（**必須寫進論文的方法段**）
────────────────────────────────────────────────────────────────────
旋轉量化後的整數係數，結果不是整數，必須再取整：

    δ = round(R(θ)·α) − α

也就是**一個整數的係數位移**，DCT-Shield 的動作空間。本模組**不是**一個
DCT-Shield 做不到的操作，而是把 δ 限制在「通過 α 的圓」上的**受約束子集**。
新穎性只能建立在「約束不同」（保長），不能建立在「動作不同」。
實測（十張）：帶內工作點 θ ≈ 1.1 上有 **92.7%** 的 δ 落在 DCT-Shield 的 ε=1
球內（`runs/dct_phase_design/ceiling.csv` 的 `delta_within_1`），該欄隨 θ
單調下降：0.9965（θ=0.08）→ 0.9268（θ=1.20）→ 0.8644（θ=π）。

梯度怎麼過量化
────────────────────────────────────────────────────────────────────
- `jpeg_encode` **不需要可微**：α 只算一次、之後是常數（與 `run_dct_shield`
  的行 2 相同）。
- 旋轉後的取整走**直通估計**：前向是真的 `round()`、反向當恆等。此處要的是
  `round(·)` 而不是 `round(·/table)`，故 `quantize_ste` 的 `table` 傳全 1。
- `jpeg_decode` 本來就完全可微。

**交付時把 STE 換成真的 `torch.round` 前向值逐位元相同**（`quantize_ste` 的
docstring 已保證），故 `render` 不需要分訓練／交付兩條路。
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from src.baselines.jpeg_codec import (
    CHANNEL_NAMES,
    jpeg_decode,
    jpeg_encode,
    normalize_quality,
)
from src.residual.texture_rephase import pixel_texture_mask

Pair = Tuple[Tuple[int, int], Tuple[int, int]]


def zigzag_order() -> List[Tuple[int, int]]:
    """JPEG 的 zigzag 掃描序（ITU-T T.81 Figure 5），`(u, v) = (列, 行)`。"""
    return sorted(((u, v) for u in range(8) for v in range(8)),
                  key=lambda c: (c[0] + c[1],
                                 c[1] if (c[0] + c[1]) % 2 == 0 else -c[1]))


def build_pairs(rule: str = "transpose", r_min: float = 0.12) -> List[Pair]:
    """回傳配對清單。DC 一律排除，徑向低於 `r_min` 的也排除。

    徑向座標取 `r = sqrt(u² + v²) / 8`，與 `texture_rephase.radial_gate` 的
    尺度一致（該處 Nyquist 歸一到 1）。**8×8 格點上最小的非零半徑是 0.125**，
    所以 `r_min = 0.12` 在這裡只排除 DC——徑向閘在 DCT 域只有 8 階解析度，
    這是換域的實質代價，不是實作疏漏。
    """
    def radius(c: Tuple[int, int]) -> float:
        return math.sqrt(c[0] ** 2 + c[1] ** 2) / 8.0

    if rule == "transpose":
        raw = [((u, v), (v, u)) for u in range(8) for v in range(u + 1, 8)]
    elif rule == "zigzag":
        order = [c for c in zigzag_order() if c != (0, 0)]
        raw = [(order[i], order[i + 1]) for i in range(0, len(order) - 1, 2)]
    else:
        raise ValueError(f"未知的配對規則 {rule!r}，可用的是 transpose／zigzag")

    out: List[Pair] = []
    for a, b in raw:
        if a == (0, 0) or b == (0, 0):
            continue
        if radius(a) < r_min or radius(b) < r_min:
            continue
        out.append((a, b))
    return out


def rotate_pairs(coef: torch.Tensor, pairs: Sequence[Pair],
                 ang: torch.Tensor) -> torch.Tensor:
    """`(N, hb, wb, 8, 8)` 的係數逐對做平面旋轉。

    `ang` 形狀 `(N, hb, wb, n_pairs)`，第 k 個切片是第 k 對的角度。
    以**新張量堆疊**而不是就地寫入：就地寫入到 `clone()` 上雖然也能回傳梯度，
    但一旦有兩對共用同一格（將來換配對規則時可能發生）就會靜默地只留最後一次
    寫入，而不會有任何症狀。這裡先驗證配對不重疊，再用 `index_put` 一次寫完。
    """
    seen: Dict[Tuple[int, int], int] = {}
    for k, (a, b) in enumerate(pairs):
        for c in (a, b):
            if c in seen:
                raise ValueError(
                    f"格 {c} 同時出現在第 {seen[c]} 與第 {k} 對；配對必須不重疊，"
                    "否則旋轉不是保長映射")
            seen[c] = k

    out = coef.clone()
    cos, sin = torch.cos(ang), torch.sin(ang)
    for k, ((u1, v1), (u2, v2)) in enumerate(pairs):
        a = coef[..., u1, v1]
        b = coef[..., u2, v2]
        out[..., u1, v1] = cos[..., k] * a - sin[..., k] * b
        out[..., u2, v2] = sin[..., k] * a + cos[..., k] * b
    return out


def block_texture_gate(x01: torch.Tensor, wb: int, sigma: float,
                       edge_power: float) -> torch.Tensor:
    """本方法的紋理閘搬到**編解碼器自己的**區塊格點上，形狀 `(N, hb, wb)`。

    **不能直接呼叫 `texture_rephase.texture_gate`**：它走 `block_mean`，會先
    reflect padding `block//2` 再以 `hop` 展開，512 上得到 65×65 個重疊視窗，
    與 `jpeg_encode` 的 64×64 個**不重疊**區塊對不齊；相乘之後錯位，而且
    **不會有任何症狀**。故走 `pixel_texture_mask`（同一條
    `(1 − coh²)^p · clamp(energy/ref)` 公式，只是把區塊平均換成高斯平滑），
    再以 `avg_pool2d` 落到編解碼器的格點上。色度平面因 4:2:0 邊長減半，
    池化核跟著加倍。
    """
    m = pixel_texture_mask(x01, sigma=sigma, energy_quantile=0.0,
                           edge_power=edge_power)              # (N,1,H,W)
    k = x01.shape[-1] // wb
    if k * wb != x01.shape[-1]:
        raise ValueError(
            f"影像邊長 {x01.shape[-1]} 不是區塊數 {wb} 的整數倍，池化會錯位")
    return F.avg_pool2d(m, kernel_size=k, stride=k).squeeze(1)


class DctRotationParam:
    """`φ = θ`（逐區塊逐配對的旋轉角），交付即參數。

    `radius` 就是 `θ_max`，`project` 是 `theta.clamp_(−θ_max, θ_max)`。
    與 `PhaseParam` 一樣，強度旗鈕與角度上界是同一個東西。
    """

    name = "dct_rotate"

    def __init__(
        self,
        radius: float = 1.1,
        qd: float = 0.85,
        pairing: str = "transpose",
        r_min: float = 0.12,
        gate: str = "texture",
        gate_sigma: float = 2.0,
        gate_edge_power: float = 1.0,
        channels: Sequence[str] = CHANNEL_NAMES,
    ):
        bad = set(channels) - set(CHANNEL_NAMES)
        if bad:
            raise ValueError(f"未知通道 {sorted(bad)}；可用的是 {CHANNEL_NAMES}")
        if not channels:
            raise ValueError("channels 不可為空——那樣 θ 沒有任何自由度")
        if gate not in ("texture", "band"):
            raise ValueError(f"未知的閘 {gate!r}，可用的是 texture／band")
        self.radius = float(radius)
        self.qd = float(qd)
        self.pairing = pairing
        self.r_min = float(r_min)
        self.gate = gate
        self.gate_sigma = float(gate_sigma)
        self.gate_edge_power = float(gate_edge_power)
        self.channels = tuple(channels)
        self.pairs = build_pairs(pairing, r_min)
        if not self.pairs:
            raise ValueError(
                f"配對規則 {pairing!r} 在 r_min={r_min} 下沒有留下任何一對")
        self.alpha: Dict[str, torch.Tensor] = {}
        self.gates: Dict[str, torch.Tensor] = {}
        self.theta: Dict[str, torch.Tensor] = {}

    # ---- Parameterization 介面 ----

    def reset(self, x01: torch.Tensor, seed: int) -> None:
        """由原圖算一次量化係數與閘，並把 θ 初始化為零。

        `θ = 0` 時 `R(0) = I`，`round(α) = α`，故輸出逐位元等於
        `jpeg_roundtrip(x, qd)`——**不是**逐位元等於原圖。這與
        `texture_rephase` 的恆等性質不同，因為交付本身就是壓縮圖。
        `tests/test_dct_rotation.py` 釘住這一點。
        """
        q = normalize_quality(self.qd)
        with torch.no_grad():
            self.alpha = {k: v.detach()
                          for k, v in jpeg_encode(x01, self.qd).items()}
        self.theta = {}
        self.gates = {}
        for name in CHANNEL_NAMES:
            a = self.alpha[name]
            n, hb, wb = a.shape[0], a.shape[1], a.shape[2]
            if self.gate == "band":
                g = torch.ones(n, hb, wb, device=a.device, dtype=a.dtype)
            else:
                g = block_texture_gate(x01, wb, self.gate_sigma,
                                       self.gate_edge_power)
            self.gates[name] = g.detach()
            if name in self.channels:
                self.theta[name] = torch.zeros(
                    n, hb, wb, len(self.pairs),
                    device=a.device, dtype=a.dtype, requires_grad=True)
        self._quality = q

    def render(self, x01: torch.Tensor) -> torch.Tensor:
        coef: Dict[str, torch.Tensor] = {}
        for name in CHANNEL_NAMES:
            a = self.alpha[name]
            if name not in self.theta:
                coef[name] = a
                continue
            ang = self.theta[name] * self.gates[name][..., None]
            r = rotate_pairs(a, self.pairs, ang)
            # 取整走直通估計：前向是真的 round，反向當恆等。
            coef[name] = r + (torch.round(r) - r).detach()
        return jpeg_decode(coef, self._quality)

    def params(self) -> List[torch.Tensor]:
        return [self.theta[n] for n in CHANNEL_NAMES if n in self.theta]

    @torch.no_grad()
    def project(self) -> None:
        for t in self.theta.values():
            t.clamp_(-self.radius, self.radius)

    def set_radius(self, r: float) -> None:
        self.radius = float(r)

    # ---- 診斷（報表用，不參與最佳化） ----

    @torch.no_grad()
    def delta_stats(self) -> Dict[str, float]:
        """交付的整數位移有多大。`delta_within_1` 是它落在 DCT-Shield ε=1 球
        內的比例——**新穎性的措辭由這個數字決定**，見模組 docstring。"""
        n_small = n_tot = 0
        n_zero = n_pair = 0
        for name in self.channels:
            a = self.alpha[name]
            ang = self.theta[name] * self.gates[name][..., None]
            delta = torch.round(rotate_pairs(a, self.pairs, ang)) - a
            for (u1, v1), (u2, v2) in self.pairs:
                dd = torch.stack([delta[..., u1, v1], delta[..., u2, v2]], dim=-1)
                n_small += int((dd.abs() <= 1).sum())
                n_tot += dd.numel()
                p = torch.stack([a[..., u1, v1], a[..., u2, v2]], dim=-1)
                n_zero += int((p.abs().sum(dim=-1) == 0).sum())
                n_pair += p[..., 0].numel()
        return {"delta_within_1": n_small / max(1, n_tot),
                "zero_pair_frac": n_zero / max(1, n_pair),
                "n_pairs": float(len(self.pairs))}


class DctRotationRandomParam(DctRotationParam):
    """同角度上界的隨機旋轉，**不最佳化**，只在 reset 時抽一次。

    與 `phase_rand`／`warp_rand`／`shading_rand` 同性質、同抽法：sign PGD 會把
    L∞ 球用滿，故隨機對照也取用滿的分布，否則「同上界」在實質上不同。
    """

    name = "dct_rotate_rand"

    def reset(self, x01: torch.Tensor, seed: int) -> None:
        super().reset(x01, seed)
        gen = torch.Generator(device="cpu").manual_seed(seed)
        with torch.no_grad():
            for name, t in self.theta.items():
                init = torch.randn(t.shape, generator=gen) * self.radius
                t.copy_(init.clamp(-self.radius, self.radius).to(
                    device=t.device, dtype=t.dtype))

    def params(self) -> List[torch.Tensor]:
        return []

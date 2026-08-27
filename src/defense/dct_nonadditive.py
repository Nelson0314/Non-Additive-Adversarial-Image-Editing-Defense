"""在 8×8 DCT 係數上做**非加性**擾動的三個形式。**探索性質。**

為什麼要有這一支
────────────────────────────────────────────────────────────────────
DCT-Shield 在同一組係數上做的是**加性**擾動（`x' = JPEG_D(α + δ)`，δ 夾在
±eps）。本專案在同一個域上能做的、而它做不到的，是**非加性**的形式。

第一次嘗試（`dct_rotation.py`，逐**固定配對**的二維旋轉）在十張上失敗：
等失真下未淨化位移只有現行做法的 60%，而且 `zigzag` 與 `transpose` 兩種配對
規則**分不開**（差 0.9%）。那個 null result 指向一個具體的懷疑——

    卡住的不是「動哪些配對」，而是**配對本身**這個限制。

固定配對只允許能量在**兩個事先指定的座標**之間搬。本檔把那個限制拿掉：
旋轉的平面改成**學出來的**。

三個形式（`mode`）
────────────────────────────────────────────────────────────────────
| mode | 是什麼 | 保長嗎 | 參數量（Y，512²） |
|---|---|---|---|
| `plane` | 逐區塊學一個二維平面與一個角度，在那個平面上旋轉 | 是（逐區塊） | 4096 × (2k+1) ≈ 52 萬 |
| `shared_plane` | 同上，但整張影像共用一個平面與角度 | 是（逐區塊） | 2k+1 ≈ 127 |
| `gain` | 逐係數乘上 `exp(g)` | **否** | 4096 × k ≈ 26 萬 |

`k` 是通帶內的 AC 係數個數（8×8 上 `r_min = 0.12` 只排除 DC，故 k = 63）。
`plane` 的參數量刻意對齊現行相位法的 59 萬——**容量匹配之後再比，才分得出
「域不對」與「自由度不夠」**。

`plane` 的閉式解（不用 `matrix_exp`）
────────────────────────────────────────────────────────────────────
給定 `u, v ∈ R^k`，取

    e1 = u / ‖u‖ ,  w = v − ⟨v, e1⟩ e1 ,  e2 = w / ‖w‖

則在 `span(e1, e2)` 上轉角度 θ 的映射是

    c' = c + (cos θ − 1)(⟨c,e1⟩ e1 + ⟨c,e2⟩ e2) + sin θ (⟨c,e1⟩ e2 − ⟨c,e2⟩ e1)

**平面外的分量完全不動**，所以 `‖c'‖ = ‖c‖` 是精確的（不是數值近似），
而且成本是 O(k)、不需要對 4096 個區塊各算一次矩陣指數。

`u, v` 決定**往哪裡搬**，θ 決定**搬多少**——兩者刻意分開，因為失真預算只由
θ 決定：`‖Δc‖ = 2‖P c‖ · sin(θ/2)`，`P` 是投影到該平面。角度夾在 ±`radius`，
與相位法的 `theta_max` 同一個語意。

與 `dct_rotation.py` 的關係
────────────────────────────────────────────────────────────────────
固定配對旋轉是本檔的**特例**：把 `u, v` 釘在兩個座標軸上就是它。所以
`plane` 若仍然打不過現行方法，「DCT 域的保長重相位」這一族就可以收掉，
而不只是「那個配對規則不好」。

**8×8 區塊不重疊、DCT 正交歸一**，所以保長在像素域也成立，而且**沒有
STFT 一致性投影誤差**——`RESULTS.md` 的 FND-040 記著現行方法的效果與
`amp_dev` 正相關（r = +0.449），也就是效果有一部分可能來自「新造出來的能量」
而不是重排。在這個域上那個管道由構造封死，量到的是**純重相位單獨能做到什麼**。

交付
────────────────────────────────────────────────────────────────────
本檔一律作用在**未量化的浮點係數**上、輸出浮點影像。量化交付是獨立的旗鈕
（`--deliver-jpeg`），與相位法的 `ours_ph_n` 對 `ours_ph_q` 完全同構。
理由：量化把 66.2% 的配對歸零，但那些配對只帶有 0.79% 的成對能量——
**量化不是容量瓶頸**（實測），把它綁進參數化只會讓變因混在一起。
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from src.baselines.jpeg_codec import block_dct, block_idct, dct_matrix
from src.residual.texture_rephase import _RGB2YCC, _YCC2RGB
from src.defense.dct_rotation import block_texture_gate

CHANNELS = ("Y", "Cb", "Cr")


def band_indices(r_min: float = 0.12, r_max: float = float("inf")
                 ) -> List[Tuple[int, int]]:
    """通帶內的 AC 係數座標。`r = sqrt(u²+v²)/8`，與 `radial_gate` 同尺度。

    8×8 格點上最小的非零半徑是 0.125，所以 `r_min = 0.12` 只排除 DC——
    徑向解析度在這個域上只有 8 階，是換域的實質代價。DC 一律排除：它是區塊的
    平均亮度，動它等於整塊變亮（`dct_shield.py` 記載那是平坦區方格的來源）。
    """
    out = []
    for u in range(8):
        for v in range(8):
            if u == 0 and v == 0:
                continue
            r = math.sqrt(u * u + v * v) / 8.0
            if r_min <= r < r_max:
                out.append((u, v))
    return out


def to_planes(x01: torch.Tensor) -> List[torch.Tensor]:
    """`[0,1]` RGB → 三個**全解析度**的 YCbCr 平面（已 level shift）。

    **刻意不做 4:2:0 色度次取樣**，與 `jpeg_encode` 在這一點上不同。
    次取樣是 JPEG 編碼器的一部分，而本檔不走 JPEG——交出去的是浮點影像。
    實測次取樣單獨造成的像素往返誤差**最大到 0.81**（`[0,1]` 值域），
    那會變成一個與參數無關的失真地板，而且 `theta = 0` 就不再是恆等映射。
    量化交付若要開，是後面獨立的 `--deliver-jpeg` 旗鈕，那時才會有 4:2:0。
    """
    m = _RGB2YCC.to(device=x01.device, dtype=x01.dtype)
    ycc = torch.einsum("ij,bjhw->bihw", m, x01 * 255.0)
    ycc = ycc + torch.tensor([0.0, 128.0, 128.0], device=x01.device,
                             dtype=x01.dtype)[None, :, None, None]
    return [ycc[:, 0:1] - 128.0, ycc[:, 1:2] - 128.0, ycc[:, 2:3] - 128.0]


def from_planes(planes: Sequence[torch.Tensor]) -> torch.Tensor:
    """三個全解析度平面 → `[0,1]` RGB。

    **末端的 `clamp` 會破壞像素域的保長**：DCT 域的旋轉是精確保長的，但轉出
    值域的像素被夾回去之後那份能量就沒了。這不是實作瑕疵，是「輸出必須是一張
    合法影像」的必然代價，與 `ShadingParam` 的亮部飽和同型。
    比例由 `DctNonAdditiveParam.clip_fraction()` 報出來供判讀。
    """
    return _to_rgb(planes).clamp(0.0, 1.0)


def _to_rgb(planes: Sequence[torch.Tensor]) -> torch.Tensor:
    """夾取之前的 RGB。**反矩陣由 `torch.linalg.inv` 算出，不用 JFIF 的常數。**

    `jpeg_codec` 沿用 JFIF 公布的正逆常數是對的——那支在模擬 libjpeg。
    本檔不是：它不走 JPEG，所以應該用精確互逆的一對，否則往返只互逆到
    **5.7e−7**（實測），而那是一筆與參數無關、白付的失真地板，也會讓
    「`theta = 0` 是恆等」這條構造保證只成立到 1e−6。`texture_rephase` 出於
    同一個理由也是這樣做的。
    """
    ycc = torch.cat([p + 128.0 for p in planes], dim=1)
    dev, dt = ycc.device, ycc.dtype
    ycc = ycc - torch.tensor([0.0, 128.0, 128.0], device=dev, dtype=dt
                             )[None, :, None, None]
    m = _YCC2RGB.to(device=dev, dtype=dt)
    return torch.einsum("ij,bjhw->bihw", m, ycc) / 255.0


def _unclamped(planes: Sequence[torch.Tensor]) -> torch.Tensor:
    """夾取之前的輸出，只給診斷用。"""
    return _to_rgb(planes)


def rotate_in_plane(c: torch.Tensor, u: torch.Tensor, v: torch.Tensor,
                    theta: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """在 `span(u, v)` 上把 `c` 轉 `theta`。全部張量的前導維可廣播。

    `c`、`u`、`v` 形狀 `(..., k)`，`theta` 形狀 `(..., 1)`。
    平面外的分量原封不動，故 `‖c'‖ = ‖c‖` 精確成立。

    `u` 或 `v` 退化（長度接近 0，或 `v` 幾乎平行 `u`）時該區塊不旋轉——
    **回傳原值而不是回傳 NaN**。這不是用條件跳過掩蓋症狀：退化的平面在數學上
    就沒有定義，`degenerate_fraction()` 會把它的比例報出來供判讀。
    """
    e1_norm = u.norm(dim=-1, keepdim=True)
    e1 = u / e1_norm.clamp_min(eps)
    w = v - (v * e1).sum(dim=-1, keepdim=True) * e1
    w_norm = w.norm(dim=-1, keepdim=True)
    e2 = w / w_norm.clamp_min(eps)

    live = ((e1_norm > eps) & (w_norm > eps)).to(c.dtype)
    a = (c * e1).sum(dim=-1, keepdim=True)
    b = (c * e2).sum(dim=-1, keepdim=True)
    cos, sin = torch.cos(theta), torch.sin(theta)
    delta = (cos - 1.0) * (a * e1 + b * e2) + sin * (a * e2 - b * e1)
    return c + delta * live


class DctNonAdditiveParam:
    """`param_pgd.Parameterization` 的實作。強度旗鈕 `radius` 即角度／增益上界。"""

    def __init__(
        self,
        radius: float = 1.0,
        mode: str = "plane",
        r_min: float = 0.12,
        r_max: float = float("inf"),
        gate: str = "texture",
        gate_sigma: float = 2.0,
        gate_edge_power: float = 1.0,
        channels: Sequence[str] = CHANNELS,
        plane_init_std: float = 1.0,
    ):
        if mode not in ("plane", "shared_plane", "gain"):
            raise ValueError(
                f"未知的 mode {mode!r}，可用的是 plane／shared_plane／gain")
        if gate not in ("texture", "band"):
            raise ValueError(f"未知的閘 {gate!r}，可用的是 texture／band")
        bad = set(channels) - set(CHANNELS)
        if bad:
            raise ValueError(f"未知通道 {sorted(bad)}；可用的是 {CHANNELS}")
        self.radius = float(radius)
        self.mode = mode
        self.idx = band_indices(r_min, r_max)
        if not self.idx:
            raise ValueError(f"r_min={r_min}／r_max={r_max} 下通帶內沒有 AC 係數")
        self.gate = gate
        self.gate_sigma = float(gate_sigma)
        self.gate_edge_power = float(gate_edge_power)
        self.channels = tuple(channels)
        self.plane_init_std = float(plane_init_std)
        self.params_: Dict[str, Dict[str, torch.Tensor]] = {}
        self.gates: Dict[str, torch.Tensor] = {}
        self._d: Optional[torch.Tensor] = None
        self._last_planes: Optional[List[torch.Tensor]] = None

    name = "dct_nonadd"

    # ---- 介面 ----

    def reset(self, x01: torch.Tensor, seed: int) -> None:
        dev, dt = x01.device, x01.dtype
        self._d = dct_matrix(dev, dt)
        k = len(self.idx)
        g = torch.Generator(device="cpu").manual_seed(seed)
        planes = to_planes(x01)
        self.params_, self.gates = {}, {}
        for i, name in enumerate(CHANNELS):
            coef = block_dct(planes[i], self._d)          # (N, hb, wb, 8, 8)
            n, hb, wb = coef.shape[0], coef.shape[1], coef.shape[2]
            if self.gate == "band":
                gt = torch.ones(n, hb, wb, device=dev, dtype=dt)
            else:
                gt = block_texture_gate(x01, wb, self.gate_sigma,
                                        self.gate_edge_power)
            self.gates[name] = gt.detach()
            if name not in self.channels:
                continue
            if self.mode == "gain":
                self.params_[name] = {
                    "g": torch.zeros(n, hb, wb, k, device=dev, dtype=dt,
                                     requires_grad=True)}
            else:
                shape = (1, 1, 1, k) if self.mode == "shared_plane" else (n, hb, wb, k)
                tshape = (1, 1, 1, 1) if self.mode == "shared_plane" else (n, hb, wb, 1)
                # **平面必須隨機起始。** 全零的 u、v 是退化平面，梯度恆為零，
                # 整批會靜默地不動——與位移場那一格同一種死法。
                mk = lambda: (torch.randn(shape, generator=g) * self.plane_init_std
                              ).to(device=dev, dtype=dt).requires_grad_(True)
                self.params_[name] = {
                    "u": mk(), "v": mk(),
                    "theta": torch.zeros(tshape, device=dev, dtype=dt,
                                         requires_grad=True)}

    def render(self, x01: torch.Tensor) -> torch.Tensor:
        planes = to_planes(x01)
        outs = []
        rows = torch.tensor([u for u, _ in self.idx], device=x01.device)
        cols = torch.tensor([v for _, v in self.idx], device=x01.device)
        for i, name in enumerate(CHANNELS):
            coef = block_dct(planes[i], self._d)
            if name in self.params_:
                p = self.params_[name]
                gt = self.gates[name][..., None]              # (N,hb,wb,1)
                sub = coef[..., rows, cols]                   # (N,hb,wb,k)
                if self.mode == "gain":
                    scale = torch.exp(p["g"] * gt)
                    new = sub * scale
                else:
                    new = rotate_in_plane(sub, p["u"], p["v"], p["theta"] * gt)
                coef = coef.clone()
                coef[..., rows, cols] = new
            outs.append(block_idct(coef, self._d))
        self._last_planes = [o.detach() for o in outs]
        return from_planes(outs)

    def params(self) -> List[torch.Tensor]:
        out = []
        for name in CHANNELS:
            if name in self.params_:
                out.extend(self.params_[name].values())
        return out

    @torch.no_grad()
    def project(self) -> None:
        """只夾**強度**，不夾平面。平面是方向、沒有預算的意義。"""
        for name, p in self.params_.items():
            key = "g" if self.mode == "gain" else "theta"
            p[key].clamp_(-self.radius, self.radius)

    def set_radius(self, r: float) -> None:
        self.radius = float(r)

    # ---- 診斷 ----

    @torch.no_grad()
    def degenerate_fraction(self, eps: float = 1e-8) -> float:
        """退化平面（`u` 或正交化後的 `v` 長度接近 0）的比例。

        報表要看這一欄：比例高代表大量區塊其實沒有在轉，那時「效果不好」
        是實作問題而不是方法問題。
        """
        if self.mode == "gain":
            return 0.0
        bad = tot = 0
        for p in self.params_.values():
            u, v = p["u"], p["v"]
            n1 = u.norm(dim=-1)
            e1 = u / n1.clamp_min(eps).unsqueeze(-1)
            w = v - (v * e1).sum(dim=-1, keepdim=True) * e1
            m = (n1 <= eps) | (w.norm(dim=-1) <= eps)
            bad += int(m.sum()); tot += m.numel()
        return bad / max(1, tot)

    @torch.no_grad()
    def clip_fraction(self) -> float:
        """最近一次 `render` 有多少比例的像素被值域夾取截掉。

        DCT 域的旋轉精確保長，但轉出 `[0,1]` 的像素被夾回去之後那份能量就沒了。
        這一欄高就代表「保長」在像素域已經名存實亡，等失真比較要據此加註。
        """
        if self._last_planes is None:
            return float("nan")
        raw = _unclamped(self._last_planes)
        return float(((raw < 0.0) | (raw > 1.0)).to(torch.float64).mean())


class DctNonAdditiveRandomParam(DctNonAdditiveParam):
    """同上界的隨機解，**不最佳化**。每一個新參數化都要有這一格。

    `RESULTS.md` 的 FND-004 就是栽在沒有同失真隨機對照上；本專案此後每一個
    參數化都配一個。抽法與 `phase_rand`／`warp_rand` 相同：sign PGD 會把
    L∞ 球用滿，故隨機對照也取用滿的分布。
    """

    name = "dct_nonadd_rand"

    def reset(self, x01: torch.Tensor, seed: int) -> None:
        super().reset(x01, seed)
        g = torch.Generator(device="cpu").manual_seed(seed + 9973)
        key = "g" if self.mode == "gain" else "theta"
        with torch.no_grad():
            for p in self.params_.values():
                t = p[key]
                init = torch.randn(t.shape, generator=g) * self.radius
                t.copy_(init.clamp(-self.radius, self.radius).to(
                    device=t.device, dtype=t.dtype))

    def params(self) -> List[torch.Tensor]:
        return []

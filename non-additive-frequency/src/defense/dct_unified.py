"""把「學出來的旋轉平面」與「量化交付」整併成**一個**模組。**探索性質。**

為什麼要有這一支
────────────────────────────────────────────────────────────────────
現行做法是兩段拼起來的：擾動在**重疊加窗的 STFT** 上設計出來（
`texture_rephase`），交付的時候再壓成 JPEG（`--deliver-jpeg`）。最佳化在一個
空間裡找解，交付把解投影到另一個空間，**投影會削掉一塊**。這一塊已經量到
兩次，而且在兩個域上都出現：

| 參數化 | 不交付 | 交付 | 代價 |
|---|---|---|---|
| 相位＋增益（STFT） | 等失真位移 0.7012 | 0.5561 | **−21%**（`ip2p_deliver_jpeg`） |
| 學習平面（浮點 DCT） | `nd_plane_31` 0.1535／0.6571 | `nd_plane_25_qd` 0.1617／0.5679 | **兩軸皆被支配**（`ip2p_dct_nonadd`） |

DCT-Shield 沒有這一段。它的參數**直接就是** JPEG 的整數係數，改完解碼出來就
是要交的圖——參數即交付，沒有投影，抗 JPEG 是白送的。它的架構之所以簡單又
有效，關鍵是這個選擇，不是省事。

本檔把同一個選擇套到我們的參數化上：**旋轉直接作用在量化後的整數係數上，
交出去的就是那組整數解碼出來的圖。**

與前兩批的關係
────────────────────────────────────────────────────────────────────
| 批次 | 旋轉平面 | 作用在 | 交付 |
|---|---|---|---|
| `dct_rotation.py` | **固定配對**（28 對） | 整數係數 | 即參數 |
| `dct_nonadditive.py` | **學出來的** | 浮點係數 | 浮點影像（或事後再壓一次） |
| **本檔** | **學出來的** | **整數係數** | **即參數** |

前兩批各自失敗於一個已查明的原因：固定配對輸在「配對」這個限制（學出來的
平面在同失真上是它的 2.1 倍），事後投影輸在可行集變小。**兩者的組合從未跑
過**，而它正是 `runs/dct_phase_design/README.md` §3.1 那條可證偽預測的載體：
若那 21% 真的來自事後投影，本檔不該付。

三件必須先寫明的事
────────────────────────────────────────────────────────────────────
1. **`theta = 0` 不是恆等映射。** 輸出逐位元等於 `jpeg_roundtrip(x, qd)`，
   而不是原圖——交付本身就是壓縮圖。這與 `texture_rephase` 的恆等性質不同，
   與 DCT-Shield 相同（它的 `δ = 0` 也是壓縮圖）。由測試釘住。
2. **旋轉之後必須再取整。** 交出去的動作因此是「一個整數的係數位移」，也就是
   DCT-Shield 的動作空間。**本檔不是一個它做不到的操作，而是一個把位移限制在
   「通過 α 的球面」上的受約束子集。** 論文的措辭只能是「約束不同」，不可以
   寫成「動作不同」（`dct_phase_design` §5.3，實測帶內工作點上 7.3% 的位移
   超過一個量化階）。
3. **保長只在量化前的整數向量上精確成立。** 取整之後有誤差，`delta_stats()`
   把它報出來。8×8 DCT 正交歸一且區塊不重疊，所以保長在像素域也成立——
   但那是取整**之前**的性質。

參數量
────────────────────────────────────────────────────────────────────
每個區塊一個平面 `span(u, v)` 與一個角度，`k` 是通帶內的 AC 係數個數
（`r_min = 0.12` 在 8×8 上只排除 DC，故 k = 63）。512² 的 Y 平面有 64×64 個
區塊，色度因 4:2:0 各 32×32 個，合計約 **62 萬**個參數，與相位法的 59 萬同級
——**容量匹配之後再比，才分得出「域不對」與「自由度不夠」**。
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import torch

from src.baselines.jpeg_codec import (
    CHANNEL_NAMES, jpeg_decode, jpeg_encode, normalize_quality, quant_table,
)
from src.defense.dct_nonadditive import band_indices, rotate_in_plane
from src.defense.dct_rotation import block_texture_gate


class DctUnifiedParam:
    """`param_pgd.Parameterization` 的實作。強度旗鈕 `radius` 即角度上界。

    `qd` 是交付品質，與 `--deliver-jpeg` 的 QD 同一個語意；差別是**這裡它是
    參數化的一部分**，不是接在後面的一步。故本條件不可再疊 `--deliver-jpeg`
    （那是壓兩次，而且第二次的品質未必等於第一次）。
    """

    name = "dct_unified"

    def __init__(
        self,
        radius: float = 2.2,
        qd: float = 0.85,
        r_min: float = 0.12,
        r_max: float = float("inf"),
        gate: str = "texture",
        gate_sigma: float = 2.0,
        gate_edge_power: float = 1.0,
        channels: Sequence[str] = CHANNEL_NAMES,
        plane_init_std: float = 1.0,
        plane_weight: str = "uniform",
    ):
        bad = set(channels) - set(CHANNEL_NAMES)
        if bad:
            raise ValueError(f"未知通道 {sorted(bad)}；可用的是 {CHANNEL_NAMES}")
        if not channels:
            raise ValueError("channels 不可為空——那樣 theta 沒有任何自由度")
        if gate not in ("texture", "band"):
            raise ValueError(f"未知的閘 {gate!r}，可用的是 texture／band")
        self.radius = float(radius)
        self.qd = float(qd)
        self.idx = band_indices(r_min, r_max)
        if not self.idx:
            raise ValueError(f"r_min={r_min}／r_max={r_max} 下通帶內沒有 AC 係數")
        self.gate = gate
        self.gate_sigma = float(gate_sigma)
        self.gate_edge_power = float(gate_edge_power)
        if plane_weight not in ("uniform", "priced"):
            raise ValueError(
                f"未知的 plane_weight {plane_weight!r}，可用的是 uniform／priced")
        self.channels = tuple(channels)
        self.plane_init_std = float(plane_init_std)
        # 旋轉的目標方向要不要依知覺定價加權。`uniform` 逐位元等於加這個
        # 旗鈕之前。理由見 `runs/integration_design/README.md` 的動作天花板：
        # 等殘差下 `priced` 的 DISTS 只有 `uniform` 的 0.69–0.73 倍。
        self.plane_weight = plane_weight
        self.weights: Dict[str, torch.Tensor] = {}
        self.alpha: Dict[str, torch.Tensor] = {}
        self.gates: Dict[str, torch.Tensor] = {}
        self.params_: Dict[str, Dict[str, torch.Tensor]] = {}
        self._quality: Optional[int] = None
        self._last_delta: Optional[Dict[str, torch.Tensor]] = None

    # ---- Parameterization 介面 ----

    def reset(self, x01: torch.Tensor, seed: int) -> None:
        """由原圖算一次整數係數與閘；平面隨機起始，角度由零起步。

        **平面必須隨機起始**：全零的 `u`、`v` 是退化平面，梯度恆為零，整批會
        靜默地不動——`ip2p_warp` 那一批就是這樣死的。角度則由零起步，使
        `theta = 0` 的恆等性質（對壓縮圖而言）在第 0 步成立。
        """
        q = normalize_quality(self.qd)
        with torch.no_grad():
            self.alpha = {k: v.detach()
                          for k, v in jpeg_encode(x01, self.qd).items()}
        g = torch.Generator(device="cpu").manual_seed(seed)
        k = len(self.idx)
        self.params_, self.gates = {}, {}
        for name in CHANNEL_NAMES:
            a = self.alpha[name]
            n, hb, wb = a.shape[0], a.shape[1], a.shape[2]
            if self.gate == "band":
                gt = torch.ones(n, hb, wb, device=a.device, dtype=a.dtype)
            else:
                # 色度平面因 4:2:0 邊長減半，`block_texture_gate` 由 `wb` 反推
                # 池化核，故同一支對 Y 與色度都對得齊。
                gt = block_texture_gate(x01, wb, self.gate_sigma,
                                        self.gate_edge_power)
            self.gates[name] = gt.detach()
            if name not in self.channels:
                continue

            # 量化步長大＝該頻率人眼不敏感＝把能量搬去那裡便宜。這張表在
            # 8×8 上是**原生的**，不必像 STFT 那一臂重採樣到 rfft2 格點。
            if self.plane_weight == "priced":
                tbl = quant_table(q, chroma=(name != "Y"),
                                  device=a.device, dtype=a.dtype)
                w = torch.tensor([float(tbl[u, v]) for u, v in self.idx],
                                 device=a.device, dtype=a.dtype)
                self.weights[name] = (w / w.mean()).detach()
            else:
                self.weights[name] = torch.ones(k, device=a.device,
                                                dtype=a.dtype)

            def mk(shape):
                t = torch.randn(shape, generator=g) * self.plane_init_std
                return t.to(device=a.device, dtype=a.dtype).requires_grad_(True)

            self.params_[name] = {
                "u": mk((n, hb, wb, k)),
                "v": mk((n, hb, wb, k)),
                "theta": torch.zeros(n, hb, wb, 1, device=a.device,
                                     dtype=a.dtype, requires_grad=True),
            }
        self._quality = q

    def render(self, x01: torch.Tensor) -> torch.Tensor:
        rows = torch.tensor([u for u, _ in self.idx], device=self.alpha["Y"].device)
        cols = torch.tensor([v for _, v in self.idx], device=self.alpha["Y"].device)
        coef: Dict[str, torch.Tensor] = {}
        delta: Dict[str, torch.Tensor] = {}
        for name in CHANNEL_NAMES:
            a = self.alpha[name]
            if name not in self.params_:
                coef[name] = a
                continue
            p = self.params_[name]
            gt = self.gates[name][..., None]
            sub = a[..., rows, cols]
            # **定價乘在方向上、不乘在角度上**：角度是預算（由 radius 夾），
            # 方向是「往哪裡搬」。乘在角度上會變成逐格的強度調整，那是另一件事。
            pw = self.weights[name]
            new = rotate_in_plane(sub, p["u"] * pw, p["v"] * pw,
                                  p["theta"] * gt)
            r = a.clone()
            r[..., rows, cols] = new
            # 取整走直通估計：前向是真的 round，反向當恆等（DiffJPEG）。
            # **交付時前向值逐位元相同**，故存檔的圖與最佳化看到的是同一張。
            q = r + (torch.round(r) - r).detach()
            coef[name] = q
            delta[name] = (q - a).detach()
        self._last_delta = delta
        return jpeg_decode(coef, self._quality)

    def params(self) -> List[torch.Tensor]:
        out: List[torch.Tensor] = []
        for name in CHANNEL_NAMES:
            if name in self.params_:
                out.extend(self.params_[name].values())
        return out

    @torch.no_grad()
    def project(self) -> None:
        """只夾角度，不夾平面。平面是方向，沒有預算的意義。"""
        for p in self.params_.values():
            p["theta"].clamp_(-self.radius, self.radius)

    def set_radius(self, r: float) -> None:
        self.radius = float(r)

    # ---- 診斷（報表用，不參與最佳化） ----

    @torch.no_grad()
    def delta_stats(self) -> Dict[str, float]:
        """交出去的整數位移長什麼樣。三個欄位都要進 CSV。

        | 欄 | 為什麼要 |
        |---|---|
        | `delta_within_1` | 位移落在 ±1 個量化階內的比例。**這一欄決定新穎性
          怎麼寫**：比例高就代表我們動的東西幾乎全在 DCT-Shield 的 ε=1 球裡，
          論文只能主張「約束不同」不能主張「動作不同」 |
        | `delta_nonzero` | 位移不為零的比例。低就代表大量區塊其實沒有動 |
        | `zero_coef_frac` | 通帶內量化值為零的係數比例。旋轉零向量還是零向量，
          這一欄是可行集稀薄程度的直接讀數 |
        """
        if self._last_delta is None:
            return {"delta_within_1": float("nan"),
                    "delta_nonzero": float("nan"),
                    "zero_coef_frac": float("nan")}
        rows = [u for u, _ in self.idx]
        cols = [v for _, v in self.idx]
        within = nonzero = tot = 0
        zeros = ztot = 0
        for name, d in self._last_delta.items():
            within += int((d.abs() <= 1.0).sum())
            nonzero += int((d.abs() > 0.0).sum())
            tot += d.numel()
            sub = self.alpha[name][..., rows, cols]
            zeros += int((sub == 0).sum())
            ztot += sub.numel()
        return {
            "delta_within_1": round(within / max(1, tot), 5),
            "delta_nonzero": round(nonzero / max(1, tot), 5),
            "zero_coef_frac": round(zeros / max(1, ztot), 5),
        }


class DctUnifiedRandomParam(DctUnifiedParam):
    """同上界的隨機解，**不最佳化**。每一個新參數化都要有這一格。

    `RESULTS.md` 的 FND-004 就是栽在沒有同失真隨機對照上。抽法與
    `dct_nonadd_rand` 相同：sign PGD 會把角度用滿，故隨機對照也取用滿的分布。
    """

    name = "dct_unified_rand"

    def reset(self, x01: torch.Tensor, seed: int) -> None:
        super().reset(x01, seed)
        g = torch.Generator(device="cpu").manual_seed(seed + 9973)
        with torch.no_grad():
            for p in self.params_.values():
                t = p["theta"]
                init = torch.randn(t.shape, generator=g) * self.radius
                t.copy_(init.clamp(-self.radius, self.radius).to(
                    device=t.device, dtype=t.dtype))

    def params(self) -> List[torch.Tensor]:
        return []

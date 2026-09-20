"""亮度平台曲線：斜率真的可以等於零的單調色調曲線。

為什麼要另寫一個參數化
────────────────────────────────────────────────────────────────────
`ColorCurveParam` 的 `θ` 夾在 `[1/K, (1+radius)/K]`，正規化之後有效斜率的
下限是 `1/(1+radius)`——radius 5 是 0.169，radius 50 也還有 0.020。
所以 `readout_terms.ToneFlatness` 想做的「把一段色調壓平」在那個載體上
**構造上不可達**，它輸給 `free` 不能當成對該機制的否定。

本模組換一個參數化來測那個機制本身：

    m(u) = Π_i [ s_i + (1 − s_i)·smoothstep((|u − c_i| − h_i)/τ) ]
    F(x) = ∫_0^x m / ∫_0^1 m

`s_i = 0` 時 `m` 在 `|u − c_i| ≤ h_i` 上**恰為零**，正規化不會把它抬起來
（0 乘任何常數仍是 0）。落在該區間的亮度值全部映到同一個輸出，
這是單調曲線唯一不可逆的破壞。

三個由構造保證的性質
────────────────────────────────────────────────────────────────────
- `m ≥ 0` 故 `F` 非遞減；`F(0) = 0`、`F(1) = 1`。
- `h_i = 0` 且 `s_i = 1` 時 `m ≡ 1`，`F` 是恆等，逐位元還原原圖。
- 平台核心之外以 `smoothstep` 過渡，`m` 連續；正規化把被壓掉的積分量
  攤回其餘區間——「其餘區間平滑補償」是這個正規化，不是另一組參數。

平台中心 `c_i` **不是最佳化變數**，由原圖（遮罩內）的亮度分布固定，見
`centres_from_luma`。可最佳化的只有每段的半寬 `h_i` 與斜率 `s_i`。

套用方式（`mode`）
────────────────────────────────────────────────────────────────────
| 值 | 作法 | 被壓平的是 |
|---|---|---|
| `luma_gain` | `out = x · F(L)/L`，`L` 為 BT.601 亮度 | 亮度，色度比例保留 |
| `per_channel` | `out_c = F(x_c)`，三通道同一條曲線 | 三個通道各自 |

`apply_where` 給定時輸出為 `w·F(x) + (1−w)·x`。遮罩核心是 `w = 1`，
該處**完全套用曲線**，不摻回任何原圖細節；`w` 只在羽化帶上取中間值。
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import torch

from .color_param import luma

EPS = 1e-6


def smoothstep01(z: torch.Tensor) -> torch.Tensor:
    """`z ≤ 0` 回 0，`z ≥ 1` 回 1，之間走 `3z² − 2z³`。"""
    t = z.clamp(0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def centres_from_luma(x01: torch.Tensor, weight: Optional[torch.Tensor] = None,
                      count: int = 2) -> List[float]:
    """由（遮罩內的）亮度分布固定平台中心。

    取加權分位數 `(2i+1)/(2·count)`：`count = 2` 時就是下四分位與上四分位。
    兩個中心因此各自坐落在一半像素質量的中央，與最佳化無關，
    也不讀任何評估結果。
    """
    if count < 1:
        raise ValueError(f'平台數至少為 1，收到 {count}')
    lum = luma(x01).reshape(-1)
    w = torch.ones_like(lum) if weight is None \
        else weight.to(lum.dtype).reshape(-1)
    if float(w.sum()) <= 0:
        raise ValueError('權重總和為零，亮度分位數沒有定義')
    order = torch.argsort(lum)
    lum_s, w_s = lum[order], w[order]
    cdf = torch.cumsum(w_s, 0) / w_s.sum()
    out = []
    for i in range(count):
        q = torch.tensor((2.0 * i + 1.0) / (2.0 * count), dtype=cdf.dtype)
        j = int(torch.searchsorted(cdf, q))
        out.append(float(lum_s[min(j, lum_s.numel() - 1)]))
    return out


class PlateauCurveParam:
    """φ = (h, s)，每段平台一個半寬與一個斜率。中心固定不進最佳化。

    `slope_floor = 0.0` 是本參數化存在的理由：`project()` 把 `s` 夾到
    `[slope_floor, 1]`，取 0 時零斜率是可行點而不是極限。
    對照組把 `slope_floor` 設成正值（例如 0.169，即 `ColorCurveParam`
    radius 5 的有效下限），載體、中心、其餘一切相同，只差平台壓不壓得平。
    """

    name = "plateau_curve"

    def __init__(self, centres: Sequence[float], pieces: int = 256,
                 transition: float = 0.03,
                 max_half_width: float = 0.25,
                 slope_floor: float = 0.0,
                 mode: str = "luma_gain",
                 apply_where: Optional[torch.Tensor] = None):
        if pieces < 8:
            raise ValueError(f'pieces 至少為 8，收到 {pieces}')
        if mode not in ('luma_gain', 'per_channel'):
            raise ValueError(
                f"mode 只能是 'luma_gain' 或 'per_channel'，收到 {mode!r}")
        if transition <= 0:
            raise ValueError(f'transition 必須為正，收到 {transition}')
        if not 0.0 <= float(slope_floor) < 1.0:
            raise ValueError(f'slope_floor 要落在 [0,1)，收到 {slope_floor}')
        self.centres = [float(c) for c in centres]
        if not self.centres:
            raise ValueError('至少要有一段平台')
        self.pieces = int(pieces)
        self.transition = float(transition)
        self.max_half_width = float(max_half_width)
        self.slope_floor = float(slope_floor)
        self.mode = mode
        self.apply_where = apply_where
        self.half_width: Optional[torch.Tensor] = None
        self.slope: Optional[torch.Tensor] = None

    def reset(self, x01: torch.Tensor, seed: int = 0,
              half_width: float = 0.0, slope: float = 1.0) -> None:
        """預設起點是恆等（半寬 0、斜率 1）。`seed` 不起作用。

        **恆等起點對半寬是零梯度點，要最佳化半寬就不能從那裡起步。**
        `m` 的每一項是 `s + (1−s)·t`，對半寬的偏導是 `(1−s)·∂t/∂h`；
        `s = 1` 時那個因子恰為零，無論 `t` 落在過渡帶的哪裡。所以恆等起點上
        `∂L/∂h ≡ 0`，只有 `∂L/∂s`（因子 `1 − t`）非零。

        後果是求解器在第 0 步只動得了斜率；斜率一離開 1，半寬才拿得到梯度。
        要讓半寬從一開始就參與，起點要給 `slope < 1`，或給一個非零的
        `half_width`——後者單獨仍不夠，`s = 1` 時半寬怎麼給都沒有梯度。
        這與 `ColorCurveParam` 的零梯度點成因不同（那裡是差向量範數在恆等處
        的次梯度為零），但要避開的做法一樣：起點不要放在恆等上。
        """
        n = len(self.centres)
        dev, dt = x01.device, x01.dtype
        self.half_width = torch.full((n,), float(half_width),
                                     device=dev, dtype=dt).requires_grad_(True)
        self.slope = torch.full((n,), float(slope),
                                device=dev, dtype=dt).requires_grad_(True)

    def params(self) -> List[torch.Tensor]:
        return [self.half_width, self.slope]

    def state_dict(self):
        return {'half_width': self.half_width.detach().clone(),
                'slope': self.slope.detach().clone()}

    def load_state_dict(self, state):
        self.half_width = state['half_width'].detach().clone().requires_grad_(True)
        self.slope = state['slope'].detach().clone().requires_grad_(True)

    @property
    def stages(self):
        return [self]

    @torch.no_grad()
    def project(self) -> None:
        self.half_width.clamp_(0.0, self.max_half_width)
        self.slope.clamp_(self.slope_floor, 1.0)

    def step_scale(self) -> float:
        return self.max_half_width

    def slope_profile(self) -> torch.Tensor:
        """(K,) 未正規化的斜率剖面，在各段的中點取值。"""
        k = self.pieces
        dev, dt = self.half_width.device, self.half_width.dtype
        u = (torch.arange(k, device=dev, dtype=dt) + 0.5) / k
        m = torch.ones_like(u)
        for i, c in enumerate(self.centres):
            d = (u - c).abs()
            t = smoothstep01((d - self.half_width[i]) / self.transition)
            m = m * (self.slope[i] + (1.0 - self.slope[i]) * t)
        return m

    def curve(self, v: torch.Tensor) -> torch.Tensor:
        """把 [0,1] 的值逐點過曲線。形狀任意，逐元素。"""
        k = self.pieces
        m = self.slope_profile()
        cum = torch.cat([m.new_zeros(1), m.cumsum(0)[:-1]])
        vc = v.clamp(0.0, 1.0)
        idx = (vc * k).floor().clamp_(0, k - 1).long()
        return (cum[idx] + (vc * k - idx.to(vc.dtype)) * m[idx]) / m.sum()

    def render(self, x01: torch.Tensor) -> torch.Tensor:
        if self.mode == 'per_channel':
            out = self.curve(x01)
        else:
            lum = luma(x01)
            out = (x01 * (self.curve(lum) / lum.clamp_min(EPS))).clamp(0.0, 1.0)
        if self.apply_where is None:
            return out
        w = self.apply_where.to(device=out.device, dtype=out.dtype)
        return w * out + (1.0 - w) * x01

    def flat_coverage(self, x01: torch.Tensor,
                      weight: Optional[torch.Tensor] = None,
                      slope_tol: float = 1e-6) -> float:
        """落在**零斜率**平台核心內的（加權）像素比例。

        只計 `s_i ≤ slope_tol` 的那些段——斜率不為零的平台沒有壓平任何東西，
        不該計進這個讀數。
        """
        v = luma(x01) if self.mode == 'luma_gain' else x01
        inside = torch.zeros_like(v, dtype=torch.bool)
        hw, sl = self.half_width.detach(), self.slope.detach()
        for i, c in enumerate(self.centres):
            if float(sl[i]) > slope_tol:
                continue
            inside |= ((v - c).abs() <= hw[i])
        f = inside.to(v.dtype)
        if weight is None:
            return float(f.mean())
        w = weight.to(device=f.device, dtype=f.dtype).expand_as(f)
        return float((f * w).sum() / w.sum().clamp_min(EPS))

    def clip_fraction(self, x01: torch.Tensor) -> float:
        """`luma_gain` 下被 clamp 吃掉的像素比例。`per_channel` 恆為 0。"""
        if self.mode == 'per_channel':
            return 0.0
        lum = luma(x01)
        raw = x01 * (self.curve(lum) / lum.clamp_min(EPS))
        return float(((raw < 0.0) | (raw > 1.0)).any(1, keepdim=True)
                     .to(x01.dtype).mean())

    def readout(self) -> dict:
        return {'plateau_centres': [round(c, 5) for c in self.centres],
                'plateau_half_width': [round(float(v), 5)
                                       for v in self.half_width.detach()],
                'plateau_slope': [round(float(v), 5)
                                  for v in self.slope.detach()],
                'plateau_transition': self.transition,
                'plateau_slope_floor': self.slope_floor,
                'plateau_mode': self.mode}

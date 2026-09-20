"""色彩重映射：把擾動放進**色彩的自由度**，而不是放進空間高頻。

一個**全域逐點色彩映射** `T` 同時繞開這兩個：

- `crop(T(x)) = T(crop(x))` **恆等成立**，沒有需要對齊的空間相位，
  同步問題在構造上不存在。
- `T` 改的是局部均值，而模糊保留局部均值；能量落在 `f_n ≲ 0.03`，
  高斯 σ=1 的核在該帶近似恆等。
- JPEG 動到的是各區塊的 **DC 係數**，位移量遠高於一個量化階，
  不受「純乘性在平坦區推不動」那條天花板（`|ΔS| = |S|·2|sin(θ/2)|`）限制。

本模組提供兩個參數化。兩者都**零半徑即逐位元恆等**，與相位族的 `θ = 0`
同性質。

| 類別 | 構造 | 參數量 | 裁切等變 |
|---|---|---|---|
| `ColorCurveParam` | 逐通道 K 段單調分段線性曲線，全域 | 3K | **精確** |
| `ColorGridParam` | 雙邊網格上的仿射色彩變換（空間 G×G × 亮度 D 格） | 12·D·G² | G=1 時精確，否則近似 |

`ColorGridParam` 在 `G = 1、D = 2、只動對角線` 的特例下就是 `ShadingParam`
那一族（乘性增益），差別是它多了偏移項，因此在暗部不受 `x · exp(m)` 的
「乘上零還是零」限制，也不像它那樣一大就把亮部推進 clamp 的死白。

移植說明
────────────────────────────────────────────────────────────────────
`ColorCurveParam` 的曲線構造逐行取自 AdvCF 的公開程式
（github.com/ZhengyuZhao/AdvColorFilter，`Journal_version/ACE.ipynb` 的
`CF()`）：

    color_curve_sum = torch.sum(param, 4) + 1e-30
    for i in range(steps):
        total_image += torch.clamp(img - 1.0*i/steps, 0, 1.0/steps) * param[...,i]
    total_image *= steps / color_curve_sum

本檔改用累積和 ＋ 逐通道索引取代那個 K 次迴圈，**輸出在浮點誤差內相同**
（`tests/test_color_param.py::test_curve_matches_advcf_loop` 逐點比對），
理由是原式會留下 K 個 (N,3,H,W) 的中間張量給 autograd，K=64、512² 下是
200 MB。

**兩處與原文不同，逐項標明**：

半徑（`radius`）在兩個類別裡的單位不同，**不可互相換算**：曲線族是斜率剖面的
動態範圍 `1 + r`，網格族是仿射係數對單位矩陣的 L∞ 偏移。等強度比較一律走
`scripts/tradeoff_curve.py` 的等失真內插。
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

# ITU-R BT.601 亮度權重。雙邊網格的導引通道用它，與 `src/metrics/acutance.py`
# 的 `_luma` 同一組係數——兩處若不一致，「導引落在哪一格」與「銳度量在哪個
# 通道」講的就不是同一個亮度。
LUMA_WEIGHTS = (0.299, 0.587, 0.114)

# 12 = 3 個輸出通道 × (3 個混合係數 ＋ 1 個偏移)。列優先，故對角線落在
# 0、5、10，偏移落在 3、7、11。
AFFINE_ENTRIES = 12
_DIAG = (0, 5, 10)


def luma(x01: torch.Tensor) -> torch.Tensor:
    """(N,3,H,W) → (N,1,H,W)，值域與輸入相同。"""
    w = torch.tensor(LUMA_WEIGHTS, device=x01.device, dtype=x01.dtype)
    return (x01 * w.view(1, 3, 1, 1)).sum(1, keepdim=True)


class ColorCurveParam:
    """φ = θ（1,3,K），逐通道 K 段單調分段線性曲線，全域套用。

    輸出

        F_c(x) = ( Σ_{i<k} θ_{c,i}/K + (x − k/K)·θ_{c,k} ) · K / Σ_i θ_{c,i}

    其中 `k = ⌊Kx⌋`。三個由構造保證的性質：

    `apply_where` (1,1,H,W) 給定時，輸出為 `w·F(x) + (1−w)·x`——`w = 0` 的地方
    逐位元保留原圖。用於「受保護主體不得改動」的設定。
    """

    name = "color_curve"

    def __init__(self, radius: float = 1.0, pieces: int = 64,
                 bound_mode: str = "symmetric",
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

    # ---- 介面 ----

    def reset(self, x01: torch.Tensor, seed: int) -> None:
        """`init_jitter = 0` 時起點是恆等曲線，`seed` 不起作用。

        多次重啟要有意義，起點必須真的不同：恆等起點在任何 `seed` 上都一樣，
        重啟只會走回同一個盆。`init_jitter` 把起點往可行盒裡的均勻抽樣插值，
        1.0 為完全隨機、0.0 為恆等。
        """
        k = self.pieces
        flat = torch.full((1, 3, k), 1.0 / k,
                          device=x01.device, dtype=x01.dtype)
        if self.init_jitter > 0:
            lo, hi = self.bounds()
            gen = torch.Generator(device="cpu").manual_seed(int(seed))
            u = torch.rand((1, 3, k), generator=gen).to(x01.device, x01.dtype)
            flat = (1.0 - self.init_jitter) * flat \
                + self.init_jitter * (lo + (hi - lo) * u)
        self.theta = flat.clone().requires_grad_(True)

    def render(self, x01: torch.Tensor) -> torch.Tensor:
        out = self._curve(x01, self.theta)
        if self.apply_where is None:
            return out
        w = self.apply_where.to(device=out.device, dtype=out.dtype)
        return w * out + (1.0 - w) * x01

    def params(self) -> List[torch.Tensor]:
        return [self.theta]

    def state_dict(self):
        """`immunise.optimise_carrier` 存可行 checkpoint 用的。"""
        return {'theta': self.theta.detach().clone()}

    def load_state_dict(self, state):
        self.theta = state['theta'].detach().clone().requires_grad_(True)

    @property
    def stages(self):
        return [self]

    def set_amplitude(self, a):
        if isinstance(a, (list, tuple)):
            if len(a) != 1:
                raise ValueError('這個載體只有一段，逐段指定時長度要是 1')
            a = a[0]
        self.amplitude = float(a)

    @torch.no_grad()
    def project(self) -> None:
        lo, hi = self.bounds()
        self.theta.clamp_(lo, hi)

    def set_radius(self, r: float) -> None:
        self.radius = r

    def step_scale(self) -> float:
        """步長公式要的是**盒寬**，不是半徑。

        本族的半徑是斜率剖面的動態範圍 `1 + r`，與 `θ` 的尺度差兩個數量級：
        K=64、r=3 時盒寬只有 0.059，而 `radius/(steps·0.25)` 會給 0.012。
        沒有這一個掛鉤的話 sign 更新五步就撞到邊界。
        """
        lo, hi = self.bounds()
        return hi - lo

    # ---- 構造 ----

    def bounds(self) -> Tuple[float, float]:
        """`θ` 的可行區間。`radius = 0` 時上下界都等於 `1/K`，即恆等。"""
        k = float(self.pieces)
        span = 1.0 + max(0.0, float(self.radius))
        if self.bound_mode == "advcf":
            return 1.0 / k, span / k
        return 1.0 / (k * span), span / k

    def _curve(self, x01: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        k = self.pieces
        th = theta.to(x01.dtype)
        # `x = 1` 落在第 K 段之外，夾到最後一段；那一點的輸出仍是 F(1) = 1。
        idx = (x01 * k).floor().clamp_(0, k - 1).long()
        s = th.sum(-1).view(1, 3, 1, 1)
        outs = []
        for c in range(3):
            th_c = th[0, c]                                    # (K,)
            # c_c[j] = Σ_{i<j} θ_i。前置一個 0，故 c_c[0] = 0。
            c_c = torch.cat([th_c.new_zeros(1), th_c.cumsum(0)[:-1]])
            kc = idx[:, c]                                     # (N,H,W)
            outs.append(c_c[kc] / k
                        + (x01[:, c] - kc.to(x01.dtype) / k) * th_c[kc])
        return torch.stack(outs, 1) * (k / s)


class ColorCurveLowFreqRandom(ColorCurveParam):
    """低頻隨機的單調曲線：**不最佳化**，只在 reset 時抽一條平滑的斜率剖面。

    為什麼不能用 i.i.d. 抽樣當隨機對照
    ────────────────────────────────────────────────────────────────
    斜率剖面逐段獨立抽樣時，累積起來的曲線會回到對角線——各段的偏差互相抵銷。
    實測 `radius = 5`、`init_jitter = 1` 的均勻抽樣在三張人像上只到 ΔE00
    **7.11／4.14／5.90**，連 16 的錨點都搆不到，拿它當對照等於送分給最佳化。

    對抗解出來的曲線不是逐段獨立的：它把整段色調一起壓平或拉開，相鄰段高度
    相關。

    構造
    ────────────────────────────────────────────────────────────────
    每個通道抽 `modes` 個低頻餘弦分量，相位與振幅隨機，疊起來後線性映到可行盒
    `[lo, hi]`。`init_jitter` 仍然是幅度旋鈕（0 為恆等、1 為滿幅），所以
    `color_amplitude.fit_curve_jitter` 可以直接二分它來對齊 ΔE00。
    """

    name = "color_curve_lowfreq_rand"

    def __init__(self, *args, modes: int = 3, **kwargs):
        super().__init__(*args, **kwargs)
        if modes < 1:
            raise ValueError(f"modes 至少為 1，收到 {modes}")
        self.modes = int(modes)

    def reset(self, x01: torch.Tensor, seed: int) -> None:
        k = self.pieces
        lo, hi = self.bounds()
        gen = torch.Generator(device="cpu").manual_seed(int(seed))
        t = torch.linspace(0.0, 1.0, k)
        prof = torch.zeros((3, k))
        for c in range(3):
            for m in range(1, self.modes + 1):
                amp = torch.rand(1, generator=gen).item() / m
                pha = torch.rand(1, generator=gen).item() * 2 * torch.pi
                prof[c] += amp * torch.cos(torch.pi * m * t + pha)
        pmin = prof.min(dim=1, keepdim=True).values
        pmax = prof.max(dim=1, keepdim=True).values
        prof = (prof - pmin) / (pmax - pmin).clamp_min(1e-9)
        full = (lo + (hi - lo) * prof).unsqueeze(0).to(x01.device, x01.dtype)
        flat = torch.full((1, 3, k), 1.0 / k,
                          device=x01.device, dtype=x01.dtype)
        j = self.init_jitter
        self.theta = ((1.0 - j) * flat + j * full).clone().requires_grad_(True)


RAND_DRAWS = ("corner", "uniform")


def _draw(shape, lo, hi, draw: str, seed: int) -> torch.Tensor:
    if draw not in RAND_DRAWS:
        raise ValueError(f"draw 只能是 {RAND_DRAWS}，收到 {draw!r}")
    gen = torch.Generator(device="cpu").manual_seed(seed)
    u = torch.rand(shape, generator=gen)
    if draw == "corner":
        return torch.where(u < 0.5, torch.as_tensor(float(lo)),
                           torch.as_tensor(float(hi)))
    return lo + (hi - lo) * u


class ColorCurveRandomParam(ColorCurveParam):
    """同半徑的隨機曲線，**不最佳化**，只在 reset 時抽一次。

    `draw` 見 `RAND_DRAWS` 上方的說明：預設抽角點，與 sign 更新的可達集合
    相同。
    """

    name = "color_curve_rand"

    def __init__(self, *args, draw: str = "corner", **kwargs):
        super().__init__(*args, **kwargs)
        if draw not in RAND_DRAWS:
            raise ValueError(f"draw 只能是 {RAND_DRAWS}，收到 {draw!r}")
        self.draw = draw

    def reset(self, x01: torch.Tensor, seed: int) -> None:
        super().reset(x01, seed)
        lo, hi = self.bounds()
        with torch.no_grad():
            self.theta.copy_(_draw(self.theta.shape, lo, hi, self.draw,
                                   seed).to(device=x01.device,
                                            dtype=x01.dtype))

    def params(self) -> List[torch.Tensor]:
        return []


class ColorGridParam:
    """φ = A（1,12,D,G,G），雙邊網格上的仿射色彩變換。

    每個像素依 `(x/W, y/H, luma)` 三線性取樣出一個 3×4 仿射矩陣，
    再作用在它自己的 RGB 上：

        y_c = Σ_j m_{cj}(x, y, luma) · x_j + b_c(x, y, luma)

    四個構造上的選擇：

    - **零半徑逐位元恆等**：`A` 初始化為 `[I | 0]`，取樣一個常數場在任何
      內插模式下都回傳該常數，故輸出等於輸入。
    - **有偏移項**。這是它與 `ShadingParam` 的關鍵差別——後者是純乘性，
      `exp(m) · 0 = 0`，暗部推不動；偏移項與像素值無關。
    - **導引用亮度不用逐通道值**。逐通道各自查表就退化成三條獨立曲線、
      失去跨通道混合；亮度導引是雙邊網格的標準選擇。
    - **`padding_mode="border"`**：亮度 0 與 1 落在網格邊界上，
      `zeros` 會在那裡把仿射係數拉向 0，產生黑邊。

    `G = 1` 時空間上是常數場，於是**對裁切精確等變**；`G > 1` 時只近似
    （裁切放大會改變空間切片的位置，但場是低頻的，位移量小）。
    """

    name = "color_grid"

    def __init__(self, radius: float = 0.10, grid: int = 8,
                 luma_bins: int = 8,
                 apply_where: Optional[torch.Tensor] = None):
        if grid < 1:
            raise ValueError(f"grid 必須至少為 1，收到 {grid}")
        if luma_bins < 2:
            raise ValueError(f"luma_bins 必須至少為 2，收到 {luma_bins}")
        self.radius = radius
        self.grid = grid
        self.luma_bins = luma_bins
        self.apply_where = apply_where
        self.a: Optional[torch.Tensor] = None

    # ---- 介面 ----

    def identity_grid(self, device, dtype) -> torch.Tensor:
        a = torch.zeros(1, AFFINE_ENTRIES, self.luma_bins, self.grid, self.grid,
                        device=device, dtype=dtype)
        for j in _DIAG:
            a[:, j] = 1.0
        return a

    def reset(self, x01: torch.Tensor, seed: int) -> None:
        a = self.identity_grid(x01.device, x01.dtype)
        self.a = a.requires_grad_(True)

    def render(self, x01: torch.Tensor) -> torch.Tensor:
        out = self._apply(x01, self.a)
        if self.apply_where is not None:
            w = self.apply_where.to(device=out.device, dtype=out.dtype)
            out = w * out + (1.0 - w) * x01
        return out

    def params(self) -> List[torch.Tensor]:
        return [self.a]

    def state_dict(self):
        return {'a': self.a.detach().clone()}

    def load_state_dict(self, state):
        self.a = state['a'].detach().clone().requires_grad_(True)

    @property
    def stages(self):
        return [self]

    def set_amplitude(self, v):
        if isinstance(v, (list, tuple)):
            if len(v) != 1:
                raise ValueError('這個載體只有一段，逐段指定時長度要是 1')
            v = v[0]
        self.amplitude = float(v)

    @torch.no_grad()
    def project(self) -> None:
        ident = self.identity_grid(self.a.device, self.a.dtype)
        self.a.copy_(ident + (self.a - ident).clamp(-self.radius, self.radius))

    def set_radius(self, r: float) -> None:
        self.radius = r

    # ---- 構造 ----

    def _slice(self, x01: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """(N,3,H,W) → (N,12,H,W)，逐像素的仿射係數。"""
        n, _, h, w = x01.shape
        dev, dt = x01.device, x01.dtype
        xs = torch.linspace(-1.0, 1.0, w, device=dev, dtype=dt)
        ys = torch.linspace(-1.0, 1.0, h, device=dev, dtype=dt)
        gx = xs.view(1, 1, w).expand(n, h, w)
        gy = ys.view(1, h, 1).expand(n, h, w)
        gz = (2.0 * luma(x01).clamp(0.0, 1.0) - 1.0).squeeze(1)   # (N,H,W)
        # grid_sample 3D：grid[...,0]↔W、[...,1]↔H、[...,2]↔D。
        grid = torch.stack([gx, gy, gz], dim=-1).unsqueeze(1)     # (N,1,H,W,3)
        samp = F.grid_sample(a.expand(n, -1, -1, -1, -1).to(dt), grid,
                             mode="bilinear", padding_mode="border",
                             align_corners=True)
        return samp[:, :, 0]                                      # (N,12,H,W)

    def _apply(self, x01: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        m = self._slice(x01, a).view(x01.shape[0], 3, 4,
                                     x01.shape[-2], x01.shape[-1])
        out = (m[:, :, 0] * x01[:, 0:1]
               + m[:, :, 1] * x01[:, 1:2]
               + m[:, :, 2] * x01[:, 2:3]
               + m[:, :, 3])
        return out.clamp(0.0, 1.0)


class ColorGridRandomParam(ColorGridParam):
    """同半徑的隨機仿射網格，**不最佳化**。存在理由同 `ColorCurveRandomParam`。

    偏移是對單位矩陣的，故盒是 `[−radius, +radius]`；`corner` 抽 ±radius。
    """

    name = "color_grid_rand"

    def __init__(self, *args, draw: str = "corner", **kwargs):
        super().__init__(*args, **kwargs)
        if draw not in RAND_DRAWS:
            raise ValueError(f"draw 只能是 {RAND_DRAWS}，收到 {draw!r}")
        self.draw = draw

    def reset(self, x01: torch.Tensor, seed: int) -> None:
        super().reset(x01, seed)
        d = _draw(self.a.shape, -self.radius, self.radius, self.draw, seed)
        with torch.no_grad():
            self.a.add_(d.to(device=x01.device, dtype=x01.dtype))

    def params(self) -> List[torch.Tensor]:
        return []

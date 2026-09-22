"""獨立畫布的 adversarial patch：先優化一張圖，再貼到衣服上算 loss。

與 `material_patch.MaterialPatchParam` 的差別
────────────────────────────────────────────────────────────────────
那一個是**在衣服上直接學一個反射色場**：參數是色盤權重，貼片的顏色被限制在
色盤的凸包內，圖案與這件衣服綁在一起。

本模組是 detector patch 文獻的標準形式：貼片是一張**獨立的 RGB 畫布**，
可以由一張現成的圖樣（布料、卡通）起始，最佳化直接動那張畫布，
再貼到衣服上、算損失、反傳回畫布。畫布與這件衣服無關，換一張照片可以重貼。

    canvas = clamp(init + ε·tanh(u), 0, 1)        （ε 給定，貼著圖樣走）
    canvas = sigmoid(u)                            （ε 為 None，自由）
    x_θ    = (1 − M)·x + M·clamp(U(canvas) ⊙ S_x + H_x, 0, 1)

`U` 是把畫布雙三次升到影像尺寸。**空間特徵的尺度由 `canvas_size` 決定**：
畫布 64²、支撐框約 180² 時一格約 2.8 像素，太細的會被 JPEG Q75 與 σ=1.0
模糊吃掉；要粗一點就把 `canvas_size` 調小。

`keep_shading` 決定貼上去像不像一件衣服
────────────────────────────────────────────────────────────────────
`True`（預設）保留原圖的明暗 `S_x` 與織紋殘差 `H_x`，貼片看起來是「印在布上」
——摺痕與光照跟著布料走。`False` 直接貼原色，看起來像一張貼紙。
兩者都留著是因為那是一個**視覺**的選擇，不是讀數能判的。

ε 是自然度與強度之間唯一的旋鈕
────────────────────────────────────────────────────────────────────
ε 小時畫布跑不遠，成品仍然認得出是原來那張圖樣；ε 大時畫布可以變成任何東西，
包含看起來像雜訊的東西。**ε 不是自然度保證**，它只是把搜尋範圍綁在圖樣附近；
成品自不自然一律由使用者看圖判定。
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn.functional as F

from .material_patch import _lowpass


def load_pattern(path, size: int, device=None) -> torch.Tensor:
    """讀一張圖樣並縮成 (1,3,size,size) 的畫布起點。"""
    import numpy as np
    from PIL import Image

    img = Image.open(path).convert('RGB').resize((size, size), Image.LANCZOS)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1)[None]
    return t.to(device) if device is not None else t


class PatchCanvasParam:
    """φ = u，(1,3,S,S) 的畫布參數。支撐、明暗、織紋固定。

    `init_jitter = 0` 且 ε 給定時，起點就是那張圖樣本身——`tanh(0) = 0`。
    那**不是零梯度點**：損失對 `u` 的梯度帶因子 `ε·(1 − tanh²(u))`，在 0 處是
    `ε`，非零。所以可以從圖樣原樣起步，不需要抖動。
    """

    name = "patch_canvas"

    def __init__(self, support: torch.Tensor, *,
                 canvas: Optional[torch.Tensor] = None,
                 canvas_size: int = 64,
                 epsilon: Optional[float] = 0.25,
                 shade_radius: int = 12,
                 keep_shading: bool = True,
                 init_jitter: float = 0.0):
        if support.ndim != 4 or support.shape[1] != 1:
            raise ValueError(
                f'support 必須是 (1,1,H,W)，收到 {tuple(support.shape)}')
        if float(support.max()) <= 0:
            raise ValueError('支撐全為零，貼片無處可放')
        if canvas_size < 4:
            raise ValueError(f'canvas_size 至少為 4，收到 {canvas_size}')
        if epsilon is not None and not 0.0 < float(epsilon) <= 1.0:
            raise ValueError(f'epsilon 要落在 (0,1]，收到 {epsilon}')
        if shade_radius < 1:
            raise ValueError(f'shade_radius 至少為 1，收到 {shade_radius}')
        if canvas is not None:
            if canvas.ndim != 4 or canvas.shape[1] != 3:
                raise ValueError(
                    f'canvas 必須是 (1,3,S,S)，收到 {tuple(canvas.shape)}')
            if float(canvas.min()) < 0 or float(canvas.max()) > 1:
                raise ValueError('canvas 的值域必須是 [0,1]')
        self.support = support
        self.canvas_size = int(canvas_size)
        self.epsilon = None if epsilon is None else float(epsilon)
        self.shade_radius = int(shade_radius)
        self.keep_shading = bool(keep_shading)
        self.init_jitter = float(init_jitter)
        self.amplitude = 1.0
        self.init = canvas
        self.u: Optional[torch.Tensor] = None
        self.shade: Optional[torch.Tensor] = None
        self.detail: Optional[torch.Tensor] = None

    def reset(self, x01: torch.Tensor, seed: int = 0) -> None:
        w = self.support.to(device=x01.device, dtype=x01.dtype)
        base = _lowpass(luma_of(x01), self.shade_radius)
        denom = (base * w).sum() / w.sum().clamp_min(1e-6)
        if float(denom) <= 0:
            raise ValueError('支撐內的平均亮度為零，明暗場沒有定義')
        self.shade = base / denom
        self.detail = x01 - _lowpass(x01, self.shade_radius)

        s = self.canvas_size
        g = torch.Generator(device='cpu').manual_seed(int(seed))
        if self.init is None:
            # 沒有給圖樣時起點是隨機畫布。ε 在這個情況下沒有意義——沒有東西
            # 可以「貼著走」——所以只走自由的 sigmoid 參數化。
            if self.epsilon is not None:
                raise ValueError(
                    'epsilon 要求一張起始圖樣：沒有 canvas 時沒有東西可以貼著走。'
                    '給 canvas，或把 epsilon 設成 None 走自由參數化。')
            u = torch.randn((1, 3, s, s), generator=g)
        else:
            self.init = F.interpolate(self.init.to(x01), size=(s, s),
                                      mode='bicubic', align_corners=False
                                      ).clamp(0.0, 1.0)
            u = torch.randn((1, 3, s, s), generator=g) * self.init_jitter
        self.u = u.to(device=x01.device, dtype=x01.dtype).requires_grad_(True)

    def params(self) -> List[torch.Tensor]:
        return [self.u]

    def state_dict(self):
        return {'u': self.u.detach().clone()}

    def load_state_dict(self, state):
        self.u = state['u'].detach().clone().requires_grad_(True)

    @property
    def stages(self):
        return [self]

    @torch.no_grad()
    def project(self) -> None:
        """參數空間無界：值域由 `tanh`／`sigmoid` 保證，這裡不需要夾。

        留著是因為 `optimise_carrier` 每一步都會呼叫它。
        """

    def step_scale(self) -> float:
        return 1.0

    def set_amplitude(self, a) -> None:
        """縮畫布相對起始圖樣的偏離。`a = 0` 回到原圖樣。"""
        if isinstance(a, (list, tuple)):
            if len(a) != 1:
                raise ValueError('這個載體只有一段，逐段指定時長度要是 1')
            a = a[0]
        if float(a) < 0:
            raise ValueError(f'amplitude 不得為負，收到 {a}')
        self.amplitude = float(a)

    def canvas(self) -> torch.Tensor:
        """(1,3,S,S) 的畫布，值域保證在 [0,1]。"""
        u = self.u * self.amplitude
        if self.epsilon is None:
            return torch.sigmoid(u)
        return (self.init + self.epsilon * torch.tanh(u)).clamp(0.0, 1.0)

    def render(self, x01: torch.Tensor) -> torch.Tensor:
        if self.shade is None:
            raise ValueError('要先呼叫 reset 凍結明暗與織紋')
        c = F.interpolate(self.canvas(), size=x01.shape[-2:], mode='bicubic',
                          align_corners=False).clamp(0.0, 1.0)
        patch = (c * self.shade + self.detail).clamp(0.0, 1.0) \
            if self.keep_shading else c
        w = self.support.to(device=x01.device, dtype=x01.dtype)
        return w * patch + (1.0 - w) * x01

    def drift(self) -> float:
        """畫布相對起始圖樣的 L∞。沒有起始圖樣時回傳 -1。"""
        if self.init is None:
            return -1.0
        return float((self.canvas() - self.init).abs().max().detach())

    def readout(self) -> dict:
        w = self.support
        return {'patch_area': round(float((w > 0.5).to(w.dtype).mean()), 5),
                'patch_weight_mean': round(float(w.mean()), 5),
                'canvas_size': self.canvas_size,
                'canvas_epsilon': -1.0 if self.epsilon is None else self.epsilon,
                'canvas_keep_shading': int(self.keep_shading),
                'canvas_drift': round(self.drift(), 5),
                'patch_amplitude': round(self.amplitude, 5)}


def luma_of(x01: torch.Tensor) -> torch.Tensor:
    from .color_param import luma
    return luma(x01)


class SupportWeightedFree:
    """`FreeObjective` 的支撐加權版，**不改 `instruction_free.py`**。

    現行 `enc`／`cond` 在整張潛圖上取範數。貼片只動 2.4%–8% 的像素，它的貢獻
    在分母裡被其餘 90% 以上未改動的像素沖掉——`runs/patch_free/` 量到代理分數
    120 步只動 1.9%，而同一個目標在全域曲線上動 40%–139%。

    這一層把權重圖換成支撐在潛空間的投影：框內 `1 + support_weight`、框外 `1`。
    背景仍然計入，只是不再等重。作法與 `FreeObjective.face_weight` 相同，
    差別只在加權到哪裡。

    `_support` 在 `super().__init__` 之前設好，因為權重圖是在那裡建的。
    """

    def __new__(cls, ip2p, x01, *, support=None, support_weight=0.0, **kw):
        from .instruction_free import FreeObjective

        class _Impl(FreeObjective):
            def __init__(self, *a, **k):
                self._support = support
                self._support_weight = float(support_weight)
                super().__init__(*a, **k)

            def _face_weight_map(self, img):
                if self._support is None or self._support_weight <= 0:
                    return super()._face_weight_map(img)
                h, w = self.cond0.shape[-2:]
                m = F.interpolate(self._support.to(torch.float32),
                                  size=(h, w), mode='area')
                return 1.0 + self._support_weight * m

        return _Impl(ip2p, x01, **kw)

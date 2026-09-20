"""從一張**具體的目標圖**出發的貼片：畫布映射到支撐的外接框。

與 `patch_canvas.PatchCanvasParam` 的分界
────────────────────────────────────────────────────────────────────
那一支把畫布雙三次放大到**整張照片**再由支撐截取。支撐只佔 8% 時，64² 的畫布
實際只用得到約 330 個像素（等效 18×18），而且一張有結構的目標圖（例如一個笑臉）
貼上去會被支撐**裁掉大部分**——畫布上的笑臉攤在整張照片上，圓形支撐只露出其中
一小塊。

這一支把畫布映射到支撐的**外接框**，所以目標圖完整落在貼片裡，解析度也全部
用在貼片上。

兩種最佳化，差別只在可行域
────────────────────────────────────────────────────────────────────
    free      canvas = sigmoid(u)，u 由目標圖的 logit 起始
    bounded   canvas = clamp(target + ε·tanh(u), 0, 1)

`free` 不受限：起點是目標圖，但最佳化可以把它變成任何東西。
`bounded` 受限：每個像素最多離開目標圖 `ε`，成品仍然認得出是原來那張圖。
**`ε` 不是自然度保證**（`runs/patch_canvas/` 量過：六個臂 drift 全部頂滿上限、
圖案變成螢光彩虹）——它只是把搜尋綁在目標圖附近。自不自然一律看圖判定。

合成式與另外三個載體共用：`M` 事前固定，`S_x`（明暗）與 `H_x`（織紋殘差）由
原圖算一次就凍結，支撐外逐位元不變、臉從不被碰。
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn.functional as F

from .material_patch import _lowpass

MODES = ('free', 'bounded')


def support_box(support: torch.Tensor):
    """支撐的外接框 `(y0, y1, x0, x1)`，右下為開區間。"""
    m = (support > 0.5)[0, 0]
    idx = torch.nonzero(m, as_tuple=False)
    if idx.numel() == 0:
        raise ValueError('支撐全為零，取不出外接框')
    y0, x0 = idx.min(0).values.tolist()
    y1, x1 = idx.max(0).values.tolist()
    return y0, y1 + 1, x0, x1 + 1


class TargetImagePatch:
    """φ = u，(1,3,h,w) 的殘差場，h×w 是支撐外接框的尺寸。

    `mode='free'` 時起點是目標圖的 logit，所以第一步的算繪結果就是目標圖本身，
    而且**不是零梯度點**——`sigmoid` 在那裡的導數非零。
    """

    def __init__(self, support: torch.Tensor, target01: torch.Tensor, *,
                 mode: str = 'bounded', epsilon: float = 0.12,
                 keep_shading: bool = True, shade_radius: int = 12,
                 init_jitter: float = 0.0):
        if support.ndim != 4 or support.shape[1] != 1:
            raise ValueError(
                f'support 必須是 (1,1,H,W)，收到 {tuple(support.shape)}')
        if float(support.max()) <= 0:
            raise ValueError('支撐全為零，貼片無處可放')
        if target01.ndim != 4 or target01.shape[1] != 3:
            raise ValueError(
                f'target 必須是 (1,3,h,w)，收到 {tuple(target01.shape)}')
        if float(target01.min()) < 0 or float(target01.max()) > 1:
            raise ValueError('target 的值域必須是 [0,1]')
        if mode not in MODES:
            raise ValueError(f'mode 要是 {MODES} 之一，收到 {mode!r}')
        if mode == 'bounded' and not 0.0 < float(epsilon) <= 1.0:
            raise ValueError(f'epsilon 要落在 (0,1]，收到 {epsilon}')
        if shade_radius < 1:
            raise ValueError(f'shade_radius 至少為 1，收到 {shade_radius}')
        self.support = support
        self.target = target01
        self.mode = mode
        self.epsilon = float(epsilon)
        self.keep_shading = bool(keep_shading)
        self.shade_radius = int(shade_radius)
        self.init_jitter = float(init_jitter)
        self.amplitude = 1.0
        self.box = support_box(support)
        self.init: Optional[torch.Tensor] = None
        self.shade: Optional[torch.Tensor] = None
        self.detail: Optional[torch.Tensor] = None
        self.u: Optional[torch.Tensor] = None

    def reset(self, x01: torch.Tensor, seed: int = 0) -> None:
        w = self.support.to(device=x01.device, dtype=x01.dtype)
        base = _lowpass(_luma(x01), self.shade_radius)
        denom = (base * w).sum() / w.sum().clamp_min(1e-6)
        if float(denom) <= 0:
            raise ValueError('支撐內的平均亮度為零，明暗場沒有定義')
        self.shade = base / denom
        self.detail = x01 - _lowpass(x01, self.shade_radius)

        y0, y1, x0, x1 = self.box
        tgt = F.interpolate(self.target.to(x01), size=(y1 - y0, x1 - x0),
                            mode='bicubic', align_corners=False)
        self.init = tgt.clamp(1e-4, 1 - 1e-4)
        g = torch.Generator(device='cpu').manual_seed(int(seed))
        noise = torch.randn(self.init.shape, generator=g).to(x01)
        if self.mode == 'free':
            u = torch.log(self.init / (1 - self.init))
        else:
            u = torch.zeros_like(self.init)
        self.u = (u + self.init_jitter * noise).requires_grad_(True)

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
        """值域由 `sigmoid`／`tanh` 保證，這裡不需要夾。"""

    def step_scale(self) -> float:
        return 1.0

    def set_amplitude(self, a) -> None:
        """把貼片往目標圖收。`a = 0` 回到目標圖原樣。"""
        if isinstance(a, (list, tuple)):
            if len(a) != 1:
                raise ValueError('這個載體只有一段，逐段指定時長度要是 1')
            a = a[0]
        if float(a) < 0:
            raise ValueError(f'amplitude 不得為負，收到 {a}')
        self.amplitude = float(a)

    def canvas(self) -> torch.Tensor:
        """(1,3,h,w) 的貼片內容，值域 [0,1]。"""
        if self.mode == 'free':
            c = torch.sigmoid(self.u)
        else:
            c = (self.init + self.epsilon * torch.tanh(self.u)).clamp(0.0, 1.0)
        if self.amplitude != 1.0:
            c = (self.init + self.amplitude * (c - self.init)).clamp(0.0, 1.0)
        return c

    def render(self, x01: torch.Tensor) -> torch.Tensor:
        if self.shade is None:
            raise ValueError('要先呼叫 reset 凍結明暗與織紋')
        y0, y1, x0, x1 = self.box
        full = torch.zeros_like(x01)
        full[..., y0:y1, x0:x1] = self.canvas()
        printed = (full * self.shade + self.detail).clamp(0.0, 1.0) \
            if self.keep_shading else full
        w = self.support.to(device=x01.device, dtype=x01.dtype)
        return w * printed + (1.0 - w) * x01

    def drift(self) -> float:
        """貼片相對目標圖的 L∞。"""
        return float((self.canvas() - self.init).abs().max().detach())

    def readout(self) -> dict:
        with torch.no_grad():
            y0, y1, x0, x1 = self.box
            hard = (self.support > 0.5).to(torch.float32)
            return {
                'target_mode': self.mode,
                'target_epsilon': self.epsilon if self.mode == 'bounded' else '',
                'target_drift': round(self.drift(), 4),
                'target_box': f'{y1 - y0}x{x1 - x0}',
                'target_keep_shading': self.keep_shading,
                'patch_area': round(float(hard.mean()), 5),
            }


def _luma(x01: torch.Tensor) -> torch.Tensor:
    from .color_param import luma
    return luma(x01)


def load_target(path, device=None) -> torch.Tensor:
    """讀一張目標圖成 (1,3,h,w) 的 [0,1] 張量。尺寸在 `reset` 裡才配到外接框。"""
    import numpy as np
    from PIL import Image

    arr = np.asarray(Image.open(path).convert('RGB'), dtype=np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1)[None]
    return t.to(device) if device is not None else t

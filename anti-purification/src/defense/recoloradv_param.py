"""ReColorAdv（Laidlaw & Feizi, NeurIPS 2019）的 CIELUV 3D 查表載體。

出處與偏離
────────────────────────────────────────────────────────────────────
論文：*Functional Adversarial Attacks*, arXiv:1906.00001，
公開原始碼 https://github.com/cassidylaidlaw/ReColorAdv。

`runs/chroma/recolor_*` 的設定檔（`configs/recolor_paper.json` 等）指名
`carrier.kind = "recoloradv"`，但 `src/defense/` 裡從來沒有這個類別——
那條路徑連同它的 dispatcher 在 `6cb0f96` 那次裁剪時一起消失了，而且那批設定
本身帶著 `instruction` 與 `target` 鍵，屬於**指令進迴圈**的舊路徑
（`runs/carrier_search_probe/` 量到那條路徑會高估防禦：2/15 對拿掉後的 1/25）。
所以那幾批的數字不能拿來代表這個載體。這個模組是重寫的，走現行的無指令路徑。

`modified_from_paper`：
- 論文的 `ε` 是逐通道的 `(0.06, 0.06, 0.06)`，作用在**各自正規化過的** LUV
  座標上。這裡沿用同一組預設值與同一個正規化（見 `LUV_SCALE`）。
- 論文用 PGD 加逐步投影；這裡把投影留在 `project()`，由呼叫端的求解器決定
  用 Adam 還是 PGD。`src/defense/immunise.py` 的 augmented Lagrangian 會在
  每一步之後呼叫它，等價於逐步投影。
- 平滑正則是論文的 `Σ ‖g(c_i) − g(c_j)‖²`（網格上相鄰點），權重由呼叫端給。
  論文在 ImageNet 上用 `smoothness_weight = 0.05`；`configs/recolor_paper.json`
  記的是 1.0，兩個值都留給設定檔決定，這裡不設預設偏好。

為什麼查表要在 LUV 而不是 Lab
────────────────────────────────────────────────────────────────────
論文選 CIELUV 是因為它的色度平面是仿射的，等亮度的線性內插仍落在合理色域內；
Lab 的色度平面不具這個性質。本模組其餘部分（`rgb_to_lab`）用 Lab 只是為了
量測色差，與載體本身無關。
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn.functional as F

# LUV 的動態範圍，用來把座標正規化到 [0,1]，`epsilon` 因此與論文同單位。
# 邊界取 sRGB 立方體在 LUV 下的實測極值（L 0–100、u −83.1–175.0、v −134.1–107.4，
# 17³ 稠密取樣）再各留 5 的餘裕。範圍取窄會讓飽和色被 clamp 到格點邊界，
# 零位移就不再是恆等映射——那是靜默的偏差，不是捨入誤差。
LUV_SCALE = (100.0, 268.0, 252.0)
LUV_SHIFT = (0.0, 88.0, 139.0)

_XYZ_FROM_RGB = ((.4124564, .3575761, .1804375),
                 (.2126729, .7151522, .0721750),
                 (.0193339, .1191920, .9503041))
_RGB_FROM_XYZ = ((3.2404542, -1.5371385, -.4985314),
                 (-.9692660, 1.8760108, .0415560),
                 (.0556434, -.2040259, 1.0572252))
_WHITE = (.95047, 1., 1.08883)


def _linear(x):
    return torch.where(x > .04045, ((x + .055) / 1.055).clamp_min(1e-10).pow(2.4),
                       x / 12.92)


def _srgb(linear):
    return torch.where(linear > .0031308,
                       1.055 * linear.clamp_min(1e-10).pow(1 / 2.4) - .055,
                       12.92 * linear)


def rgb_to_luv(x01: torch.Tensor) -> torch.Tensor:
    """(B,3,H,W) sRGB [0,1] → CIELUV。D65 白點，與 skimage 同一組常數。"""
    x = x01.float() if x01.dtype in (torch.float16, torch.bfloat16) else x01
    m = x.new_tensor(_XYZ_FROM_RGB)
    xyz = torch.einsum('ij,bjhw->bihw', m, _linear(x))
    X, Y, Z = xyz.unbind(1)
    denom = (X + 15 * Y + 3 * Z).clamp_min(1e-10)
    up, vp = 4 * X / denom, 9 * Y / denom
    wx, wy, wz = _WHITE
    wd = wx + 15 * wy + 3 * wz
    un, vn = 4 * wx / wd, 9 * wy / wd
    yr = Y / wy
    L = torch.where(yr > (6 / 29) ** 3, 116 * yr.clamp_min(1e-10).pow(1 / 3) - 16,
                    (29 / 3) ** 3 * yr)
    return torch.stack((L, 13 * L * (up - un), 13 * L * (vp - vn)), 1)


def luv_to_rgb(luv: torch.Tensor) -> torch.Tensor:
    """CIELUV → sRGB [0,1]，未裁切；裁切由呼叫端決定要不要做。"""
    L, u, v = luv.unbind(1)
    wx, wy, wz = _WHITE
    wd = wx + 15 * wy + 3 * wz
    un, vn = 4 * wx / wd, 9 * wy / wd
    Y = torch.where(L > 8, wy * ((L + 16) / 116).pow(3), wy * L * (3 / 29) ** 3)
    safe = L.abs().clamp_min(1e-6) * torch.where(L < 0, -1.0, 1.0)
    up, vp = u / (13 * safe) + un, v / (13 * safe) + vn
    vp = vp.clamp_min(1e-6)
    X = Y * 9 * up / (4 * vp)
    Z = Y * (12 - 3 * up - 20 * vp) / (4 * vp)
    xyz = torch.stack((X, Y, Z), 1)
    m = luv.new_tensor(_RGB_FROM_XYZ)
    return _srgb(torch.einsum('ij,bjhw->bihw', m, xyz))


class ReColorAdvParam:
    """CIELUV 上的 3D 查表色彩變換。介面與其他載體一致。

    參數 `delta` 的形狀是 `(1, 3, D_L, D_U, D_V)`，是**查表輸出相對恆等的位移**，
    單位是正規化後的 LUV。零位移逐位元等於恆等映射：`grid_sample` 取一個恆等
    座標場在三線性內插下回傳原座標。
    """

    name = 'recoloradv'

    def __init__(self, x01: torch.Tensor, *, resolution=(16, 32, 32),
                 radius: float = 0.06, amplitude: float = 1.0,
                 apply_where: Optional[torch.Tensor] = None):
        if len(resolution) != 3 or any(int(r) < 2 for r in resolution):
            raise ValueError(f'resolution 要是三個 ≥ 2 的整數，收到 {resolution}')
        self.resolution = tuple(int(r) for r in resolution)
        self.radius = float(radius)
        self.amplitude = float(amplitude)
        self.apply_where = apply_where
        self.delta = torch.zeros(1, 3, *self.resolution, dtype=torch.float32,
                                 device=x01.device, requires_grad=True)

    @property
    def stages(self):
        return [self]

    def set_amplitude(self, a):
        if isinstance(a, (list, tuple)):
            if len(a) != 1:
                raise ValueError('這個載體只有一段，逐段指定時長度要是 1')
            a = a[0]
        self.amplitude = float(a)

    def reset(self, x01: torch.Tensor, seed: int = 0):
        """起點是恆等查表，防禦圖一開始逐位元等於原圖。"""
        with torch.no_grad():
            self.delta.zero_()

    def params(self):
        return [self.delta]

    def project(self):
        """逐通道的 `L∞` 投影，與論文的 `ε` 同單位。"""
        with torch.no_grad():
            self.delta.clamp_(-self.radius, self.radius)

    def _normalised(self, x01: torch.Tensor) -> torch.Tensor:
        luv = rgb_to_luv(x01)
        shift = luv.new_tensor(LUV_SHIFT)[None, :, None, None]
        scale = luv.new_tensor(LUV_SCALE)[None, :, None, None]
        return ((luv + shift) / scale).clamp(0.0, 1.0)

    def lookup(self, x01: torch.Tensor) -> torch.Tensor:
        """回傳每個像素查到的位移，形狀 (B,3,H,W)，單位是正規化 LUV。"""
        n = self._normalised(x01)
        grid = (n.permute(0, 2, 3, 1) * 2.0 - 1.0)
        grid = grid[..., [2, 1, 0]].unsqueeze(1)
        out = F.grid_sample(self.amplitude * self.delta, grid, mode='bilinear',
                            padding_mode='border', align_corners=True)
        return out.squeeze(2)

    def smoothness(self) -> torch.Tensor:
        """論文的平滑正則：網格上相鄰點輸出差的平方和，三個軸都算。"""
        d = self.amplitude * self.delta
        total = d.new_zeros(())
        for axis in (2, 3, 4):
            if d.shape[axis] < 2:
                continue
            a = d.narrow(axis, 0, d.shape[axis] - 1)
            b = d.narrow(axis, 1, d.shape[axis] - 1)
            total = total + (b - a).pow(2).sum()
        return total

    def render(self, x: torch.Tensor) -> torch.Tensor:
        off = self.lookup(x).to(x.dtype)
        n = self._normalised(x).to(x.dtype)
        scale = x.new_tensor(LUV_SCALE)[None, :, None, None]
        shift = x.new_tensor(LUV_SHIFT)[None, :, None, None]
        luv = (n + off) * scale - shift
        y = luv_to_rgb(luv).to(x.dtype)
        if self.apply_where is not None:
            w = self.apply_where.to(x.dtype)
            y = w * y + (1.0 - w) * x
        return y.clamp(0.0, 1.0)

    def state_dict(self):
        return {'delta': self.delta.detach().clone()}

    def load_state_dict(self, state):
        self.delta = state['delta'].detach().clone().requires_grad_(True)

    @torch.no_grad()
    def diagnostics(self, x: torch.Tensor) -> Dict[str, float]:
        off = self.lookup(x)
        return {'name': self.name, 'resolution': 'x'.join(map(str, self.resolution)),
                'radius': self.radius, 'amplitude': self.amplitude,
                'lut_linf': float((self.amplitude * self.delta).abs().max()),
                'lut_used_linf': float(off.abs().max()),
                'lut_used_mean': float(off.abs().mean()),
                'lut_smoothness': float(self.smoothness())}

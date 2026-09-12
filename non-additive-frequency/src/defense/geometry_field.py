"""幾何載體：低頻平滑的取樣位移場。

為什麼換到幾何
────────────────────────────────────────────────────────────────────
顏色載體上量到的取捨是單調的：位移場越均勻，防禦越弱（`runs/immunise_tv*/`，
TV 2.0 → id −0.440、TV 1.0 → +0.498、TV 0.5 → +0.943）。根因在自然照片對顏色
有強先驗——同一個表面不該出現說不出理由的漸層——而全域的顏色映射保住每一條
邊與每一道梯度，人臉幾何原封不動，身分嵌入因此讀不到差異。
`runs/objective_pilot/` 的 CIELUV 3D LUT（16×32×32、含平滑正則）在 ΔE≈6 上
身分降幅中位 +0.0005，`runs/color_scaleup_search/` 的全域仿射在 ΔE 6.3 上
−0.002～+0.0043，兩個獨立量測指向同一件事。

幾何沒有這個對立。「一張自然的照片」不約束那張臉長什麼形狀：觀看者手上沒有
原圖，一張被平滑扭過的臉就是另一個人的臉。而平滑本身正是自然的來源——低頻的
位移場就是鏡頭畸變、輕微透視與視差在做的事。於是**平滑性與防禦力同向**，
不像顏色那樣反向。

參數化
────────────────────────────────────────────────────────────────────
自由參數是 `grid × grid × 2` 的控制點（單位是像素），雙三次上採樣到全解析度。
平滑是**結構性**的，做不出高頻 warp，不需要靠 TV 去壓。邊界 `taper` 像素內
以 smoothstep 漸縮為零，畫幅與構圖不動，也不會在邊緣拉出重複邊。

會讓 warp 露餡的兩件事都在臉以外，而且都寫得成約束：場景裡的直線被扭彎
（`affine_residual`——臉框外的位移對最佳全域仿射的殘差），以及摺疊撕裂
（`fold_measure`——`det J` 掉到正的下界以下）。預算的分配因此與顏色相反：
**臉上自由、背景近剛體**，而身分正好住在臉上。
"""
from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F


def smoothstep(t: torch.Tensor) -> torch.Tensor:
    t = t.clamp(0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def border_window(size: Tuple[int, int], taper: float, *, dtype, device
                  ) -> torch.Tensor:
    """邊界 `taper` 像素內漸縮為零的窗，C¹ 連續。

    `taper <= 0` 時回傳全一；此時畫幅邊緣的像素會沿 `padding_mode='border'`
    重複，那是肉眼看得出來的人造痕跡，所以預設不走這條。
    """
    h, w = size
    if taper <= 0:
        return torch.ones(1, 1, h, w, dtype=dtype, device=device)
    ys = torch.arange(h, dtype=dtype, device=device)
    xs = torch.arange(w, dtype=dtype, device=device)
    dy = torch.minimum(ys, (h - 1) - ys) / taper
    dx = torch.minimum(xs, (w - 1) - xs) / taper
    return (smoothstep(dy)[None, None, :, None]
            * smoothstep(dx)[None, None, None, :])


def base_grid(size: Tuple[int, int], *, dtype, device) -> torch.Tensor:
    """`align_corners=False` 下的恆等取樣格，形狀 (1, H, W, 2)。"""
    h, w = size
    ys = (2.0 * torch.arange(h, dtype=dtype, device=device) + 1.0) / h - 1.0
    xs = (2.0 * torch.arange(w, dtype=dtype, device=device) + 1.0) / w - 1.0
    gy, gx = torch.meshgrid(ys, xs, indexing='ij')
    return torch.stack([gx, gy], dim=-1)[None]


def forward_diff(f: torch.Tensor, dim: int) -> torch.Tensor:
    """同尺寸的前向差分，最後一格複製前一格。"""
    d = f.diff(dim=dim)
    return torch.cat([d, d.narrow(dim, d.shape[dim] - 1, 1)], dim=dim)


def det_jacobian(flow: torch.Tensor) -> torch.Tensor:
    """取樣映射 `p ↦ p + flow(p)` 的 Jacobian 行列式，逐像素。

    `flow` 的單位是像素，形狀 (1, 2, H, W)，通道 0 是 x、通道 1 是 y。
    行列式掉到零以下代表映射在該處摺疊，輸出會出現撕裂或鏡像——那是 warp
    唯一真正屬於人造痕跡的失效模式，所以它進約束而不只是回報。
    """
    dxdx = forward_diff(flow[:, 0:1], dim=-1)
    dxdy = forward_diff(flow[:, 0:1], dim=-2)
    dydx = forward_diff(flow[:, 1:2], dim=-1)
    dydy = forward_diff(flow[:, 1:2], dim=-2)
    return (1.0 + dxdx) * (1.0 + dydy) - dxdy * dydx


def affine_residual(flow: torch.Tensor, support: torch.Tensor) -> torch.Tensor:
    """位移場扣掉支撐上最佳的全域仿射之後，剩下的位移量，逐像素。

    全域仿射的位移是相機換位置或鏡頭換焦段就會產生的東西，直線仍是直線，
    人眼讀不出異常；真正會露餡的是**直線被扭彎**，也就是仿射以外的殘差。
    以加權正規方程解 3×3，權重是支撐遮罩，整條可微。
    """
    h, w = flow.shape[-2:]
    dtype = flow.dtype
    ys = torch.arange(h, dtype=dtype, device=flow.device) / max(h - 1, 1)
    xs = torch.arange(w, dtype=dtype, device=flow.device) / max(w - 1, 1)
    gy, gx = torch.meshgrid(ys, xs, indexing='ij')
    design = torch.stack([gx.reshape(-1), gy.reshape(-1),
                          torch.ones_like(gx.reshape(-1))], dim=1)
    weight = support.to(device=flow.device, dtype=dtype)[:, 0].reshape(-1)
    total = weight.sum()
    if float(total) <= 0:
        raise ValueError('支撐的總權重為零，仿射殘差沒有定義')
    target = flow.reshape(2, -1).transpose(0, 1)
    xtw = design.transpose(0, 1) * weight[None]
    gram = xtw @ design
    gram = gram + 1e-6 * torch.eye(3, dtype=dtype, device=flow.device)
    coeff = torch.linalg.solve(gram, xtw @ target)
    residual = target - design @ coeff
    return residual.transpose(0, 1).reshape(1, 2, h, w)


def dilate(mask: torch.Tensor, radius: float) -> torch.Tensor:
    """把遮罩向外膨脹 `radius` 像素（方形結構元，可分離的最大值濾波）。

    為什麼需要它
    ────────────────────────────────────────────────────────────────
    「背景的直線不可以被扭彎」這道約束，支撐若直接取臉框的補集，就會把**臉框
    外緣的過渡帶**也算進背景。位移場是平滑的，臉要整體移動就必然拉動它周圍的
    一圈；那一圈的位移本來就不可能是全域仿射的一部分，於是背景那道約束把臉的
    移動一起擋掉。實測 `face_rigid_16` 允許臉框走 16 px，實際只走到 1.4–2.0，
    而背景殘差貼死在上限。

    膨脹之後補集裡只剩真正的背景，過渡帶兩邊都不管——它既不是要保直線的背景，
    也不是要保形狀的臉。
    """
    if radius <= 0:
        return mask
    k = 2 * int(round(radius)) + 1
    m = F.max_pool2d(mask, (1, k), stride=1, padding=(0, k // 2))
    return F.max_pool2d(m, (k, 1), stride=1, padding=(k // 2, 0))


class FlowFieldParam:
    """低頻取樣位移場。介面與 `CompositeParam` 對各段的要求一致。"""

    name = 'flow_field'

    def __init__(self, x01, *, grid: int = 16, taper: float = 16.0,
                 amplitude: float = 1.0, box: float = 64.0):
        if grid < 2:
            raise ValueError('控制網格至少要 2×2，否則整個場是空間常數的平移')
        self.grid = int(grid)
        self.taper = float(taper)
        self.amplitude = float(amplitude)
        self.box = float(box)
        self.size = tuple(int(v) for v in x01.shape[-2:])
        self.theta = torch.zeros(1, 2, self.grid, self.grid,
                                 dtype=torch.float32, device=x01.device,
                                 requires_grad=True)

    @property
    def radius(self):
        return self.box

    @property
    def stages(self):
        """單段載體也要有 `stages`，`immunise.fit_caps` 的退路是照段取幅度的。"""
        return [self]

    def set_amplitude(self, a):
        """純量或長度 1 的序列；後者是 `fit_caps` 逐段指定幅度的形狀。"""
        if isinstance(a, (list, tuple)):
            if len(a) != 1:
                raise ValueError('這個載體只有一段，逐段指定時長度要是 1')
            a = a[0]
        self.amplitude = float(a)

    def reset(self, x01, seed: int = 0):
        """起點是恆等映射。

        零場是確定性的起點，防禦圖一開始逐位元等於原圖，分數的起點因此就是
        未防禦的值，場的每一點變化都來自梯度，不含亂數抽樣的混淆。
        """
        self.size = tuple(int(v) for v in x01.shape[-2:])
        with torch.no_grad():
            self.theta.zero_()

    def params(self):
        return [self.theta]

    def project(self):
        with torch.no_grad():
            self.theta.clamp_(-self.box, self.box)

    def flow(self, size=None) -> torch.Tensor:
        size = tuple(size or self.size)
        field = F.interpolate(self.theta, size=size, mode='bicubic',
                              align_corners=False)
        window = border_window(size, self.taper, dtype=field.dtype,
                               device=field.device)
        return self.amplitude * field * window

    def magnitude(self, size=None) -> torch.Tensor:
        f = self.flow(size)
        return (f * f).sum(dim=1, keepdim=True).clamp_min(1e-12).sqrt()

    def fold_measure(self, size=None, floor: float = 0.5) -> torch.Tensor:
        """`det J` 低於 `floor` 的部分，逐像素取平均。

        用與下界的差而不是「低於下界的像素比例」：比例限制的是面積，不是
        嚴重程度，一小塊深度摺疊照樣藏得住，這與 `uniformity` 裡選 CVaR 而不選
        分位數是同一個理由。
        """
        return (floor - det_jacobian(self.flow(size))).clamp_min(0.0).mean()

    def render(self, x):
        size = tuple(int(v) for v in x.shape[-2:])
        f = self.flow(size).to(x.dtype)
        h, w = size
        offset = torch.stack([2.0 * f[:, 0] / w, 2.0 * f[:, 1] / h], dim=-1)
        grid = base_grid(size, dtype=x.dtype, device=x.device) + offset
        return F.grid_sample(x, grid, mode='bilinear', padding_mode='border',
                             align_corners=False)

    def state_dict(self):
        return {'theta': self.theta.detach().clone()}

    def load_state_dict(self, state):
        self.theta = state['theta'].detach().clone().requires_grad_(True)

    @torch.no_grad()
    def diagnostics(self, x) -> Dict[str, float]:
        size = tuple(int(v) for v in x.shape[-2:])
        f = self.flow(size)
        mag = self.magnitude(size)
        det = det_jacobian(f)
        return {'name': self.name, 'grid': self.grid, 'taper': self.taper,
                'amplitude': self.amplitude,
                'flow_px_max': float(mag.max()),
                'flow_px_mean': float(mag.mean()),
                'flow_det_min': float(det.min()),
                'flow_det_mean': float(det.mean()),
                'flow_theta_norm': float(self.theta.norm())}

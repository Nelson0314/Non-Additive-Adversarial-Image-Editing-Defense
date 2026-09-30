"""幾何印花貼片：自然度由參數化保證，不靠事後投影也不靠正則項。

為什麼要第三個參數化
────────────────────────────────────────────────────────────────────
`material_patch.MaterialPatchParam` 把顏色鎖在衣物色盤的凸包內，低彩度衣物
推不出振幅；`patch_canvas.PatchCanvasParam` 用 `ε` 綁住相對圖樣的 L∞，
`runs/patch_canvas/` 量到六個臂的 drift 全部頂滿 0.345、圖案變成螢光彩虹斑。
`ε` 管得住「每個像素最多變多少」，管不住「變成螢光色」也管不住「變成散斑」。

這一份把兩件事都放進參數化本身：

1. **顏色**在 CIELab 上參數化，彩度以 `chroma_max·sigmoid` 為上限，
   所以螢光色不在可行域裡。亮度不設限——現實的印花本來就可以是黑白高對比，
   而高對比對 VAE 是比高彩度更強的輸入。
2. **空間結構**由一組固定的低階幾何波場張成，色塊指派走 softmax。
   圖案因此只能是條紋、格紋、圓點、同心圓這一類東西，逐點散斑不在可行域裡。

    score_k(u,v) = b_k + Σ_j A[k,j]·φ_j(u,v)
    w            = softmax(score / τ)
    pattern      = Σ_k w_k · c_k
    x_θ          = (1 − M)·x + M·clamp(pattern ⊙ S_x + H_x, 0, 1)

`φ_j` 是固定的：三個週期 × 兩個相位的徑向餘弦（同心圓）、四個方位 × 三個週期 ×
兩個相位的平面波（條紋；兩個正交平面波經 softmax 門檻後給圓點與格紋），外加一個
常數場。同頻率的正交相位對是刻意的——單一相位的線性場只有兩端勝得出，K 個色塊
會有 K−2 個永遠用不到。
**基底不學**——讓頻率可學等於把散斑放回可行域。

合成式與另外兩個載體共用：`M` 事前固定，`S_x`（明暗）與 `H_x`（織紋殘差）
由原圖算一次就凍結，所以圖案看起來是印在布上而不是貼上去的一張紙。
支撐外逐位元不變，臉從不被碰。
"""

from __future__ import annotations

import math
from typing import List, Optional

import torch

from .material_patch import _lowpass
from .ncf_param import lab_to_rgb, rgb_to_lab

RADIAL_PERIODS = (1.0, 0.5, 0.25)
PLANE_ANGLES = (0.0, 0.25 * math.pi, 0.5 * math.pi, 0.75 * math.pi)
PLANE_PERIODS = (1.0, 0.5, 0.25)
PHASES = (0.0, 0.5 * math.pi)
ARCHETYPES = ('rings', 'stripes', 'checks', 'dots', 'random')
RADIAL_COUNT = len(RADIAL_PERIODS) * len(PHASES)
BASIS_COUNT = (RADIAL_COUNT
               + len(PLANE_ANGLES) * len(PLANE_PERIODS) * len(PHASES)
               + 1)


def basis_bank(shape, centre, scale, device, dtype) -> torch.Tensor:
    """(1,J,H,W) 的固定幾何場。座標以支撐外接框的半寬正規化。"""
    h, w = shape
    yy = torch.arange(h, device=device, dtype=dtype)
    xx = torch.arange(w, device=device, dtype=dtype)
    gy, gx = torch.meshgrid(yy, xx, indexing='ij')
    u = (gx - float(centre[1])) / float(scale)
    v = (gy - float(centre[0])) / float(scale)
    r = (u * u + v * v).clamp_min(1e-12).sqrt()
    fields = [torch.cos(2 * math.pi * r / period + phase)
              for period in RADIAL_PERIODS for phase in PHASES]
    for ang in PLANE_ANGLES:
        p = u * math.cos(ang) + v * math.sin(ang)
        for period in PLANE_PERIODS:
            for phase in PHASES:
                fields.append(torch.cos(2 * math.pi * p / period + phase))
    fields.append(torch.ones_like(u))
    return torch.stack(fields)[None]


def support_frame(support: torch.Tensor):
    """支撐外接框的中心與半寬，給基底場當座標原點與尺度。"""
    m = (support > 0.5)[0, 0]
    idx = torch.nonzero(m, as_tuple=False)
    if idx.numel() == 0:
        raise ValueError('支撐全為零，取不出外接框')
    y0, x0 = idx.min(0).values.tolist()
    y1, x1 = idx.max(0).values.tolist()
    centre = (0.5 * (y0 + y1), 0.5 * (x0 + x1))
    scale = 0.5 * max(y1 - y0, x1 - x0)
    if scale < 1.0:
        raise ValueError(f'支撐外接框只有 {scale} 像素寬，放不下任何圖案')
    return centre, scale


def archetype_weights(kind: str, colours: int, generator=None):
    """把一個印花原型寫成 (K,J) 的基底權重與 (K,) 的偏置。

    強度取到 softmax 後色塊已經接近平面色；`dots` 的偏置是負的，因為圓點
    要的是「兩個正交平面波同時接近波峰」那一小塊區域，不是棋盤格的一半面積。
    """
    a = torch.zeros(colours, BASIS_COUNT)
    b = torch.zeros(colours)
    plane0 = RADIAL_COUNT
    per_angle = len(PLANE_PERIODS) * len(PHASES)
    if kind in ('rings', 'stripes'):
        cosine = 2 if kind == 'rings' else plane0 + 1 * per_angle + 2
        sine = cosine + 1
        for k in range(colours):
            ang = 2 * math.pi * k / colours
            a[k, cosine] = 6.0 * math.cos(ang)
            a[k, sine] = 6.0 * math.sin(ang)
    elif kind in ('checks', 'dots'):
        s1 = plane0 + 0 * per_angle + 2
        s2 = plane0 + 2 * per_angle + 2
        gain = 6.0 if kind == 'checks' else 9.0
        bias = 0.0 if kind == 'checks' else -4.0
        for k in range(colours):
            ang = 2 * math.pi * k / colours
            a[k, s1] = gain * math.cos(ang)
            a[k, s2] = gain * math.sin(ang)
            b[k] = bias * (k % 2)
    elif kind == 'random':
        if generator is None:
            raise ValueError('random 原型需要一個 generator 才可重現')
        a = torch.randn(colours, BASIS_COUNT, generator=generator) * 2.0
    else:
        raise ValueError(f'未知的原型 {kind!r}，要是 {ARCHETYPES} 之一')
    return a, b


class GeometricPrintParam:
    """φ = (A, b, 色盤的 Lab 參數)。支撐、明暗、織紋固定。

    `chroma_max` 是**結構上的**彩度上限而不是懲罰項：色盤條目的彩度走
    `chroma_max·sigmoid(·)`，最佳化再怎麼推也越不過去。亮度不設限是刻意的，
    黑白高對比的印花在現實裡完全正常，而它對 VAE 的輸入強度比高彩度大。

    `tau` 越小色塊邊界越硬。太小會讓梯度只剩邊界那一圈像素，所以預設 0.15
    是「看起來是平面色」與「梯度還流得動」之間的折衷。
    """

    def __init__(self, support: torch.Tensor, *, colours: int = 4,
                 tau: float = 0.15, chroma_max: float = 28.0,
                 archetype: str = 'dots', keep_shading: bool = True,
                 shade_radius: int = 12, lightness_init=None,
                 lightness_span: float = 34.0,
                 palette_init: str = 'garment'):
        if support.ndim != 4 or support.shape[1] != 1:
            raise ValueError(
                f'support 必須是 (1,1,H,W)，收到 {tuple(support.shape)}')
        if float(support.max()) <= 0:
            raise ValueError('支撐全為零，貼片無處可放')
        if colours < 2:
            raise ValueError(f'色盤至少兩色，收到 {colours}')
        if tau <= 0:
            raise ValueError(f'tau 必須為正，收到 {tau}')
        if chroma_max <= 0:
            raise ValueError(f'chroma_max 必須為正，收到 {chroma_max}')
        if archetype not in ARCHETYPES:
            raise ValueError(
                f'archetype 要是 {ARCHETYPES} 之一，收到 {archetype!r}')
        if shade_radius < 1:
            raise ValueError(f'shade_radius 至少為 1，收到 {shade_radius}')
        if lightness_span <= 0:
            raise ValueError(f'lightness_span 必須為正，收到 {lightness_span}')
        if palette_init not in ('garment', 'random'):
            raise ValueError(
                f'palette_init 要是 garment 或 random，收到 {palette_init!r}')
        self.support = support
        self.colours = int(colours)
        self.tau = float(tau)
        self.chroma_max = float(chroma_max)
        self.archetype = archetype
        self.keep_shading = bool(keep_shading)
        self.shade_radius = int(shade_radius)
        self.lightness_init = lightness_init
        self.lightness_span = float(lightness_span)
        self.palette_init = palette_init
        self.amplitude = 1.0
        self.basis: Optional[torch.Tensor] = None
        self.shade: Optional[torch.Tensor] = None
        self.detail: Optional[torch.Tensor] = None
        self.mean_rgb: Optional[torch.Tensor] = None
        self.a: Optional[torch.Tensor] = None
        self.b: Optional[torch.Tensor] = None
        self.pl: Optional[torch.Tensor] = None
        self.pc: Optional[torch.Tensor] = None
        self.ph: Optional[torch.Tensor] = None

    def reset(self, x01: torch.Tensor, seed: int = 0) -> None:
        w = self.support.to(device=x01.device, dtype=x01.dtype)
        base = _lowpass(_luma(x01), self.shade_radius)
        denom = (base * w).sum() / w.sum().clamp_min(1e-6)
        if float(denom) <= 0:
            raise ValueError('支撐內的平均亮度為零，明暗場沒有定義')
        self.shade = base / denom
        self.detail = x01 - _lowpass(x01, self.shade_radius)

        centre, scale = support_frame(self.support)
        self.basis = basis_bank(x01.shape[-2:], centre, scale,
                                x01.device, x01.dtype)

        hard = (w > 0.5).to(x01.dtype)
        self.mean_rgb = ((x01 * hard).sum((0, 2, 3))
                         / hard.sum().clamp_min(1e-6))
        lab_mean = rgb_to_lab(self.mean_rgb.view(1, 3, 1, 1).float())

        g = torch.Generator().manual_seed(int(seed))
        a, b = archetype_weights(self.archetype, self.colours, g)
        self.a = a.to(x01).requires_grad_(True)
        self.b = b.to(x01).requires_grad_(True)

        if self.palette_init == 'random':
            # **失真對齊的對照要能走到最佳化臂走得到的地方。** 起點色盤取自
            # 衣物時彩度恆等於衣物自己的彩度，`chroma_max` 對起點毫無作用，
            # 未最佳化的 ΔE00 因此封頂在 29 而最佳化臂在 40——兩者比不了。
            # 這一支從同一個可行域裡隨機抽色盤，抽完再由呼叫端縮幅度對齊失真。
            u = torch.rand((3, self.colours), generator=g)
            self.pl = torch.log((0.05 + 0.9 * u[0]) / (1 - 0.05 - 0.9 * u[0])
                                ).to(x01).requires_grad_(True)
            self.pc = torch.log((0.1 + 0.8 * u[1]) / (1 - 0.1 - 0.8 * u[1])
                                ).to(x01).requires_grad_(True)
            self.ph = (2 * math.pi * u[2]).to(x01).requires_grad_(True)
            return

        lightness = self.lightness_init
        if lightness is None:
            centre_l = float(lab_mean[0, 0, 0, 0])
            # **不強制對稱。** 深色衣物的平均明度只有三十幾，對稱展開會被
            # `centre_l − 1` 夾住，`lightness_span` 從 34 推到 88 讀數完全不動。
            # 往兩邊各自夾到色域邊界，暗的那一側用完就全部讓給亮的那一側。
            lo = max(1.0, centre_l - self.lightness_span)
            hi = min(99.0, centre_l + self.lightness_span)
            lightness = [lo + (hi - lo) * i / max(1, self.colours - 1)
                         for i in range(self.colours)]
        lightness = torch.as_tensor(lightness, dtype=torch.float32)
        if lightness.numel() != self.colours:
            raise ValueError(
                f'lightness_init 要有 {self.colours} 個值，'
                f'收到 {lightness.numel()}')
        lv = lightness.clamp(1.0, 99.0) / 100.0
        self.pl = torch.log(lv / (1 - lv)).to(x01).requires_grad_(True)

        c0 = float((lab_mean[0, 1, 0, 0] ** 2
                    + lab_mean[0, 2, 0, 0] ** 2).clamp_min(1e-6).sqrt())
        h0 = float(torch.atan2(lab_mean[0, 2, 0, 0], lab_mean[0, 1, 0, 0]))
        frac = min(max(c0 / self.chroma_max, 0.02), 0.9)
        self.pc = torch.full((self.colours,), math.log(frac / (1 - frac))
                             ).to(x01).requires_grad_(True)
        self.ph = (torch.full((self.colours,), h0)
                   + torch.arange(self.colours, dtype=torch.float32)
                   * (0.5 * math.pi / max(1, self.colours - 1))
                   ).to(x01).requires_grad_(True)

    def params(self) -> List[torch.Tensor]:
        return [self.a, self.b, self.pl, self.pc, self.ph]

    def state_dict(self):
        return {k: getattr(self, k).detach().clone()
                for k in ('a', 'b', 'pl', 'pc', 'ph')}

    def load_state_dict(self, state):
        for k in ('a', 'b', 'pl', 'pc', 'ph'):
            setattr(self, k, state[k].detach().clone().requires_grad_(True))

    @property
    def stages(self):
        return [self]

    @torch.no_grad()
    def project(self) -> None:
        """值域由 sigmoid 與 softmax 保證，這裡不需要夾。

        留著是因為 `optimise_carrier` 每一步都會呼叫它。
        """

    def step_scale(self) -> float:
        return 1.0

    def set_amplitude(self, a) -> None:
        """把整個圖案往支撐內的平均色收。`a = 0` 是沒有圖案的素色。"""
        if isinstance(a, (list, tuple)):
            if len(a) != 1:
                raise ValueError('這個載體只有一段，逐段指定時長度要是 1')
            a = a[0]
        if float(a) < 0:
            raise ValueError(f'amplitude 不得為負，收到 {a}')
        self.amplitude = float(a)

    def palette(self) -> torch.Tensor:
        """(K,3) 的 RGB 色盤。彩度上限由參數化保證。"""
        lightness = 100.0 * torch.sigmoid(self.pl)
        chroma = self.chroma_max * torch.sigmoid(self.pc)
        lab = torch.stack([lightness,
                           chroma * torch.cos(self.ph),
                           chroma * torch.sin(self.ph)])
        rgb = lab_to_rgb(lab.view(1, 3, self.colours, 1)).clamp(0.0, 1.0)
        return rgb[0, :, :, 0].permute(1, 0)

    def weights(self) -> torch.Tensor:
        """(1,K,H,W) 的色塊指派。"""
        if self.basis is None:
            raise ValueError('要先呼叫 reset 建基底場')
        score = torch.einsum('kj,bjhw->bkhw', self.a, self.basis)
        return torch.softmax((score + self.b[None, :, None, None]) / self.tau,
                             dim=1)

    def pattern(self) -> torch.Tensor:
        """(1,3,H,W) 的圖案本身，值域 [0,1]。"""
        p = torch.einsum('bkhw,kc->bchw', self.weights(), self.palette())
        if self.amplitude != 1.0:
            flat = self.mean_rgb.view(1, 3, 1, 1)
            p = flat + self.amplitude * (p - flat)
        return p.clamp(0.0, 1.0)

    def render(self, x01: torch.Tensor) -> torch.Tensor:
        if self.shade is None:
            raise ValueError('要先呼叫 reset 凍結明暗與織紋')
        p = self.pattern()
        printed = (p * self.shade + self.detail).clamp(0.0, 1.0) \
            if self.keep_shading else p
        w = self.support.to(device=x01.device, dtype=x01.dtype)
        return w * printed + (1.0 - w) * x01

    def readout(self) -> dict:
        with torch.no_grad():
            pal = self.palette()
            lab = rgb_to_lab(pal.permute(1, 0).view(1, 3, self.colours, 1)
                             .float())
            chroma = (lab[0, 1] ** 2 + lab[0, 2] ** 2).clamp_min(0).sqrt()
            hard = (self.support > 0.5).to(torch.float32)
            share = ((self.weights().float() * hard).sum((0, 2, 3))
                     / hard.sum().clamp_min(1e-6))
        return {
            'print_colours': self.colours,
            'print_archetype': self.archetype,
            'print_tau': self.tau,
            'print_chroma_max': self.chroma_max,
            'print_chroma_used': round(float(chroma.max()), 3),
            'print_lightness_span': round(
                float(lab[0, 0].max() - lab[0, 0].min()), 3),
            'print_share_min': round(float(share.min()), 4),
            'print_patch_area': round(float(hard.mean()), 5),
        }


def _luma(x01: torch.Tensor) -> torch.Tensor:
    from .color_param import luma
    return luma(x01)


def safe_mask(x01: torch.Tensor, *, kind: str = 'upper', inset: float = 0.12,
              erode: int = 10, face_margin: int = 12,
              device=None) -> torch.Tensor:
    """衣物內部 ∩ 裁切安全區 ∩ 排除臉框。幾何支撐都放在這塊區域裡。

    與 `material_patch.patch_support` 的前半段相同，抽出來是因為幾何支撐
    要先看這塊區域的形狀才決定圓心與邊長，不能沿用那一支的散斑放置。
    """
    from src.metrics.identity import face_boxes

    from .carrier_mask import carrier_mask, erode_mask, exclude_boxes

    if not 0.0 <= inset < 0.5:
        raise ValueError(f'inset 要落在 [0,0.5)，收到 {inset}')
    dev = device or x01.device
    m = carrier_mask(x01, kind, device=dev).to(x01.dtype)
    m = erode_mask(m, erode)
    h, w = x01.shape[-2:]
    box = torch.zeros_like(m)
    y0, y1 = int(round(h * inset)), int(round(h * (1.0 - inset)))
    x0, x1 = int(round(w * inset)), int(round(w * (1.0 - inset)))
    box[..., y0:y1, x0:x1] = 1.0
    return exclude_boxes(m * box, face_boxes(x01, dev), margin=face_margin)


def _distance_transform(mask01, metric: str):
    import numpy as np
    from scipy.ndimage import distance_transform_cdt, distance_transform_edt

    arr = (mask01 > 0.5)[0, 0].detach().cpu().numpy()
    # **畫面邊界要算成背景。** 遮罩碰到邊界時距離轉換在陣列外沒有零可以量，
    # 貼邊的像素會拿到很大的內接距離，畫出來的圓於是被畫面切掉一半——
    # 外接框從 222×222 變成 109×217 才看得出來。補一圈零再算，算完切掉。
    arr = np.pad(arr, 1, mode='constant', constant_values=False)
    if metric == 'euclidean':
        d = distance_transform_edt(arr)
    elif metric == 'chessboard':
        d = distance_transform_cdt(arr, metric='chessboard')
    else:
        raise ValueError(f'未知的距離度量 {metric!r}')
    return np.asarray(d[1:-1, 1:-1], dtype=np.float32)


def place_shapes(mask: torch.Tensor, *, shape: str = 'disc',
                 area: float = 0.06, count: int = 1,
                 feather: int = 6) -> torch.Tensor:
    """把 `count` 個整塊的圓或正方形放進 `mask` 內部，總面積為 `area`。

    與 `carrier_mask.scatter_support` 的分界：那一支圓斑先畫好再與安全區
    相交，外形因此是圓與衣物輪廓的交集——圖上讀起來像污漬。這一支反過來，
    先算安全區的**內接距離**，再把整個形狀放進去，所以邊界是完整的幾何形。

    `count > 1` 時貪婪地放：每放一個就把它從可用區挖掉再找下一個內接最大處。
    放不下時**拋錯**，不自動縮小。`mask` 是呼叫端算好的安全區，這一支不自己
    跑分割——分割每呼叫一次就載入一次模型，掃描時要付好幾十秒。
    """
    import numpy as np

    if shape not in ('disc', 'square'):
        raise ValueError(f'shape 要是 disc 或 square，收到 {shape!r}')
    if count < 1:
        raise ValueError(f'count 至少為 1，收到 {count}')
    from .carrier_mask import feather_inward

    avail = float((mask > 0.5).to(torch.float32).mean())
    if avail < area:
        raise ValueError(
            f'可用的安全區只有 {avail:.4f}，放不下 {area:.4f} 的貼片。'
            '**不自動縮小、不移到臉上**：這張影像記成適用性失敗。')

    dev, dtype = mask.device, mask.dtype
    h, w = mask.shape[-2:]
    per = area * h * w / count
    want = math.sqrt(per / math.pi) if shape == 'disc' else 0.5 * math.sqrt(per)
    metric = 'euclidean' if shape == 'disc' else 'chessboard'

    free = (mask > 0.5).to(torch.float32)
    core = torch.zeros((1, 1, h, w), dtype=dtype, device=dev)
    yy = torch.arange(h, device=dev, dtype=torch.float32)[:, None]
    xx = torch.arange(w, device=dev, dtype=torch.float32)[None, :]
    for i in range(count):
        dist = _distance_transform(free, metric)
        flat = int(np.argmax(dist))
        cy, cx = divmod(flat, w)
        reach = float(dist[cy, cx])
        if reach < want:
            raise ValueError(
                f'第 {i + 1} 個 {shape} 的內接半徑只有 {reach:.1f} 像素，'
                f'放不下要求的 {want:.1f}。**不自動縮小**：'
                '這個形狀與面積的組合在這張影像上記成適用性失敗。')
        dy, dx = yy - cy, xx - cx
        core = torch.maximum(core, _blob(dy, dx, want, shape).to(dtype)[None,
                                                                       None])
        # 挖掉下一輪的可用區時直接畫一個放大的形狀，不走最大池化：半徑上百時
        # 池化核心是幾百見方，那一步慢到像當掉（512² × 227² 次比較）。
        free = free * (1.0 - _blob(dy, dx, want + 2.0,
                                   shape).to(torch.float32)[None, None])
    return feather_inward(core, feather)


def _blob(dy, dx, radius: float, shape: str):
    if shape == 'disc':
        return dy * dy + dx * dx <= radius * radius
    return (dy.abs() <= radius) & (dx.abs() <= radius)


def geometric_support(x01: torch.Tensor, *, shape: str = 'disc',
                      area: float = 0.06, count: int = 1,
                      kind: str = 'upper', inset: float = 0.12,
                      erode: int = 10, feather: int = 6,
                      face_margin: int = 12, device=None) -> torch.Tensor:
    """`safe_mask` 與 `place_shapes` 串起來的便利函式。"""
    dev = device or x01.device
    m = safe_mask(x01, kind=kind, inset=inset, erode=erode,
                  face_margin=face_margin, device=dev)
    return place_shapes(m, shape=shape, area=area, count=count,
                        feather=feather)



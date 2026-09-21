"""具有解析證書的亮度索引色盤；兩道界都在參數域內。

本族採用 brief 明列的 ``T(c) = P(L(c))`` 選項。三組 K 維參數分別
控制 Lab 的三個方向，預設共有 192 個自由度；不是三個全域常數。
它不宣稱逐通道單調，也不宣稱擾動必然低頻。色盤使用 C2 cubic B-spline，
沒有膚色遮罩、色相閘門、座標、影像統計或可學的結構常數。

膚色帶的精確定義
────────────────────────────────────────────────────────────────────
令 q 為線性 Y，A 為 D65 正規化 XYZ → 線性 RGB，
v = A @ (0.2, 0, -0.4)。固定暖色色盤為

    R(q) = Lab(q·1 + q(1-q)·v)。

它的 L 與輸入相同。膚色帶是 RGB 立方體內滿足
``||(a,b) - R_ab(Y)||₂ <= skin_width`` 的顏色；預設半徑為 8 Lab。
這是明確的軌跡管帶，不等於所有人臉像素，也不是舊版的整個色相扇區。
preview 另外回報涵蓋率。帶內仍可移動，render 不判斷像素是否在帶內。

全域證書
────────────────────────────────────────────────────────────────────
Lab 的 f 滿足 f'(u) <= 1/(3 u**(2/3))。設
``a = max(q**(1/3)*(1-q)) = 3/(4*4**(1/3))``，則上述暖色色盤的
彩度，以及把 v 乘 λ 後相對原色盤的 Lab 位移，分別不超過

    B·λ+ε，B·(1-λ)，其中 B = a*sqrt(100²+80²)/(3*0.599**(2/3))。

使用與 skimage 相同的 sRGB 矩陣。其正規化白點分量不小於 0.999，故
0.599 是所有 λ 下 XYZ 相對係數的下界；灰軸因矩陣捨入而帶有的彩度
ε < 0.006，由 ``sqrt((500|sX-1|)²+(200|sZ-1|)²)/(3*0.999**(2/3))``
界住。ε 以及 Lab 分段係數的捨入差均包含在 margin 內。

λ = min(1, 0.9*(C_max-margin)/B) 是固定設定。令 P0 為縮色後的色盤，
``r <= min(C_max-margin-Bλ,
           (d_max-margin)/κ - skin_width - B(1-λ))``，κ = 3/sqrt(2)。
非正的餘額會在建構時報錯，不能悄悄放寬界或把自由度歸零。

每個控制點為 ``z_i = tanh((θL_i, θa_i, θb_i))/sqrt(3)``。
四個非負且總和為 1 的 B-spline 權重給出 ``||z(L)||₂ <= 1``。
最後的 Lab 色盤是 ``P0(L) + ρ(L) z(L)``，其中 0 <= ρ <= r。
因此 C* <= Bλ+ε+r，帶內 Lab 位移 <= skin_width+B(1-λ)+r。
CIEDE2000 的 SL,SC,SH >= 1、|RT| <= 2、1+G <= 1.5，以及
``ΔC'²+ΔH'² = Δa'²+Δb²``，給出 ΔE00 <= κ·||ΔLab||₂。

色域也由參數域保證：Lab → XYZ 的逆函式在鄰域有明確導數上界 H。
P0 的線性 RGB 到六個面的距離至少為 m，故 Lab 球半徑 g=m/H 安全。
使用 ``ρ = r*g/(r+g)``，直接生成安全球內的顏色，沒有先生成再修補。
margin=0.02 僅保留標準矩陣四捨五入與浮點誤差的餘裕，不是自然度門檻。

起點
────────────────────────────────────────────────────────────────────
低彩度全域上界排除了全域恆等映射，所以預設零參數是 P0，不冒稱恆等。
另提供 ``base='identity'`` 以測試及使用真正的恆等起點；此模式要求
C_max 大於整個 RGB 立方體的解析彩度界，且位移界對所有顏色成立。
兩種模式的零參數都在 tanh 內點，三組參數的一階導數均不為零。
set_amplitude 只縮放球內的偏移，不混回可能超過 C_max 的原圖。
證書針對連續浮點 render；8-bit PNG 的量化誤差另外量測。
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch


MARGIN = 0.02
DE_LIPSCHITZ = 3.0 / math.sqrt(2.0)
PALETTE_BOUND = (3.0 / (4.0 * 4.0 ** (1.0 / 3.0))
                 * math.hypot(100.0, 80.0) / (3.0 * 0.599 ** (2.0 / 3.0)))
CUBE_CHROMA_BOUND = math.hypot(500.0, 200.0) * (1.0 - 4.0 / 29.0)
DEFAULT_BOUNDS = ((16.0, 54.0), (24.0, 38.0), (32.0, 24.0), (48.0, 24.0))

# 與 skimage.color.colorconv.xyz_from_rgb 完全相同的 D65 sRGB 矩陣。
RGB_TO_XYZ = ((0.412453, 0.357580, 0.180423),
              (0.212671, 0.715160, 0.072169),
              (0.019334, 0.119193, 0.950227))
WHITE = (0.95047, 1.0, 1.08883)
LAB_TO_F = ((1.0 / 116.0, 1.0 / 500.0, 0.0),
            (1.0 / 116.0, 0.0, 0.0),
            (1.0 / 116.0, 0.0, -1.0 / 200.0))


def _matrices(x):
    """只建立 3×3 常數；使用同一組正反矩陣避免色域端點的係數不一致。"""
    forward = x.new_tensor(RGB_TO_XYZ) / x.new_tensor(WHITE)[:, None]
    return forward, torch.linalg.inv(forward)


def _linear(x):
    """sRGB 解碼；未選到的冪次分支使用正數，避免 autograd 的 NaN。"""
    high = x > 0.04045
    safe = torch.where(high, (x + 0.055) / 1.055, torch.ones_like(x))
    return torch.where(high, safe.pow(2.4), x / 12.92)


def _encode(x):
    """線性 RGB 編碼；分支只是 sRGB 定義，沒有色域修補。"""
    high = x > 0.0031308
    safe = torch.where(high, x, torch.ones_like(x))
    return torch.where(high, 1.055 * safe.pow(1.0 / 2.4) - 0.055, 12.92 * x)


def _f(x):
    """CIELab 的分段函式，黑點的 backward 也保持有限。"""
    high = x > (6.0 / 29.0) ** 3
    safe = torch.where(high, x, torch.ones_like(x))
    return torch.where(high, safe.pow(1.0 / 3.0),
                       x / (3.0 * (6.0 / 29.0) ** 2) + 4.0 / 29.0)


def _inverse_f(x):
    return torch.where(x > 6.0 / 29.0, x.pow(3),
                       3.0 * (6.0 / 29.0) ** 2 * (x - 4.0 / 29.0))


def _linear_to_lab(x):
    forward, _ = _matrices(x)
    f = _f(torch.einsum('ij,bjhw->bihw', forward, x))
    return torch.stack((116 * f[:, 1] - 16,
                        500 * (f[:, 0] - f[:, 1]),
                        200 * (f[:, 1] - f[:, 2])), 1)


def rgb_to_lab(x):
    """本載體證書所用的 D65 CIELab；不裁切輸入。"""
    return _linear_to_lab(_linear(x))


def _lab_f(lab):
    l, a, b = lab.unbind(1)
    fy = (l + 16.0) / 116.0
    return torch.stack((fy + a / 500.0, fy, fy - b / 200.0), 1)


def lab_to_rgb(lab):
    """直接反轉 Lab；呼叫端的參數域已保證色域，不裁切輸出。"""
    _, inverse = _matrices(lab)
    return _encode(torch.einsum('ij,bjhw->bihw', inverse, _inverse_f(_lab_f(lab))))


def reference_lab(lightness):
    """固定膚色軌跡，輸入及輸出分別為 (N,1,H,W)、(N,3,H,W)。"""
    q = _inverse_f((lightness + 16.0) / 116.0)
    _, inverse = _matrices(lightness)
    v = inverse @ lightness.new_tensor((0.2, 0.0, -0.4))
    return _linear_to_lab(q + q * (1.0 - q) * v[None, :, None, None])


@dataclass(frozen=True)
class PreparedColours:
    """固定輸入的色盤座標；僅快取逐像素資料，不含可學參數或影像統計。"""

    base_lab: torch.Tensor
    radius: torch.Tensor
    index: torch.Tensor
    fraction: torch.Tensor
    dtype: torch.dtype


class BoundedCastCurveParam:
    """192 維的有界色盤載體；project 是空操作，θ 可取任意有限實數。"""

    name = 'bounded_cast_curve'

    def __init__(self, C_max=32.0, d_max=24.0, pieces=64, skin_width=8.0,
                 init_jitter=0.0, base='palette', apply_where=None):
        if isinstance(pieces, bool) or int(pieces) != pieces or pieces < 4:
            raise ValueError('pieces 必須是至少 4 的整數')
        if base not in ('palette', 'identity'):
            raise ValueError("base 必須是 'palette' 或 'identity'")
        if apply_where is not None:
            raise ValueError('純顏色全域映射不接受 apply_where')
        for name, value in (('C_max', C_max), ('d_max', d_max),
                            ('skin_width', skin_width), ('init_jitter', init_jitter)):
            if not math.isfinite(float(value)):
                raise ValueError(f'{name} 必須有限')
        if C_max <= MARGIN or d_max <= MARGIN or skin_width <= 0:
            raise ValueError('兩道界必須大於數值餘裕，skin_width 必須為正')
        if not 0 <= init_jitter <= 1:
            raise ValueError('init_jitter 必須落在 [0,1]')
        self.C_max, self.d_max = float(C_max), float(d_max)
        self.skin_width = float(skin_width)
        self.pieces, self.init_jitter, self.base = int(pieces), float(init_jitter), base
        self.amplitude = 1.0
        if base == 'identity':
            if C_max < CUBE_CHROMA_BOUND + MARGIN:
                raise ValueError('全域恆等模式需要容納整個 RGB 立方體的彩度界')
            self.palette_scale = 1.0
            self.offset_radius = (d_max - MARGIN) / DE_LIPSCHITZ
        else:
            self.palette_scale = min(1.0, 0.9 * (C_max - MARGIN) / PALETTE_BOUND)
            chroma_room = C_max - MARGIN - PALETTE_BOUND * self.palette_scale
            skin_room = ((d_max - MARGIN) / DE_LIPSCHITZ - skin_width
                         - PALETTE_BOUND * (1.0 - self.palette_scale))
            self.offset_radius = min(chroma_room, skin_room)
            if self.offset_radius <= 0:
                raise ValueError('這組界在本族的解析證書下沒有內點；請增大 C_max 或 d_max')
        self.theta_l = self.theta_a = self.theta_b = None

    def reset(self, x01, seed):
        """零抖動為內點；抖動只使用 seed，不依賴輸入的空間或顏色分布。"""
        generator = torch.Generator(device='cpu').manual_seed(int(seed))
        for name in ('theta_l', 'theta_a', 'theta_b'):
            p = torch.zeros((1, 1, self.pieces), dtype=x01.dtype, device=x01.device)
            if self.init_jitter:
                noise = torch.randn(p.shape, generator=generator, dtype=torch.float64)
                p = self.init_jitter * noise.to(device=x01.device, dtype=x01.dtype)
            setattr(self, name, p.detach().requires_grad_(True))

    def params(self):
        if self.theta_l is None:
            raise RuntimeError('請先 reset 或 load_state_dict')
        return [self.theta_l, self.theta_a, self.theta_b]

    def state_dict(self):
        """同時保存結構設定；跨不同界載入時拒絕悄悄改變證書。"""
        return {'params': [p.detach().clone() for p in self.params()],
                'config': (self.C_max, self.d_max, self.pieces, self.skin_width, self.base),
                'amplitude': self.amplitude}

    def load_state_dict(self, state):
        expected = (self.C_max, self.d_max, self.pieces, self.skin_width, self.base)
        if tuple(state['config']) != expected:
            raise ValueError('checkpoint 的結構常數與此載體不同')
        values = state['params']
        if len(values) != 3 or any(p.shape != (1, 1, self.pieces)
                                   or not torch.isfinite(p).all() for p in values):
            raise ValueError('checkpoint 需要三組有限的 (1,1,K) 參數')
        self.set_amplitude(state['amplitude'])
        self.theta_l, self.theta_a, self.theta_b = [
            p.detach().clone().requires_grad_(True) for p in values]

    @property
    def stages(self):
        return [self]

    def set_amplitude(self, value):
        """縮放可學的球內位移；零幅度仍保留低彩度的固定基底。

        僅接受 [0,1] 以維持結構證書；不保證相對原圖的 ΔE00 單調。
        solver 的 cap 二分必須先確認零幅度可行，不能把它當成恆等映射。
        """
        if isinstance(value, (list, tuple)):
            if len(value) != 1:
                raise ValueError('本載體只有一段')
            value = value[0]
        value = float(value)
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError('amplitude 必須落在 [0,1]')
        self.amplitude = value

    def project(self):
        """介面相容的空操作；結構界對任意有限 θ 都成立。"""

    def step_scale(self):
        return 1.0

    def bounds(self):
        """回報原始參數域；tanh 之前的 θ 沒有有限盒界。"""
        return -math.inf, math.inf

    def skin_band(self, x01):
        """只供報告與測試；render 不使用這個判斷。"""
        lab = rgb_to_lab(x01.double())
        ref = reference_lab(lab[:, :1])
        return (lab[:, 1:] - ref[:, 1:]).square().sum(1) <= self.skin_width ** 2

    def prepare(self, x01):
        """預先計算固定 RGB 顏色的 chart；float64 減少色域端點的捨入誤差。"""
        if x01.ndim != 4 or x01.shape[1] != 3 or not x01.is_floating_point():
            raise ValueError('輸入必須是浮點 (N,3,H,W) RGB')
        x = x01.double()
        linear = _linear(x)
        lab = _linear_to_lab(linear)
        _, inverse = _matrices(x)
        if self.base == 'palette':
            q = _inverse_f((lab[:, :1] + 16.0) / 116.0)
            v = self.palette_scale * (inverse @ x.new_tensor((0.2, 0.0, -0.4)))
            linear = q + q * (1.0 - q) * v[None, :, None, None]
            base_lab = _linear_to_lab(linear)
            lower = q * (1.0 + (1.0 - q) * v.min())
            upper = (1.0 - q) * (1.0 - q * v.max())
        else:
            base_lab = lab
            lower = linear.amin(1, keepdim=True)
            upper = 1.0 - linear.amax(1, keepdim=True)
        denominator = lower + upper
        safe = torch.where(denominator > 0, denominator, torch.ones_like(denominator))
        margin = lower * upper / safe
        a_norm = x.new_tensor(LAB_TO_F).square().sum(1).sqrt()
        max_f = torch.maximum(_lab_f(base_lab).amax(1, keepdim=True),
                              x.new_tensor(6.0 / 29.0))
        f_bound = max_f + a_norm.max() * self.offset_radius
        h = 3.0 * f_bound.square() * (inverse.abs() @ a_norm).max()
        gamut_radius = margin / h
        radius = self.offset_radius * gamut_radius / (self.offset_radius + gamut_radius)
        s = (lab[:, :1] / 100.0) * (self.pieces - 3)
        # L=100 是最後一段的右端；這只是索引定義，不是顏色裁切。
        index = torch.where(s >= self.pieces - 3, self.pieces - 4, s.floor()).long()
        return PreparedColours(base_lab, radius, index, s - index, x01.dtype)

    def render_prepared(self, prepared):
        """在已證明安全的 Lab 球內直接生成輸出。"""
        controls = torch.cat(self.params(), dim=1).to(prepared.base_lab)
        controls = controls.tanh()[0] / math.sqrt(3.0)
        u = prepared.fraction
        weights = ((1 - u) ** 3 / 6,
                   (3 * u ** 3 - 6 * u ** 2 + 4) / 6,
                   (-3 * u ** 3 + 3 * u ** 2 + 3 * u + 1) / 6,
                   u ** 3 / 6)
        z = torch.zeros_like(prepared.base_lab)
        for j, weight in enumerate(weights):
            values = controls[:, prepared.index[:, 0] + j].permute(1, 0, 2, 3)
            z = z + weight * values
        lab = prepared.base_lab + self.amplitude * prepared.radius * z
        return lab_to_rgb(lab).to(prepared.dtype)

    def render(self, x01):
        return self.render_prepared(self.prepare(x01))

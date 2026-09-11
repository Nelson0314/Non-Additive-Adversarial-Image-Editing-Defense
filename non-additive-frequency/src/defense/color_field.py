"""空間變化的顏色場：控制點網格上的 Lab 仿射，帶寬是可搜尋的參數。

與 `ChromaAffineParam` 的關係
────────────────────────────────────────────────────────────────────
`ChromaAffineParam` 是整張圖共用一個 2x2，四個參數。`runs/color_scaleup_search/`
的 1440 列量到那一族在 ΔE00 6.3 上的主體身分降幅是 −0.002 到 0.0043：全域的
顏色變換對擴散編輯近似等變，不移除主體資訊、也不與指令衝突。

這個類別把同一個映射放到 `grid` × `grid` 的控制點上，每點各有自己的 2x2 與
位移，雙線性放大到影像尺寸之後再用 `sigma` 的高斯模糊決定帶寬。`grid = 1` 且
`sigma` 很大時逐位元退化成全域仿射，所以舊的結果是這一族的一個角落。

**帶寬不設限。** `sigma = 0` 代表不模糊，控制點的邊界會直接進到空間高頻。
高頻要付的抗淨化代價由 `src/purify/` 的算子量出來，不由這個類別擋。
`highfreq_report` 仍逐列報。

亮度
────────────────────────────────────────────────────────────────────
`lock_luminance` 為真時 L 逐像素照抄輸入，與顏色線先前的設定一致；為假時 L 也
進入仿射，射程大得多但亮度的高頻也會動。兩者都是可搜尋的設定。
"""
from __future__ import annotations

import torch

from .lowfreq_color import soft_gamut
from .ncf_param import lab_to_rgb, mk_matrix, rgb_to_lab, regularize_covariance
from .lowfreq_color import highfreq_report
from src.purify.ops import gaussian_blur


def _upsample(field: torch.Tensor, size) -> torch.Tensor:
    return torch.nn.functional.interpolate(field, size=size, mode='bilinear',
                                           align_corners=True)


class ColorFieldParam:
    """(1,1,H,W) 支撐上的空間變化顏色映射。

    `grid` 是每邊的控制點數，`sigma` 是放大後權重圖的高斯模糊標準差（像素）。
    `lock_luminance` 決定 L 通道動不動。`radius` 是可學增量的 L∞ 盒，
    `max_gain` 是奇異值上界，兩者都可以設成 None 表示不約束。
    """

    name = 'color_field'

    def __init__(self, target_mean, target_cov, *, support, grid=8, sigma=8.0,
                 lock_luminance=True, radius=0.5, max_gain=None, cov_floor=1e-4,
                 amplitude=1.0, gamut='soft'):
        if int(grid) < 1:
            raise ValueError('grid 至少為 1')
        if float(sigma) < 0:
            raise ValueError('sigma 不可為負；0 代表不模糊')
        if gamut not in ('soft', 'clip'):
            raise ValueError("gamut 必須是 'soft' 或 'clip'")
        self.target_mean = torch.as_tensor(target_mean, dtype=torch.float64)
        self.target_cov = torch.as_tensor(target_cov, dtype=torch.float64)
        self.support = support
        self.grid = int(grid)
        self.sigma = float(sigma)
        self.lock_luminance = bool(lock_luminance)
        self.radius = None if radius is None else float(radius)
        self.max_gain = None if max_gain is None else float(max_gain)
        self.cov_floor = cov_floor
        self.gamut = gamut
        self.set_amplitude(amplitude)

    @property
    def channels(self) -> int:
        return 2 if self.lock_luminance else 3

    def set_amplitude(self, a):
        if not (0. <= float(a) <= 1.):
            raise ValueError('amplitude 必須落在 [0,1]')
        self.amplitude = float(a)

    def reset(self, x01, seed=0):
        if x01.ndim != 4 or x01.shape[:2] != (1, 3):
            raise ValueError('ColorField 需要單張 RGB 影像')
        if self.support.shape != (1, 1, *x01.shape[-2:]):
            raise ValueError('support 必須是 1x1xHxW')
        w = self.support.to(device=x01.device, dtype=torch.float64)[0, 0]
        lab = rgb_to_lab(x01).double()
        flat, wf = lab[0].flatten(1), w.flatten()
        total = wf.sum()
        if float(total) < 2:
            raise ValueError('支撐的權重不足兩個像素')
        self.source_mean = (flat * wf).sum(1) / total
        centred = flat - self.source_mean[:, None]
        source, self.source_audit = regularize_covariance(
            (centred * wf) @ centred.T / total, self.cov_floor)
        target, _ = regularize_covariance(
            self._to_like(self.target_cov, source), self.cov_floor)
        t0 = mk_matrix(source, target).to(lab.device)
        c = self.channels
        lo = 3 - c
        self.T0 = t0[lo:, lo:].contiguous()
        self.target_shift = self._to_like(self.target_mean, source)[lo:]
        g = self.grid
        self.delta = torch.zeros(1, c * c + c, g, g, dtype=torch.float64,
                                 device=lab.device).requires_grad_(True)

    @staticmethod
    def _to_like(v, ref):
        return torch.as_tensor(v, dtype=ref.dtype, device=ref.device)

    def params(self):
        return [self.delta]

    def increment(self):
        return self.delta

    @torch.no_grad()
    def project(self):
        """L∞ 盒與奇異值上界，兩者都可以關掉。

        與 `ChromaAffineParam.project` 一樣**不是幂等的**：奇異值那一步作用在
        `T0 + delta` 上，減回去之後 `delta` 可以稍微超出 `radius`。
        """
        if self.radius is not None:
            self.delta.clamp_(-self.radius, self.radius)
        if self.max_gain is None:
            return
        c = self.channels
        m = self.delta[0, :c * c].permute(1, 2, 0).reshape(-1, c, c)
        m = m + self.T0[None]
        u, s, vh = torch.linalg.svd(m)
        m = (u * s.clamp(max=self.max_gain)[..., None, :]) @ vh
        m = (m - self.T0[None]).reshape(self.grid, self.grid, c * c)
        self.delta[0, :c * c].copy_(m.permute(2, 0, 1))

    def effective_sigma(self, size) -> float:
        """實際用得上的模糊標準差。

        `gaussian_blur` 的反射填充要求核半徑小於影像邊長，所以超過
        `(min(size) - 1) / 3` 的 sigma 沒有辦法照字面執行，這裡夾住並把夾過的
        值寫進 `diagnostics`，不讓宣稱的帶寬與實際用的帶寬分開。控制點只有一個
        時整個場是空間常數，模糊對它是恆等，故回傳 0。
        """
        if self.grid == 1 or self.sigma <= 0:
            return 0.0
        return min(self.sigma, (min(size) - 1) / 3.0)

    def _maps(self, size):
        c = self.channels
        field = _upsample(self.delta, size)
        field = gaussian_blur(field, self.effective_sigma(size))
        m = field[:, :c * c].reshape(1, c, c, *size) + \
            self.T0[None, :, :, None, None]
        t = field[:, c * c:]
        return m, t

    def raw_rgb(self, x):
        lab = rgb_to_lab(x).double()
        c = self.channels
        lo = 3 - c
        src = lab[:, lo:]
        centred = src - self.source_mean[lo:].to(lab)[None, :, None, None]
        m, t = self._maps(tuple(x.shape[-2:]))
        mapped = torch.einsum('bijhw,bjhw->bihw', m, centred)
        mapped = mapped + self.target_shift.to(mapped)[None, :, None, None] + t
        out = src + self.amplitude * (mapped - src)
        if self.lock_luminance:
            out = torch.cat([lab[:, :1], out], dim=1)
        return lab_to_rgb(out).to(x.dtype)

    def render(self, x):
        w = self.support.to(device=x.device, dtype=x.dtype)
        raw = self.raw_rgb(x)
        out = w * (soft_gamut(raw) if self.gamut == 'soft' else raw.clamp(0, 1)) \
            + (1 - w) * x
        return torch.where(w > 0, out, x)

    def state_dict(self):
        return {'delta': self.delta.detach().clone()}

    def load_state_dict(self, state):
        self.delta = state['delta'].detach().clone().requires_grad_(True)

    @torch.no_grad()
    def diagnostics(self, x):
        c = self.channels
        m, _ = self._maps(tuple(x.shape[-2:]))
        sv = torch.linalg.svdvals(m.reshape(c, c, -1).permute(2, 0, 1))
        raw = self.raw_rgb(x)
        return {'name': self.name, 'grid': self.grid, 'sigma': self.sigma,
                'sigma_effective': self.effective_sigma(tuple(x.shape[-2:])),
                'lock_luminance': int(self.lock_luminance),
                'amplitude': self.amplitude, 'gamut': self.gamut,
                'max_gain': '' if self.max_gain is None else self.max_gain,
                'radius': '' if self.radius is None else self.radius,
                'chroma_gain_max': float(sv.max()),
                'chroma_gain_min': float(sv.min()),
                'field_spread': float(self.delta.detach().std()),
                'clipping_fraction': float(((raw < 0) | (raw > 1)).double().mean()),
                'clipping_max': float((raw - raw.clamp(0, 1)).abs().max()),
                'T_distance': float(self.delta.detach().norm()),
                'support_area': float(self.support.to(torch.float64).mean()),
                **highfreq_report(x, self.render(x))}

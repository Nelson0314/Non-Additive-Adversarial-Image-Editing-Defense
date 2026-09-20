"""衣物材質貼片：固定支撐、固定明暗與細紋，只學粗尺度的印花反射色。

與本專案既有載體的差別
────────────────────────────────────────────────────────────────────
`ColorCurveParam`、`ColorGridParam`、`geometry_field` 都是**全域**載體：把一份
小預算攤到整張圖。`runs/advcf_objective/` 量到那一族落在一條單一的
「代價→位移」直線上（`LPIPS/ΔE00` 0.0163–0.0170），十六種目標函數改法沒有一個
贏過原本的 `free`。

本載體是反過來的操作點：**面積約 3%、局部振幅大、全域 ΔE00 小**。那個組合
不在上面那條直線取樣過的範圍內，所以那條線不構成對它的否定。

構造
────────────────────────────────────────────────────────────────────
    x_θ = (1 − M)·x + M·clamp(ρ_θ ⊙ S_x + H_x, 0, 1)

    ρ_θ = Σ_k softmax_k(U(θ))·p_k

`M` 是**事前固定**的支撐（衣物內部 ∩ 裁切安全區 ∩ 排除臉框），最佳化不動它。
`S_x` 是原圖的明暗（亮度低通，在支撐上正規化成均值 1），`H_x` 是細紋殘差
（原圖減去低通）：兩者都凍結，所以布料的摺痕、織紋與光照留在原處，
換掉的只有「這塊布印的是什麼顏色」。

`U` 是把 `(1, K, g, g)` 的係數雙三次升到影像尺寸。**空間尺度由 `g` 決定**
而不是由最佳化決定：512² 上 `g = 16` 給約 32 像素的特徵、`g = 8` 給約 64 像素。
這個帶寬是刻意的——單像素噪聲與密集細點過不了 JPEG Q75 與 σ=1.0 的模糊。

兩個由構造保證的性質
────────────────────────────────────────────────────────────────────
- **支撐外逐位元不變。** `M = 0` 的地方輸出恆等於輸入。
- **每個貼片像素都是色盤的凸組合。** `softmax` 的權重非負且和為 1，
  所以 `ρ` 落在色盤的凸包內。色盤取自原衣物時凸包就是該衣物的色域。
  這縮小了搜尋空間，**但它不是自然度保證**——`chroma_gain > 1` 與
  `mode='print'` 都會把色盤推出原色域，而且凸包內的圖案照樣可以醜。
  自然度一律由使用者看圖判定。
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence

import torch
import torch.nn.functional as F

from .color_param import LUMA_WEIGHTS as LUMA_RGB
from .color_param import luma


def _lowpass(t: torch.Tensor, radius: int) -> torch.Tensor:
    k = 2 * radius + 1
    return F.avg_pool2d(F.pad(t, (radius,) * 4, mode='reflect'),
                        kernel_size=k, stride=1)


PALETTE_MODES = ('garment', 'print')


def garment_palette(x01: torch.Tensor, support: torch.Tensor, count: int,
                    chroma_gain: float = 1.0, mode: str = 'garment',
                    chroma: float = 40.0) -> torch.Tensor:
    """從支撐內的像素取 `count` 個色盤條目，回傳 (count, 3) 的 RGB。

    取法是沿支撐內色彩的**第一主方向**排序後取等分位數：這條方向通常就是
    明暗加上主色調的變化，等分位數因此涵蓋該塊布料實際出現過的範圍，
    而不是只取平均色。

    兩個模式，差在色盤能不能離開這件衣服的色域
    ────────────────────────────────────────────────────────────────
    `garment`（預設）只用衣物自己的顏色，`chroma_gain > 1` 在 Lab 上放大 a/b。
    **這在低彩度的衣物上會退化**：白襯衫的 a/b 本來就接近零，乘任何倍數仍然
    接近零，貼片因此推不出振幅。這不是瑕疵而是那個結構保證的代價。

    `print` 保留衣物的亮度剖面，把 a/b 設成色相均勻分布在色環上、半徑**至多**
    `chroma` 的一組值。理由是現實裡白 T 上的印花本來就可以是任何顏色，亮度
    仍然跟著布料走所以摺痕的明暗不會打架。**代價是色盤離開了原衣物的色域，
    「凸包」那個結構性的自然度論證在這個模式下不成立**，只剩看圖判定。

    半徑是「至多」而不是「等於」：Lab 的亮度極端處（很暗或很亮）放不下
    半徑 40 的彩度，直接設下去再 `clip` 會把 L 一起改掉——實測最亮與最暗
    那兩個條目的亮度被裁掉 0.52，「保留亮度剖面」那句話就不成立了。
    所以逐條目二分找**留在 sRGB 色域內**的最大彩度，寧可彩度不足也不裁。
    """
    if count < 2:
        raise ValueError(f'色盤至少要兩個條目，收到 {count}')
    if chroma_gain <= 0:
        raise ValueError(f'chroma_gain 必須為正，收到 {chroma_gain}')
    if mode not in PALETTE_MODES:
        raise ValueError(f'mode 必須是 {PALETTE_MODES} 之一，收到 {mode!r}')
    if chroma <= 0:
        raise ValueError(f'chroma 必須為正，收到 {chroma}')
    m = (support > 0.5)[0, 0]
    if int(m.sum()) < count:
        raise ValueError(
            f'支撐內只有 {int(m.sum())} 個像素，取不出 {count} 個色盤條目')
    px = x01[0].permute(1, 2, 0)[m]
    centred = px - px.mean(0, keepdim=True)
    _, _, v = torch.pca_lowrank(centred, q=min(3, centred.shape[0]))
    axis = v[:, 0]
    # **主方向的正負號是任意的**，而 `torch.pca_lowrank` 走隨機化 SVD，所以
    # 同一份輸入跑兩次可能拿到相反的方向。`print` 模式按索引配色相，號一翻
    # 就換成另一組色盤，整批因此不可重現。把號釘在「投影與亮度正相關」上。
    w = torch.tensor(LUMA_RGB, device=px.device, dtype=px.dtype)
    if float(axis @ w) < 0:
        axis = -axis
    proj = centred @ axis
    order = torch.argsort(proj)
    idx = [order[int(round(i * (len(order) - 1) / (count - 1)))]
           for i in range(count)]
    pal = px[torch.stack(idx)]
    if mode == 'garment' and chroma_gain == 1.0:
        return pal.contiguous()

    from skimage.color import lab2rgb, rgb2lab

    lab = rgb2lab(pal.detach().cpu().float().numpy()[None])
    if mode == 'garment':
        lab[..., 1:] *= float(chroma_gain)
    else:
        for i in range(count):
            ang = 2.0 * math.pi * i / count
            lab[0, i, 1], lab[0, i, 2] = _fit_chroma(
                float(lab[0, i, 0]), ang, float(chroma))
    out = lab2rgb(lab)[0].clip(0.0, 1.0)
    return torch.as_tensor(out, device=pal.device, dtype=pal.dtype).contiguous()


def _fit_chroma(lightness: float, angle: float, ceiling: float,
                tol: float = 1e-3, iters: int = 20):
    """在固定亮度與色相上，二分找留在 sRGB 色域內的最大彩度。

    回傳 `(a, b)`。上限本身就在色域內時直接回傳上限，不做多餘的搜尋。
    """
    import numpy as np
    from skimage.color import lab2rgb

    def inside(c: float) -> bool:
        rgb = lab2rgb(np.array([[[lightness,
                                  c * math.cos(angle),
                                  c * math.sin(angle)]]]))
        return bool(rgb.min() >= -tol and rgb.max() <= 1.0 + tol)

    if inside(ceiling):
        return ceiling * math.cos(angle), ceiling * math.sin(angle)
    lo, hi = 0.0, ceiling
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if inside(mid):
            lo = mid
        else:
            hi = mid
    return lo * math.cos(angle), lo * math.sin(angle)


class MaterialPatchParam:
    """φ = θ，(1, K, g, g) 的色盤權重場。支撐、色盤、明暗、細紋全部固定。

    `θ = 0` 時 `softmax` 是均勻的，`ρ` 等於色盤的平均色——貼片變成一塊
    帶原布料明暗與織紋的素色。那**不是零梯度點**（`softmax` 在均勻處的
    Jacobian 非零），但它也不是恆等：支撐內的顏色已經被換掉了。
    要逐位元恆等只能把支撐設成全零。
    """

    name = "material_patch"

    def __init__(self, support: torch.Tensor, palette: torch.Tensor, *,
                 grid: int = 16, shade_radius: int = 12,
                 logit_cap: float = 6.0, init_jitter: float = 1.0):
        if support.ndim != 4 or support.shape[1] != 1:
            raise ValueError(
                f'support 必須是 (1,1,H,W)，收到 {tuple(support.shape)}')
        if float(support.max()) <= 0:
            raise ValueError('支撐全為零，貼片無處可放')
        if palette.ndim != 2 or palette.shape[1] != 3:
            raise ValueError(
                f'palette 必須是 (K,3)，收到 {tuple(palette.shape)}')
        if grid < 2:
            raise ValueError(f'grid 至少為 2，收到 {grid}')
        if shade_radius < 1:
            raise ValueError(f'shade_radius 至少為 1，收到 {shade_radius}')
        if logit_cap <= 0:
            raise ValueError(f'logit_cap 必須為正，收到 {logit_cap}')
        self.support = support
        self.palette = palette
        self.grid = int(grid)
        self.shade_radius = int(shade_radius)
        self.logit_cap = float(logit_cap)
        self.init_jitter = float(init_jitter)
        self.amplitude = 1.0
        self.theta: Optional[torch.Tensor] = None
        self.shade: Optional[torch.Tensor] = None
        self.detail: Optional[torch.Tensor] = None

    def reset(self, x01: torch.Tensor, seed: int = 0) -> None:
        """凍結明暗與細紋，並抽一個非均勻的起點。

        `init_jitter = 0` 給均勻 `softmax`（素色貼片）；重啟要有意義就得讓
        不同種子真的走到不同的盆，所以預設是 1.0。
        """
        w = self.support.to(device=x01.device, dtype=x01.dtype)
        self.palette = self.palette.to(device=x01.device, dtype=x01.dtype)
        lum = luma(x01)
        base = _lowpass(lum, self.shade_radius)
        denom = (base * w).sum() / w.sum().clamp_min(1e-6)
        if float(denom) <= 0:
            raise ValueError('支撐內的平均亮度為零，明暗場沒有定義')
        self.shade = base / denom
        self.detail = x01 - _lowpass(x01, self.shade_radius)
        k = self.palette.shape[0]
        g = torch.Generator(device='cpu').manual_seed(int(seed))
        t = torch.randn((1, k, self.grid, self.grid), generator=g)
        self.theta = (t * self.init_jitter).to(
            device=x01.device, dtype=x01.dtype).requires_grad_(True)

    def params(self) -> List[torch.Tensor]:
        return [self.theta]

    def state_dict(self):
        return {'theta': self.theta.detach().clone()}

    def load_state_dict(self, state):
        self.theta = state['theta'].detach().clone().requires_grad_(True)

    @property
    def stages(self):
        return [self]

    @torch.no_grad()
    def project(self) -> None:
        """夾住 logit 的幅度。

        不夾的話 `softmax` 會飽和成硬邊，那種邊緣過不了 JPEG 與模糊，
        而讀數上看起來仍然在進步——把限制放在參數上，不要事後再縮。
        """
        self.theta.clamp_(-self.logit_cap, self.logit_cap)

    def step_scale(self) -> float:
        return 2.0 * self.logit_cap

    def set_radius(self, r: float) -> None:
        self.logit_cap = float(r)

    def set_amplitude(self, a) -> None:
        """縮 logit。`a = 0` 讓 `softmax` 回到均勻，貼片褪成色盤的平均色。

        `immunise.optimise_carrier` 在還原 checkpoint 之後呼叫這個還原幅度，
        `fit_caps` 則用它去找可行點。**這裡是真的旋鈕不是記帳欄**：
        `reflectance` 會用到它，所以 `fit_caps` 在這個載體上縮得動東西。
        由 0 到 1 單調把貼片由素色推到完整圖案。
        """
        if isinstance(a, (list, tuple)):
            if len(a) != 1:
                raise ValueError('這個載體只有一段，逐段指定時長度要是 1')
            a = a[0]
        if float(a) < 0:
            raise ValueError(f'amplitude 不得為負，收到 {a}')
        self.amplitude = float(a)

    def reflectance(self, size) -> torch.Tensor:
        """(1,3,H,W)，色盤的凸組合。"""
        logits = F.interpolate(self.theta * self.amplitude, size=size,
                               mode='bicubic', align_corners=False)
        w = torch.softmax(logits, dim=1)
        return torch.einsum('bkhw,kc->bchw', w, self.palette)

    def render(self, x01: torch.Tensor) -> torch.Tensor:
        if self.shade is None:
            raise ValueError('要先呼叫 reset 凍結明暗與細紋')
        rho = self.reflectance(x01.shape[-2:])
        patch = (rho * self.shade + self.detail).clamp(0.0, 1.0)
        w = self.support.to(device=x01.device, dtype=x01.dtype)
        return w * patch + (1.0 - w) * x01

    def readout(self) -> dict:
        w = self.support
        return {'patch_area': round(float((w > 0.5).to(w.dtype).mean()), 5),
                'patch_weight_mean': round(float(w.mean()), 5),
                'patch_grid': self.grid,
                'patch_palette': self.palette.shape[0],
                'patch_logit_cap': self.logit_cap,
                'patch_shade_radius': self.shade_radius,
                'patch_amplitude': round(self.amplitude, 5)}


def patch_support(x01: torch.Tensor, *, kind: str = 'upper',
                  area: float = 0.03, blobs: int = 2,
                  inset: float = 0.12, erode: int = 10, feather: int = 6,
                  face_margin: int = 12, seed: int = 0,
                  device=None) -> torch.Tensor:
    """事前固定的貼片支撐：衣物內部 ∩ 裁切安全區 ∩ 排除臉框。

    `inset` 是畫面每一側留出的比例。取 0.12 而不是 0.10 是因為攻擊者的
    `crop_resize` 每邊裁 10%：貼片完全落在 12%–88% 內，裁切不會切到它。
    **這不表示裁切對貼片無效**——裁後升回原尺寸會改變它與 VAE 網格的對齊，
    那要靠評估量，不能由「沒被裁掉」推出來。

    面積放不下時**拋錯**，不自動縮小也不改放到臉上：那時記成這張影像的
    適用性失敗。
    """
    from src.defense.carrier_mask import (carrier_mask, erode_mask,
                                          exclude_boxes, feather_inward,
                                          match_area, scatter_support)
    from src.metrics.identity import face_boxes

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
    m = m * box
    m = exclude_boxes(m, face_boxes(x01, dev), margin=face_margin)
    avail = float((m > 0.5).to(m.dtype).mean())
    if avail < area:
        raise ValueError(
            f'可用的衣物安全區只有 {avail:.4f}，放不下 {area:.4f} 的貼片。'
            '**不自動縮小、不移到臉上**：這張影像記成適用性失敗。')
    core = scatter_support(m, blobs, area, seed)
    return feather_inward(core, feather)

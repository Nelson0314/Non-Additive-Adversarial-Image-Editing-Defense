"""EOT：在最佳化迴圈裡對防禦圖抽隨機的前處理，再算目標。

為什麼需要
────────────────────────────────────────────────────────────────────
`runs/field_purify/` 量到幾何載體的抗淨化形狀與顏色相反：`flow_08` 在
identity 6/25、JPEG 5/25、模糊 7/25，但 **crop_resize 只剩 1/25**。裁切加縮放
本身就是一個幾何變換，它重取樣了位移場；顏色載體在同一道上是 16/25。

現行的解完全沒有看過任何前處理——`noise_seed` 也固定在 0。EOT 把「攻擊者會
先做一件事」放進期望值裡：每一步抽一組參數，目標在那組參數下算，梯度因此指向
**在一整族前處理下都成立**的解，而不是只在未處理的那一點上成立的解。

這不是把指令放回迴圈
────────────────────────────────────────────────────────────────────
硬約束管的是**編輯指令**：防禦圖必須在不知道攻擊者會下什麼指令的情況下產生。
前處理不是指令，它是攻擊者在編輯**之前**對影像做的事，與他想改什麼無關。
`scripts/immunise_field.py` 的設定檔守門照舊拒絕任何 `instruction`／`prompt` 鍵。

**但評估的誠實性要另外交代。** EOT 抽的是隨機參數（裁切比例在一個區間裡均勻
抽、模糊 σ 在一個區間裡抽），而評估用的是固定的那幾組（DIA 的 10%、σ=1.0）。
所以**算子的族是看過的，具體的參數不是**。報告要照這個講，不可以宣稱評估的
淨化是完全未見的。JPEG 不進 EOT：它不可微，直通估計會讓梯度與真實算子脫節，
而且它本來就撐得住（5/25）。
"""
from __future__ import annotations

from typing import Dict, List, Optional

import torch

from src.purify.ops import crop_resize, gaussian_blur


def random_crop_resize(x: torch.Tensor, generator: torch.Generator,
                       lo: float = 0.02, hi: float = 0.15) -> torch.Tensor:
    """隨機比例的中心裁切加縮放。`lo` 不取 0：恆等由 `identity` 那一支負責。

    比例以像素為單位量化——`crop_resize` 內部就是 `int(round(h*fraction))`，
    連續抽樣在 512 px 上只落在 77 個不同的裁切量上，所以直接抽那個整數，
    抽樣分布與實際作用的變換一致。
    """
    h = x.shape[-2]
    k_lo = max(1, int(round(h * lo)))
    k_hi = max(k_lo + 1, int(round(h * hi)))
    k = int(torch.randint(k_lo, k_hi + 1, (1,), generator=generator,
                          device=generator.device).item())
    return crop_resize(x, fraction=k / h)


def random_blur(x: torch.Tensor, generator: torch.Generator,
                lo: float = 0.4, hi: float = 1.6) -> torch.Tensor:
    u = torch.rand(1, generator=generator, device=generator.device).item()
    return gaussian_blur(x, lo + (hi - lo) * u)


TRANSFORMS = {
    'identity': lambda x, g: x,
    'crop_resize': random_crop_resize,
    'blur': random_blur,
}


class EOTObjective:
    """把目標包成「在一組隨機前處理上的平均」。

    `kinds` 是要抽的算子名稱，`samples` 是每一步抽幾個。`identity` 一定在
    第一個位置且**不計入隨機抽樣**：少了它，解可以把分數押在「只有被處理過
    才成立」的地方，而攻擊者不處理也照樣編輯得動。

    亂數走自己的 `Generator`：載體的初始化與 `FreeObjective` 的噪聲都是固定
    種子的確定性量，EOT 的隨機性不可以污染那兩個，否則逐步的分數變化分不出
    是最佳化走了還是抽到了不同的變換。
    """

    def __init__(self, base, *, kinds: Optional[List[str]] = None,
                 samples: int = 1, seed: int = 0, device=None,
                 include_identity: bool = True):
        unknown = [k for k in (kinds or []) if k not in TRANSFORMS]
        if unknown:
            raise ValueError(f'不認得的前處理 {unknown}；'
                             f'可用的是 {sorted(TRANSFORMS)}')
        self.base = base
        self.kinds = [k for k in (kinds or []) if k != 'identity']
        if not self.kinds:
            raise ValueError('EOT 至少要有一個非恆等的前處理，否則它沒有作用')
        self.samples = int(samples)
        if self.samples < 1:
            raise ValueError('每一步至少要抽一個樣本')
        self.include_identity = bool(include_identity)
        self.ip2p = base.ip2p
        self.timesteps = base.timesteps
        self.weights = base.weights
        dev = device or getattr(base, 'device', None) or torch.device('cpu')
        self.generator = torch.Generator(device=dev).manual_seed(int(seed))

    def _views(self, x_def: torch.Tensor) -> List[torch.Tensor]:
        views = [x_def] if self.include_identity else []
        for _ in range(self.samples):
            i = int(torch.randint(len(self.kinds), (1,),
                                  generator=self.generator,
                                  device=self.generator.device).item())
            views.append(TRANSFORMS[self.kinds[i]](x_def, self.generator))
        return views

    def terms(self, x_def: torch.Tensor) -> Dict[str, torch.Tensor]:
        views = self._views(x_def)
        acc: Dict[str, torch.Tensor] = {}
        for v in views:
            for k, t in self.base.terms(v).items():
                acc[k] = t if k not in acc else acc[k] + t
        return {k: v / len(views) for k, v in acc.items()}

    def score(self, x_def: torch.Tensor) -> torch.Tensor:
        views = self._views(x_def)
        return torch.stack([self.base.score(v) for v in views]).mean()

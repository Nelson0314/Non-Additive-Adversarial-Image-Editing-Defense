"""把獎勵移到貼片外面：空文字鏈的**支撐外**輸出位移。

為什麼不是現行的 `out`
────────────────────────────────────────────────────────────────────
`readout_terms.OutputDisplacement` 取的是整張圖的 LPIPS。貼片自己變得夠花
就能把那個數字推上去，而那不是防禦——`runs/patch_canvas/` 量到最佳化臂的
LPIPS 高 45%、失真也高 75%，換算成每單位失真的位移沒有勝過不最佳化的對照。
整張圖的讀數給了一條捷徑：改貼片自己。

這一份只在支撐外（再往外留一圈緩衝）算位移，那條捷徑因此不存在。獎勵只來自
「貼片把它以外的地方推開了」，也就是局部載體唯一可能構成防禦的通道。

三個帶分開報
────────────────────────────────────────────────────────────────────
`near` 是貼片外緣的一圈，`far` 是其餘畫面，`face` 是臉框。顏色從貼片暈開
只會動 `near`；真正有意義的是 `far` 與 `face`。**只看合計會把顏色擴散
讀成防禦**，所以三個帶一律分開記。

位移用 RGB 的 L1 而不是 LPIPS：LPIPS 的感受野會跨過遮罩邊界，把貼片自己的
變化算進支撐外的分數裡，那條捷徑就又回來了。LPIPS 只當輔助讀數。
"""

from __future__ import annotations

from typing import Optional

import torch

from .subject_mask import _dilate


def band_masks(support: torch.Tensor, x01: torch.Tensor, *,
               margin: int = 16, near_width: int = 48,
               device=None) -> dict:
    """支撐外的三個帶：`near`、`far`、`face`。皆為 (1,1,H,W) 的 0/1。

    `margin` 是支撐外先排除的緩衝寬度。留它是因為羽化帶與 VAE 的 8 倍降採樣
    會讓緊貼邊界的像素本來就跟著貼片走，把那一圈算進來等於把貼片自己算進來。
    """
    from src.metrics.identity import face_boxes

    if margin < 0:
        raise ValueError(f'margin 不得為負，收到 {margin}')
    if near_width < 1:
        raise ValueError(f'near_width 至少為 1，收到 {near_width}')
    dev = device or support.device
    hard = (support > 0.5).to(torch.float32).to(dev)
    buffered = _dilate(hard, margin)
    near_outer = _dilate(hard, margin + near_width)
    near = (near_outer - buffered).clamp(0.0, 1.0)
    outside = (1.0 - buffered).clamp(0.0, 1.0)
    far = (outside - near).clamp(0.0, 1.0)

    face = torch.zeros_like(hard)
    h, w = hard.shape[-2:]
    for box in face_boxes(x01, dev):
        x0, y0, x1, y1 = (int(round(float(v))) for v in box)
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(w, max(x0 + 1, x1)), min(h, max(y0 + 1, y1))
        face[..., y0:y1, x0:x1] = 1.0
    face = face * outside
    for name, m in (('near', near), ('far', far), ('outside', outside)):
        if float(m.sum()) <= 0:
            raise ValueError(f'{name} 帶是空的，支撐把整張畫面吃光了')
    return {'near': near, 'far': far, 'face': face, 'outside': outside}


class OutsideDisplacement:
    """`‖(1−M⁺)·(N(x_def) − N(x))‖₁ / |1−M⁺|`，越大代表把貼片以外推得越開。

    `N` 是目標自己的空文字鏈 `null_edit`。兩側共用同一組噪聲：參考值只在
    抽樣換過之後才重算，與 `readout_terms.OutputDisplacement` 的快取規則相同。
    不共用噪聲的話讀到的是取樣的隨機性，不是防禦造成的位移。
    """

    name = 'outside'

    def __init__(self, objective, x01: torch.Tensor, support: torch.Tensor, *,
                 margin: int = 16, near_width: int = 48,
                 weight_far: float = 1.0, weight_face: float = 1.0,
                 weight_near: float = 0.0):
        self.objective = objective
        self.x01 = x01
        self.bands = band_masks(support, x01, margin=margin,
                                near_width=near_width, device=x01.device)
        self.weights = {'far': float(weight_far), 'face': float(weight_face),
                        'near': float(weight_near)}
        if all(v <= 0 for v in self.weights.values()):
            raise ValueError('三個帶的權重全為零，這一項沒有定義')
        self._cache_key = None
        self._reference: Optional[torch.Tensor] = None

    def reference(self) -> torch.Tensor:
        key = (int(self.objective.draws), int(self.objective.seed))
        if key != self._cache_key:
            with torch.no_grad():
                self._reference = self.objective.null_edit(self.x01).float()
            self._cache_key = key
        return self._reference

    def _reduce(self, diff: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        m = mask.to(device=diff.device, dtype=diff.dtype)
        total = m.sum() * diff.shape[1]
        return (diff.abs() * m).sum() / total.clamp_min(1.0)

    def __call__(self, x_def: torch.Tensor) -> torch.Tensor:
        a = self.reference()
        b = self.objective.null_edit(x_def).float()
        diff = b.clamp(0, 1) - a.clamp(0, 1)
        out = None
        for name, w in self.weights.items():
            if w <= 0:
                continue
            term = w * self._reduce(diff, self.bands[name])
            out = term if out is None else out + term
        return out

    @torch.no_grad()
    def report(self, x_def: torch.Tensor) -> dict:
        """三個帶各自的位移，寫進 CSV 用。**不進梯度。**"""
        a = self.reference()
        b = self.objective.null_edit(x_def).float()
        diff = b.clamp(0, 1) - a.clamp(0, 1)
        return {f'outside_{k}': round(float(self._reduce(diff,
                                                         self.bands[k])), 6)
                for k in ('near', 'far', 'face', 'outside')}


class PrintObjective:
    """`FreeObjective` 外掛 `OutsideDisplacement`，**不改 `readout_terms.py`**。

    介面與 `CompositeObjective` 相同的那四個方法：`score`／`terms`／
    `eval_score`／`eval_terms`。自成一份是為了跟顏色線那邊的同名檔案脫鉤，
    兩條線同時在改的時候不互相踩。

    符號慣例與專案其餘部分相同：分數一律是要最小化的方向，所以越大越好的項
    以負號進分數。
    """

    def __init__(self, base, *, outside_term=None, weights=None):
        self.base = base
        self.outside_term = outside_term
        self.weights = dict(base.weights, **(weights or {}))
        self.ip2p = base.ip2p
        self.timesteps = base.timesteps

    def _w(self, name):
        return float(self.weights.get(name, 0.0) or 0.0)

    def _extra(self, x_def, terms):
        if self.outside_term is not None and self._w('outside') > 0:
            terms['outside'] = self.outside_term(x_def)
        return terms

    def _compose(self, terms):
        s = (-self._w('enc') * torch.tanh(terms['enc'])
             - self._w('cond') * torch.tanh(terms['cond']))
        if 'id' in terms:
            s = s + self._w('id') * terms['id']
        if 'outside' in terms:
            s = s - self._w('outside') * terms['outside']
        return s

    def terms(self, x_def):
        return self._extra(x_def, dict(self.base.terms(x_def)))

    def score(self, x_def):
        return self._compose(self.terms(x_def))

    def eval_terms(self, x_def):
        """固定驗證抽樣上的逐項值。外掛項也要在同一個抽樣狀態裡算。

        `FreeObjective.eval_terms` 回傳前就把訓練用的 `draws` 還原了，而
        `OutsideDisplacement` 用 `objective.seed + objective.draws` 產生取樣鏈
        的噪聲。先呼叫 `base.eval_terms` 再算外掛項，`outside` 會落在**訓練
        抽樣**上——選 checkpoint 的分數因此帶著訓練端的隨機性。這裡自己進出
        驗證狀態，所有項在同一個抽樣上算完才離開。
        """
        base = self.base
        val = getattr(base, 'val', None)
        if val is None:
            return self._extra(x_def, dict(base.terms(x_def)))
        keep = (base.noise, base.timesteps, base.eps0, base.eps0_norm,
                base.draws, base.resample)
        (base.noise, base.timesteps, base.eps0, base.eps0_norm,
         base.draws) = val
        base.resample = False
        try:
            return self._extra(x_def, dict(base.terms(x_def)))
        finally:
            (base.noise, base.timesteps, base.eps0, base.eps0_norm,
             base.draws, base.resample) = keep

    def eval_score(self, x_def):
        return self._compose(self.eval_terms(x_def))

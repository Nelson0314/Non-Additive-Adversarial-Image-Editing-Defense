"""兩個新的無指令項：對準讀數的輸出位移，以及單調曲線專屬的色調壓縮。

為什麼要這兩項
────────────────────────────────────────────────────────────────────
`runs/advcf_objective/` 量到：純半徑那一族的代理分數對**量到的** LPIPS 位移是
Pearson **r = −0.996**，幾乎完美預測；把改過目標函數的 `deep` 兩臂加進來就掉到
**−0.893**——那兩臂把代理最佳化得很好，增益卻沒有轉成編輯輸出的位移。
相關性正好在「改過目標函數的臂」上斷掉，這是代理設定錯誤的直接證據。

錯在哪很具體：`FreeObjective` 的 `id` 項量的是 `null_edit` 輸出的**人臉餘弦**，
而判準那一側量的是**整張編輯輸出的位移**。兩者不是同一件事。

`out` — 對準讀數的輸出位移
────────────────────────────────────────────────────────────────────
直接最大化 `LPIPS(null_edit(x), null_edit(x_def))`：同一條取樣鏈、同一組噪聲、
**不含任何文字**，量的就是「模型在沒有指令時畫出來的東西移動了多少」，與
`scripts/edit_distance_panel.py` 的主讀數同一個度量。

參考側 `null_edit(x)` 必須跟著當前的抽樣重算，否則重抽時兩側的噪聲不一致，
差值裡混進的是噪聲差而不是影像差。代價是每一次求值多跑一條無梯度的鏈。

`tone` — 單調曲線唯一的不可逆武器
────────────────────────────────────────────────────────────────────
全域單調分段線性曲線做不出空間結構，它唯一能做的**不可逆**破壞是把某一段色調
的斜率壓到近乎零：落在那一段的像素被壓成同一個值，任何淨化都救不回來。
`enc`／`cond`／`id` 沒有一項獎勵這件事。

這一項量的是**主體像素所在色調上的有效斜率**：

    tone = Σ_c Σ_k w_{c,k} · θ_{c,k} · K / Σ_k θ_{c,k}

`w_{c,k}` 是主體框內、通道 c 的像素落在第 k 段的比例，`θ` 與正規化沿用
`ColorCurveParam._curve` 的同一套。**越小代表主體色調被壓得越平**，所以它是
要被最小化的項。直方圖在原圖上算一次就凍結，不隨最佳化移動——否則「把像素推到
斜率大的段」與「把段壓平」會互相抵銷。
"""
from __future__ import annotations

from typing import Optional

import torch


class OutputDisplacement:
    """`LPIPS(null_edit(x), null_edit(x_def))`，越大代表推得越開。

    `metric` 用呼叫端傳進來的可微 LPIPS 本體（`MetricSuite.lpips_module`），
    量測與最佳化因此是同一份權重。
    """

    name = 'out'

    def __init__(self, objective, x01: torch.Tensor, metric):
        self.objective = objective
        self.x01 = x01
        self.metric = metric
        self._cache_key = None
        self._reference = None

    def reference(self) -> torch.Tensor:
        """當前抽樣下的 `null_edit(x)`。抽樣沒動就重用上一次的結果。"""
        key = (int(self.objective.draws), int(self.objective.seed))
        if key != self._cache_key:
            with torch.no_grad():
                self._reference = self.objective.null_edit(self.x01).float()
            self._cache_key = key
        return self._reference

    def __call__(self, x_def: torch.Tensor) -> torch.Tensor:
        a = self.reference()
        b = self.objective.null_edit(x_def).float()
        return self.metric(b.clamp(0, 1), a.clamp(0, 1))


class TargetedOutputDisplacement:
    """`LPIPS(null_edit(x_def), null_edit(target))`，**越小代表越靠近目標**。

    為什麼要 targeted 版
    ────────────────────────────────────────────────────────────────
    `OutputDisplacement` 是**無目標**的：只要求「離原來畫出來的東西遠」，梯度
    在任何方向上都拿得到分，會隨著離開原點而轉向。targeted 版整條路徑指向同一
    個點，同樣的預算走得比較遠。本專案在 encoder 那一側已經量到同一個形狀
    （`enc_target` 的 PhotoGuard 臂贏過無目標的 `enc`）；這一項把那個形狀搬到
    現在最好的那個讀數對齊項上。

    目標側也要跟著抽樣重算
    ────────────────────────────────────────────────────────────────
    `null_edit` 的取樣噪聲由 `objective.seed + objective.draws` 決定，而
    `FreeObjective` 的 `resample` 每一次求值都會把 `draws` 往前推。目標側只算
    一次就永久快取的話，兩側落在**不同的取樣噪聲**上，差值裡混進來的是噪聲差
    而不是影像差。快取要跟著 `draws` 這個抽樣版本走，與參考側 `OutputDisplacement`
    同一套。

    目標用 `mainstream_terms.texture_target`：程序化、可重現、不帶身分。
    """

    name = 'out_target'

    def __init__(self, objective, target01: torch.Tensor, metric):
        self.objective = objective
        self.target01 = target01
        self.metric = metric
        self._cache_key = None
        self._reference = None

    def reference(self) -> torch.Tensor:
        """當前抽樣下的 `null_edit(target)`。抽樣沒動就重用上一次的結果。"""
        key = (int(self.objective.draws), int(self.objective.seed))
        if key != self._cache_key:
            with torch.no_grad():
                self._reference = self.objective.null_edit(
                    self.target01).float()
            self._cache_key = key
        return self._reference

    def __call__(self, x_def: torch.Tensor) -> torch.Tensor:
        a = self.reference()
        b = self.objective.null_edit(x_def).float()
        return self.metric(b.clamp(0, 1), a.clamp(0, 1))


class ToneFlatness:
    """主體色調上的有效斜率，**越小代表壓得越平**。

    `box` 是原圖主體框；`pieces` 要與載體的 `ColorCurveParam.pieces` 相同，
    不同就是在對一條不存在的曲線加權。
    """

    name = 'tone'

    def __init__(self, x01: torch.Tensor, *, box=None, pieces: int = 64,
                 support: Optional[torch.Tensor] = None):
        self.pieces = int(pieces)
        with torch.no_grad():
            self.weights = self._histogram(x01, box, support)

    def _histogram(self, x01, box, support) -> torch.Tensor:
        k = self.pieces
        if support is not None:
            m = support.to(x01.device)
        else:
            m = torch.zeros_like(x01[:, :1])
            if box is None:
                m[:] = 1.0
            else:
                x0, y0, x1, y1 = (int(round(float(v))) for v in box)
                h, w = x01.shape[-2:]
                x0, y0 = max(0, x0), max(0, y0)
                x1, y1 = min(w, max(x0 + 1, x1)), min(h, max(y0 + 1, y1))
                m[:, :, y0:y1, x0:x1] = 1.0
        idx = (x01 * k).floor().clamp_(0, k - 1).long()
        out = torch.zeros((3, k), device=x01.device, dtype=torch.float32)
        flat_m = m[0, 0].reshape(-1).float()
        for c in range(3):
            out[c].scatter_add_(0, idx[0, c].reshape(-1), flat_m)
        total = out.sum(dim=1, keepdim=True).clamp_min(1e-6)
        return out / total

    def __call__(self, carrier) -> torch.Tensor:
        th = carrier.theta[0].float()
        k = float(self.pieces)
        eff = (self.weights * th).sum(dim=1) * k / th.sum(dim=1).clamp_min(1e-9)
        return eff.mean()


class CompositeObjective:
    """`FreeObjective` 外掛上依賴載體的項。介面與 `FreeObjective` 相同。

    `optimise_carrier` 只會呼叫 `score`／`terms`／`eval_score`／`eval_terms`，
    所以這一層只要把那四個補齊，其餘欄位轉發。
    """

    def __init__(self, base, *, carrier=None, out_term=None, tone_term=None,
                 mainstream=None, out_target_term=None, sds_term=None,
                 weights=None):
        self.base = base
        self.carrier = carrier
        self.out_term = out_term
        self.tone_term = tone_term
        self.mainstream = mainstream
        self.out_target_term = out_target_term
        self.sds_term = sds_term
        self.weights = dict(base.weights, **(weights or {}))
        self.ip2p = base.ip2p
        self.timesteps = base.timesteps

    def _w(self, name):
        return float(self.weights.get(name, 0.0) or 0.0)

    def _extra(self, x_def, terms):
        """外掛項。權重為零的項不計算——每一項都要跑 UNet，關掉的不該付成本。"""
        if self.out_term is not None:
            terms['out'] = self.out_term(x_def)
        if self.out_target_term is not None:
            terms['out_target'] = self.out_target_term(x_def)
        if self.sds_term is not None:
            terms['sds'] = self.sds_term(x_def)
        if self.tone_term is not None:
            terms['tone'] = self.tone_term(self.carrier)
        if self.mainstream is not None:
            if self._w('enc_target') > 0:
                terms['enc_target'] = self.mainstream.enc_target(x_def)
            if self._w('diffusion') > 0:
                terms['diffusion'] = self.mainstream.diffusion(x_def)
            if self._w('diffusion_target') > 0:
                terms['diffusion_target'] = self.mainstream.diffusion_target(
                    x_def)
        return terms

    def _compose(self, terms):
        """符號慣例：分數一律是要最小化的那個方向。

        `enc_target` 越小越好（要靠近目標），所以以 `+w·tanh` 進分數；
        `diffusion` 越大越好（要模型還原不了），以 `−w·tanh` 進；
        `out` 越大越好，`tone` 越小越好，`out_target` 越小越好。
        `sds` 是 `diffusion` 的同一個損失換一種梯度，所以符號與縮放與
        `diffusion` 逐字相同——兩者只差在梯度，比較才成立。
        """
        w = self.weights
        s = (-w.get('enc', 0.0) * torch.tanh(terms['enc'])
             - w.get('cond', 0.0) * torch.tanh(terms['cond']))
        if 'id' in terms:
            s = s + w.get('id', 0.0) * terms['id']
        if 'out' in terms:
            s = s - w.get('out', 0.0) * terms['out']
        if 'out_target' in terms:
            s = s + self._w('out_target') * terms['out_target']
        if 'sds' in terms:
            s = s - self._w('sds') * torch.tanh(terms['sds'])
        if 'tone' in terms:
            s = s + w.get('tone', 0.0) * terms['tone']
        if 'enc_target' in terms:
            s = s + self._w('enc_target') * torch.tanh(terms['enc_target'])
        if 'diffusion' in terms:
            s = s - self._w('diffusion') * torch.tanh(terms['diffusion'])
        if 'diffusion_target' in terms:
            s = s + self._w('diffusion_target') * torch.tanh(
                terms['diffusion_target'])
        return s

    def terms(self, x_def):
        return self._extra(x_def, dict(self.base.terms(x_def)))

    def score(self, x_def):
        return self._compose(self.terms(x_def))

    def eval_terms(self, x_def):
        """固定驗證抽樣上的逐項值。**外掛項也要在那個狀態裡算。**

        `FreeObjective.eval_terms` 在返回前就把訓練用的 `draws` 與 `resample`
        還原了，而 `OutputDisplacement` 是用 `objective.seed + objective.draws`
        產生取樣鏈的噪聲。先呼叫 `base.eval_terms` 再算外掛項，`out` 會落在
        **訓練抽樣**上而不是固定驗證抽樣上——選 checkpoint 的分數因此帶著
        訓練端的隨機性，等於用抖動的數挑點。這裡自己進出驗證狀態，
        三個項在同一個抽樣上算完才離開。
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

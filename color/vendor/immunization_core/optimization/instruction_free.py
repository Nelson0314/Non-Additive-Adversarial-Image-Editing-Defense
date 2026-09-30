"""不含編輯指令的防禦目標；文字條件一律為空字串。

三個項皆與攻擊指令無關：

`enc`   ‖E(x_def) − E(x)‖ ÷ ‖E(x)‖（VAE 編碼器位移）。
`cond`  mean_t ‖ε(ẑ_t, t, c_img(x_def), ∅) − ε(ẑ_t, t, c_img(x), ∅)‖ ÷ ‖ε_orig‖；
        兩側共用噪聲 ẑ_t 與時刻 t。
`id`    空指令短取樣鏈的輸出，在原圖主體框的固定座標上取人臉嵌入，與原圖嵌入
        的餘弦；只對鏈末端 `grad_steps` 次 UNet 呼叫反傳。

    score = w_id · id − w_enc · tanh(enc) − w_cond · tanh(cond)      （最小化）

`tanh` 使無上界的兩個位移項飽和。此分數為最佳化代理；編輯結果的讀數在回報時
以真實指令另行計算。
"""
from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.utils.checkpoint as ckpt

DEFAULT_WEIGHTS = {'id': 1.0, 'enc': 0.5, 'cond': 1.0}


def null_text_embedding(ip2p) -> torch.Tensor:
    """空字串的文字嵌入。這是整個模組唯一碰到 text encoder 的地方。"""
    with torch.no_grad():
        return ip2p.pipe._encode_prompt('', ip2p.device, 1, False)


def pick_timesteps(scheduler, k: int, steps: int) -> List[int]:
    """在完整時間表上取 k 個位置，最後一個一定是最低的那一格。

    涵蓋高噪聲到低噪聲：高時刻決定構圖、低時刻決定細節，只取一端會偏。
    """
    if k < 1:
        raise ValueError('k 至少為 1')
    scheduler.set_timesteps(steps)
    ts = list(scheduler.timesteps)
    if k > len(ts):
        raise ValueError(f'k={k} 超過時間表長度 {len(ts)}')
    idx = [round(i * (len(ts) - 1) / max(1, k - 1)) for i in range(k)]
    return [ts[i] for i in dict.fromkeys(idx)]


def _sigma_at(scheduler, t) -> torch.Tensor:
    if not (hasattr(scheduler, 'sigmas') and scheduler.sigmas is not None):
        raise ValueError('這個排程器沒有 sigmas，無法把噪聲縮到該時刻的強度')
    i = (scheduler.timesteps == t).nonzero()[0, 0]
    return scheduler.sigmas[i]


def _scaled_sample(scheduler, noise, t) -> torch.Tensor:
    """時刻 t 上、已經縮成 UNet 輸入的 latent：`noise · σ_t / √(σ_t²+1)`。

    不呼叫 `scheduler.scale_model_input`：diffusers 的
    `EulerAncestralDiscreteScheduler` 只在第一次呼叫時依 timestep 初始化
    `step_index`，逐時刻的獨立探針因此會全部套用第一個時刻的 σ。本函式以該時刻
    的 σ 做無狀態縮放。
    """
    s = _sigma_at(scheduler, t).to(noise)
    return noise * s / (s * s + 1).sqrt()


class FreeObjective:
    """一張影像上的不含指令目標。原圖那一側的量全部先算好並凍結。

    `box` 是原圖主體框，由呼叫端用偵測器取一次；最佳化過程中不再偵測，
    所以「把臉推到偵測器失手」不是一個可以被最佳化的狀態。
    """

    def __init__(self, ip2p, x01, *, box=None, k: int = 4, steps: int = 50,
                 seed: int = 0, weights: Optional[Dict[str, float]] = None,
                 chain_steps: int = 6, grad_steps: int = 1, s_i: float = 1.5,
                 use_ckpt: bool = True, device=None, resample: bool = False,
                 face_weight: float = 0.0):
        from copy import deepcopy

        self.ip2p = ip2p
        self.device = device or ip2p.device
        self.box = box
        self.weights = dict(DEFAULT_WEIGHTS, **(weights or {}))
        self.use_ckpt = use_ckpt
        self.chain_steps = int(chain_steps)
        self.grad_steps = int(grad_steps)
        self.s_i = float(s_i)
        self.seed = int(seed)
        self.k = int(k)
        self.steps = int(steps)
        self.resample = bool(resample)
        self.face_weight = float(face_weight)
        self.text = null_text_embedding(ip2p)
        self.scheduler = deepcopy(ip2p.pipe.scheduler)
        self.timesteps = pick_timesteps(self.scheduler, k, steps)
        self.scheduler.set_timesteps(self.steps)
        self.full_timesteps = list(self.scheduler.timesteps)
        self.draws = 0
        self.gen = torch.Generator(device=self.device).manual_seed(int(seed))

        with torch.no_grad():
            self.z0 = ip2p.encode_image(x01).float()
            self.cond0 = ip2p.image_latents(x01).to(self.text.dtype)
            self.weight_map = self._face_weight_map(x01)
            self.z0_norm = self._wnorm(self.z0).clamp_min(1e-6)
            gen = torch.Generator(device=self.device).manual_seed(int(seed))
            self.noise = torch.randn(self.cond0.shape, device=self.device,
                                     dtype=self.text.dtype, generator=gen)
            self._freeze_original()
            self.id0 = None
            if box is not None:
                from immunization_core.metrics.identity import embed_box
                self.id0 = embed_box(x01, box, self.device).float()
            self.val = self._build_validation()

    def _build_validation(self):
        """一組與訓練抽樣不重疊的固定 (噪聲, 時刻)，只用來判收斂與選 checkpoint。

        重抽把訓練目標變成隨機量，逐步分數因此本來就會抖；拿抖動的分數挑
        checkpoint 等於挑到運氣好的那一個抽樣。收斂與選點一律走這一組固定的
        抽樣，它在整個最佳化過程中不變，也不參與梯度。
        """
        if not self.resample:
            return None
        gen = torch.Generator(device=self.device).manual_seed(self.seed + 90001)
        idx = torch.randperm(len(self.full_timesteps), generator=gen,
                             device=self.device)[:self.k]
        ts = [self.full_timesteps[int(i)] for i in idx.tolist()]
        noise = torch.randn(self.cond0.shape, device=self.device,
                            dtype=self.text.dtype, generator=gen)
        keep = (self.noise, self.timesteps, self.eps0, self.eps0_norm)
        self.noise, self.timesteps = noise, ts
        self._freeze_original()
        out = (noise, ts, self.eps0, self.eps0_norm, self.seed + 90001)
        self.noise, self.timesteps, self.eps0, self.eps0_norm = keep
        return out

    def eval_score(self, x_def: torch.Tensor) -> torch.Tensor:
        """固定抽樣上的分數。`resample` 關閉時與 `score` 完全相同。"""
        if self.val is None:
            return self.score(x_def)
        keep = (self.noise, self.timesteps, self.eps0, self.eps0_norm,
                self.draws, self.resample)
        (self.noise, self.timesteps, self.eps0, self.eps0_norm,
         self.draws) = self.val
        self.resample = False
        try:
            return self.score(x_def)
        finally:
            (self.noise, self.timesteps, self.eps0, self.eps0_norm,
             self.draws, self.resample) = keep

    def eval_terms(self, x_def: torch.Tensor):
        """`eval_score` 的逐項版本，寫進 CSV 用。"""
        if self.val is None:
            return self.terms(x_def)
        keep = (self.noise, self.timesteps, self.eps0, self.eps0_norm,
                self.draws, self.resample)
        (self.noise, self.timesteps, self.eps0, self.eps0_norm,
         self.draws) = self.val
        self.resample = False
        try:
            return self.terms(x_def)
        finally:
            (self.noise, self.timesteps, self.eps0, self.eps0_norm,
             self.draws, self.resample) = keep

    def _face_weight_map(self, x01):
        """(1,1,h,w) 的潛空間權重圖。`face_weight = 0` 時回傳 None，即均勻。

        `enc` 與 `cond` 原本在整張潛圖上取範數，但連言裡唯一會垮的那一項住在
        臉上。全域 tone curve 沒有空間自由度，能調的只有「斜率的預算放在哪一段
        色調」——把範數往臉的座標加權，等於要求最佳化只認臉上出現的那些色調。
        權重在框內是 `1 + face_weight`、框外是 `1`，背景仍然計入，只是不再等重。
        """
        if self.face_weight <= 0 or self.box is None:
            return None
        h, w = self.cond0.shape[-2:]
        sy = h / x01.shape[-2]
        sx = w / x01.shape[-1]
        x0, y0, x1, y1 = (float(v) for v in self.box)
        c0 = max(0, min(w - 1, int(x0 * sx)))
        r0 = max(0, min(h - 1, int(y0 * sy)))
        c1 = max(c0 + 1, min(w, int(round(x1 * sx))))
        r1 = max(r0 + 1, min(h, int(round(y1 * sy))))
        m = torch.ones((1, 1, h, w), device=self.device, dtype=torch.float32)
        m[:, :, r0:r1, c0:c1] = 1.0 + self.face_weight
        return m

    def _wnorm(self, d: torch.Tensor) -> torch.Tensor:
        """加權的 Frobenius 範數。`weight_map` 為 None 時與 `.norm()` 相同。

        平方和在進 `sqrt` 之前夾到一個正的下界。起點是恆等曲線時 `z − z0` 逐
        位元為零，`sqrt(0)` 的導數是無限大，第一步的梯度就不是有限值；
        `torch.norm` 自己在零點回傳零次梯度，這裡要跟它一致。
        """
        d = d.float()
        if self.weight_map is None:
            return d.flatten().norm()
        return (self.weight_map * d.pow(2)).sum().clamp_min(1e-24).sqrt()

    def _freeze_original(self) -> None:
        """在目前的 `noise` 與 `timesteps` 上重算原圖那一側的 ε。"""
        self.eps0, self.eps0_norm = {}, {}
        for t in self.timesteps:
            e = self._eps(self.cond0, t, grad=False)
            self.eps0[int(t)] = e
            self.eps0_norm[int(t)] = self._wnorm(e).clamp_min(1e-6)

    def _redraw(self) -> None:
        """抽一組新的噪聲與時刻，並把原圖那一側搬到新的抽樣上。

        凍結單一抽樣等於用一個樣本估計 `E_{t,ε}`。載體只有 3K 個參數，解會貼著
        那一個抽樣走，而評估抽的是別的種子。每一次求值重抽把目標換回期望值，
        代價是每一步多 k 次無梯度的 UNet 呼叫。
        """
        self.draws += 1
        idx = torch.randperm(len(self.full_timesteps), generator=self.gen,
                             device=self.device)[:self.k]
        self.timesteps = [self.full_timesteps[int(i)] for i in idx.tolist()]
        with torch.no_grad():
            self.noise = torch.randn(self.cond0.shape, device=self.device,
                                     dtype=self.text.dtype, generator=self.gen)
            self._freeze_original()

    def _run_unet(self, model_input, t, text):
        return self.ip2p.unet(model_input, t, encoder_hidden_states=text,
                              return_dict=False)[0]

    def _eps(self, cond, t, grad: bool = True) -> torch.Tensor:
        z = _scaled_sample(self.scheduler, self.noise, t)
        model_input = torch.cat([z, cond.to(z.dtype)], dim=1)
        if grad and self.use_ckpt:
            return ckpt.checkpoint(self._run_unet, model_input, t, self.text,
                                   use_reentrant=False)
        return self._run_unet(model_input, t, self.text)

    def terms(self, x_def: torch.Tensor) -> Dict[str, torch.Tensor]:
        """三個項，全部是張量，梯度可以直接對 `x_def` 反傳。"""
        if self.resample:
            self._redraw()
        z = self.ip2p.encode_image(x_def).float()
        out = {'enc': self._wnorm(z - self.z0) / self.z0_norm}

        cond = self.ip2p.image_latents(x_def).to(self.text.dtype)
        gaps = []
        for t in self.timesteps:
            e = self._eps(cond, t)
            gaps.append(self._wnorm(e.float() - self.eps0[int(t)].float())
                        / self.eps0_norm[int(t)])
        out['cond'] = torch.stack(gaps).mean()

        if self.id0 is not None:
            from immunization_core.metrics.identity import embed_box_differentiable
            img = self.null_edit(x_def).float()
            emb = embed_box_differentiable(img, self.box, self.device).float()
            out['id'] = torch.nn.functional.cosine_similarity(
                emb[None], self.id0[None]).squeeze()
        return out

    def null_edit(self, x_def: torch.Tensor) -> torch.Tensor:
        """空指令下的取樣輸出：模型在沒有任何文字時會把這張圖畫成什麼。

        文字為空時 IP2P 的三分支退化——`text` 分支與 `image` 分支的條件相同，
        `s_t·(ε_text − ε_image)` 消失，只剩 `ε̃ = ε_uncond + s_i·(ε_image − ε_uncond)`，
        所以這裡只跑兩個分支。鏈長 `chain_steps`，只對末端 `grad_steps` 次呼叫
        反傳，與 `IP2PWrapper.edit_differentiable` 同一套截斷。
        """
        from copy import deepcopy

        sch = deepcopy(self.scheduler)
        sch.set_timesteps(self.chain_steps, device=self.device)
        cond = self.ip2p.image_latents(x_def).to(self.text.dtype)
        gen = torch.Generator(device=self.device).manual_seed(
            self.seed + self.draws)
        z = torch.randn(cond.shape, device=self.device, dtype=self.text.dtype,
                        generator=gen) * sch.init_noise_sigma
        text = torch.cat([self.text, self.text])
        cut = len(sch.timesteps) - self.grad_steps
        outer = torch.is_grad_enabled()
        for i, t in enumerate(sch.timesteps):
            active = outer and i >= cut
            with torch.set_grad_enabled(active):
                c = cond if active else cond.detach()
                image_batch = torch.cat([c, torch.zeros_like(c)])
                noise_batch = sch.scale_model_input(torch.cat([z, z]), t)
                model_input = torch.cat([noise_batch, image_batch], dim=1)
                eps = (ckpt.checkpoint(self._run_unet, model_input, t, text,
                                       use_reentrant=False)
                       if active and self.use_ckpt
                       else self._run_unet(model_input, t, text))
                e_img, e_un = eps.chunk(2)
                guided = e_un + self.s_i * (e_img - e_un)
                z = sch.step(guided, t, z, generator=gen,
                             return_dict=False)[0]
        return self.ip2p.decode_latent(z.to(self.ip2p.vae.dtype),
                                       use_ckpt=self.use_ckpt)

    def score(self, x_def: torch.Tensor) -> torch.Tensor:
        t = self.terms(x_def)
        w = self.weights
        s = -w['enc'] * torch.tanh(t['enc']) - w['cond'] * torch.tanh(t['cond'])
        if 'id' in t:
            s = s + w['id'] * t['id']
        return s

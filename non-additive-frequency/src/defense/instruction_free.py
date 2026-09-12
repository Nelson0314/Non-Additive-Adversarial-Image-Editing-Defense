"""不含指令的目標函數：整條最佳化路徑上沒有任何編輯指令參與。

為什麼要這樣改
────────────────────────────────────────────────────────────────────
先前的 `scripts/carrier_search.py` 每評估一個候選點就用**那一格的攻擊指令**
跑一次真編輯，再用編輯輸出算判準。指令因此是最佳化的輸入之一，防禦是對
「這句話」調出來的。真實的攻擊者會打一句沒看過的話，所以那個設定高估了防禦。

這個模組把指令整個拿掉。可用的訊號只剩下**與文字無關的那一側**：IP2P 的
三分支 CFG 裡，`image` 分支是 `ε(ẑ, t, c_img, ∅_text)`——影像條件在、文字是空
字串。編輯要成立，模型必須先從影像條件裡讀出這張臉；破壞那一步不需要知道
攻擊者想做什麼。

三個項，全部不含文字
────────────────────────────────────────────────────────────────────
`enc`   ‖E(x_def) − E(x)‖ ÷ ‖E(x)‖。PhotoGuard 的 encoder attack，最便宜，
        只碰 VAE。

`cond`  mean_t ‖ε(ẑ_t, t, c_img(x_def), ∅) − ε(ẑ_t, t, c_img(x), ∅)‖ ÷ ‖ε_orig‖。
        兩側共用同一組噪聲 ẑ_t 與同一個 t，唯一的差別是影像條件，量的就是
        「換了這張圖之後，模型在無文字條件下的預測偏掉多少」。

`id`    用**空指令**跑一條短的取樣鏈（IP2P 的三分支在文字為空時退化成
        `ε̃ = ε_uncond + s_i·(ε_image − ε_uncond)`，只剩影像導引），把輸出解碼成
        影像，在**原圖主體框的固定座標**上裁下來取人臉嵌入，與原圖的嵌入取餘弦。
        那是模型在沒有任何指令時會畫出來的東西，所以這一項問的是「模型還認不認得
        這是誰」。只對鏈末端的 `grad_steps` 次呼叫反傳，與 `edit_differentiable`
        同一套截斷。

合成
────────────────────────────────────────────────────────────────────
    score = w_id · id − w_enc · tanh(enc) − w_cond · tanh(cond)      要最小化

位移項套 `tanh` 是為了飽和：`enc` 與 `cond` 沒有上界，不飽和的話最佳化會把
全部的預算押在其中一項上，另外兩項的梯度被壓掉。`id` 本來就落在 [−1,1]。

**這是代理，不是判準。** 判準仍然是「攻擊者拿不拿得到可用、認得出、指令完成
的照片」，只能用真編輯量，而真編輯需要指令——所以判準只在**回報**時計算，
指令是純粹的測試資料。
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

    **不呼叫 `scheduler.scale_model_input`。** diffusers 的
    `EulerAncestralDiscreteScheduler` 是有狀態的：`step_index` 只在第一次呼叫時
    依 timestep 初始化，之後要靠 `step()` 前進。這裡是逐時刻的獨立探針、不走
    取樣鏈，所以連續呼叫會全部套用**第一個**時刻的 σ。實測 50 步時間表上
    四個時刻回傳的縮放都是 0.06826（σ_max = 14.6146），最低時刻因此被多縮了
    約 14.6 倍。改成自己用該時刻的 σ 做無狀態縮放。
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
                 use_ckpt: bool = True, device=None):
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
        self.text = null_text_embedding(ip2p)
        self.scheduler = deepcopy(ip2p.pipe.scheduler)
        self.timesteps = pick_timesteps(self.scheduler, k, steps)

        with torch.no_grad():
            self.z0 = ip2p.encode_image(x01).float()
            self.z0_norm = self.z0.flatten().norm().clamp_min(1e-6)
            cond0 = ip2p.image_latents(x01).to(self.text.dtype)
            gen = torch.Generator(device=self.device).manual_seed(int(seed))
            self.noise = torch.randn(cond0.shape, device=self.device,
                                     dtype=self.text.dtype, generator=gen)
            self.eps0, self.eps0_norm = {}, {}
            for t in self.timesteps:
                e = self._eps(cond0, t, grad=False)
                self.eps0[int(t)] = e
                self.eps0_norm[int(t)] = e.flatten().norm().clamp_min(1e-6)
            self.id0 = None
            if box is not None:
                from src.metrics.identity import embed_box
                self.id0 = embed_box(x01, box, self.device).float()

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
        z = self.ip2p.encode_image(x_def).float()
        out = {'enc': (z - self.z0).flatten().norm() / self.z0_norm}

        cond = self.ip2p.image_latents(x_def).to(self.text.dtype)
        gaps = []
        for t in self.timesteps:
            e = self._eps(cond, t)
            gaps.append((e.float() - self.eps0[int(t)].float()).flatten().norm()
                        / self.eps0_norm[int(t)])
        out['cond'] = torch.stack(gaps).mean()

        if self.id0 is not None:
            from src.metrics.identity import embed_box_differentiable
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
        gen = torch.Generator(device=self.device).manual_seed(self.seed)
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

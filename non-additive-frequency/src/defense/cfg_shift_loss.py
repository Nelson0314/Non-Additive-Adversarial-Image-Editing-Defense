"""把 IP2P **完整三分支引導預測**對影像條件的依賴壓掉。

與 `image_guidance_loss` 的關係：後者只壓了其中一項
────────────────────────────────────────────────────────────────────
IP2P 的取樣式（`image_guidance_loss` 已逐行核對管線 444-447 行）是

    g(c, p) = u + s_T·[ eps(z, c, p) − eps(z, c, ∅) ]
                + s_I·[ eps(z, c, ∅) − u ],        u = eps(z, 0, ∅)

`image_guidance` 最小化的是 `‖eps(z, c, ∅) − u‖²`，也就是**只有 s_I 那一項**。
但 `eps(z, c, p) − eps(z, c, ∅)` 這個文字項**也帶著影像條件 c**：UNet 的
cross-attention 會因為影像通道的內容而改變它對文字的反應。把 s_I 那一項壓到
零，文字項仍可以獨力把指令畫出來。

    L_cfg(x') = E_{t, ε, p} ‖ g(c(x'), p) − g(0, p) ‖²

展開之後（`u` 與 `g(0,p)` 的第一項都不依賴 c，相消）：

    Δ = s_T·[ eps(z,c,p) − eps(z,c,∅) ]
      − s_T·[ eps(z,0,p) − eps(z,0,∅) ]
      + s_I·[ eps(z,c,∅) − eps(z,0,∅) ]

前兩項的差就是「影像條件改變了文字項多少」，第三項就是 `image_guidance`。

一個精確的退化關係，可用來釘住實作
────────────────────────────────────────────────────────────────────
`p = ∅` 時 `eps(z,c,p) = eps(z,c,∅)` 且 `eps(z,0,p) = eps(z,0,∅)`，於是

    Δ = s_I·[ eps(z,c,∅) − eps(z,0,∅) ]      即  L_cfg = s_I² · L_ig

`tests/test_cfg_shift_loss.py` 用這個等式驗接線；兩者差一個常數倍是構造使然，
不是巧合。

`text_embeds` 是必填的
────────────────────────────────────────────────────────────────────
上面那個等式同時說明：只餵空字串時本項退化成 `image_guidance`，多花的兩次
UNet 呼叫什麼也沒買到。本項存在的理由就是文字項，所以文字條件必填。

防禦方**不知道**攻擊者會下哪一句，故餵的是一疊可能的指令、每一步抽一個，
對指令的分布取期望（與 `image_guidance` 的 `text_embeds` 同一個機制與同一個
生成器慣例）。這不假設知道特定那一句，威脅模型未變。

成本
────────────────────────────────────────────────────────────────────
每個樣本四次 UNet 呼叫：兩次帶梯度（`eps(z,c,p)`、`eps(z,c,∅)`），兩次是
不依賴 `x'` 的常數（`no_grad` 並 detach）。`image_guidance` 是一次帶梯度加
一次常數，故約兩倍；相對於走完整條編輯軌跡的任務損失仍便宜一個數量級。
`make_fixed` 的固定評估會把兩個常數分支快取起來，因為那裡的 `(t, ε, p)` 不變。
"""

from __future__ import annotations

from typing import Callable, Optional

import torch

from src.defense.fixedpoint_loss import _null_embedding, _scheduler_of
# **共用同一個空間加權實作**，不另寫一份：兩份各自維護時，其中一份改了而
# 另一份沒改，被最佳化的量與收斂監看的量會悄悄變成兩個東西。
from src.defense.image_guidance_loss import ZT_MODES, _INHERIT, _weighted_mean


def make_cfg_shift_loss(
    ip2p,
    *,
    zt_mode: str,
    text_embeds: torch.Tensor,
    s_t: float,
    s_i: float,
    x_clean: Optional[torch.Tensor] = None,
    t_min: int = 1,
    t_max: int = 1000,
    samples: int = 1,
    seed: int = 0,
    weight: Optional[torch.Tensor] = None,
    normalise: bool = False,
    norm_eps: float = 1e-3,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """回傳 `loss(x_def01) -> 純量`，**要最小化**。

    `zt_mode`、`x_clean`、`t_min`／`t_max`、`samples`、`weight`、`normalise`
    的語意與 `make_image_guidance_loss` 逐項相同。
    """
    if zt_mode not in ZT_MODES:
        raise ValueError(f"未知的 zt_mode：{zt_mode!r}，必須是 {ZT_MODES}")
    if not 1 <= t_min <= t_max:
        raise ValueError(f"需要 1 <= t_min <= t_max，收到 {t_min}／{t_max}")
    if samples < 1:
        raise ValueError(f"samples 必須為正整數，收到 {samples}")
    if zt_mode == "diffuse_src" and x_clean is None:
        raise ValueError(
            "zt_mode='diffuse_src' 需要 x_clean（**原圖**）。"
            "用防禦圖當錨會讓取樣軌跡隨最佳化漂移，而且不會有症狀。")
    if norm_eps <= 0:
        raise ValueError("norm_eps 必須為正，它是分母的下界守門")
    if not (s_t > 1 and s_i >= 1):
        raise ValueError(
            f"三分支 CFG 需要 s_t > 1 且 s_i >= 1，收到 {s_t}／{s_i}；"
            "本項壓的是實際取樣式，強度必須是攻擊者真的會用的那一組")

    unet = ip2p.unet
    device = ip2p.device
    sched = _scheduler_of(ip2p)
    abar = sched.alphas_cumprod.to(device=device, dtype=torch.float32)
    if t_max > len(abar):
        raise ValueError(f"t_max={t_max} 超出排程長度 {len(abar)}")
    null_emb = _null_embedding(ip2p)

    if text_embeds is None:
        raise ValueError(
            "text_embeds 必填：只餵空字串時本項退化成 image_guidance 乘上 "
            "s_i²（見模組 docstring 的等式），多出來的兩次 UNet 呼叫什麼也"
            "沒買到。")
    if text_embeds.dim() != 3 or text_embeds.shape[0] < 1:
        raise ValueError(
            f"text_embeds 必須是 (K,L,D) 且 K>=1，收到 {tuple(text_embeds.shape)}")
    if text_embeds.shape[1:] != null_emb.shape[1:]:
        raise ValueError(
            f"text_embeds 的形狀 {tuple(text_embeds.shape[1:])} 與空字串嵌入的 "
            f"{tuple(null_emb.shape[1:])} 不合——兩者要能互換才是同一個位置的條件。")
    text_embeds = text_embeds.to(device=device).detach()

    gen = torch.Generator(device="cpu").manual_seed(int(seed))

    def _pick_emb(g, dtype):
        k = int(torch.randint(text_embeds.shape[0], (1,), generator=g))
        return text_embeds[k:k + 1].to(dtype)

    def _scale(z_img: torch.Tensor) -> torch.Tensor:
        """`normalise` 開著時的分母：影像條件自己的能量。"""
        if not normalise:
            return z_img.new_tensor(1.0)
        return z_img.pow(2).mean() + norm_eps

    z_src = None
    if zt_mode == "diffuse_src":
        with torch.no_grad():
            # 走 encode_image（**有**乘 scaling_factor）：這一份是噪聲 latent
            # 那一側，與拼進去的影像條件不同側。
            z_src = ip2p.encode_image(x_clean).detach()

    def _zt_from(step: int, eps: torch.Tensor) -> torch.Tensor:
        if zt_mode == "noise":
            return eps
        a = abar[step].to(eps.dtype)
        return z_src.to(eps.dtype) * a.sqrt() + eps * (1.0 - a).sqrt()

    def _delta(z_t, tt, emb_p, z_img, zero):
        """`g(c, p) − g(0, p)`。兩個不依賴 `x'` 的分支在 `no_grad` 下算。"""
        emb_0 = null_emb.to(z_t.dtype)
        with torch.no_grad():
            e_0p = unet(torch.cat([z_t, zero], dim=1), tt,
                        encoder_hidden_states=emb_p).sample.detach()
            e_00 = unet(torch.cat([z_t, zero], dim=1), tt,
                        encoder_hidden_states=emb_0).sample.detach()
        e_cp = unet(torch.cat([z_t, z_img], dim=1), tt,
                    encoder_hidden_states=emb_p).sample
        e_c0 = unet(torch.cat([z_t, z_img], dim=1), tt,
                    encoder_hidden_states=emb_0).sample
        return s_t * (e_cp - e_c0) - s_t * (e_0p - e_00) + s_i * (e_c0 - e_00)

    def make_fixed(n_draws: int, eval_seed: int):
        """**決定性**的評估函數，語意與 `make_image_guidance_loss.make_fixed` 相同。

        `(t, ε, p)` 抽一次就固定，於是 `g(0, p)` 的兩個分支是常數，快取起來
        每次評估省兩次 UNet 呼叫。快取的鍵就是抽樣索引本身——固定評估的
        `z_t` 不隨 `x'` 改變，這一點由 `_zt_from` 只吃 `(step, eps)` 保證。
        """
        g2 = torch.Generator(device="cpu").manual_seed(int(eval_seed))
        steps_fixed = [int(torch.randint(t_min - 1, t_max, (1,), generator=g2))
                       for _ in range(n_draws)]
        embeds_fixed = [_pick_emb(g2, text_embeds.dtype) for _ in range(n_draws)]
        eps_fixed = [None] * n_draws
        const_cache = [None] * n_draws

        def fixed(x_def01: torch.Tensor, eval_weight=_INHERIT) -> torch.Tensor:
            w = weight if eval_weight is _INHERIT else eval_weight
            z_img = ip2p.image_latents(x_def01)
            zero = torch.zeros_like(z_img)
            emb_0 = null_emb.to(z_img.dtype)
            total = None
            for k, step in enumerate(steps_fixed):
                if eps_fixed[k] is None:
                    eps_fixed[k] = torch.randn(
                        z_img.shape, generator=g2, dtype=torch.float32
                    ).to(device=z_img.device, dtype=z_img.dtype)
                z_t = _zt_from(step, eps_fixed[k])
                tt = torch.tensor([step], device=device, dtype=torch.long)
                emb_p = embeds_fixed[k].to(z_t.dtype)
                if const_cache[k] is None:
                    with torch.no_grad():
                        e_0p = unet(torch.cat([z_t, zero], dim=1), tt,
                                    encoder_hidden_states=emb_p).sample.detach()
                        e_00 = unet(torch.cat([z_t, zero], dim=1), tt,
                                    encoder_hidden_states=emb_0).sample.detach()
                    const_cache[k] = (e_0p, e_00)
                e_0p, e_00 = const_cache[k]
                e_cp = unet(torch.cat([z_t, z_img], dim=1), tt,
                            encoder_hidden_states=emb_p).sample
                e_c0 = unet(torch.cat([z_t, z_img], dim=1), tt,
                            encoder_hidden_states=emb_0).sample
                delta = (s_t * (e_cp - e_c0) - s_t * (e_0p - e_00)
                         + s_i * (e_c0 - e_00))
                term = _weighted_mean(delta.pow(2), w) / _scale(z_img)
                total = term if total is None else total + term
            return total / len(steps_fixed)

        return fixed

    def loss(x_def01: torch.Tensor) -> torch.Tensor:
        z_img = ip2p.image_latents(x_def01)
        zero = torch.zeros_like(z_img)
        total = None
        for _ in range(samples):
            step = int(torch.randint(t_min - 1, t_max, (1,), generator=gen))
            eps = torch.randn(z_img.shape, generator=gen, dtype=torch.float32
                              ).to(device=z_img.device, dtype=z_img.dtype)
            z_t = _zt_from(step, eps)
            tt = torch.tensor([step], device=device, dtype=torch.long)
            emb_p = _pick_emb(gen, z_t.dtype)
            delta = _delta(z_t, tt, emb_p, z_img, zero)
            term = _weighted_mean(delta.pow(2), weight) / _scale(z_img)
            total = term if total is None else total + term
        return total / samples

    loss.make_fixed = make_fixed
    return loss


def instruction_embeddings(ip2p, instructions) -> torch.Tensor:
    """一疊指令 → (K, L, D) 的文字嵌入，供 `text_embeds` 使用。

    走與空字串嵌入同一條路徑（同一個 tokenizer、同樣的 padding 到
    `model_max_length`），兩者才能在同一個位置互換。

    tokenizer 的取法與 `fixedpoint_loss._null_embedding` 逐字相同：
    `IP2PWrapper` **沒有** `.tokenizer`，要走它底下的 pipeline。
    """
    if not instructions or any(not isinstance(s, str) or not s.strip()
                               for s in instructions):
        raise ValueError("instructions 必須是非空字串的非空序列")
    tok = None
    for attr in ("tokenizer", "_pipe", "pipe", "pipeline"):
        obj = getattr(ip2p, attr, None)
        if obj is None:
            continue
        tok = obj if hasattr(obj, "model_max_length") else getattr(obj, "tokenizer", None)
        if tok is not None:
            break
    if tok is None:
        raise AttributeError("找不到 tokenizer，無法把指令編成文字嵌入")
    ids = tok(list(instructions), padding="max_length",
              max_length=tok.model_max_length, truncation=True,
              return_tensors="pt").input_ids.to(ip2p.device)
    with torch.no_grad():
        return ip2p.text_encoder(ids)[0].detach()

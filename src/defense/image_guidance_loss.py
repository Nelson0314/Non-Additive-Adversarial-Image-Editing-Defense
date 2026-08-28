"""把「讓影像引導項消失」寫成損失。**探索性質。**

為什麼是這一項
────────────────────────────────────────────────────────────────────
專案量到的三個不變量說：擋下率幾乎完全由位移決定（`RESULTS.md` 一節）、
「每單位 LPIPS 換到多少位移」跨四個參數化的變異係數只有 0.108
（`distortion_axis_analysis`）、位移在 0.70 飽和（`PENDING.md` 〇節）。
合起來就是**在「損失只讀 VAE 編碼器」這個威脅面上，換參數化不是槓桿**。
還沒被動過的是損失讀哪裡——本專案至今所有損失都只經過 VAE，從未碰過 UNet。

機制：IP2P 的影像引導是一個差
────────────────────────────────────────────────────────────────────
`pipeline_stable_diffusion_instruct_pix2pix.py:444-447`（已逐行核對）：

    eps_tilde = eps(z_t, 0, null)
              + s_T * [ eps(z_t, c_I, c_T) - eps(z_t, c_I, null) ]
              + s_I * [ eps(z_t, c_I, null) - eps(z_t, 0,   null) ]

三份批次的順序是 [text, image, uncond]（同檔 :443），文字嵌入是
[prompt, negative, negative]，影像 latent 是 [img, img, zeros]（同檔 :898 的
`uncond_image_latents = torch.zeros_like(image_latents)`）。

由此確定兩件事：**影像無條件分支用的就是零影像 latent**，而且影像分支與
無條件分支**共用同一組文字嵌入**（空字串），兩者的差只來自影像條件。

於是現行的 `latent_norm`（把 ‖E(x')‖ 壓向零）可以精確地描述：它把影像條件
**逐元素**推向 UNet 的無條件分支，影像引導項因此消失，IP2P 退化成純文生圖
——這就是 `RESULTS.md` 記的「輸出即 prompt 被畫出來」。它一直是這個機制的
逐點版本。

本項是**函數版本**：不要求 E(x') = 0，只要求 UNet 對兩者的反應相同。

    L_ig(x') = E_{t, eps} || eps(z_t, E_img(x'), null) - eps(z_t, 0, null) ||^2

`{x' : eps(z_t, E_img(x'), null) = eps(z_t, 0, null)}` 是 UNet 的一個等位集，
比單點大得多，故同一個失真預算下更容易落進去。

與已否決的三個 reward 的差別，必須主動聲明
────────────────────────────────────────────────────────────────────
分類器／latent／CLIP 三種 reward 已全數否決（`GOAL.md`），這是第四種，歷史
基底率不利。差別有兩點且都可查：那三種都在 **SDEdit 線**上做的
（`runs/sdedit_reward_clip`、`sdedit_reward_latent`），而**SDEdit 沒有影像
引導分支**——它把原圖以「被噪聲稀釋的殘影」餵進去，沒有 `eps(z_t, c_I, ·)`
這個物件；其次那三種量的是語意對齊，本項量的是條件通道的代數性質，不涉及
任何語意空間。

兩個實作上的坑
────────────────────────────────────────────────────────────────────
1. **拼進 UNet 的影像 latent 不乘 scaling_factor**（`IP2PWrapper.image_latents`
   的 docstring 記了這件事）。用 `encode_image`（有乘）去拼會讓影像條件的
   強度整個跑掉，補錯不會拋錯。
2. **無條件那一支不依賴 x'**，故在 `no_grad` 底下算並 detach。這不是最佳化
   技巧而是正確性的一部分：它是常數，讓它進計算圖只會多一次反傳。

`z_t` 的抽法是必填的
────────────────────────────────────────────────────────────────────
IP2P 由純噪聲起步，中間步的 `z_t` 分布依賴條件、無法解析，兩個候選都是近似：

    diffuse_src   z_t = sqrt(abar_t) E(x_clean) + sqrt(1-abar_t) eps
    noise         z_t = eps

按 CLAUDE.md「查不到的參數設為必填，不要填看起來合理的預設」，工廠不給預設。
`x_clean` 一律是**原圖**：`z_t` 是取樣軌跡上的點，用防禦圖當錨會讓軌跡隨
最佳化漂移。
"""

from __future__ import annotations

from typing import Callable, Optional

import torch

from src.defense.fixedpoint_loss import _null_embedding, _scheduler_of

# 兩個候選都是近似，沒有一個是「對的」，故並列而不設預設。
ZT_MODES = ("diffuse_src", "noise")


def make_image_guidance_loss(
    ip2p,
    *,
    zt_mode: str,
    x_clean: Optional[torch.Tensor] = None,
    t_min: int = 1,
    t_max: int = 1000,
    samples: int = 1,
    seed: int = 0,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """回傳 `loss(x_def01) -> 純量`，**要最小化**。

    值越小代表 UNet 對「防禦圖當影像條件」與「沒有影像條件」的反應越接近，
    也就是 IP2P 取樣式裡 `s_I * [eps(z_t, c_I, null) - eps(z_t, 0, null)]`
    這一項越接近零。

    `samples > 1` 時每次呼叫平均多組 `(t, eps)`，代價線性增加。
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

    unet = ip2p.unet
    device = ip2p.device
    sched = _scheduler_of(ip2p)
    abar = sched.alphas_cumprod.to(device=device, dtype=torch.float32)
    if t_max > len(abar):
        raise ValueError(f"t_max={t_max} 超出排程長度 {len(abar)}")
    null_emb = _null_embedding(ip2p)
    gen = torch.Generator(device="cpu").manual_seed(int(seed))

    z_src = None
    if zt_mode == "diffuse_src":
        with torch.no_grad():
            # **走 encode_image（有乘 scaling_factor）**：這一份是噪聲 latent
            # 那一側，與拼進去的影像條件不同側，尺度也不同。IP2P 原本就有
            # 這個不對稱，見 `IP2PWrapper.image_latents` 的 docstring。
            z_src = ip2p.encode_image(x_clean).detach()

    def _sample_zt(ref: torch.Tensor):
        step = int(torch.randint(t_min - 1, t_max, (1,), generator=gen))
        eps = torch.randn(ref.shape, generator=gen, dtype=torch.float32
                          ).to(device=ref.device, dtype=ref.dtype)
        if zt_mode == "noise":
            return step, eps
        a = abar[step].to(ref.dtype)
        return step, z_src.to(ref.dtype) * a.sqrt() + eps * (1.0 - a).sqrt()

    def make_fixed(n_draws: int, eval_seed: int):
        """回傳一個**決定性**的評估函數：抽樣一次就固定，之後每次呼叫都相同。

        存在的理由是收斂判定。訓練用的損失每一步重抽 `(t, eps)`，逐步值本來
        就會抖 0.16–0.61（實測），那是取樣變異不是參數在漂；拿它判收斂會判錯，
        本專案已經犯過一次。評估必須把噪聲固定住，曲線才讀得出趨勢。
        """
        g2 = torch.Generator(device="cpu").manual_seed(int(eval_seed))
        steps_fixed = [int(torch.randint(t_min - 1, t_max, (1,), generator=g2))
                       for _ in range(n_draws)]
        eps_fixed = [None] * n_draws

        def fixed(x_def01: torch.Tensor) -> torch.Tensor:
            z_img = ip2p.image_latents(x_def01)
            total = None
            for k, step in enumerate(steps_fixed):
                if eps_fixed[k] is None:
                    eps_fixed[k] = torch.randn(
                        z_img.shape, generator=g2, dtype=torch.float32
                    ).to(device=z_img.device, dtype=z_img.dtype)
                eps = eps_fixed[k]
                if zt_mode == "noise":
                    z_t = eps
                else:
                    a = abar[step].to(z_img.dtype)
                    z_t = z_src.to(z_img.dtype) * a.sqrt() + eps * (1.0 - a).sqrt()
                tt = torch.tensor([step], device=device, dtype=torch.long)
                emb = null_emb.to(z_t.dtype)
                base = unet(torch.cat([z_t, torch.zeros_like(z_t)], dim=1), tt,
                            encoder_hidden_states=emb).sample
                cond = unet(torch.cat([z_t, z_img], dim=1), tt,
                            encoder_hidden_states=emb).sample
                term = (cond - base).pow(2).mean()
                total = term if total is None else total + term
            return total / len(steps_fixed)

        return fixed

    def loss(x_def01: torch.Tensor) -> torch.Tensor:
        # 拼進 UNet 前 4 個新通道的影像條件：**不乘 scaling_factor**。
        z_img = ip2p.image_latents(x_def01)
        total = None
        for _ in range(samples):
            step, z_t = _sample_zt(z_img)
            tt = torch.tensor([step], device=device, dtype=torch.long)
            emb = null_emb.to(z_t.dtype)
            with torch.no_grad():
                # 無條件分支不依賴 x_def，是常數。影像 latent 補零、文字取
                # 空字串——與管線第 898 行的 uncond 分支逐字相同。
                base = unet(torch.cat([z_t, torch.zeros_like(z_t)], dim=1),
                            tt, encoder_hidden_states=emb).sample.detach()
            cond = unet(torch.cat([z_t, z_img], dim=1), tt,
                        encoder_hidden_states=emb).sample
            term = (cond - base).pow(2).mean()
            total = term if total is None else total + term
        return total / samples

    loss.make_fixed = make_fixed
    return loss

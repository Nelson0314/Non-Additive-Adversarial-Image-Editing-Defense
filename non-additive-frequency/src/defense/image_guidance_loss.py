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

# `make_fixed` 回傳的評估函數用它區分「沒有給 weight」與「明確指定不加權」。
# 沒有給時沿用建構時的 `weight`——收斂監看的必須是**正在被最佳化的那個量**，
# 否則曲線走平不代表訓練走平。明確傳 `None` 才是全域均勻。
_INHERIT = object()


def _weighted_mean(sq: torch.Tensor,
                   weight: Optional[torch.Tensor]) -> torch.Tensor:
    """殘差平方圖的空間平均。`weight` 為 None 時就是 `.mean()`。

    `weight` 是像素域的 (1,1,H,W) 軟遮罩，以 `adaptive_avg_pool2d` 降到 latent
    解析度，於是每一格的值是該格落在區域內的面積比例。

    **只有這一個實作**：訓練損失與固定評估共用它。兩邊各寫一份的話，其中一份
    改了而另一份沒改時，`best_eval` 與被最佳化的量會悄悄變成兩個東西。
    """
    if weight is None:
        return sq.mean()
    import torch.nn.functional as F

    w = F.adaptive_avg_pool2d(weight.to(sq), sq.shape[-2:])
    denom = w.sum() * sq.shape[1]
    if float(denom) <= 0.0:
        raise ValueError(
            "weight 降到 latent 解析度之後總和為 0：區域太小或全為零，"
            "分區殘差沒有定義。")
    return (sq * w).sum() / denom


def make_image_guidance_loss(
    ip2p,
    *,
    zt_mode: str,
    x_clean: Optional[torch.Tensor] = None,
    t_min: int = 1,
    t_max: int = 1000,
    samples: int = 1,
    seed: int = 0,
    weight: Optional[torch.Tensor] = None,
    text_embeds: Optional[torch.Tensor] = None,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """回傳 `loss(x_def01) -> 純量`，**要最小化**。

    值越小代表 UNet 對「防禦圖當影像條件」與「沒有影像條件」的反應越接近，
    也就是 IP2P 取樣式裡 `s_I * [eps(z_t, c_I, null) - eps(z_t, 0, null)]`
    這一項越接近零。

    `samples > 1` 時每次呼叫平均多組 `(t, eps)`，代價線性增加。

    `weight`：像素域的 (1,1,H,W) 軟遮罩，把殘差的空間平均改成加權平均。
    `None`（預設）時逐位元等同原本的 `.mean()`。

    為什麼會有這個參數：`runs/ig_probe/by_region_*.csv` 量到**原圖殘差的 64%
    落在受保護主體自己的 latent token 上**，而主體只佔 53% 的面積。均勻平均
    等於把預算平均花在整張畫面上，其中將近一半花在對「主體會不會被編輯」
    影響較小的地方。把 `weight` 設成主體遮罩，就是要求最佳化只對那些 token
    負責。**這是消融，不是預設**——`--ig-weight uniform` 仍是主線。

    `text_embeds`：(K, L, D) 的一疊文字嵌入，每一步隨機抽一個當文字條件，
    也就是對**攻擊指令的分布**取期望（EOT）。`None`（預設）時一律用空字串，
    與此前逐位元相同。

    為什麼這是一個方法而不是一個旋鈕：現行損失只見過空字串，於是它對「模型
    會被要求做什麼」完全無知——它學到的是「讓影像條件在無文字的情況下失效」。
    真正的攻擊一定帶著文字條件，而 UNet 的 cross-attention 會因文字而改變它
    讀影像的方式。對一組**可能的**指令取期望，仍然不假設知道特定那一句
    （威脅模型的前提未變），但讓防禦見過文字條件存在這件事。

    **抽樣要與 `(t, ε)` 用同一個生成器**：各用一個的話「這一步用了哪一句」
    與「這一步抽到哪個時間」會變成兩條獨立的序列，重跑時對不回去。
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
    if text_embeds is not None:
        if text_embeds.dim() != 3 or text_embeds.shape[0] < 1:
            raise ValueError(
                f"text_embeds 必須是 (K,L,D) 且 K>=1，收到 "
                f"{tuple(text_embeds.shape)}")
        if text_embeds.shape[1:] != null_emb.shape[1:]:
            raise ValueError(
                f"text_embeds 的形狀 {tuple(text_embeds.shape[1:])} 與空字串嵌入的 "
                f"{tuple(null_emb.shape[1:])} 不合——兩者要能互換才是同一個位置的"
                "條件。")
        text_embeds = text_embeds.to(device=device).detach()

    def _pick_emb(g, dtype):
        """這一步的文字條件。`text_embeds` 為 None 時恆為空字串嵌入。"""
        if text_embeds is None:
            return null_emb.to(dtype)
        k = int(torch.randint(text_embeds.shape[0], (1,), generator=g))
        return text_embeds[k:k + 1].to(dtype)

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

        回傳的函數接受第二個選用參數 `weight`：一張 (1,1,H,W) 的**像素域**
        軟遮罩，會被降到 latent 解析度當空間權重，於是同一組固定抽樣可以
        問「殘差落在畫面的哪一塊」。`weight=None` 時走 `.mean()`，
        與加上這個參數之前**逐位元相同**（`tests/test_ig_regional_eval.py`
        釘住）。共用同一條路徑而不另寫一份，是因為兩份各自抽樣就不是同一
        條軸，而且看不出來。
        """
        g2 = torch.Generator(device="cpu").manual_seed(int(eval_seed))
        steps_fixed = [int(torch.randint(t_min - 1, t_max, (1,), generator=g2))
                       for _ in range(n_draws)]
        eps_fixed = [None] * n_draws

        def fixed(x_def01: torch.Tensor, eval_weight=_INHERIT) -> torch.Tensor:
            w = weight if eval_weight is _INHERIT else eval_weight
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
                emb = _pick_emb(g2, z_t.dtype)
                base = unet(torch.cat([z_t, torch.zeros_like(z_t)], dim=1), tt,
                            encoder_hidden_states=emb).sample
                cond = unet(torch.cat([z_t, z_img], dim=1), tt,
                            encoder_hidden_states=emb).sample
                term = _weighted_mean((cond - base).pow(2), w)
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
            # **兩支共用同一個文字嵌入**：影像引導項的定義就是「同文字條件下，
            # 有影像 vs 沒影像」的差。兩支用不同的文字會讓那個差混進文字的貢獻。
            emb = _pick_emb(gen, z_t.dtype)
            with torch.no_grad():
                # 無條件分支不依賴 x_def，是常數。影像 latent 補零、文字取
                # 空字串——與管線第 898 行的 uncond 分支逐字相同。
                base = unet(torch.cat([z_t, torch.zeros_like(z_t)], dim=1),
                            tt, encoder_hidden_states=emb).sample.detach()
            cond = unet(torch.cat([z_t, z_img], dim=1), tt,
                        encoder_hidden_states=emb).sample
            term = _weighted_mean((cond - base).pow(2), weight)
            total = term if total is None else total + term
        return total / samples

    loss.make_fixed = make_fixed
    return loss

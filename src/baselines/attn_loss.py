"""注意力抑制損失：Lo et al., CVPR 2024 的式 (3)(4)(5)，接到像素臂上。

    L = ‖ Att(x_def, c_a) ⊙ M ‖₁ ，  M = I( Att(x, c_a) / max > τ )

`Att` 是把 UNet 各層 cross-attention 對內容 token 的質量上採樣後相加
（`aggregate_token_attention`），`M` 由**原圖**的注意力圖決定故為常數
（`attention_region_mask`），損失只計遮罩內的反應（`masked_attention_l1`）。
三個函式都在 `src/models/attention.py`，此處只負責把它們接成 PGD 的損失。

為什麼要接這一條
────────────────────────────────────────────────────────────────────
它是指導者 Ling Lo 的《Distraction is All You Need》（CVPR 2024）的核心目標，
是本專案的對齊基準。先前只以「APA 階段二的 reward」的形式測過，且被 FND-024
以四條證據否決（最硬的一條：壓掉 94% 的注意力之後編輯成功率反而由 67% 升到
100%）。此處是**像素臂的 PGD 損失**，與 encoder-targeted、latent、CLIP 在
同一個迴圈、同樣的步數與種子下互換，唯一變因是損失——那是把基準論文的目標
放進本專案預算對齊協定的唯一方式。

成本
────────────────────────────────────────────────────────────────────
每個 PGD 步要一次 UNet 前向＋反向（其餘三個損失都不碰 UNet）。實測 0.80 s
一次（由 photoguard_c 的 6429 s/張 ÷ 8000 次反推），100 步即約 80 s/張。

時間點固定在 SDEdit 的 t₀，噪聲固定種子
────────────────────────────────────────────────────────────────────
`t = int(1000 · strength)` 與威脅模型同一點，噪聲以固定種子抽一次並全程共用。
兩者都固定是為了讓損失是 φ 的確定性函數——每步重抽會讓 `sign(grad)` 的方向
帶上取樣噪聲，而 FND-024 第 3 條已實測抑制在不同 t 上只差 1–3 個百分點，
多取樣買不到東西。
"""

from __future__ import annotations

from typing import Callable

import torch

from src.models.attention import (
    CrossAttentionRecorder, aggregate_token_attention, attention_region_mask,
    masked_attention_l1, token_span,
)


def make_attn_loss(sd, content: str, x_ref01: torch.Tensor, *,
                   strength: float = 0.8, tau: float = 0.5,
                   seed: int = 0) -> Callable:
    """回傳 `loss(x_def01) -> 純量`，要最小化。

    `content` 是內容詞 c_a（資料集的 `content` 欄，例如 "cat"／"man"）。
    """
    device = x_ref01.device
    emb = sd.encode_text(content)
    span = token_span(sd.tokenizer, content)
    if span[1] <= span[0]:
        raise ValueError(f"content={content!r} 沒有內容 token，注意力損失無從施力")

    abar = sd.alphas_cumprod(device)
    t = torch.tensor(min(int(1000 * strength), 999), device=device)
    rec = CrossAttentionRecorder(sd.unet)

    def _att(x01: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        z = sd.encode_image(x01)
        z_t = abar[t].sqrt() * z + (1 - abar[t]).sqrt() * noise
        rec.clear()
        with rec:
            sd._eps(z_t, t, emb)
        return aggregate_token_attention(rec.maps, span)

    with torch.no_grad():
        z_ref = sd.encode_image(x_ref01)
        g = torch.Generator(device="cpu").manual_seed(seed)
        noise = torch.randn(z_ref.shape, generator=g,
                            dtype=z_ref.dtype).to(device)
        mask = attention_region_mask(_att(x_ref01, noise), tau=tau).detach()

    def loss(x_def01: torch.Tensor) -> torch.Tensor:
        return masked_attention_l1(_att(x_def01, noise), mask)

    return loss

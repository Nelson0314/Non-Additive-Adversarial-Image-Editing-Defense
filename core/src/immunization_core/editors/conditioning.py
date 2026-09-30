"""SD 與 SDXL 文字條件的批次操作；SDXL 序列與 pooled 嵌入共同傳遞。"""

from dataclasses import dataclass
from typing import Sequence

import torch


@dataclass(frozen=True)
class SDXLPrompt:
    """SDXL 的文字條件：一條序列嵌入加上一個 pooled 嵌入。

    SDXL 有兩個 text encoder。CLIP-L 的倒數第二層隱狀態（768 維）與
    OpenCLIP-bigG 的倒數第二層隱狀態（1280 維）在最後一維串接成
    (B, 77, 2048)，即 `unet.config.cross_attention_dim`；bigG 的投影輸出
    另外給出一個 (B, 1280) 的 pooled 嵌入，經 `added_cond_kwargs` 與
    micro-conditioning 的 time_ids 一起進入 UNet 的 additive embedding，
    **不進 cross-attention**。

    兩者必須綁在一起傳遞，理由是 CFG：`_eps_cfg` 對條件與無條件各做一次
    前向，兩次的 pooled 不同。若 pooled 改由 wrapper 的狀態提供，無條件那
    一支會拿到條件的 pooled，而症狀只是「CFG 的效果怪怪的」，沒有任何錯誤
    訊息。綁成一個物件之後，走錯配對在型別上就不可能發生。

    `emb_residual`（文字嵌入）的作用對象**只有 `embeds`**——那是 cross-attention
    讀的序列，也是文字定位發生的地方。`__add__` 因此只加在 `embeds` 上，
    使 `generator.py` 的 `emb = emb + d_emb` 不必修改即有正確語意。
    """

    embeds: torch.Tensor      # (B, 77, 2048)
    pooled: torch.Tensor      # (B, 1280)

    @property
    def shape(self):
        """對外表現為序列嵌入的形狀，讓 `EmbeddingResidual` 的建構不需改動。"""
        return self.embeds.shape

    @property
    def dtype(self):
        return self.embeds.dtype

    @property
    def device(self):
        return self.embeds.device

    def detach(self) -> "SDXLPrompt":
        return SDXLPrompt(self.embeds.detach(), self.pooled.detach())

    def to(self, *a, **kw) -> "SDXLPrompt":
        return SDXLPrompt(self.embeds.to(*a, **kw), self.pooled.to(*a, **kw))

    def __add__(self, delta: torch.Tensor) -> "SDXLPrompt":
        if not isinstance(delta, torch.Tensor):
            raise TypeError(
                f"SDXLPrompt 只能加上序列嵌入的殘差張量，收到 {type(delta).__name__}"
            )
        if delta.shape[-1] != self.embeds.shape[-1]:
            raise ValueError(
                f"殘差的最後一維 {delta.shape[-1]} 與序列嵌入的 "
                f"{self.embeds.shape[-1]} 不符。文字嵌入 的作用對象是那條 2048 維"
                "的序列，不是 1280 維的 pooled 嵌入"
            )
        return SDXLPrompt(self.embeds + delta, self.pooled)


def concatenate_conditioning(items: Sequence):
    """把數個文字條件沿批次維串接，裸張量與 `SDXLPrompt` 都能處理。

    存在的理由：`src/baselines/` 的 AdvPaint 與 PromptFlare 需要自建兩列
    批次。在 SD v1.x 上那就是 `torch.cat([...])`，但 SDXL 的條件是
    `SDXLPrompt`（序列嵌入 + pooled 嵌入兩件），`torch.cat` 會直接拋
    `TypeError`，而 `.repeat()` 這種張量方法在 `SDXLPrompt` 上根本不存在。

    寫成函式而非要求呼叫端 `isinstance` 判斷，是因為那種判斷會漏掉
    pooled 那一半——症狀是 UNet 的 additive embedding 只拿到一列條件，
    兩列共用同一組 pooled，看起來像「CFG 效果怪怪的」而沒有錯誤訊息。
    """
    items = list(items)
    if not items:
        raise ValueError("concatenate_conditioning 收到空序列")
    first = items[0]
    if isinstance(first, SDXLPrompt):
        return SDXLPrompt(
            torch.cat([p.embeds for p in items]),
            torch.cat([p.pooled for p in items]),
        )
    return torch.cat(items)


def expand_conditioning(emb, batch: int):
    """把單列的文字條件擴成 `batch` 列。裸張量與 `SDXLPrompt` 都能處理。

    對應 SD v1.x 的 `emb.expand(batch, -1, -1)`。`SDXLPrompt` 沒有 `expand`，
    且 pooled 是二維（B, 1280）與序列嵌入的三維不同，兩者要分別處理。
    """
    if isinstance(emb, SDXLPrompt):
        return SDXLPrompt(
            emb.embeds.expand(batch, -1, -1), emb.pooled.expand(batch, -1)
        )
    return emb.expand(batch, -1, -1)

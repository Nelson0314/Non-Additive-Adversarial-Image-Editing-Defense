"""保護 reward — SPEC.md §4.2 方案一（注意力抑制）。

三個非加性方法共用（STRUCTURE.md 未列此檔，為避免重複而抽出）。

R = −‖A_concept‖₁，其中 A_concept 為 concept token 之跨層聚合注意力圖
（DAYN 式 3：各層 bicubic 上採樣至最大解析度後逐像素相加；式 5：取一次 ℓ1）。
maximize R ⇔ 抑制 concept 的 cross-attention。梯度僅穿過單步 UNet，
計算圖淺、記憶體可控（SPEC §4.2 方案一之設計理由）。

註：此為本專案自行實作於 editing pipeline 之 reward，非 DAYN 完整方法之重現
（SPEC §3.4）。
"""

import math

import torch
import torch.nn.functional as F


def concept_token_positions(tokenizer, concept: str) -> list[int]:
    """concept 於 padded prompt 中之 token 位置（排除 BOS/EOS 特殊 token）。"""
    n_tokens = len(tokenizer(concept).input_ids) - 2  # 去除首尾特殊 token
    return list(range(1, 1 + n_tokens))


def aggregate_attention(maps: list[dict], token_positions: list[int]) -> torch.Tensor:
    """跨層聚合（DAYN 式 3）：各層取 concept token 之注意力、head 平均、
    reshape 回空間圖、bicubic 上採樣至最大解析度、逐像素相加。

    回傳 (1, 1, S, S)。
    """
    spatial = []
    for m in maps:
        probs = m["probs"]                       # (batch*heads, q, L)
        a = probs[..., token_positions].sum(-1)  # (batch*heads, q)
        a = a.mean(0)                            # head 平均 → (q,)
        s = int(math.isqrt(a.shape[0]))
        spatial.append(a.reshape(1, 1, s, s))
    size = max(x.shape[-1] for x in spatial)
    up = [
        F.interpolate(x, size=(size, size), mode="bicubic", align_corners=False)
        for x in spatial
    ]
    return torch.stack(up).sum(0)


def attention_reward_latent(z: torch.Tensor, t, sd, concept: str) -> torch.Tensor:
    """於 latent z、timestep t 上計算注意力抑制 reward（可反傳至 z）。"""
    emb = sd.encode_text(concept).detach()
    with sd.capture_cross_attention() as maps:
        sd.unet(z, t, encoder_hidden_states=emb)
    positions = concept_token_positions(sd.tokenizer, concept)
    agg = aggregate_attention(maps, positions)
    return -agg.abs().sum()  # 式 (5)：ℓ1；取負使 maximize = 抑制


def attention_reward_image(x_pm1: torch.Tensor, sd, concept: str, t: int = 1) -> torch.Tensor:
    """於影像（[-1,1]，可含計算圖）上計算 reward：encode 後於小 t 評估。"""
    z = sd.encode_image(x_pm1)
    return attention_reward_latent(z, t, sd, concept)

"""inpainting 威脅模型的遮罩：攻擊方想換掉的那一塊。

為什麼要自己產
──────────────────────────────────────────────────────────────────────
`data/lo_aligned/` 是 25 張 CC0 真實照片，**沒有標註遮罩**；PIE-Bench 有
（`src/data/pie_bench.py` 的 `mask` 欄「保留供 inpainting 方向」）但遠端機器
連不上 HuggingFace。故遮罩必須由本專案產生，而「怎麼產」是一個會影響全部
結果的設計決定，不能隨手取一個中央方框。

判準：遮罩要蓋住**攻擊方想換掉的那個物件**
──────────────────────────────────────────────────────────────────────
資料集每一類都宣告了 `content`（Lo et al. 的 c_a，防禦方選擇要保護的詞），
而攻擊 prompt 的第一個正是把它換掉（dog → "a cat"）。攻擊方在 inpainting
情境下會把遮罩畫在那個物件上。

本模組以**模型自己對該詞的 cross-attention** 定出那塊區域，即 Lo et al.
(CVPR 2024) 式 (3)(4)，兩者都已實作於 `src/models/attention.py`：

    Att(x, c_a) = Σ_l upsample(A_l(x, c_a))      式 (3)
    M = I(Att / max(Att) > tau)                  式 (4)，以峰值正規化

取這條路而不是外接一個分割模型，有三個理由：

1. **可重現且無外部相依。** 分割模型是另一組權重、另一個版本相依，且它的
    失敗模式（分不出主體）與本研究無關卻會污染結果。
2. **與本專案已對齊的協定同源。** `attention_region_mask` 已由
    `tests/test_lo_protocol.py` 釘住，包含「遮罩永遠至少含峰值、不可能為空」
    這個性質。
3. **它就是攻擊方會有的資訊。** 白盒設定下攻擊方持有模型，用模型自己的
    注意力找物件是他能做的事；不必假設他有人工標註。

**已知偏離**：真實攻擊方畫的遮罩是人工的、邊界規整；此處的遮罩由注意力
閾值得到，形狀不規則且可能有孔洞。故本模組提供形態學閉運算與外接矩形兩種
後處理，由 `mask_mode` 選擇，並一律記錄所選的模式與覆蓋率——遮罩面積直接
決定攻擊方能改多少，那是必須進報告的量。
"""

from __future__ import annotations

from typing import Dict

import torch

from src.models.attention import (
    aggregate_token_attention,
    attention_region_mask,
    token_span,
)

MASK_MODES = ("attention", "attention_box", "center_box")


def _closing(m: torch.Tensor, k: int = 5) -> torch.Tensor:
    """二值遮罩的形態學閉運算（先膨脹再侵蝕），填掉注意力圖上的小孔。

    以最大池化實作膨脹、對補集最大池化實作侵蝕；`k` 為奇數方形結構元素。
    孔洞會讓攻擊方在物件內部留下不能改的小塊，那不是任何真實攻擊方會畫的
    遮罩，且它造成的斑點在輸出上看起來像防禦生效，是一個假陽性的來源。
    """
    pad = k // 2
    d = torch.nn.functional.max_pool2d(m, k, stride=1, padding=pad)
    e = 1.0 - torch.nn.functional.max_pool2d(1.0 - d, k, stride=1, padding=pad)
    return e


def _bounding_box(m: torch.Tensor) -> torch.Tensor:
    """遮罩的外接矩形。最接近人工畫框的形態。"""
    idx = (m[0, 0] > 0.5).nonzero()
    if idx.numel() == 0:
        raise ValueError("遮罩為空，取不出外接矩形")
    y0, x0 = idx.min(dim=0).values.tolist()
    y1, x1 = idx.max(dim=0).values.tolist()
    out = torch.zeros_like(m)
    out[..., y0:y1 + 1, x0:x1 + 1] = 1.0
    return out


def center_box(x01: torch.Tensor, frac: float = 0.5) -> torch.Tensor:
    """置中方框。**只作為對照**，不是主要作法。

    它與影像內容無關，故「防禦有沒有效」會混入「物件有沒有落在框內」這個
    與本研究無關的變因。保留它是為了讓「遮罩的選法影響多大」有一個下界可比。
    """
    h, w = x01.shape[-2:]
    bh, bw = int(h * frac), int(w * frac)
    y0, x0 = (h - bh) // 2, (w - bw) // 2
    m = torch.zeros(1, 1, h, w, device=x01.device, dtype=x01.dtype)
    m[..., y0:y0 + bh, x0:x0 + bw] = 1.0
    return m


def content_mask(
    sd,
    x01: torch.Tensor,
    content: str,
    *,
    mode: str = "attention_box",
    tau: float = 0.5,
    timestep: int = 500,
    seed: int = 0,
    close_k: int = 5,
) -> Dict[str, object]:
    """由模型對 `content` 的 cross-attention 產生遮罩。

    回傳 `{"mask": (1,1,H,W) [0,1], "coverage": float, "mode": str, ...}`。
    **1 表示要重畫的區域**，與 `SDWrapper.inpaint` 及 diffusers 同一約定。

    實際運算，逐步：

    1. 把 `content` 編碼成 prompt（`token_span` 取出它在 77 格中的區間，
       不含 BOS/EOS——padding 與 EOS 仍會分到可觀的注意力質量卻不承載語意）。
    2. 把原圖編碼成 latent，加噪到 `timestep`。取單一 timestep 而非平均：
       注意力圖在中段 timestep 最能反映物件位置，而逐 t 平均會把早期尚未
       成形的分佈混進來。t=500 是 [0,1000] 的中點。
    3. 跑一次 UNet 前向並以 `CrossAttentionRecorder` 記下全部 attn2 層。
    4. `aggregate_token_attention` 逐層取該詞的質量、上採樣後相加（式 3）。
    5. `attention_region_mask` 以峰值正規化後取 `> tau`（式 4）。
    6. 依 `mode` 後處理，再上採樣到影像尺寸。

    注意力圖的邊長由**實際掃到的層**決定，不是 latent 邊長：SDXL 在 1024²
    下 latent 是 128² 而最細的一層沒有 attention，聚合圖是 64²
    （`attention.cross_attention_resolutions`）。故第 6 步的上採樣是必要的。
    """
    if mode not in MASK_MODES:
        raise ValueError(f"mask_mode 只接受 {MASK_MODES}，收到 {mode!r}")
    if mode == "center_box":
        m = center_box(x01)
        return {"mask": m, "coverage": float(m.mean()), "mode": mode,
                "content": content, "tau": None}
    if not content:
        raise ValueError(
            "content 為空：注意力遮罩需要一個要保護的詞（Lo et al. 的 c_a）。"
            "資料集的 content 欄為必填，此處不落回置中方框——那會讓兩種"
            "來源不同的遮罩混在同一張表上而看不出差別"
        )

    from src.models.attention import CrossAttentionRecorder

    emb = sd.encode_text(content).detach()
    span = token_span(sd.tokenizer, content)
    if span[1] <= span[0]:
        raise ValueError(
            f"content={content!r} 沒有內容 token（tokenizer 只給出 BOS/EOS）"
        )

    with torch.no_grad():
        z = sd.encode_image(x01)
        noise = sd.sample_edit_noise(z, seed=seed)
        abar = sd.alphas_cumprod(x01.device)
        t = torch.tensor(int(timestep), device=x01.device)
        zt = abar[t].sqrt() * z + (1 - abar[t]).sqrt() * noise
        if sd.is_inpainting:
            # 9 通道權重下，只為了取注意力而跑前向時仍須補滿後 5 個通道。
            # 遮罩此時尚未存在（正在產生它），故給全 1 與原圖的 latent：
            # 那等於「整張都要重畫」，對 attn2 的 query 沒有偏好，不會把
            # 一個尚未決定的遮罩偷偷寫進它自己的來源。
            m1 = torch.ones_like(zt[:, :1])
            zt = torch.cat([zt, m1, z], dim=1)
        rec = CrossAttentionRecorder(sd.unet)
        with rec:
            sd._eps(zt, t, emb)
        att = aggregate_token_attention(rec.maps, span, reduce="mean")
        rec.clear()

    m = attention_region_mask(att, tau=tau).unsqueeze(1)     # (1,1,s,s)
    if mode == "attention":
        m = _closing(m, close_k)
    elif mode == "attention_box":
        m = _bounding_box(_closing(m, close_k))

    m = torch.nn.functional.interpolate(
        m, size=x01.shape[-2:], mode="nearest").to(x01.dtype)
    return {"mask": m, "coverage": float(m.mean()), "mode": mode,
            "content": content, "tau": tau, "timestep": int(timestep),
            "attn_side": int(att.shape[-1])}

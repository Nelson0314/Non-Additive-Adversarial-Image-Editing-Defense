"""把「交付到淨化器的不動點集合」寫成一個可微的損失項。**探索性質。**

為什麼是這一項
────────────────────────────────────────────────────────────────────
不動點框架（`runs/fixedpoint_framework/README.md`）說：擾動能不能撐過某個淨化
算子，取決於交出去的影像能不能落在那個算子的不動點集合 `Fix(T)` 上。四個算子
裡三個已經有答案——JPEG 的 `Fix` 是稠密的量化格點（可行且有效）、模糊的 `Fix`
退化成編碼器看不見的低頻（結構性不可行）、裁切的 `Fix` 存在但寬度為零。

第四個是擴散淨化，而它的 `Fix` 是**擴散先驗的高似然集合**：模型認為已經乾淨
的影像，它沒有東西要修。這一項就是把那句話寫成損失。

用誰的先驗：**攻擊方自己的**
────────────────────────────────────────────────────────────────────
評測用的 GrIDPure 建在 guided-diffusion 的 256² 無條件模型上，但本項用的是
**攻擊方編輯模型自己的先驗**（IP2P 底下的 SD 1.5，把文字與影像兩個條件都丟掉
就是它的無條件先驗）。三個理由：

1. **威脅模型如此。** 白盒假設給的是攻擊方的編輯模型（`GOAL.md`），不是
   「攻擊方會挑哪一個淨化器」。針對評測用的那一個淨化器最佳化就是 co-adapt，
   而那是已否決的方向。
2. **這讓結果變成轉移性的證據。** 用 A 模型的先驗做出來的圖，若在 B 模型建的
   淨化器上仍然有效，那是跨模型的結論；反過來則只是對單一算子過擬合。
3. 成本低一個數量級：latent 空間 64² 的一次前向，而不是像素空間 256² 的
   550M 參數模型。

**一筆必須記下的環境事實**：本專案的環境（torch 2.13.0+cu126）對
guided-diffusion 模型的**輸入**反傳會拋
`RuntimeError: One of the differentiated Tensors does not require grad`，
在 `use_checkpoint` 全部為 False、參數全部凍結的情況下仍然發生，**根因未查明**。
這不是選擇本設計的主要理由（上面三點才是），但它排除了另一條路。

損失的形式
────────────────────────────────────────────────────────────────────
標準的去噪分數匹配目標，在 latent 空間上：

    z  = E(x)                                   （VAE 編碼，含 scaling_factor）
    z_t = sqrt(a_t) z + sqrt(1 - a_t) eps
    L_fix(x) = || eps_theta(z_t, t, 空文字, 零影像條件) - eps ||^2

`t` 在一個小範圍內抽樣，與淨化器實際使用的噪聲尺度對齊——GrIDPure 每一格只走
很小的一步，所以這裡也取小 `t`。抽樣而不是固定，理由與階段二的算子輪替相同：
固定一步會讓解只對那一步過度特化。

**兩個條件都要丟掉**才是先驗：IP2P 的 UNet 吃 8 個輸入通道（4 噪聲 ＋ 4 影像
latent），無條件分支就是把影像 latent 補零、文字取空字串的嵌入。**用防禦圖
自己當影像條件是循環的**，量到的不是先驗。

**與最接近的前例方向相反**：AntiPure（ICCV 2025）是**破壞**淨化器（削弱它對
高頻的支配力、破壞跨時間步一致性）；本項是**迎合**淨化器。損失的符號是反的。
"""

from __future__ import annotations

from typing import Callable, Optional

import torch


def make_manifold_term(
    ip2p,
    *,
    t_max: int = 100,
    t_min: int = 1,
    seed: int = 0,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """回傳 `term(x01) -> 純量`。值越小＝在攻擊方的擴散先驗底下越像乾淨影像。

    `t_min`／`t_max` 界定抽樣的時間步。預設取前 100 步（總共 1000 步），對應
    淨化器實際會走的那一小段噪聲尺度。

    **梯度只走輸入**：UNet 與 VAE 的參數在 `IP2PWrapper` 裡已經凍結。
    """
    if not 1 <= t_min <= t_max:
        raise ValueError(f"需要 1 <= t_min <= t_max，收到 {t_min}／{t_max}")
    unet = ip2p.unet
    device = ip2p.device
    sched = _scheduler_of(ip2p)
    abar = sched.alphas_cumprod.to(device=device, dtype=torch.float32)
    if t_max > len(abar):
        raise ValueError(f"t_max={t_max} 超出排程長度 {len(abar)}")
    null_emb = _null_embedding(ip2p)
    gen = torch.Generator(device="cpu").manual_seed(int(seed))

    def term(x01: torch.Tensor) -> torch.Tensor:
        z = ip2p.encode_image(x01)
        step = int(torch.randint(t_min - 1, t_max, (1,), generator=gen))
        a = abar[step].to(z.dtype)
        eps = torch.randn(z.shape, generator=gen, dtype=torch.float32
                          ).to(device=z.device, dtype=z.dtype)
        z_t = z * a.sqrt() + eps * (1.0 - a).sqrt()
        # 無條件分支：影像 latent 補零、文字取空字串。**兩個都要丟**，
        # 用防禦圖自己當影像條件量到的不是先驗而是循環。
        model_in = torch.cat([z_t, torch.zeros_like(z_t)], dim=1)
        tt = torch.tensor([step], device=device, dtype=torch.long)
        pred = unet(model_in, tt,
                    encoder_hidden_states=null_emb.to(z_t.dtype)).sample
        return (pred - eps).pow(2).mean()

    return term


def _scheduler_of(ip2p):
    """取排程。`IP2PWrapper` 沒有直接暴露時，走它底下的 pipeline。"""
    for attr in ("scheduler", "_pipe", "pipe", "pipeline"):
        obj = getattr(ip2p, attr, None)
        if obj is None:
            continue
        if hasattr(obj, "alphas_cumprod"):
            return obj
        inner = getattr(obj, "scheduler", None)
        if inner is not None and hasattr(inner, "alphas_cumprod"):
            return inner
    raise AttributeError(
        "找不到擴散排程（alphas_cumprod）。**不要用自己算的 beta 表代替**"
        "——那會與攻擊方實際用的噪聲尺度不一致，而且不會有症狀。")


def _null_embedding(ip2p) -> torch.Tensor:
    """空字串的文字嵌入，快取在 wrapper 上。"""
    cached = getattr(ip2p, "_null_emb_cache", None)
    if cached is not None:
        return cached
    tok = None
    for attr in ("tokenizer", "_pipe", "pipe", "pipeline"):
        obj = getattr(ip2p, attr, None)
        if obj is None:
            continue
        tok = obj if hasattr(obj, "model_max_length") else getattr(
            obj, "tokenizer", None)
        if tok is not None:
            break
    if tok is None:
        raise AttributeError("找不到 tokenizer，無法取空字串的文字嵌入")
    ids = tok([""], padding="max_length", max_length=tok.model_max_length,
              truncation=True, return_tensors="pt").input_ids.to(ip2p.device)
    with torch.no_grad():
        emb = ip2p.text_encoder(ids)[0]
    ip2p._null_emb_cache = emb
    return emb


def make_normalised_term(
    ip2p, x_clean: torch.Tensor, **kw
) -> Callable[[torch.Tensor], torch.Tensor]:
    """把 `make_manifold_term` 除以它在**乾淨影像**上的值，使起點約為 1。

    存在理由：`--manifold-weight` 要能跨影像比較。分母逐圖算一次、之後固定，
    **不隨最佳化更新**——它是尺度不是目標。分母用同一組抽樣種子重新建立的項
    去量，故它與訓練時看到的時間步分布一致。
    """
    raw = make_manifold_term(ip2p, **kw)
    with torch.no_grad():
        ref = float(raw(x_clean))
    if not ref > 0:
        raise ValueError(f"乾淨影像上的不動點殘差不為正（{ref}），無法正規化")
    live = make_manifold_term(ip2p, **kw)

    def term(x01: torch.Tensor) -> torch.Tensor:
        return live(x01) / ref

    term.reference = ref
    return term

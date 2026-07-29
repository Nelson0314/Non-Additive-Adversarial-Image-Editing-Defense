"""site E — 文字嵌入空間 —— 由使用者提出，對應 APA 第二階段的另一個選項。

    c' = c + Δ_emb ,   Δ_emb = Σ_{i=1..r} u_i ⊗ v_iᵀ ,  u ∈ ℝ^77, v ∈ ℝ^768

`c` 是防禦生成時餵給 UNet 的條件嵌入。目前 `prompt_def` 預設為空字串，
其 CLIP 編碼即 `[BOS] [EOS] [PAD]×75` 的無條件嵌入（classifier-free
guidance 用的那一個）。本模塊不是「換一個 token」，而是**把那個 77×768
的張量本身變成可訓練參數**。

**與其他注入位置的關係**

| 位置 | φ 住在哪 | 影響範圍 |
|---|---|---|
| P / PF | 像素 | 逐像素，加性 |
| L | 每步的預測噪聲 ε̂ | 沿去噪軌跡傳播 |
| **E** | **條件嵌入 c** | **每一層 cross-attention × 每一個 timestep** |

槓桿明顯大於 site L：latent 注入實測 φ 的效果比重建誤差小 20–36 倍，而
嵌入會經由 `K = W_K·c`、`V = W_V·c` 進入每一層 cross-attention 的每一步。

**擾動天生空間相干。** 改嵌入不是改某幾個像素，是改動整條去噪軌跡的條件，
產生的像素殘差必然是空間上連貫的。若「低秩＝空間相干＝耐淨化」這個假設
成立，此處應是最極端的驗證點。

**三個必須先講清楚的限制**

1. 它在生成路徑上，輸出經 VAE 解碼，故保真度受該位置的重建誤差牽制。
   對策不是錨定（那會消掉非加性），而是先跑階段一保真對齊（見
   `optimize.align`）。
2. 改條件會改**內容**而非只改紋理，優化器可能把狗變成貓。「有效果」與
   「還是同一張圖」之間有沒有可用的區間，是本模塊要回答的主要問題，
   不是可以假設的前提。
3. **φ=0 對照必須從第一天就跑。** site L 白跑了 36 格才發現 φ 貢獻為零，
   此處不得重蹈。V 初始為零使 Δ_emb = 0，模塊停用與 φ=0 皆逐元素等價於
   未注入，該不變量由測試把關。

**參數量**：r·(77 + 768)。r=16 為 13,520，比像素低秩的 49,152 更少。

**快取不變量**：`prepare()` 的 DDIM inversion 必須用**未擾動**的 `c`，
否則 z_inv 依賴 φ、快取失效。故本模塊只在 `generate()` 的去噪段生效，
不參與 inversion。此約束由 generator 實作，測試另行把關。
"""

from typing import Optional

import torch

from src.residual.base import ResidualModule
from src.residual.lowrank import LowRankResidual

# SD v1.x 的 CLIP text encoder：77 個 token、每個 768 維
CLIP_TOKENS = 77
CLIP_DIM = 768


class EmbeddingResidual(ResidualModule):
    site = "E"

    def __init__(
        self,
        tokens: int = CLIP_TOKENS,
        dim: int = CLIP_DIM,
        max_rank: int = 32,
        const_rank: int = 8,
        scale: float = 1.0,
        init_std: float = 0.02,
        seed: int = None,
    ):
        super().__init__()
        # 沿用 LowRankResidual：channels=1、height=tokens、width=dim，其輸出
        # (1, tokens, dim) 恰為嵌入張量的形狀。重用已測過的外積構造，秩仍由
        # 架構保證，不需在此重寫一遍。
        self.tensor = LowRankResidual(
            steps=1,
            channels=1,
            height=tokens,
            width=dim,
            max_rank=max_rank,
            init_std=init_std,
            seed=seed,
        )
        self.const_rank = const_rank
        self.scale = scale

    def delta(self) -> torch.Tensor:
        """回傳 (1, tokens, dim) 的嵌入殘差。"""
        return self.scale * self.tensor(step=0, rank=self.const_rank)

    def emb_residual(self, emb: torch.Tensor) -> Optional[torch.Tensor]:
        """回傳要加到條件嵌入上的殘差，或 None 表示不提供此能力。

        形狀不符時直接報錯而非廣播：嵌入的 token 數與維度由 text encoder
        決定，對不上代表模塊建錯了，靜默廣播會讓錯誤延後到數值階段。
        """
        if not self.enabled:
            return None
        d = self.delta().to(emb.dtype)
        if d.shape[-2:] != emb.shape[-2:]:
            raise ValueError(
                f"嵌入殘差形狀 {tuple(d.shape)} 與條件嵌入 {tuple(emb.shape)} "
                "不符；請以 text encoder 的實際輸出形狀建立模塊"
            )
        return d

    def raw_residual(self) -> None:
        """嵌入空間沒有對應的像素空間量，與 site L 同。"""
        return None

    def rank_trace(self, ts=None, steps=None) -> list:
        return [self.const_rank]

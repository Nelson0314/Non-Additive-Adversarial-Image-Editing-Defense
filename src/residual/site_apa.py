"""APA 移植（條件 N3）— `docs/DESIGN_2026-08-05.md` §4。

## 與原論文的對應

APA（*Adversary Preferences Alignment*, arXiv:2506.01511）是兩階段框架：

| 階段 | 原論文 | 本專案的移植 |
|---|---|---|
| 一：視覺一致性 | `R_s(Δθ) = E_{t,ε}[−‖ε − ε_{θ+Δθ}(z_t,t,c)‖²]`，LoRA | **原樣保留** |
| 二：攻擊有效性 | `R_a(z_T) = L(f_φ(x_adv), y)`，替代分類器的 cross-entropy | **替換**，見下 |
| 攻擊點 | latent `z_T` 優於 prompt embedding（黑盒 ASR 88.02% vs 62.08%） | **取 latent** |

**階段二為何必須替換**：原論文的 reward 是分類器標籤的 cross-entropy，
而抗編輯場景沒有分類器，該式在字面上不可執行。替換為

    R_a(z_T) = −‖SDEdit_sub(x_adv; c_∅) − x_target‖²

即代理編輯管線在**空 prompt** 下的輸出對固定目標影像的距離。正當性有兩層：

1. **結構位置相同。** 原論文的 substitute classifier 是「防禦方手上、與攻擊方
   不同的代理評分器」；`SDEdit_sub` 佔同一個位置——防禦方用自己的一條編輯
   管線取得回饋，不假設攻擊方會用哪個 prompt。
2. **形式取自對比論文而非自創。** targeted 的輸出距離正是 PhotoGuard-c 的
   diffusion attack 與 Mist 的 textural loss 所用的形式，故 N3 與 B1／B2 在
   損失形式上可直接對照，差別只在參數化。

損失本身在 `src/defense/objective.py`；本模組只負責參數化。

## 已量測的適用範圍限制

APA 走生成路徑，`x_def` 經 `decode(encode(x))`，該來回本身即
**LPIPS 0.1434 / PSNR 27.51 dB**（實測）。故本條件在 τ = 0.05 與 0.10 上
**結構上不可能達成**——其最小失真已超過該預算。這兩格標為 `skipped` 而非
`failed`（`CODE` §2.2），並在論文中作為適用範圍限制陳述，不隱藏。

主表因此設在 τ = 0.20，使三個非加性條件與全部 baseline 同時在場。
"""

from typing import Optional

from src.residual.composite import CompositeResidual
from src.residual.site_latent import LatentResidual
from src.residual.site_weight import APA_BLOCKS, WeightResidual

STAGE1, STAGE2 = "stage1", "stage2"

# 階段一的 LoRA 設定，取自官方實作 https://github.com/deep-kaixun/APA
# （2026-08-05 查證，見 `docs/SOURCE_AUDIT_2026-08-05.md`）。
# **這些值論文中不存在**——APA 的 arXiv v1 是唯一版本且沒有 Appendix，
# 論文四處引用它卻查無該節，故階段一的全部超參數只能取自程式碼。
APA_LORA_RANK = 8
APA_LORA_ALPHA = 8
APA_STAGE1_LR = 1e-4          # AdamW，constant schedule
APA_STAGE1_STEPS = 200
APA_NOISE_OFFSET = 0.1        # 論文未提，只在程式碼中


def build_apa(
    unet,
    steps: int,
    latent_size: int,
    latent_channels: int = 4,
    lora_rank: int = APA_LORA_RANK,
    lora_alpha: Optional[float] = APA_LORA_ALPHA,
    lora_blocks: tuple = APA_BLOCKS,
    latent_max_rank: int = 32,
    latent_const_rank: int = 8,
    latent_scale: float = 1.0,
    latent_init_std: float = 0.02,
    seed: Optional[int] = None,
) -> CompositeResidual:
    """回傳 `names = ["stage1", "stage2"]` 的複合模塊。

    `latent_size` 是 latent 的邊長，即影像邊長 ÷ 8。SDXL 在 1024² 下為 128，
    SD v1.x 在 512² 下為 64。**呼叫端必須明給**，不由影像尺寸推導——
    推導會在換模型時靜默給出錯誤的形狀，而那是本專案已知的缺陷型態。

    `steps` 必須等於防禦生成的去噪步數：`LatentResidual` 逐步持有一組殘差，
    步數不符時它會在 `eps_hook` 就拋出，不會靜默對齊。

    兩個階段的**學習率必須分別校準**（`ARCH` §6）。LoRA 的參數是矩陣元素、
    latent 注入的參數是噪聲量級，量綱不同；Adam 每步位移約等於 lr，
    同一個值對兩者代表兩種步長。實測不同注入位置的校準值可差 12.5 倍。
    """
    # blocks 預設為 APA_BLOCKS = ("attn1", "attn2")：官方實作的 LoRA
    # target_modules 掛在整個 UNet 上，同時涵蓋 self- 與 cross-attention。
    # 本專案原本寫死只掃 attn2，照它實作會少掉一半的目標層。
    if latent_init_std <= 0.0:
        raise ValueError(
            f"latent_init_std={latent_init_std} 會讓外積參數化的兩個因子同時為零，"
            "梯度永久為零而不會報錯——階段二將完全訓練不動，"
            "N3 靜默退化成「只有 LoRA」的條件。見本模組對該參數的說明。"
        )
    lora = WeightResidual(unet, rank=lora_rank, alpha=lora_alpha, seed=seed,
                          blocks=lora_blocks)
    latent = LatentResidual(
        steps=steps,
        channels=latent_channels,
        size=latent_size,
        max_rank=latent_max_rank,
        const_rank=latent_const_rank,
        scale=latent_scale,
        # **`init_std` 不可為 0。** 它控制的是外積參數化中「高斯的那一半」
        # （U），另一半（V）本來就初始化為零。兩個因子同時為零時
        #
        #     ∂(U⊗V)/∂U = V = 0    且    ∂(U⊗V)/∂V = U = 0
        #
        # 兩邊的梯度**永久**為零，階段二完全訓練不動——而且不會報錯：
        # 損失算得出來、優化器跑得完、日誌看起來正常，只是 φ 從頭到尾沒變。
        # N3 會靜默退化成「只有 LoRA 保真對齊」的條件。
        #
        # 實測（`LowRankResidual`，rank 2）：
        #     init_std=0.0  → gradU 0.0     gradV 0.0
        #     init_std=0.02 → gradU 0.0     gradV 0.1896
        #
        # 非零的 `init_std` 不會破壞「φ=0 逐位元等同未注入」：V 為零使外積
        # 恆為零，殘差仍精確是 0。這正是 LoRA（Hu et al., ICLR 2022）採用
        # 「一半高斯、一半零」的理由，`site_weight.py` 的 `_LoRAHook` 同一慣例。
        #
        # 2026-08-05 修正。before：`init_std=0.0`，註解誤稱「此處給 0 指的是
        # 整體殘差為零」。after：沿用 `LatentResidual` 的預設 0.02。
        init_std=latent_init_std,
        seed=seed,
    )
    return CompositeResidual([lora, latent], [STAGE1, STAGE2])

"""針對淨化最佳化：把可微分的淨化算子放進防禦的前向路徑。

為什麼要有這個
────────────────────────────────────────────────────────────────────
本專案至今的抗淨化都是**事後量測**——先把防禦做好，再看它過了淨化還剩多少。
擾動從來沒有「知道」自己會被壓縮過。

FND-061 給了一個很說明問題的對照：DCT-Shield 在 JPEG 下大勝紋理重相位
（淨增益 +0.519 對 +0.135），而它並沒有做任何 min-max 最佳化——它只是把
擾動**長在 JPEG 的量化格點上**。也就是「與淨化算子對齊」這件事本身就足以
換到抗性。反過來，同一個方法在高斯模糊下淨增益只剩 +0.010，因為模糊不吃
那套格點。

所以最直接的強化路徑是：**把淨化算子放進最佳化迴圈，讓梯度自己去找活得下來
的位置。** 作法取自 MetaCloak-JPEG（arXiv:2604.18537）：把可微分 JPEG 放進
前向，並讓品質因子沿著一條 curriculum 由高走低。

與 min-max 的差別
────────────────────────────────────────────────────────────────────
這**不是** min-max。攻擊方的淨化算子在此是固定的、已知的、可微的，我們只是
在它的複合函數上做一般的 PGD。真正的 min-max 需要對淨化參數也做內層最佳化，
成本高一個數量級，且 2026-08-13 已否決過（見 CLAUDE.md 的否決清單）。

課程排程為什麼由高品質走向低品質
────────────────────────────────────────────────────────────────────
高品質的 JPEG 量化步長細，幾乎所有頻率都留得住，梯度看到的地形接近沒有淨化
的情形；低品質步長粗，只有落在保留頻帶上的擾動才活得下來。先易後難讓擾動
先找到有效的方向、再被逼進活得下來的頻帶。直接從低品質起步時，早期的梯度
幾乎全被量化吃掉，最佳化沒有方向可循。

**排程的端點是本專案指定的**：MetaCloak-JPEG 的摘要只說「由 95 降到 50」，
未載明衰減形狀。此處取線性，並把它寫成參數。
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence

import torch

from src.baselines.jpeg_codec import jpeg_roundtrip_ste

# MetaCloak-JPEG 摘要載明的端點；衰減形狀為本專案指定（線性）
CURRICULUM_Q_HI = 95
CURRICULUM_Q_LO = 50


def jpeg_quality_at(step: int, steps: int, *, q_hi: int = CURRICULUM_Q_HI,
                    q_lo: int = CURRICULUM_Q_LO) -> int:
    """第 `step` 步該用的 JPEG 品質。線性由 `q_hi` 降到 `q_lo`。

    端點都取得到：`step = 0` 給 `q_hi`，`step = steps - 1` 給 `q_lo`。
    `steps = 1` 時退化為 `q_hi`（沒有可衰減的區間）。
    """
    if steps <= 0:
        raise ValueError(f"steps 必須為正，收到 {steps}")
    if not 0 <= step < steps:
        raise ValueError(f"step={step} 超出 [0, {steps})")
    if q_lo > q_hi:
        raise ValueError(f"q_lo={q_lo} 高於 q_hi={q_hi}，課程應由高走低")
    if steps == 1:
        return q_hi
    frac = step / (steps - 1)
    return int(round(q_hi + (q_lo - q_hi) * frac))


def make_jpeg_transform(steps: int, *, q_hi: int = CURRICULUM_Q_HI,
                        q_lo: int = CURRICULUM_Q_LO
                        ) -> Callable[[torch.Tensor, int], torch.Tensor]:
    """回傳 `transform(x01, step)`，把影像過一次該步品質的可微分 JPEG 往返。

    前向值逐位元等於真實的 JPEG 往返（`jpeg_codec.jpeg_roundtrip`），只有
    `round()` 的反向被當成恆等，故**量到的失真與最終存檔的一致**。
    """

    def transform(x01: torch.Tensor, step: int) -> torch.Tensor:
        return jpeg_roundtrip_ste(x01, jpeg_quality_at(step, steps,
                                                       q_hi=q_hi, q_lo=q_lo))

    return transform


def make_fixed_jpeg_transform(quality: int
                              ) -> Callable[[torch.Tensor, int], torch.Tensor]:
    """固定品質的版本，用來把「課程」與「有沒有放 JPEG」兩個變因分開。"""

    def transform(x01: torch.Tensor, step: int) -> torch.Tensor:
        return jpeg_roundtrip_ste(x01, quality)

    return transform


def make_eot_jpeg_transform(qualities: Sequence[int], seed: int = 0
                            ) -> Callable[[torch.Tensor, int], torch.Tensor]:
    """每一步從 `qualities` 裡**隨機抽一個**品質。

    這是 expectation-over-transformation 的最省事版本：不對每步的多個品質求
    平均（那樣每步要多算好幾次 VAE），而是讓抽樣在步與步之間攤平。**與
    MetaCloak-JPEG 的課程排程是兩種不同的做法**，不要混在同一列報表上。
    """
    if not qualities:
        raise ValueError("qualities 不可為空")
    gen = torch.Generator(device="cpu").manual_seed(int(seed))

    def transform(x01: torch.Tensor, step: int) -> torch.Tensor:
        i = int(torch.randint(len(qualities), (1,), generator=gen))
        return jpeg_roundtrip_ste(x01, qualities[i])

    return transform


def describe(transform: Optional[Callable]) -> str:
    """報表用的一行說明。`None` 代表沒有針對淨化最佳化。"""
    return "none" if transform is None else getattr(
        transform, "__qualname__", "custom").split(".")[0]

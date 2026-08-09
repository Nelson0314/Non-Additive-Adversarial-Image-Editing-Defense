"""site W — 權重空間 —— spec §4.3 列出但從未實作的第三個注入位置。

    W' = W + (α/r)·B·A      B ∈ ℝ^{out×r}, A ∈ ℝ^{r×in}
    φ = { A, B }  於 UNet cross-attention 的 to_q / to_k / to_v / to_out

即 LoRA。防禦圖生成時以 `W'` 跑去噪；`B = 0` 時 `W' = W` 逐元素完全
相等，模型行為零改變。

為什麼需要它（E9 的直接結果）

階段一保真對齊的目的是把 latent 位置的重建誤差吸收進 φ，讓生成式的
`x_def` 拿到可用的保真度。E9 實測低秩 ε 注入容量不足：

    car_00  LPIPS 0.2032 → 0.0950（200 步，仍在緩慢下降）
    car_01  LPIPS 0.2095 → 0.1723（第 50 步即停住，其後 150 步無改善）

兩者都遠不及像素位置實際運作的 0.063。car_01 的平台是容量天花板而非步數
不足。可用的替代載體只剩權重空間：

| 載體 | 參數量（SD v1.4, 512², r=16） |
|---|---|
| 低秩 ε 注入 | 163,840 —— 實測容量不足 |
| 嵌入空間 site E | 13,520 —— 比現況更少，不會更好 |
| 像素全秩 site PF（對照用，非生成式） | 786,432 |
| 權重空間 site W | 1,591,296（16 個 transformer block × 4 個 Linear）|

site W 的參數量是低秩 ε 注入的 9.7 倍。r=4 時為 397,824，仍為 2.4 倍。

實作方式：forward hook，不改動模型權重本身

`_LoRAHook` 掛在每個目標 `nn.Linear` 上，把 `B(A(x))` 加到其輸出。這樣
SD 的權重張量從頭到尾沒有被寫入過，`disable()` 即刻還原，且同一個
`SDWrapper` 可以在多個實驗之間重複使用而不需重新載入模型。

與其他位置的能力差異：本模塊既不提供 `pixel_residual`、也不提供
`eps_hook` 或 `emb_residual` —— φ 不走任何顯式的殘差路徑，而是直接改動
模型。故另外宣告 `patches_model()`，供 generator 判斷 φ 是否進得了計算圖。

限制：在生成路徑上，輸出經 VAE 解碼，保真度受該位置的重建誤差牽制。
必須搭配階段一使用；單獨啟用只是換一個地方注入同樣的問題。
"""

from typing import List, Optional

import torch
import torch.nn as nn

from src.residual.base import ResidualModule

# cross-attention（attn2）才吃文字條件；attn1 是 self-attention。
DEFAULT_TARGETS = ("to_q", "to_k", "to_v", "to_out.0")

# 要掛載的 attention 區塊。預設只有 cross-attention，維持既有行為。
#
# **APA 移植必須用 ("attn1", "attn2")。** 2026-08-05 的原始碼查證
# （`docs/reference/SOURCE_AUDIT.md`）確認 APA 官方實作的 LoRA
# `target_modules=["to_k","to_q","to_v","to_out.0"]` 是掛在整個 UNet 上的，
# 因此同時涵蓋 self-attention（attn1）與 cross-attention（attn2）。
# 原本此處寫死 `.attn2.`，照它實作會少掉一半的目標層，是一個不會有症狀的
# 容量差異——訓練跑得完、曲線正常，只是階段一的視覺一致性對齊能力被削弱。
DEFAULT_BLOCKS = ("attn2",)
APA_BLOCKS = ("attn1", "attn2")


class _LoRAHook(nn.Module):
    """單一 Linear 的低秩修正。輸出 += (α/r)·B(A(x))。"""

    def __init__(self, in_features: int, out_features: int, rank: int,
                 alpha: float, init_std: float, gen: torch.Generator):
        super().__init__()
        # A 高斯、B 零：初始修正量恆為零（W' = W），而 ∂L/∂B = (∂L/∂out)·A(x)
        # 不為零，梯度可正常流動。此慣例出自 LoRA（Hu et al., ICLR 2022）。
        # 若 A 也為零則兩者的梯度都恆為零，永遠訓練不動。
        self.A = nn.Parameter(
            torch.randn(rank, in_features, generator=gen) * init_std)
        self.B = nn.Parameter(torch.zeros(out_features, rank))
        self.scaling = alpha / rank

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """輸入先轉成 A／B 的 dtype。**參數不跟著骨幹降精度。**

        骨幹在本輪是 bf16 而 A／B 是 fp32，兩者相乘會以
        `RuntimeError: expected mat1 and mat2 to have the same dtype, but got:
        c10::BFloat16 != float` 中止（2026-08-06 於段 0 的 N3 探測實測）。

        修法是轉輸入而不是把參數降成 bf16：A／B 就是 φ 本身，是**唯一**的
        可訓練參數。bf16 只有 8 位尾數，B 由零起步、Adam 的單步更新量遠小於
        權重量級，降精度會讓更新被捨入吃掉——訓練仍然跑得完、曲線仍然像樣，
        只是學不動。凍結的骨幹走半精度、可訓練參數留 fp32，是混合精度的
        標準配置。

        回程由 `_make_hook` 的 `.to(output.dtype)` 負責，那一行本來就在，
        也就是說「LoRA 自己一個精度、進出各轉一次」原本就是設計意圖，
        只是入口那一半漏了。

        2026-08-06 新增。before：`linear(linear(x, self.A), self.B) * scaling`，
        直接以 `x` 進入。fp32 全程時 dtype 相同，故本機的 CPU 測試不會暴露。
        """
        xa = x.to(self.A.dtype)
        return torch.nn.functional.linear(
            torch.nn.functional.linear(xa, self.A), self.B) * self.scaling


class WeightResidual(ResidualModule):
    site = "W"

    def __init__(
        self,
        unet: nn.Module,
        rank: int = 16,
        alpha: Optional[float] = None,
        targets: tuple = DEFAULT_TARGETS,
        init_std: float = 0.02,
        seed: int = None,
        blocks: tuple = DEFAULT_BLOCKS,
    ):
        super().__init__()
        self.hooks = nn.ModuleDict()
        self._handles = []
        # alpha 預設等於 rank，使 scaling = 1.0：掃描 rank 時不會同時改動
        # 修正量的整體尺度，否則「秩」與「強度」兩個變因會糾纏在一起。
        self.alpha = float(rank if alpha is None else alpha)

        gen = torch.Generator().manual_seed(seed) if seed is not None else None
        named = dict(unet.named_modules())
        selected = [
            (n, m) for n, m in named.items()
            if any(f".{b}." in n for b in blocks) and isinstance(m, nn.Linear)
            and any(n.endswith(t) for t in targets)
        ]
        if not selected:
            raise ValueError(
                f"在 UNet 中找不到符合 blocks={blocks}、targets={targets} 的 Linear；"
                "此模型的模組命名可能與 diffusers 的 attn1/attn2 慣例不同"
            )

        for name, lin in selected:
            # ModuleDict 的鍵不得含 '.'，改用 '/' 保留可讀的原始路徑
            key = name.replace(".", "/")
            self.hooks[key] = _LoRAHook(
                lin.in_features, lin.out_features, rank,
                self.alpha, init_std, gen,
            )
            self._handles.append(
                lin.register_forward_hook(self._make_hook(key))
            )

        self.rank = rank
        self.targets = targets
        self.blocks = blocks
        self.n_layers = len(selected)

    def _make_hook(self, key: str):
        def hook(module, args, output):
            if not self.enabled:
                # 停用時原樣回傳。回傳 None 也可，但明確回傳 output 讓
                # 「停用 ⟹ 逐元素等同無此模塊」在讀程式時就看得出來。
                return output
            return output + self.hooks[key](args[0]).to(output.dtype)
        return hook

    def patches_model(self) -> bool:
        return True

    def remove(self) -> None:
        """卸除所有 forward hook，把 UNet 還原成未掛載狀態。

        必須顯式呼叫：hook 是註冊在 SD 的模組上的，模塊本身被垃圾回收
        並不會移除它們，殘留的 hook 會污染同一個 SDWrapper 的後續實驗。
        """
        for h in self._handles:
            h.remove()
        self._handles = []

    def raw_residual(self) -> None:
        """權重空間沒有對應的像素空間量。"""
        return None

    def rank_trace(self, ts=None, steps=None) -> list:
        return [self.rank]

"""求解目標：色彩映射與 VAE 重建的不交換量。

問的是什麼
────────────────────────────────────────────────────────────────────
令 `V = Dec∘Enc`（posterior mean，決定性）。對一個**全域**色彩映射 `T_θ`，
如果編碼器把顏色當成它已經學得很好的低頻資訊，那麼先上色再重建、與先重建
再上色，結果應該幾乎一樣：`V(T x) ≈ T(V x)`。兩者差多少就是這個載體讓
自編碼器「算錯」了多少：

    J(θ) = LPIPS( V(T_θ x),  T_θ(V x) )

擴散編輯的第一步就是把輸入送進這個 `V` 的編碼器，所以 `J` 大代表防禦圖在
進入 UNet 之前就已經被表示錯了，而且**錯的方向與任何一句指令無關**——
這個目標不讀攻擊者的 prompt，也不跑淨化分支。

兩邊都對 θ 微分
────────────────────────────────────────────────────────────────────
`T_θ(V x)` 這一側常被寫成常數（先算好一張圖）。那會讓梯度只推左邊，等價於
在最小化「重建後的圖離某個固定目標多遠」，與不交換量不是同一件事。這裡
`V x` 是常數（不含 θ），但 `T_θ` 作用在它上面仍是 θ 的函數，兩側都留在圖上。

方向
────────────────────────────────────────────────────────────────────
防禦方要的是**大**的不交換量，而 `optimise_carrier` 最小化 `score`，
所以 `score = −J`。`terms` 回報的是 `J` 本身（正值），CSV 的
`free_term_vae_noncommute` 因此越大越好，與 `free_score_*` 的符號相反。

決定性
────────────────────────────────────────────────────────────────────
posterior mean 沒有抽樣，`eval_score` 與 `score` 是同一個量；早停盯的固定
抽樣讀數因此就是目標本身，不需要另外固定種子。
"""
from __future__ import annotations

from typing import Dict

import torch


class VAENonCommutation:
    """`J(θ) = LPIPS(V(T_θ x), T_θ(V x))`，以 `−J` 當分數。"""

    name = 'vae_noncommute'

    def __init__(self, ip2p, x01: torch.Tensor, carrier, metric,
                 use_ckpt: bool = True):
        self.ip2p = ip2p
        self.carrier = carrier
        self.metric = metric
        self.use_ckpt = bool(use_ckpt)
        with torch.no_grad():
            self.vx = self.reconstruct(x01).detach()
        self.draws = 0
        self.seed = 0

    def reconstruct(self, x01: torch.Tensor) -> torch.Tensor:
        """`V = Dec∘Enc`，posterior mean。`decode_latent` 已把輸出夾回 [0,1]。"""
        latent = self.ip2p.encode_image(x01, use_ckpt=self.use_ckpt)
        return self.ip2p.decode_latent(latent, use_ckpt=self.use_ckpt)

    def value(self, y: torch.Tensor) -> torch.Tensor:
        left = self.reconstruct(y).float().clamp(0, 1)
        right = self.carrier.render(self.vx).float().clamp(0, 1)
        return self.metric(left, right)

    def score(self, y: torch.Tensor) -> torch.Tensor:
        return -self.value(y)

    def terms(self, y: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {'vae_noncommute': self.value(y)}

    # posterior mean 沒有抽樣，固定抽樣的評估就是它自己。
    def eval_score(self, y: torch.Tensor) -> torch.Tensor:
        return self.score(y)

    def eval_terms(self, y: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.terms(y)

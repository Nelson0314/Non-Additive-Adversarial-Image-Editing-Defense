"""主流影像免疫法的兩個目標項。**都不含文字，也不含任何編輯指令。**

`instruction_free.FreeObjective` 現行的 `enc` 與 `cond` 都是**無目標**的相對
位移：它們只要求「離原來的地方遠」，不指定要去哪裡。文獻上被採用的兩個作法
都不是這個形狀，而且兩個的失效模式不同，所以分開實作、分開量。

`enc_target` — PhotoGuard 的 targeted encoder attack
────────────────────────────────────────────────────────────────────
最小化 `‖E(x_def) − z_target‖ / ‖z_target‖`，把防禦圖的 VAE latent 拉到一個
**固定的目標 latent** 上。與無目標版的差別在梯度的形狀：無目標版在任何方向上
走都拿得到分，梯度會隨著離開原點而轉向；targeted 版整條路徑指向同一個點，
所以同樣的預算走得比較遠。目標由呼叫端給一張固定影像（本專案用中性灰），
**與攻擊者要下什麼指令無關**，也不隨影像內容變。

`diffusion` — AdvDM／Mist 的擴散訓練損失
────────────────────────────────────────────────────────────────────
最大化 `E_t ‖ε_θ(z_t, t, c_img) − ε‖²`，其中 `z_t` 由**防禦圖自己的 latent**
加噪而成。前兩項問的是「模型的預測偏掉多少」，這一項問的是「模型有多不會
還原這張圖」——被推向的是模型的訓練分布之外，而不只是離原圖遠。

IP2P 的 UNet 吃九個通道，所以 `z_t` 與影像條件**兩側都用防禦圖**：攻擊者拿到
的就是這張圖，兩側本來就一致。噪聲與時刻沿用 `FreeObjective` 已經凍結的那一組，
兩個項因此在同一組隨機性上比較。

加噪走 `σ` 的形式而不是 `scheduler.add_noise`
────────────────────────────────────────────────────────────────────
`EulerAncestralDiscreteScheduler` 是有狀態的（見 `instruction_free._scaled_sample`
的說明），`add_noise` 與 `scale_model_input` 都會踩到 `step_index` 只在第一次
呼叫時初始化的問題。這裡用該時刻自己的 σ 做無狀態的加噪與縮放：

    z_t = (z_0 + σ_t · ε) / √(σ_t² + 1)

分母就是 `scale_model_input` 對這一族排程器做的事。預測的目標在 σ 參數化下是
ε 本身，所以殘差直接對 ε 取，不需要再換參數。
"""
from __future__ import annotations

from typing import Dict

import torch
import torch.utils.checkpoint as ckpt

from .instruction_free import _sigma_at


def grey_target(x01: torch.Tensor, value: float = 0.5) -> torch.Tensor:
    """與 `x01` 同形狀的中性灰影像，當 targeted encoder attack 的目標。

    取中性灰而不是取另一張真實照片：真實照片會把一個具體身分寫進目標，
    防禦圖就被拉向**那個人**，而目標圖的選擇會變成一個沒有理由的自由度。
    中性灰在 latent 上是一個低變異的點，不對應任何身分。
    """
    return torch.full_like(x01, float(value))


class MainstreamTerms:
    """掛在 `FreeObjective` 旁邊的兩個項。共用同一組噪聲、時刻與排程器。"""

    def __init__(self, objective, x01, *, target: torch.Tensor = None,
                 use_ckpt: bool = True):
        self.obj = objective
        self.ip2p = objective.ip2p
        self.use_ckpt = use_ckpt
        with torch.no_grad():
            tgt = grey_target(x01) if target is None else target
            self.z_target = self.ip2p.encode_image(tgt).float()
            self.z_target_norm = self.z_target.flatten().norm().clamp_min(1e-6)

    def _noised(self, z0, t):
        s = _sigma_at(self.obj.scheduler, t).to(z0)
        return (z0 + s * self.obj.noise.to(z0)) / (s * s + 1).sqrt()

    def enc_target(self, x_def: torch.Tensor) -> torch.Tensor:
        z = self.ip2p.encode_image(x_def).float()
        return (z - self.z_target).flatten().norm() / self.z_target_norm

    def diffusion(self, x_def: torch.Tensor) -> torch.Tensor:
        z0 = self.ip2p.encode_image(x_def)
        cond = self.ip2p.image_latents(x_def).to(self.obj.text.dtype)
        noise = self.obj.noise
        losses = []
        for t in self.obj.timesteps:
            zt = self._noised(z0.to(noise.dtype), t)
            model_input = torch.cat([zt, cond.to(zt.dtype)], dim=1)
            if self.use_ckpt:
                eps = ckpt.checkpoint(self.obj._run_unet, model_input, t,
                                      self.obj.text, use_reentrant=False)
            else:
                eps = self.obj._run_unet(model_input, t, self.obj.text)
            losses.append(((eps.float() - noise.float()) ** 2).mean())
        return torch.stack(losses).mean()

    def terms(self, x_def: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {'enc_target': self.enc_target(x_def),
                'diffusion': self.diffusion(x_def)}


class CombinedObjective:
    """`FreeObjective` 的項，加上要用到的主流項，權重由設定檔給。

    `enc_target` **越小越好**（要靠近目標），所以它以 `+w·tanh` 進分數；
    `diffusion` **越大越好**（要模型還原不了），以 `−w·tanh` 進分數。與
    `FreeObjective.score` 的符號慣例一致：分數一律是要最小化的那個方向。

    權重為零的項**不計算**：每一項都要跑一輪 UNet，關掉的項不該付那個成本。
    """

    def __init__(self, base, extra, weights: Dict[str, float]):
        self.base = base
        self.extra = extra
        self.weights = dict(weights)
        self.ip2p = base.ip2p
        self.timesteps = base.timesteps

    def _w(self, name: str) -> float:
        return float(self.weights.get(name, 0.0) or 0.0)

    def terms(self, x_def: torch.Tensor) -> Dict[str, torch.Tensor]:
        out = dict(self.base.terms(x_def))
        if self._w('enc_target') > 0:
            out['enc_target'] = self.extra.enc_target(x_def)
        if self._w('diffusion') > 0:
            out['diffusion'] = self.extra.diffusion(x_def)
        return out

    def score(self, x_def: torch.Tensor) -> torch.Tensor:
        t = self.terms(x_def)
        w = self.weights
        s = (-self._w('enc') * torch.tanh(t['enc'])
             - self._w('cond') * torch.tanh(t['cond']))
        if 'id' in t:
            s = s + self._w('id') * t['id']
        if 'enc_target' in t:
            s = s + self._w('enc_target') * torch.tanh(t['enc_target'])
        if 'diffusion' in t:
            s = s - self._w('diffusion') * torch.tanh(t['diffusion'])
        return s

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


def texture_target(x01: torch.Tensor, *, seed: int = 20259,
                   octaves: int = 6, low: float = 0.05,
                   high: float = 0.95) -> torch.Tensor:
    """與 `x01` 同形狀的程序化高頻紋理，當 targeted 項的目標。

    為什麼中性灰是錯的目標
    ────────────────────────────────────────────────────────────────
    `enc_target` 拉的是 VAE latent，**靜態編碼**，拉到哪個點都算數，所以中性灰
    不害它。`diffusion_target` 拉的是**去噪行為**：中性灰在任何時刻的 ε 預測都
    低變異、接近平庸，把行為拉向平庸等於要求模型**更容易**還原這張圖，與
    「讓模型還原不了」正好相反。Mist 原論文的目標是一張高頻紋理圖。

    構造
    ────────────────────────────────────────────────────────────────
    `octaves` 層均勻雜訊，第 `o` 層的區塊邊長是 `2**o`，以最近鄰放大到原尺寸後
    等權相加，再逐通道 min-max 正規化到 `[low, high]`。三個通道各自獨立抽，
    所以色度也帶高頻。最細的一層是逐像素雜訊，最粗的一層在 VAE 下採樣之後仍
    看得見，因此潛空間與像素空間兩邊都有結構。

    三個性質是刻意的：**可重現**（固定種子、不讀外部素材）、**不帶身分**
    （沒有任何人臉或物件結構，不會把防禦圖拉向某個具體的人）、**不隨影像內容
    變**（只用到形狀）。端點留 `low`／`high` 的邊距，避免目標貼在 VAE 的
    飽和端上。
    """
    h, w = int(x01.shape[-2]), int(x01.shape[-1])
    gen = torch.Generator().manual_seed(int(seed))
    acc = torch.zeros((1, 3, h, w), dtype=torch.float64)
    for o in range(int(octaves)):
        s = 1 << o
        gh, gw = -(-h // s), -(-w // s)
        n = torch.rand((1, 3, gh, gw), generator=gen, dtype=torch.float64)
        up = n.repeat_interleave(s, -2).repeat_interleave(s, -1)
        acc = acc + up[..., :h, :w]
    lo = acc.amin(dim=(-2, -1), keepdim=True)
    hi = acc.amax(dim=(-2, -1), keepdim=True)
    unit = (acc - lo) / (hi - lo).clamp_min(1e-9)
    out = float(low) + (float(high) - float(low)) * unit
    return out.to(device=x01.device, dtype=x01.dtype)


TARGETS = {'grey': grey_target, 'texture': texture_target}


def build_target(name: str, x01: torch.Tensor, **options) -> torch.Tensor:
    """依名字造一張目標影像。名字不認得就拋錯，不要靜默退回中性灰。"""
    key = str(name)
    if key not in TARGETS:
        raise ValueError(f'不認得的目標 {key!r}，可用的是 {sorted(TARGETS)}')
    return TARGETS[key](x01, **options)


class MainstreamTerms:
    """掛在 `FreeObjective` 旁邊的兩個項。共用同一組噪聲、時刻與排程器。"""

    def __init__(self, objective, x01, *, target: torch.Tensor = None,
                 use_ckpt: bool = True, enc_target_norm: str = 'target'):
        """`enc_target_norm` 決定 `enc_target` 用什麼當分母。

        `'target'`（原行為）：`‖z − z_target‖ / ‖z_target‖`。分母跟著目標換，
        所以**換一張目標就換一個量綱**，中性灰與高頻紋理的數值不可並列。
        實測中性灰上收在 1.26–1.29、紋理上收在 1.69–1.97，而這一項是以
        `+w·tanh(·)` 進分數的——`tanh′` 在 1.7 只剩 0.12，**87% 的梯度被吃掉**，
        紋理目標把它推得更深。

        `'start'`：`‖z − z_target‖ / ‖z_x − z_target‖`，分母是**原圖到目標的
        距離**。起點恆為 1.0、與目標的尺度無關，`tanh′` 從 0.42 出發而且隨著
        靠近目標往 1 上升。兩張目標因此落在同一個量綱上，可以並列。
        """
        if enc_target_norm not in ('target', 'start'):
            raise ValueError(
                f"enc_target_norm 只能是 'target' 或 'start'，"
                f"收到 {enc_target_norm!r}")
        self.obj = objective
        self.ip2p = objective.ip2p
        self.use_ckpt = use_ckpt
        self.enc_target_norm = enc_target_norm
        with torch.no_grad():
            tgt = grey_target(x01) if target is None else target
            self._tgt01 = tgt
            self._tgt_eps = None
            self._tgt_draw = None
            self.z_target = self.ip2p.encode_image(tgt).float()
            if enc_target_norm == 'start':
                z_x = self.ip2p.encode_image(x01).float()
                self.enc_scale = (z_x - self.z_target).flatten().norm()
            else:
                self.enc_scale = self.z_target.flatten().norm()
            self.enc_scale = self.enc_scale.clamp_min(1e-6)
            self.z_target_norm = self.z_target.flatten().norm().clamp_min(1e-6)

    def _noised(self, z0, t):
        s = _sigma_at(self.obj.scheduler, t).to(z0)
        return (z0 + s * self.obj.noise.to(z0)) / (s * s + 1).sqrt()

    def enc_target(self, x_def: torch.Tensor) -> torch.Tensor:
        z = self.ip2p.encode_image(x_def).float()
        return (z - self.z_target).flatten().norm() / self.enc_scale

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

    def _target_eps(self, tgt01: torch.Tensor) -> Dict[int, torch.Tensor]:
        """目標影像在**當前抽樣**的每個時刻上的 ε 預測。

        不能只算一次就永久快取：`FreeObjective` 的 `resample` 每一次求值都重抽
        時刻**與噪聲**，而 `_noised` 用的是 `self.obj.noise`。快取要跟著
        `obj.draws` 這個抽樣版本走，換抽樣就重算，否則舊時刻查不到（KeyError）、
        而查得到的那些也配在錯誤的噪聲上。代價是每次重抽多 k 次無梯度的
        UNet 呼叫。

        兩側都用**各自的** `z_t` 與各自的影像條件，噪聲與時刻共用
        `FreeObjective` 已經抽好的那一組。這與 `cond` 的構造不同：那一項固定
        `z_t` 只換條件，量的是「換了圖之後預測偏多少」；這一項要的是「模型在
        我的照片上的去噪行為變成它在目標上的行為」，所以 `z_t` 必須跟著各自
        的 latent 走，否則比較的是兩個不同的東西。
        """
        out = {}
        with torch.no_grad():
            z0 = self.ip2p.encode_image(tgt01)
            cond = self.ip2p.image_latents(tgt01).to(self.obj.text.dtype)
            for t in self.obj.timesteps:
                zt = self._noised(z0.to(self.obj.noise.dtype), t)
                model_input = torch.cat([zt, cond.to(zt.dtype)], dim=1)
                out[int(t)] = self.obj._run_unet(
                    model_input, t, self.obj.text).float()
        return out

    def diffusion_target(self, x_def: torch.Tensor) -> torch.Tensor:
        """Mist 的 targeted 版：把去噪行為拉到目標影像上，**越小越好**。

        無目標的 `diffusion` 只要求「模型還原不了」，梯度在任何方向上都拿得到
        分，會隨著離開原點而轉向；targeted 版整條路徑指向同一個點。文獻上
        targeted 一致比 untargeted 強，本專案在 encoder 那一側也量到同一個形狀
        （`enc_target` 的 PhotoGuard 臂贏過無目標的 `enc`）。

        目標沿用 `grey_target`：中性灰不對應任何身分，不會把一個具體的人寫進
        目標裡。
        """
        if self._tgt_eps is None or self._tgt_draw != self.obj.draws:
            self._tgt_eps = self._target_eps(self._tgt01)
            self._tgt_draw = self.obj.draws
        z0 = self.ip2p.encode_image(x_def)
        cond = self.ip2p.image_latents(x_def).to(self.obj.text.dtype)
        losses = []
        for t in self.obj.timesteps:
            zt = self._noised(z0.to(self.obj.noise.dtype), t)
            model_input = torch.cat([zt, cond.to(zt.dtype)], dim=1)
            if self.use_ckpt:
                eps = ckpt.checkpoint(self.obj._run_unet, model_input, t,
                                      self.obj.text, use_reentrant=False)
            else:
                eps = self.obj._run_unet(model_input, t, self.obj.text)
            ref = self._tgt_eps[int(t)]
            losses.append(((eps.float() - ref) ** 2).mean()
                          / ref.pow(2).mean().clamp_min(1e-9))
        return torch.stack(losses).mean()

    def terms(self, x_def: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {'enc_target': self.enc_target(x_def),
                'diffusion': self.diffusion(x_def),
                'diffusion_target': self.diffusion_target(x_def)}


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

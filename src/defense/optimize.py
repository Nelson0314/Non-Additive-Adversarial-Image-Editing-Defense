"""逐圖優化 φ — spec §5、§7。

每張影像獨立優化一組 φ。防禦者擁有原圖與完整權重（spec §2 威脅模型），
故逐圖優化符合威脅模型，不需泛化到未見影像。

**成本模型（E0 實測，V100 32GB, SD v1-4, 512², fp32, UNet+VAE checkpoint）**：

    seconds ≈ 1.05 + 0.384·k_inv + 0.304·n_edit·n_eot
    peak    ≈ 9.95 GB，於 k_inv、n_edit ∈ [5,50] 幾乎不變

記憶體與步數無關是因為兩條 UNet 鏈與三次 VAE 呼叫都已 checkpoint；時間則
與步數線性相關。故 n_eot 直接乘在時間上，是本迴圈最貴的旋鈕。
"""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch

from src.defense.generator import DefenseGenerator
from src.defense.objective import DefenseObjective, LossConfig
from src.models.sd import SDWrapper
from src.purify.ops import Purifier
from src.residual.base import ResidualModule


@dataclass
class OptimConfig:
    steps: int = 100
    lr: float = 0.05
    k_inv: int = 10
    t_max: Optional[int] = None   # inversion 的 timestep 上限，見 E0c
    n_edit: int = 10
    n_eot: int = 1              # 每步的 (淨化, 噪聲) 取樣數
    strength: float = 0.5
    prompt_def: str = ""        # 防禦生成的 prompt，spec §1.1 要求也測空 prompt
    prompt_edit: str = "a photo"
    seed: int = 20260728
    unet_ckpt: bool = True      # E0：512² 下關閉必 OOM，故預設開啟
    vae_ckpt: bool = True       # E0b：三次 VAE 呼叫的激活由總和降為最大值
    log_every: int = 10
    grad_clip: float = 1.0


@dataclass
class OptimResult:
    history: List[Dict] = field(default_factory=list)
    x_def: Optional[torch.Tensor] = None
    x0_trace: List[torch.Tensor] = field(default_factory=list)
    seconds: float = 0.0
    steps_done: int = 0


def optimize(
    sd: SDWrapper,
    module: ResidualModule,
    x01: torch.Tensor,
    cfg: OptimConfig,
    loss_cfg: LossConfig,
    purifiers: List[Purifier],
) -> OptimResult:
    """優化 φ 使編輯偏移最大化，同時維持 x_def 與 x 的保真。

    `y_orig` 對 φ 為常數（spec §5.1），故對每個噪聲取樣預先算好並快取；
    這省下每步一條 n_edit 長度的無梯度 UNet 鏈。
    """
    device = x01.device
    gen = DefenseGenerator(sd, module, k_inv=cfg.k_inv, t_max=cfg.t_max)
    obj = DefenseObjective(loss_cfg, device)
    opt = torch.optim.Adam(module.parameters(), lr=cfg.lr)

    lat = sd.latent_shape(x01.shape[-2], x01.shape[-1])
    emb_edit = sd.encode_text(cfg.prompt_edit).detach()

    # 每個 EOT 取樣配一組固定噪聲。噪聲跨 step 固定而非每步重抽：目標是
    # 對「這組噪聲下的編輯」造成偏移，每步換噪聲會讓梯度訊號被取樣噪音淹沒。
    noises = [
        sd.sample_edit_noise(torch.empty(lat, device=device), seed=cfg.seed + i)
        for i in range(cfg.n_eot)
    ]

    # y_orig 不依賴 φ，整個優化過程只算一次
    with torch.no_grad():
        y_origs = [
            sd.sdedit(x01, emb_edit, n, cfg.n_edit, strength=cfg.strength)
            for n in noises
        ]

    result = OptimResult()
    t0 = time.perf_counter()

    for step in range(cfg.steps):
        opt.zero_grad(set_to_none=True)

        ctx = gen.prepare(x01, prompt_def=cfg.prompt_def)
        x_def = gen.generate(
            x01, ctx, use_ckpt=cfg.unet_ckpt, vae_ckpt=cfg.vae_ckpt,
            collect_x0=(step == cfg.steps - 1),
        )

        y_defs, y_refs = [], []
        for i in range(cfg.n_eot):
            # 淨化算子輪替取樣，即 spec §5.1 對 𝒫 的期望值估計
            p = purifiers[(step * cfg.n_eot + i) % len(purifiers)]
            y_defs.append(
                sd.sdedit(
                    p.forward(x_def), emb_edit, noises[i], cfg.n_edit,
                    strength=cfg.strength,
                    use_ckpt=cfg.unet_ckpt, vae_ckpt=cfg.vae_ckpt,
                )
            )
            y_refs.append(y_origs[i])

        total, log = obj(x_def, x01, y_defs, y_refs)
        total.backward()

        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(module.parameters(), cfg.grad_clip)
        opt.step()

        log["step"] = step
        log["grad_norm"] = float(
            torch.stack(
                [p.grad.norm() for p in module.parameters() if p.grad is not None]
            ).norm()
        )
        result.history.append(log)

        if step % cfg.log_every == 0 or step == cfg.steps - 1:
            print(
                f"  step {step:>4d}  loss={log['loss']:.4f}  "
                f"L_def={log['L_def']:.4f}  L_fid={log['L_fid']:.4f}  "
                f"shift={log['edit_shift']:.4f}  psnr={log['fid_psnr']:.2f}  "
                f"linf={log['fid_linf']:.4f}  |g|={log['grad_norm']:.3e}",
                flush=True,
            )

        if step == cfg.steps - 1:
            result.x_def = x_def.detach().clone()
            result.x0_trace = [t.detach().clone() for t in ctx.x0_trace]

        del x_def, y_defs, total
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    result.seconds = time.perf_counter() - t0
    result.steps_done = cfg.steps
    return result

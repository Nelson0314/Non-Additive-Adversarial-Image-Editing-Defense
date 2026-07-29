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
    n_eot: int = 1              # 每步的噪聲取樣數
    # 淨化算子的取樣方式：
    #   "rotate" — 每步只用一個算子，逐步輪替（原始行為）
    #   "all"    — 每步對全部算子求梯度後平均，即 spec §5.1 對 𝒫 的期望值
    #              以完整列舉估計，而非以輪替近似
    # 改用 "all" 的理由：主網格中模糊 σ=1.0 就在訓練集裡，測試時防禦在該
    # 條件下卻只保留 4.4%。輪替下 Adam 的動量在三個目標之間來回拉扯，等於
    # 三個互相衝突的梯度訊號輪流覆寫，而非求其共同下降方向；25 步分給三個
    # 算子，每個只有約 8 次更新。成本為每步時間乘以算子數。
    purify_mode: str = "rotate"
    strength: float = 0.5
    prompt_def: str = ""        # 防禦生成的 prompt，spec §1.1 要求也測空 prompt
    prompt_edit: str = "a photo"
    seed: int = 20260728
    unet_ckpt: bool = True      # E0：512² 下關閉必 OOM，故預設開啟
    vae_ckpt: bool = True       # E0b：三次 VAE 呼叫的激活由總和降為最大值
    log_every: int = 10
    grad_clip: float = 1.0


def eot_pairs(mode: str, step: int, n_eot: int, n_purifiers: int) -> List[tuple]:
    """該步要評估的 (淨化算子索引, 噪聲索引) 清單。

    抽成純函數是為了能在沒有 SD 模型的情況下驗證取樣邏輯——這段決定了
    梯度訊號的構成，是 spec §5.1 期望值估計的實作，不該只能靠跑完整實驗
    才看得出對錯。
    """
    if mode == "all":
        return [(pi, i) for pi in range(n_purifiers) for i in range(n_eot)]
    if mode == "rotate":
        return [((step * n_eot + i) % n_purifiers, i) for i in range(n_eot)]
    raise ValueError(f"未知的 purify_mode: {mode!r}，只接受 'rotate' 或 'all'")


@dataclass
class OptimResult:
    history: List[Dict] = field(default_factory=list)
    x_def: Optional[torch.Tensor] = None
    x_base: Optional[torch.Tensor] = None   # G(x; φ=0)，該 site 的保真地板
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

        # x_base = G(x; φ=0)，即該 site 未施加防禦時就已產生的圖。site P 為
        # x 本身；site L 為 inversion + VAE 來回的重建。用於把 L∞ hinge 的
        # 對象限定在「φ 造成的改變」（見 objective.fidelity_term）。停用模塊
        # 即可取得，之後還原原本的啟用狀態。
        was_enabled = module.enabled
        module.disable()
        try:
            ctx0 = gen.prepare(x01, prompt_def=cfg.prompt_def)
            x_base = gen.generate(x01, ctx0).detach()
        finally:
            if was_enabled:
                module.enable()

    result = OptimResult()
    t0 = time.perf_counter()

    for step in range(cfg.steps):
        opt.zero_grad(set_to_none=True)

        ctx = gen.prepare(x01, prompt_def=cfg.prompt_def)
        x_def = gen.generate(
            x01, ctx, use_ckpt=cfg.unet_ckpt, vae_ckpt=cfg.vae_ckpt,
            collect_x0=(step == cfg.steps - 1),
        )

        # (淨化算子索引, 噪聲索引) 的取樣清單。defense_term 對清單取平均，
        # 故 "all" 模式下一次 backward 得到的就是全算子的平均梯度，而不是
        # 輪替下「這一步只朝某一個算子下降」的訊號。
        pairs = eot_pairs(cfg.purify_mode, step, cfg.n_eot, len(purifiers))

        y_defs, y_refs = [], []
        for pi, i in pairs:
            p = purifiers[pi]
            y_defs.append(
                sd.sdedit(
                    p.forward(x_def), emb_edit, noises[i], cfg.n_edit,
                    strength=cfg.strength,
                    use_ckpt=cfg.unet_ckpt, vae_ckpt=cfg.vae_ckpt,
                )
            )
            y_refs.append(y_origs[i])

        total, log = obj(x_def, x01, y_defs, y_refs, x_base=x_base)
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

    result.x_base = x_base
    result.seconds = time.perf_counter() - t0
    result.steps_done = cfg.steps
    return result

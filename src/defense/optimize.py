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
from dataclasses import dataclass, field, replace
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
    # True 改用 BDIA 精確反演（arXiv 2307.10829）取代 DDIM。tiny-SD 實測
    # latent 空間來回誤差由 1.41 降到 1.37e-04（k=20, t_max=500）。
    # **影像空間的地板不會歸零**：VAE 編解碼來回誤差（真實 SD 實測
    # 27.51 dB / LPIPS 0.143）與反演無關，BDIA 不觸及該項。預期效果是把
    # 重建地板由 LPIPS 0.194 降到 0.143，仍高於像素側加性位置實際運作的
    # 0.063，故此旗標是量測反演佔地板多少的工具，不是解除封鎖的萬靈丹。
    exact_inversion: bool = False
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

    # ---- 階段一：保真對齊 ----
    # 先訓練 φ 使 G(x; φ) 逼近 x，再以該 φ 熱啟動防禦訓練。
    #
    # 借鑑 APA（arXiv 2506.01511）「先立保真、再訓攻擊」的順序，其餘不同：
    # APA 在 UNet 另掛一組 LoRA、以噪聲空間的 ‖ε − ε̂‖² 訓練、階段二凍結它；
    # 此處不加任何新模組，就用同一組 φ，損失直接是影像空間的 L_fid(G(x;φ), x)。
    # 我們的 G 是固定長度的可微 DDIM 鏈，能端對端反傳到輸出影像本身，不需要
    # 噪聲空間的代理量。兩階段之間只重置 Adam 狀態，不換參數集。
    #
    # 動機是實測結果：latent 注入的 φ=0 對照與訓練 25 步的結果在每個淨化條件
    # 下都相同到小數點後三位，而該位置的重建誤差比 φ 的效果大 20–36 倍。
    # 若階段一能把重建誤差吸收掉，階段二的注入才有可能不被淹沒。
    #
    # align_steps = 0 表示不執行階段一（既有行為，保留可比性）。
    align_steps: int = 0
    align_lr: float = 0.008
    # 階段一專用的 PSNR 係數，覆蓋 LossConfig.gamma_psnr（防禦階段為 0.0）。
    # 理由見 align() 的 docstring：重建對齊是逐像素準確度確實重要的場合。
    align_gamma_psnr: float = 1.0
    strength: float = 0.5
    # site S 專用：位移場的硬上界，單位為**像素**。空間變形的保真度預算是
    # 位移量而非 L∞（把一條邊緣移動一像素，L∞ 可接近 1.0 卻幾乎看不出來），
    # 故此值必須與 tau_lpips 併列記錄，才能說清楚該格的失真預算是什麼。
    warp_max_disp: float = 1.5
    # site S 的 grid_sample 插值模式。預設維持 bilinear 使 E13–E19 可重現；
    # E20 §5.2 量出 bicubic 可把銳利度保留率由 85.0% 拉到 99.9%。
    warp_resample: str = "bilinear"
    # site C 專用：色度矩陣場偏離單位陣的硬上界。與 warp_max_disp 同一角色
    # ——本位置的保真度預算是矩陣偏離量而非 L∞，故必須與 tau_lpips 併列記錄。
    color_max_dev: float = 0.15
    # ---- cross-attention 目標專用 ----
    # "divergence" — 把防禦圖的注意力分佈推離原圖的（改變綁定的指向）
    # "entropy"    — 直接把分佈推向均勻（瓦解綁定本身，不需要參考分佈）
    # "suppress"   — 降低內容 token 分到的注意力質量（需要 attn_content_only）
    #
    # **預設由 "divergence" 改為 "suppress"（2026-08-01，E25）。** divergence
    # 在 φ=0 的梯度精確為零（KL 在最小值處），最佳化永遠離不開起點，拿它去
    # 跑會安靜地什麼都不做（E20 §9 實測 grad_norm = 0.000e+00）。留著它是為了
    # 讓該缺陷有具名的位置與釘住它的測試，但它不該是預設。suppress 的最佳點
    # 不在 φ=0，故起步梯度非零，見 src/models/attention.py 的同名函式。
    attn_mode: str = "suppress"
    # 每步取樣幾個 timestep。cross-attention 的綁定強度隨 t 變化，只在單一
    # t 上施力等於只防住編輯鏈的其中一步。取樣點均分於 [0, t_edit]，
    # t_edit 為 SDEdit 在該 strength 下的起始 timestep。
    attn_timesteps: int = 4
    # 只作用在 prompt 的內容 token 上（排除 BOS/EOS/padding），見 token_span
    attn_content_only: bool = True
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
    # 防禦訓練所用的保真基準。未執行階段一時為 G(x; φ=0)；執行後為
    # G(x; φ_align)，即階段一結束時該位置實際能做到的重建。
    x_base: Optional[torch.Tensor] = None
    x_base0: Optional[torch.Tensor] = None  # 恆為 G(x; φ=0)，作為對照保留
    x0_trace: List[torch.Tensor] = field(default_factory=list)
    align_history: List[Dict] = field(default_factory=list)
    align_seconds: float = 0.0
    seconds: float = 0.0
    steps_done: int = 0


def align(
    sd: SDWrapper,
    module: ResidualModule,
    x01: torch.Tensor,
    cfg: OptimConfig,
    loss_cfg: LossConfig,
    gen: DefenseGenerator,
) -> tuple[torch.Tensor, List[Dict]]:
    """階段一：訓練 φ 使 G(x; φ) 逼近 x。回傳 (x_align, history)。

    損失以 `fidelity_term(G(x;φ), x, x_base=None)` 為基礎，即對原圖的絕對
    保真度。`x_base=None` 使兩道 hinge 的對象為原圖本身；hinge 在容差內
    不施力是此處要的行為——對齊到 τ 以內即停止，不必把重建誤差壓到零。

    **但係數與階段二不同：`gamma_psnr` 由 `cfg.align_gamma_psnr` 覆蓋。**
    防禦階段把 PSNR 移出梯度是對的（逐像素平方誤差與人眼可辨性關聯薄弱，
    見 objective 的修訂之二），但**重建對齊正是逐像素準確度確實重要的場合**，
    同一組係數不該同時適用兩者。

    這是 E9 直接量到的問題：200 步對齊後 car_00 的 LPIPS 由 0.2032 降到
    0.0950，PSNR 卻只由 19.62 動到 20.54；car_01 的 PSNR 甚至掉了 1.84 dB
    （22.28 → 20.44）。PSNR 當時完全不在損失裡，那些數字反映的是自由漂移，
    不是容量限制，因此無法用來判斷任何事。

    **這個階段可能失敗**，而失敗本身是結果：低秩 ε 注入未必有足夠容量吸收
    VAE 與 DDIM 的重建誤差。E9 實測 car_01 在第 50 步即停在 LPIPS 0.166，
    其後 150 步無改善——那是容量天花板，不是步數不足。回傳的 history 記錄
    逐步的 LPIPS 與 PSNR，呼叫端據此判斷，不得假設對齊必然成功。
    """
    # 只覆蓋 gamma_psnr，其餘係數與階段二一致：保真度的「定義」不變，
    # 變的是逐像素項在這個階段要不要參與梯度。
    align_cfg = replace(loss_cfg, gamma_psnr=cfg.align_gamma_psnr)
    obj = DefenseObjective(align_cfg, x01.device)
    opt = torch.optim.Adam(module.parameters(), lr=cfg.align_lr)
    history: List[Dict] = []
    best_loss, best_step, best_state = float("inf"), -1, None

    for step in range(cfg.align_steps):
        opt.zero_grad(set_to_none=True)
        ctx = gen.prepare(x01, prompt_def=cfg.prompt_def)
        x_gen = gen.generate(x01, ctx, use_ckpt=cfg.unet_ckpt,
                             vae_ckpt=cfg.vae_ckpt)
        loss, parts = obj.fidelity_term(x_gen, x01, x_base=None)
        loss.backward()
        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(module.parameters(), cfg.grad_clip)
        opt.step()

        # 保留軌跡最佳的 φ，最後還原它，而不是拿最後一步的。
        #
        # E12 實測 8 個 (載體 × 影像) 組合中有 7 個的最後一步**比自己的最佳
        # 值差**，且劣化幅度隨參數量遞增：site L（163,840 參數）+0.0316、
        # site W r=4（397,824）+0.0404、site W r=16（1,591,296）+0.0526。
        # 拿最後一步等於系統性低報每個載體的能力，而且低報的程度與參數量
        # 相關——那會讓「容量」與「優化穩定性」兩個變因無法分離。
        #
        # 選擇判準用總損失而非單一 LPIPS：損失才是這個階段在最小化的量，
        # 挑 LPIPS 最低的一步可能挑到 PSNR 被犧牲掉的那一步。
        if float(loss) < best_loss:
            best_loss, best_step = float(loss), step
            best_state = {k: v.detach().clone()
                          for k, v in module.state_dict().items()}

        parts["step"] = step
        parts["align_loss"] = float(loss)
        history.append(parts)
        if step % cfg.log_every == 0 or step == cfg.align_steps - 1:
            print(f"  [align] step {step:>4d}  loss={float(loss):.4f}  "
                  f"lpips={parts['fid_lpips']:.4f}  "
                  f"psnr={parts['fid_psnr_total']:.2f}", flush=True)

        del x_gen, loss
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if best_state is not None and best_step != cfg.align_steps - 1:
        module.load_state_dict(best_state)
        print(f"  [align] 還原第 {best_step} 步的 φ（loss={best_loss:.4f}）；"
              f"最後一步為 {history[-1]['align_loss']:.4f}", flush=True)
    for h in history:
        h["align_best_step"] = best_step

    with torch.no_grad():
        ctx = gen.prepare(x01, prompt_def=cfg.prompt_def)
        x_align = gen.generate(x01, ctx).detach()
    return x_align, history


def optimize(
    sd: SDWrapper,
    module: ResidualModule,
    x01: torch.Tensor,
    cfg: OptimConfig,
    loss_cfg: LossConfig,
    purifiers: List[Purifier],
    y_target: Optional[torch.Tensor] = None,
) -> OptimResult:
    """優化 φ 使編輯偏移最大化，同時維持 x_def 與 x 的保真。

    `y_target` 僅在 `loss_cfg.defense_mode == "targeted"` 時使用，為編輯
    結果要被推向的固定目標影像。無目標模式下傳入會被忽略。

    `y_orig` 對 φ 為常數（spec §5.1），故對每個噪聲取樣預先算好並快取；
    這省下每步一條 n_edit 長度的無梯度 UNet 鏈。
    """
    device = x01.device
    gen = DefenseGenerator(sd, module, k_inv=cfg.k_inv, t_max=cfg.t_max,
                          exact_inversion=cfg.exact_inversion)
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
            x_base0 = gen.generate(x01, ctx0).detach()
        finally:
            if was_enabled:
                module.enable()

    result = OptimResult(x_base0=x_base0)
    x_base = x_base0

    # ---- 階段一：保真對齊 ----
    if cfg.align_steps > 0:
        if torch.equal(x_base0, x01):
            # G(x; φ=0) 已逐元素等於原圖，沒有重建誤差可吸收。以數值判定而非
            # 依 site 名稱分支：判準是「這個位置有沒有重建誤差」，不是「這是
            # 哪個 site」，將來新增位置不必改這裡。
            print("  [align] G(x; φ=0) 已逐元素等於 x，略過階段一", flush=True)
        else:
            ta = time.perf_counter()
            x_base, result.align_history = align(
                sd, module, x01, cfg, loss_cfg, gen)
            result.align_seconds = time.perf_counter() - ta
            # 保真基準改為階段一實際達成的重建。階段一若成功，x_base ≈ x，
            # 相對 hinge 與絕對 hinge 自然合流；若失敗，x_base 仍是誠實的
            # 基準，不會把階段一沒解決的重建誤差算到防禦頭上。
            opt = torch.optim.Adam(module.parameters(), lr=cfg.lr)

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

        total, log = obj(x_def, x01, y_defs, y_refs,
                         x_base=x_base, y_target=y_target)
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


def optimize_encoder(
    sd: SDWrapper,
    module: ResidualModule,
    x01: torch.Tensor,
    cfg: OptimConfig,
    loss_cfg: LossConfig,
    purifiers: List[Purifier],
    z_target: Optional[torch.Tensor] = None,
) -> OptimResult:
    """VAE 編碼器目標（PhotoGuard 的 encoder attack 形式）。

        min_φ  ‖E_vae(P(x_def)) − z_target‖²  +  λ_fid · L_fid(x_def, x)

    **與 optimize() 是不同的方法，不是它的一個選項**，故獨立成一個函式：
    這裡完全沒有 SDEdit、沒有 y_orig、沒有編輯 prompt，共用同一個迴圈只會
    讓兩者的差異被埋在條件分支裡。

    成本差距來自 E0 的成本模型 `秒 ≈ 1.05 + 0.384·k_inv + 0.304·n_edit`：
    此處 `n_edit` 與 `k_inv` 兩項都不存在，每步只剩一次 VAE 編碼。1000 步
    的成本仍低於 optimize() 的 25 步。

    同時消除本專案最大的兩個過擬合來源——目標不依賴任何特定的編輯 prompt
    或噪聲取樣（實測噪聲過擬合 3.3 倍，prompt 過擬合從未量過）。代價是它
    不再針對「這個編輯」最佳化，是泛化性換特異性的取捨，須實測而非假設。

    `z_target` 預設為零張量。零是 latent 空間的一個退化點，把 x_def 推向
    它等同要求 VAE 把防禦圖看成「沒有內容」，這是 PhotoGuard encoder attack
    的常見選擇；呼叫端可傳入其他目標。
    """
    device = x01.device
    gen = DefenseGenerator(sd, module, k_inv=cfg.k_inv, t_max=cfg.t_max,
                          exact_inversion=cfg.exact_inversion)
    obj = DefenseObjective(loss_cfg, device)
    opt = torch.optim.Adam(module.parameters(), lr=cfg.lr)

    with torch.no_grad():
        was_enabled = module.enabled
        module.disable()
        try:
            x_base0 = gen.generate(x01, gen.prepare(x01, prompt_def=cfg.prompt_def))
            x_base0 = x_base0.detach()
        finally:
            if was_enabled:
                module.enable()
        if z_target is None:
            z_target = torch.zeros_like(sd.encode_image(x01))

    result = OptimResult(x_base0=x_base0, x_base=x_base0)
    t0 = time.perf_counter()

    for step in range(cfg.steps):
        opt.zero_grad(set_to_none=True)
        ctx = gen.prepare(x01, prompt_def=cfg.prompt_def)
        x_def = gen.generate(x01, ctx, use_ckpt=cfg.unet_ckpt,
                             vae_ckpt=cfg.vae_ckpt)

        # 淨化仍以 EOT 取樣：編碼器目標不會自動帶來耐淨化性，那是兩件事
        pairs = eot_pairs(cfg.purify_mode, step, 1, len(purifiers))
        z_defs = [sd.encode_image(purifiers[pi].forward(x_def),
                                  use_ckpt=cfg.vae_ckpt)
                  for pi, _ in pairs]
        l_def = torch.stack([obj.encoder_term(z, z_target) for z in z_defs]).mean()
        l_fid, parts = obj.fidelity_term(x_def, x01, x_base=x_base0)
        total = loss_cfg.lam_def * l_def + loss_cfg.lam_fid * l_fid
        total.backward()

        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(module.parameters(), cfg.grad_clip)
        opt.step()

        log = {"step": step, "loss": float(total), "L_def": float(l_def),
               "L_fid": float(l_fid), "edit_shift": float("nan"),
               "defense_mode": "encoder", **parts}
        log["grad_norm"] = float(torch.stack(
            [p.grad.norm() for p in module.parameters() if p.grad is not None]
        ).norm())
        result.history.append(log)

        if step % cfg.log_every == 0 or step == cfg.steps - 1:
            print(f"  [enc] step {step:>4d}  loss={float(total):.4f}  "
                  f"L_def={float(l_def):.4f}  psnr={parts['fid_psnr_total']:.2f}  "
                  f"|g|={log['grad_norm']:.3e}", flush=True)

        if step == cfg.steps - 1:
            result.x_def = x_def.detach().clone()

        del x_def, z_defs, total
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    result.seconds = time.perf_counter() - t0
    result.steps_done = cfg.steps
    return result


def optimize_crossattn(
    sd: SDWrapper,
    module: ResidualModule,
    x01: torch.Tensor,
    cfg: OptimConfig,
    loss_cfg: LossConfig,
    purifiers: List[Purifier],
) -> OptimResult:
    """cross-attention 目標：破壞 prompt token 與影像位置之間的綁定。

        max_φ  E_t[ D( A(P(x_def), c, t), A(x, c, t) ) ]  −  λ_fid · L_fid

    **著力點與 optimize() 不同。** optimize() 在輸出端量測「編輯結果被推開
    多少」；此處直接作用在使文字編輯得以定位的機制上——UNet 的 cross-attention
    把每個 token 綁到影像的特定區域，綁定被破壞則編輯無從落點。
    （Xu et al., arXiv 2509.10359, ACM MM 2025 採取相同的著力點。）

    **只做單步 UNet 前向，不走完整的 SDEdit 鏈。** 綁定是逐 timestep 的性質，
    在取樣到的 t 上把它破壞掉即可，不需要把整條 n_edit 步的鏈跑完。成本因此
    由 `0.304·n_edit`（10 步）降為 `attn_timesteps` 次單步前向。

    **這條前向不能開 UNet checkpoint。** 實測（tiny-SD）：開了之後 backward
    以 RuntimeError 中止，訊息為

        A different number of tensors was saved during the original forward
        and recomputation. Number of tensors saved during forward: 477
        Number of tensors saved during recomputation: 459.

    原因是 hook 在原前向時是掛著的，它額外算的 to_q / to_k / QKᵀ 進了
    checkpoint 區塊的圖；而 backward 觸發重算時 context manager 早已離開、
    hook 已卸除，重算的圖少了那一段，兩次存檔的張量數對不上。此處以
    `sd._eps(...)` 不傳 use_ckpt（預設 False）迴避，單步前向不開 checkpoint
    的記憶體是可負擔的——真正需要 checkpoint 的是 n_edit 步串接的那條鏈。

    參考分佈 A(x, c, t) 不依賴 φ，故對每個取樣到的 t 只算一次並快取。
    兩條分支共用同一組加噪 ε（spec §5.1），否則量到的差異主要來自噪聲不同。
    """
    from src.models.attention import (
        CrossAttentionRecorder, attention_content_suppression,
        attention_divergence, attention_entropy, token_span,
    )

    device = x01.device
    gen = DefenseGenerator(sd, module, k_inv=cfg.k_inv, t_max=cfg.t_max,
                          exact_inversion=cfg.exact_inversion)
    obj = DefenseObjective(loss_cfg, device)
    opt = torch.optim.Adam(module.parameters(), lr=cfg.lr)
    rec = CrossAttentionRecorder(sd.unet)

    emb_edit = sd.encode_text(cfg.prompt_edit).detach()
    span = token_span(sd.tokenizer, cfg.prompt_edit) if cfg.attn_content_only else None
    if span is not None and span[1] <= span[0]:
        # 空 prompt 沒有內容 token；對空區間取平均會得到 nan，須明確落回全域
        print("  [attn] prompt 無內容 token，改用全部 77 格", flush=True)
        span = None
    if cfg.attn_mode == "suppress" and span is None:
        # 落回全域對 suppress 不是「比較粗糙」而是「恆等於常數 0」：全部 77 格
        # 的注意力質量和恆為 1。此處提前拒絕，而不是讓它跑完才發現什麼都沒動。
        raise ValueError(
            "attn_mode='suppress' 需要非空的內容 token 區間，"
            f"但 attn_content_only={cfg.attn_content_only}、"
            f"prompt_edit={cfg.prompt_edit!r} 得到的 span 為空。"
            "全部 token 的注意力質量和恆為 1，該目標會退化成常數"
        )

    # 取樣 timestep：均分於 [0, t_edit]，即 SDEdit 在該 strength 下實際走過
    # 的區間。超出該區間的 t 對攻擊者的編輯不起作用，在那裡施力是浪費預算。
    t_edit = min(int(sd.num_train_timesteps * cfg.strength),
                 sd.num_train_timesteps - 1)
    t_list = torch.linspace(0, t_edit, cfg.attn_timesteps + 1)[1:].round().long()
    abar = sd.alphas_cumprod(device)

    lat = sd.latent_shape(x01.shape[-2], x01.shape[-1])
    noises = [
        sd.sample_edit_noise(torch.empty(lat, device=device), seed=cfg.seed + i)
        for i in range(len(t_list))
    ]

    with torch.no_grad():
        was_enabled = module.enabled
        module.disable()
        try:
            x_base0 = gen.generate(
                x01, gen.prepare(x01, prompt_def=cfg.prompt_def)).detach()
        finally:
            if was_enabled:
                module.enable()

        # 參考分佈：原圖在同一組 (t, ε) 下的注意力
        z_orig = sd.encode_image(x01)
        ref_maps = []
        for t, n in zip(t_list, noises):
            zt = abar[t].sqrt() * z_orig + (1 - abar[t]).sqrt() * n
            with rec:
                sd._eps(zt, t, emb_edit)
            ref_maps.append([m.detach() for m in rec.maps])

    result = OptimResult(x_base0=x_base0, x_base=x_base0)
    t0 = time.perf_counter()

    for step in range(cfg.steps):
        opt.zero_grad(set_to_none=True)
        ctx = gen.prepare(x01, prompt_def=cfg.prompt_def)
        x_def = gen.generate(x01, ctx, use_ckpt=cfg.unet_ckpt,
                             vae_ckpt=cfg.vae_ckpt)

        pairs = eot_pairs(cfg.purify_mode, step, len(t_list), len(purifiers))
        divs = []
        for pi, ti in pairs:
            z_def = sd.encode_image(purifiers[pi].forward(x_def),
                                    use_ckpt=cfg.vae_ckpt)
            t, n = t_list[ti], noises[ti]
            zt = abar[t].sqrt() * z_def + (1 - abar[t]).sqrt() * n
            with rec:
                sd._eps(zt, t, emb_edit)      # 見 docstring：此處不開 checkpoint
            if cfg.attn_mode == "entropy":
                divs.append(attention_entropy(rec.maps, span))
            elif cfg.attn_mode == "divergence":
                divs.append(attention_divergence(rec.maps, ref_maps[ti], span))
            elif cfg.attn_mode == "suppress":
                divs.append(attention_content_suppression(rec.maps, span))
            else:
                raise ValueError(
                    f"未知的 attn_mode: {cfg.attn_mode!r}，"
                    "只接受 'divergence'、'entropy' 或 'suppress'"
                )
            rec.clear()

        # 目標是把散度/熵推大，故取負號後最小化，與 defense_term 的符號一致
        l_def = -torch.stack(divs).mean()
        l_fid, parts = obj.fidelity_term(x_def, x01, x_base=x_base0)
        total = loss_cfg.lam_def * l_def + loss_cfg.lam_fid * l_fid
        total.backward()

        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(module.parameters(), cfg.grad_clip)
        opt.step()

        log = {"step": step, "loss": float(total), "L_def": float(l_def),
               "L_fid": float(l_fid), "edit_shift": float("nan"),
               "attn_div": float(-l_def), "defense_mode": "crossattn",
               "attn_mode": cfg.attn_mode, **parts}
        log["grad_norm"] = float(torch.stack(
            [p.grad.norm() for p in module.parameters() if p.grad is not None]
        ).norm())
        result.history.append(log)

        if step % cfg.log_every == 0 or step == cfg.steps - 1:
            # div 以科學記號輸出：實測其量級跨越數個數量級，定點格式在
            # 小的那一端全部印成 0.0000，等於沒有記錄
            print(f"  [attn] step {step:>4d}  loss={float(total):.4f}  "
                  f"div={-float(l_def):.4e}  lpips={parts['fid_lpips']:.4f}  "
                  f"psnr={parts['fid_psnr_total']:.2f}  "
                  f"|g|={log['grad_norm']:.3e}", flush=True)

        if step == cfg.steps - 1:
            result.x_def = x_def.detach().clone()

        del x_def, divs, total
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    result.seconds = time.perf_counter() - t0
    result.steps_done = cfg.steps
    return result

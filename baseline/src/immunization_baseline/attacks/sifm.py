"""SIFM —— Dong, Zhang, Zhao, Shan, Chen，arXiv:2512.14320v1（2025-12-16）。

*Semantic Mismatch and Perceptual Degradation: A New Perspective on Image
Editing Immunity*。逐條佐證見 `docs/reference/AUDIT_SIFM.md`。

**原作沒有公開程式。** 全文（含結論與參考文獻）沒有 repo 連結、沒有匿名
repo、沒有 implementation details 一節；`docs/reference/AUDIT_SIFM.md` §0 記
了查過哪些地方。因此本檔是**依論文正文與 Algorithm 1 重建**，不是官方路徑
的重現，`SPEC_PAPER.modified_from_paper` 為真。

論文怎麼定義
──────────────────────────────────────────────────────────────────────

中間特徵（§V 式 (3)）。`φ_t(x,c)` 是噪聲預測網路（U-Net 或 Diffusion
Transformer）內 M 個指定層的輸出，對層平均：

    φ_t(x,c) = (1/M) Σ_{j=1..M} L_j( E(x,t)_noisy , c , t )

其中 `E(x,t)_noisy` 是「把影像 x 取得其對應 t 的帶噪 latent」。

語意發散項（§V-A 式 (4)）：

    L_dist(δ;t) = Dist( φ_t(x₀+δ, c), φ_t(x₀, c) )

`Dist` 論文寫「a distance metric (e.g., MSE)」。

知覺劣化項（§V-B 式 (5)）：

    L_norm(δ;t) = ‖ φ_t(x₀+δ, c) ‖₁

合成（§V-C 式 (6)(7)）與 Algorithm 1 第 9、13 行：

    L_SIFM(δ;t)   = L_norm(δ;t) − λ · L_dist(δ;t)
    L_total(δ)    = (1/|T|) Σ_{t∈T} L_SIFM(δ;t)
    δ ← clip( δ − α·sign(∇ L_total), −ε, ε )
    x_imu ← clip_{0,1}( x_orig + δ )

即**最小化** L_total：壓低特徵的 L1 範數，同時推高與原圖特徵的距離。
`objective="minimize"` 與骨幹的 `x − step·sign(g)` 正是這一行。

預算與步數（§VII 首段）：「All methods were constrained to a perturbation
budget of ε=0.03 and limited to 100 optimization iterations.」
λ=0.1（§VII-C 表 VII，該節說明 0.1 是 ISR 最高的設定，全文其餘實驗都用它）。

值域
──────────────────────────────────────────────────────────────────────

Algorithm 1 第 14 行寫 `clip_{0,1}(x_orig + δ)`，故論文的最佳化值域就是
`[0,1]`，ε=0.03 不必換算（≈7.65/255）。這一點與 Mist／DIA／AdvPaint 三篇
（都在 `[-1,1]` 上量）不同，`pgd.py` 模組 docstring 的換算表即為此存在。

與論文對不上、或論文沒寫的地方
──────────────────────────────────────────────────────────────────────

1. **L1 還是 Frobenius。** §I 寫「SIFM minimizes the Frobenius norm of
   targeted intermediate features」，§V-B 與式 (5)、Algorithm 1 第 9 行寫
   L1（`‖·‖₁¹`），且 §V-B 整段的理由是 sparsity-inducing。**此處實作 L1**
   （可執行的式子優先於引言的敘述），落差記在 `discrepancy_note`。
2. **兩項的 reduction 不同一個尺度。** L_norm 是絕對值**和**（512² 影像、
   SD v1.x 的 mid_block 特徵有 1280×8×8 = 81,920 個元素），`Dist` 取 MSE 是
   **平均**。λ=0.1 的意義完全取決於這兩個 reduction，而論文沒有寫出
   `Dist` 的 reduction 也沒有寫 L_norm 是否除以元素數。照式子逐字實作，
   落差記在 `discrepancy_note`。
3. **M 與層的身分未給。** 論文只說「deeper layers」「critical semantic
   bottlenecks」，沒有列出任何層名、也沒有給 M。本檔取 `M=1`、層為
   U-Net 的 `mid_block` 輸出（見 `DEFAULT_LAYERS`）。
4. **T 未給。** 論文只寫 `T={t₁,…,t_k}`，沒有 k、沒有取值、沒有取法。
   本檔取 `(200, 400, 600, 800)`（見 `DEFAULT_TIMESTEPS`）。
5. **步長 α 未給。** Algorithm 1 把 α 列為輸入，正文與實驗節都沒有給值。
   本檔取 `1/255`（見 `DEFAULT_STEP_SIZE`）。
6. **受害模型不同。** 論文跑 SD3（MMDiT）、HQ-Edit、InstructPix2Pix；本專案
   的 baseline 管線是 SD v1.4 的 U-Net。「哪一層是語意瓶頸」是隨架構改變的
   問題，MMDiT 上的對應層本檔未處理。
7. **帶噪 latent 的取法未給。** 論文沒說 VAE 取 sample 還是 mode、噪聲每個
   iteration 重抽還是固定。本檔取 mode（`sd.encode_image`）與**固定噪聲**
   （見 `prepare`）。
8. **CFG 未給。** 論文沒說 φ_t 是在有條件、無條件還是 CFG 合成後的前向上取。
   本檔取單次有條件前向。
9. **ISR 沒有實作。** ISR（§VI 式 (8)）要兩個 MLLM（Gemini 2.5 Pro 與
   Flash）各自判定再取嚴格一致，本專案的評測管線沒有這條路，本檔只產生
   防禦圖，不產生 ISR。

上面 3–8 每一項都是**本專案自己決定的**，不是論文的值；全部寫進
`SPEC_PAPER.modification_note` 與 `extras`，報表上不得被讀成原論文設定。

為什麼不提供 `use_ckpt`
──────────────────────────────────────────────────────────────────────

Mist 與 DIA 用 `torch.utils.checkpoint` 壓峰值記憶體。**這一篇不能用**：
checkpoint 區塊的前向在 `no_grad` 下執行，forward hook 攔到的特徵張量
因此不在計算圖上，反向時重算的圖也接不回 hook 取走的那個張量——梯度會
靜默變成零，而輸出仍是一張合理的防禦圖。峰值由 `|T|` 次前向的圖同時留存
決定，要省記憶體只能減少 `timesteps` 的長度（這會改變方法本身，必須記錄）。

接進 `scripts/baseline_run.py`
──────────────────────────────────────────────────────────────────────

本檔不改那個檔（多個 baseline 同時在改它）。要接進去需要三處：

    # 1. import 行
    from immunization_baseline.attacks import dia, mist, photoguard, sifm

    # 2. CONDITIONS
    CONDITIONS = ["photoguard_c", "photoguard_linf", "mist", "dia_r", "sifm"]

    # 3. run_additive 的 spec 表與 kwargs
    spec = {..., "sifm": sifm.SPEC_PAPER}[name]
    ...
    elif name == "sifm":
        # SIFM 的 φ_t 以編輯 prompt 為條件（§VII-A：以該圖原本的 prompt
        # 產生擾動），故必須把 prompt 傳進去，沒有預設值。
        kw = {"prompt": item["prompt"]}
"""

from typing import Dict, List, Optional, Sequence, Tuple

import torch

from immunization_baseline.attacks.pgd import BaselineSpec, ValueRange

# Algorithm 1 第 14 行 `x_imu ← clip_{0,1}(x_orig + δ)`：論文的最佳化值域
# 就是 [0,1]，與本專案的張量介面相同，故 eps 不需換算。
SIFM_RANGE = ValueRange(
    0.0,
    1.0,
    "Algorithm 1 第 14 行 clip_{0,1}(x_orig+δ)；§VII 的 ε=0.03 與 Algorithm 1 的 α 同在此值域",
)

# ---- 論文給出的值 ----

PAPER_EPS = 0.03        # §VII 首段
PAPER_STEPS = 100       # §VII 首段 "limited to 100 optimization iterations"
PAPER_LAMBDA = 0.1      # §VII-C 表 VII

# ---- 論文未給、由本專案決定的值 ----

# 論文只說取「deeper layers」的輸出、在「critical semantic bottlenecks」上
# 擾動，沒有列層名也沒有給 M。取 M=1、層為 `mid_block`：那是 U-Net 空間解析度
# 最低、通道數最多的一點，是論文敘述唯一能指認的對象；取 M>1 還必須再自行
# 決定另外 M−1 層的身分，那是第二個沒有依據的決定。
#
# 換成多層時，式 (3) 的「對層平均」要求各層輸出形狀相同（U-Net 各 stage 的
# 通道數與解析度都不同），`FeatureRecorder.aggregate` 會逐一核對並在不符時
# 拋錯——不核對的話會 broadcast 成另一個張量而不報錯。
DEFAULT_LAYERS: Tuple[str, ...] = ("mid_block",)

# 論文只寫 T={t₁,…,t_k}，沒有 k、沒有取值、沒有取法。取訓練排程 [0,1000) 上的
# 四個等距內點。四個的理由是成本：|T| 次前向的計算圖必須同時留存（見模組
# docstring「為什麼不提供 use_ckpt」）。
DEFAULT_TIMESTEPS: Tuple[int, ...] = (200, 400, 600, 800)

# Algorithm 1 把 α 列為輸入，正文與實驗節都沒有給值。取 1/255：本專案已查證的
# 七篇 baseline 中，Mist、DIA、PromptFlare、DiffusionGuard 的步長在各自值域下
# 都是 1/255 或 2/255。ε=0.03 ≈ 7.65/255 而步數 100，此步長足以讓投影生效。
DEFAULT_STEP_SIZE = 1.0 / 255.0

# 論文沒說 φ_t 的前向用什麼 prompt 分支。取單次有條件前向（不做 CFG），
# 與 DIA 的 `cfg=1` 單分支同一個形狀，且式 (3)(4) 都只寫一個 c。
DEFAULT_SEED = 0

DIST_METRICS = ("mse",)


class FeatureRecorder:
    """以 forward hook 取指定層的輸出，並依式 (3) 對層平均。

    hook 而非改寫 forward：式 (3) 要的是「第 j 個目標層的輸出」，那就是該層
    `forward` 的回傳值本身，hook 取到的與該層實際算出的是同一個張量。

    diffusers 的 `DownBlock2D`／`CrossAttnDownBlock2D` 回傳
    `(hidden_states, output_states)` 二元組，其第 0 元素才是該 block 的輸出；
    `mid_block` 直接回傳張量。兩者都接受，其餘型別拋錯——猜一個元素出來會
    取到 skip connection 的堆疊而不是該層的輸出。
    """

    def __init__(self, unet, layer_names: Sequence[str]):
        if not layer_names:
            raise ValueError("SIFM 的目標層不得為空：式 (3) 的 M ≥ 1")
        self.layer_names = tuple(layer_names)
        available = dict(unet.named_modules())
        missing = [n for n in self.layer_names if n not in available]
        if missing:
            raise ValueError(
                f"UNet 中找不到目標層 {missing}。可用的頂層模組為 "
                f"{[n for n in available if n and '.' not in n]}。"
                "層名寫錯時 hook 不會裝上，特徵永遠是空的，而 PGD 照跑"
            )
        self._layers = [(n, available[n]) for n in self.layer_names]
        self._handles: List = []
        self.features: Dict[str, torch.Tensor] = {}

    def _make_hook(self, name: str):
        def hook(module, args, output):
            if isinstance(output, tuple):
                if not output or not isinstance(output[0], torch.Tensor):
                    raise TypeError(
                        f"目標層 {name} 回傳的二元組第 0 元素不是張量"
                    )
                out = output[0]
            elif isinstance(output, torch.Tensor):
                out = output
            else:
                raise TypeError(
                    f"目標層 {name} 的輸出型別為 {type(output)}，"
                    "本實作只處理張量與 (hidden_states, output_states) 二元組"
                )
            self.features[name] = out

        return hook

    def __enter__(self) -> "FeatureRecorder":
        for name, layer in self._layers:
            self._handles.append(layer.register_forward_hook(self._make_hook(name)))
        return self

    def __exit__(self, *exc) -> bool:
        for h in self._handles:
            h.remove()
        self._handles = []
        return False

    def clear(self) -> None:
        self.features = {}

    def aggregate(self) -> torch.Tensor:
        """式 (3)：`φ = (1/M) Σ_j L_j`。

        缺層即拋錯：hook 沒被觸發（前向路徑改了、層名對但沒進到那條分支）
        時，少平均一項不會有任何症狀。形狀不符也拋錯，理由見 `DEFAULT_LAYERS`。
        """
        missing = [n for n in self.layer_names if n not in self.features]
        if missing:
            raise RuntimeError(
                f"目標層 {missing} 的 hook 沒有被觸發。UNet 的前向路徑沒有經過"
                "這些層，式 (3) 的 M 項會少掉幾項而不報錯"
            )
        feats = [self.features[n].float() for n in self.layer_names]
        ref = feats[0].shape
        for name, f in zip(self.layer_names, feats):
            if f.shape != ref:
                raise ValueError(
                    f"目標層 {name} 的輸出形狀 {tuple(f.shape)} 與 "
                    f"{self.layer_names[0]} 的 {tuple(ref)} 不同，無法依式 (3) "
                    "對層平均。形狀不同時相加會 broadcast 成另一個張量而不報錯"
                )
        return torch.stack(feats, dim=0).mean(dim=0)


class SIFMContext:
    def __init__(self, sd, spec, emb, timesteps, noise, phi_orig, lam,
                 dist_metric, recorder, layer_names):
        self.spec = spec
        self.emb = emb
        self.timesteps = timesteps          # List[int]
        self.noise = noise                  # Dict[int, Tensor]，固定不重抽
        self.phi_orig = phi_orig            # Dict[int, Tensor]，Algorithm 1 第 3 行
        self.lam = float(lam)
        self.dist_metric = dist_metric
        self.recorder = recorder
        self.layer_names = tuple(layer_names)

    def close(self) -> None:
        self.recorder.clear()


def _noisy_latent(sd, z0: torch.Tensor, t: int, noise: torch.Tensor) -> torch.Tensor:
    """式 (3) 的 `E(x,t)_noisy`：標準前向加噪。

        z_t = √ᾱ_t · z₀ + √(1−ᾱ_t) · ε

    論文只寫「the process of obtaining the noisy latent representation」，
    沒有寫式子；這是 DDPM 的前向式，與本專案 `mist._semantic_loss` 用的
    同一條。
    """
    abar = sd.alphas_cumprod(z0.device)[t].to(z0.dtype)
    return abar.sqrt() * z0 + (1.0 - abar).sqrt() * noise.to(z0.dtype)


def _phi(sd, x01: torch.Tensor, ctx: SIFMContext, t: int) -> torch.Tensor:
    """式 (3) 的 `φ_t(x,c)`。`x01` 在 `[0,1]`（＝本篇的值域）。"""
    z0 = sd.encode_image(x01)
    z_t = _noisy_latent(sd, z0, t, ctx.noise[t])
    ctx.recorder.clear()
    with ctx.recorder:
        # `sd.unet_forward` 而非 `sd.unet`：SDXL 的條件是 SDXLPrompt，UNet 另需
        # added_cond_kwargs，少傳會直接報錯而在 SD v1.x 上卻能跑。
        sd.unet_forward(z_t, torch.tensor(t, device=z_t.device), ctx.emb)
    return ctx.recorder.aggregate()


def prepare(
    sd,
    x01: torch.Tensor,
    spec: BaselineSpec,
    *,
    prompt: Optional[str] = None,
    lam: float = PAPER_LAMBDA,
    timesteps: Sequence[int] = DEFAULT_TIMESTEPS,
    layers: Sequence[str] = DEFAULT_LAYERS,
    dist_metric: str = "mse",
    seed: int = DEFAULT_SEED,
    **_,
) -> SIFMContext:
    """Algorithm 1 第 1–4 行：δ←0（由骨幹的 `init_rule="none"` 負責）、
    預先算好 `Φ_orig = {φ_t(x_orig,c) | t∈T}`。

    `prompt` **沒有預設值**。§VII-A 寫「Each image's original prompt from
    this collection was used for perturbation generation」，即擾動是在該圖
    自己的編輯 prompt 上求的；式 (3)(4) 的 φ_t 也都以 c 為條件。空字串會
    產生另一個方法（無條件去噪器的特徵），且沒有任何症狀。

    **噪聲固定**：每個 t 在此抽一次並存起來，100 次迭代共用。理由是
    Algorithm 1 第 3 行把 `Φ_orig` 預先算完並在迴圈外保持不變——若 φ_t^imu
    每次用新噪聲，第 9 行的 `Dist(φ_t^imu, Φ_orig[t])` 比的就是兩個不同噪聲
    實現下的特徵，該距離的大部分來自噪聲而不是 δ。論文沒有寫這一點。
    """
    if prompt is None:
        raise ValueError(
            "SIFM 需要編輯 prompt：式 (3)(4) 的 φ_t 以 c 為條件，§VII-A 指定用"
            "該圖原本的 prompt 產生擾動。傳空字串會變成無條件去噪器的特徵，"
            "那是另一個方法，而結果看起來一樣是一張防禦圖"
        )
    if dist_metric not in DIST_METRICS:
        raise ValueError(
            f"Dist 只實作了 {DIST_METRICS}，收到 {dist_metric!r}。"
            "論文 §V-A 寫「a distance metric (e.g., MSE)」，未指定其他選項"
        )
    if lam <= 0:
        raise ValueError(
            f"式 (6) 要求 λ>0，收到 {lam}。λ=0 是表 VI 的消融列（只有 L_norm），"
            "要跑那一列請用另一個 spec，不要靠傳參數把主設定改掉"
        )
    ts = [int(t) for t in timesteps]
    if not ts:
        raise ValueError("式 (7) 對 T 取平均，|T| 不得為 0")
    n_train = sd.num_train_timesteps
    for t in ts:
        if not 0 <= t < n_train:
            raise ValueError(f"timestep {t} 不在 [0, {n_train}) 內")

    emb = sd.encode_text(prompt).detach()
    recorder = FeatureRecorder(sd.unet, layers)

    # 產生器固定在 CPU：與 `run_pgd` 的起點同一個理由（CUDA 的 randn 逐版本
    # 可能不同，而可重現性是報表要宣稱的東西）。
    gen = torch.Generator(device="cpu").manual_seed(seed)
    with torch.no_grad():
        z0 = sd.encode_image(x01.detach())
        noise = {
            t: torch.randn(z0.shape, generator=gen, dtype=torch.float32).to(
                device=z0.device, dtype=z0.dtype
            )
            for t in ts
        }
        ctx = SIFMContext(sd, spec, emb, ts, noise, {}, lam, dist_metric,
                          recorder, layers)
        # Algorithm 1 第 3 行：Φ_orig 取自**原圖**，整個迴圈不再更新。
        with sd.conditioning_for(x01, mask=torch.ones_like(x01[:, :1])):
            for t in ts:
                ctx.phi_orig[t] = _phi(sd, x01.detach(), ctx, t).detach()
    recorder.clear()
    return ctx


def _dist(a: torch.Tensor, b: torch.Tensor, metric: str) -> torch.Tensor:
    """式 (4) 的 `Dist(·,·)`。論文寫「e.g., MSE」，本檔即取 MSE。"""
    if metric == "mse":
        return ((a - b) ** 2).mean()
    raise ValueError(f"未知的 Dist {metric!r}")


def sifm_terms(
    phi_imu: torch.Tensor, phi_orig: torch.Tensor, lam: float,
    dist_metric: str = "mse",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """式 (5)(4)(6) 三個量，回傳 `(L_norm, L_dist, L_SIFM)`。

        L_norm = ‖φ_imu‖₁            （絕對值之和，式 (5)）
        L_dist = Dist(φ_imu, φ_orig) （式 (4)）
        L_SIFM = L_norm − λ·L_dist   （式 (6)、Algorithm 1 第 9 行）

    分成獨立函式是為了讓兩項的方向可以在 CPU 上單獨釘住：整個損失寫反號
    只會讓攻擊變成替攻擊方最佳化，而曲線照樣會動、圖照樣產得出來。

    `float()` 不是形式：L_norm 是數萬到數百萬個元素的絕對值和，fp16 的
    上限 65504 會直接溢位成 inf。
    """
    a = phi_imu.float()
    b = phi_orig.float()
    l_norm = a.abs().sum()
    l_dist = _dist(a, b, dist_metric)
    return l_norm, l_dist, l_norm - lam * l_dist


def loss_fn(sd, x_adv: torch.Tensor, ctx: SIFMContext) -> torch.Tensor:
    """式 (7)：`(1/|T|) Σ_{t∈T} [ ‖φ_t‖₁ − λ·Dist(φ_t, Φ_orig[t]) ]`。

    由骨幹以 `objective="minimize"` 走，即 Algorithm 1 第 13 行的
    `δ ← clip(δ − α·sign(g))`。**不要把符號吸收進損失**——報表上的損失曲線
    要是論文那個量。

    Algorithm 1 第 7–12 行是逐 t 求梯度再平均；此處回傳逐 t 損失的平均再由
    骨幹一次反向。兩者的梯度相同（線性），差別只在 |T| 條計算圖同時留存，
    見模組 docstring。

    9 通道（inpainting）權重下的後 5 個通道取**全 1 遮罩**，理由與 Mist、DIA
    相同：本篇的 φ_t 取自一個**沒有影像條件**的噪聲預測器，全 1 遮罩下
    `masked_image_latents` 是 `encode(0)`，UNet 退化為純文字條件的去噪器，
    那才是原形式的對應物。
    """
    with sd.conditioning_for(x_adv, mask=torch.ones_like(x_adv[:, :1])):
        total = None
        for t in ctx.timesteps:
            phi = _phi(sd, x_adv, ctx, t)
            _, _, l_t = sifm_terms(phi, ctx.phi_orig[t], ctx.lam, ctx.dist_metric)
            total = l_t if total is None else total + l_t
        return total / len(ctx.timesteps)


SPEC_PAPER = BaselineSpec(
    name="sifm",
    source="arXiv:2512.14320v1（2025-12-16）；無官方程式碼",
    value_range=SIFM_RANGE,
    eps=PAPER_EPS,              # §VII 首段
    eps_pixel01=PAPER_EPS,      # 值域已是 [0,1]（Algorithm 1 第 14 行），不需換算
    norm="linf",                # §III-A「usually p=∞」＋ Algorithm 1 第 13 行逐元素 clip
    steps=PAPER_STEPS,          # §VII 首段
    step_size=DEFAULT_STEP_SIZE,  # 論文未給，本專案決定，見 DEFAULT_STEP_SIZE
    step_schedule="constant",   # Algorithm 1 的 α 不隨 n 改變
    update_rule="sign",         # Algorithm 1 第 13 行 sign(g)
    objective="minimize",       # Algorithm 1 第 13 行 δ − α·sign(g)
    init_rule="none",           # Algorithm 1 第 1 行 δ ← 0
    grad_reps=1,                # Algorithm 1 每個 n 只走一次 T
    needs_target_image=False,
    needs_mask=False,
    modified_from_paper=True,
    modification_note=(
        "無官方程式碼（arXiv:2512.14320v1 全文無 repo 連結、無匿名 repo、"
        "無 implementation details 一節），本檔依論文正文與 Algorithm 1 重建。"
        "論文未給而由本專案決定的項目："
        "(a) 目標層與 M——論文只說「deeper layers」「semantic bottleneck」，"
        "本檔取 M=1、層為 U-Net 的 mid_block 輸出；"
        "(b) 時間步集合 T——論文只寫 T={t1,…,tk}，本檔取 (200,400,600,800)；"
        "(c) 步長 α——Algorithm 1 列為輸入但全文未給值，本檔取 1/255；"
        "(d) 帶噪 latent 的取法——本檔用 VAE 的 mode 與**固定**噪聲（每個 t 在 "
        "prepare 抽一次，100 次迭代共用），論文未述；"
        "(e) φ_t 的前向用單次有條件前向，不做 CFG，論文未述；"
        "(f) Dist 取 MSE（§V-A 的例子）；"
        "(g) 受害模型為 SD v1.4 的 U-Net，論文跑的是 SD3（MMDiT）／HQ-Edit／"
        "InstructPix2Pix，「哪一層是語意瓶頸」隨架構改變。"
        "另：論文的 ISR（§VI 式 (8)，兩個 Gemini 2.5 模型嚴格一致）未實作，"
        "本檔只產生防禦圖。"
    ),
    discrepancy_note=(
        "§I 寫「minimizes the Frobenius norm of targeted intermediate features」，"
        "§V-B 與式 (5)、Algorithm 1 第 9 行寫 L1（且整段理由是 sparsity-inducing），"
        "本檔實作 L1；"
        "式 (5) 的 L_norm 是絕對值之和、式 (4) 的 Dist 取 MSE 是平均，兩項的 "
        "reduction 差了元素數個數量級，而 λ=0.1 的意義完全取決於這兩個 reduction，"
        "論文兩者都沒有寫明；"
        "§VII 的 ε=0.03 與 100 步是「all methods」共用的預算，不是 SIFM 專屬的設定。"
    ),
    prepare=prepare,
    loss_fn=loss_fn,
    extras={
        "lambda": PAPER_LAMBDA,
        "lambda_source": "§VII-C 表 VII（ISR 最高，全文其餘實驗沿用）",
        "layers": list(DEFAULT_LAYERS),
        "layers_source": "論文未給，本專案決定（見 DEFAULT_LAYERS）",
        "M": len(DEFAULT_LAYERS),
        "timesteps": list(DEFAULT_TIMESTEPS),
        "timesteps_source": "論文未給，本專案決定（見 DEFAULT_TIMESTEPS）",
        "step_size_source": "論文未給，本專案決定（見 DEFAULT_STEP_SIZE）",
        "dist_metric": "mse",
        "norm_reduction": "sum（式 (5) 的 ‖·‖₁ 逐字）",
        "noise_policy": "每個 t 在 prepare 抽一次後固定",
        "cfg": "無（單次有條件前向）",
        "model_paper": "StableDiffusion-3 / HQ-Edit / InstructPix2Pix",
        "model_here": "CompVis/stable-diffusion-v1-4",
        "dataset_paper": "100 張（人像 35／風景 35／畫作 30）取自 "
                         "instructpix2pix-clip-filtered，另生成 5 條未見過的 prompt",
        "metric_paper": "PSNR/SSIM/VIFp/FSIM/LPIPS ＋ ISR（Gemini 2.5 Pro 與 Flash 嚴格一致）",
        "isr_implemented": False,
    },
)

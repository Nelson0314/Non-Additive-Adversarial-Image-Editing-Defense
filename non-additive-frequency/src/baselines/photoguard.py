"""PhotoGuard-c（complex / diffusion attack）—— Salman et al., ICML 2023。

出處：`MadryLab/photoguard`，HEAD `686bea75c786cb46c88fc396a0cd0ee3d7d28c2e`，
`notebooks/demo_complex_attack_inpainting.ipynb`。逐字佐證見
`docs/_audit_promptflare_photoguard.md` §PhotoGuard 1–4，裁決見
`docs/reference/SOURCE_AUDIT.md` §3。

三項必須照原始碼而非論文正文的地方
──────────────────────────────────────────────────────────────────────

1. **實跑的是 L2 不是 ℓ∞。** cell 8 的 `super_l2` 用
   `torch.renorm(d_x, p=2, dim=0, maxnorm=eps)`，更新規則是歸一化梯度
   而非 sign。cell 11 的 ℓ∞ 版本整段 17 行以 `#` 開頭，從未被執行。
   論文 Table 9 寫的「ℓ∞ / 16/255 / step 2/255 / 200 步」這組參數，
   **repo 中不存在任何一處同時滿足四欄的程式碼**。
2. **target 實際被歸零**（cell 10）：
   `target_image_tensor = 0*target_image_tensor.cuda()`。
   下載的 pinimg 圖片不參與計算，損失退化為 `‖f(x_adv)‖₂`，
   即把 pipeline 的輸出壓成 `[-1,1]` 的中性灰。不需要外部影像，
   也**不可**與 Mist 的 `MIST.png` 混用。
3. **損失沒有平方。** 論文 Eq. 5 寫 `‖·‖₂²`，cell 8 是
   `loss = (image_nat - target_image).norm(p=2)`。此處照原始碼。

img2img 版由我方移植（`modified_from_paper=True`）
──────────────────────────────────────────────────────────────────────

repo 的 complex attack **只有 inpainting 版**（GitHub tree API 全檔案清單
確認，`_audit_promptflare_photoguard.md` §5）。本專案的威脅模型是
img2img／SDEdit，故 `attack_forward` 必須由我方改寫。改動逐項如下：

| 原始（inpainting） | 本移植（img2img） | 理由 |
|---|---|---|
| latent 由 `torch.randn` 初始化，跑滿 `num_inference_steps` | latent 由 `E(x_adv)` 加噪到 `strength` 對應的 t | img2img 的定義；這正是 `strength` 進入攻擊的原因 |
| 9 通道 `cat([latents, mask, masked_image_latents])` | 4 通道 | img2img 的 UNet 沒有 mask 與 masked-image 通道 |
| `grad = ... * (1 - cur_mask)`，擾動只在非重繪區 | 全圖 | 無 mask 的威脅模型 |
| `eta=1`（DDIM 隨機性）+ 未固定的 latent，`grad_reps=10` 平均 | DDIM 為決定性，改為**每個 rep 換一個編輯噪聲種子**，仍平均 10 次 | 保留「對前向的隨機性取期望」這個作用；決定性 DDIM 下若不換種子，10 個 rep 會完全相同，`grad_reps=10` 形同 1 |
| `strength=0.7` 在 cell 10 被賦值但**未傳入** `super_l2` | 由呼叫端提供，無預設 | inpainting 版根本不用 strength，故原始碼中沒有這個數；憑空給一個值會讓失真預算與威脅模型脫節 |

另兩處與原始碼不同、但不是我們選的（皆源自 `src/models/sd.py` 的既有語意）：

1. `SDWrapper.decode_latent` 對輸出 `clamp(0,1)`，而 `attack_forward` 回傳
   未裁切的 `vae.decode(...).sample`。被裁切的像素其梯度為零。
2. `SDWrapper.encode_image`（`sd.py:156,160`）固定取 `latent_dist.mean`，
   而 `attack_forward` 取 `latent_dist.sample()`
   （`_audit_promptflare_photoguard.md:392-393`）。原作的 `grad_reps=10`
   是對「DDIM `eta=1` 的隨機性 + 未固定的 `latents` + VAE 後驗抽樣」
   三重隨機性取期望，本移植只保留了編輯噪聲一項，期望的對象與原作不同。

兩者都不在此繞過：繞過要改 `src/models/sd.py`，而 `encode_image` 取 mean
是全專案（含防禦端與既有 53 個 run）共用的決定性前提；且 `sdedit` 以
`vae_ckpt=True` 走 `torch.utils.checkpoint`，反向時會**重算**前向，若在
其中抽樣，重算會抽到另一組雜訊，梯度將對應到與前向值不同的函式
（`preserve_rng_state` 只還原全域 RNG，不還原顯式傳入的 `Generator`）。
要做成忠實版必須把雜訊改為在外部抽好再傳入，屬於介面改動，需另案決定。
Mist 與 DIA 因為不經 `sdedit`，才能各自另寫取樣路徑
（`mist.py:66-74`、`dia.py:130-136`）。

以上記錄於此並列入 `modification_note`，在報表註明。
"""

from typing import Optional

import torch

from src.baselines.pgd import BaselineSpec, ValueRange

# `notebooks/utils.py:28-42` 的 `prepare_mask_and_masked_image`：
# `image / 127.5 - 1.0`；`super_l2` 的 `clamp_min=-1, clamp_max=1`。
PHOTOGUARD_RANGE = ValueRange(
    -1.0,
    1.0,
    "notebooks/utils.py:31 `image / 127.5 - 1.0`；cell 10 clamp_min=-1, clamp_max=1",
)


class PhotoGuardContext:
    """與 δ 無關的常量。每個 rep 換一個編輯噪聲，見模組 docstring 的移植表。"""

    def __init__(self, sd, spec, target, emb, emb_uncond, latent_shape,
                 strength, num_inference_steps, guidance_scale, seed,
                 mask=None):
        self.spec = spec
        # inpainting 遮罩。給定時本篇回到**原生形態**：9 通道輸入、由純噪聲
        # 起跑、不吃 strength。那也是 repo 唯一存在的版本。
        self.mask = mask
        self.target = target
        self.emb = emb
        self.emb_uncond = emb_uncond
        self.latent_shape = latent_shape
        self.strength = strength
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.seed = seed
        self._rep = 0
        self._sd = sd

    def next_noise(self) -> torch.Tensor:
        z_like = torch.empty(
            self.latent_shape, device=self.target.device, dtype=self.target.dtype
        )
        noise = self._sd.sample_edit_noise(z_like, seed=self.seed + self._rep)
        self._rep += 1
        return noise


def prepare(
    sd,
    x01: torch.Tensor,
    spec: BaselineSpec,
    *,
    strength: Optional[float] = None,
    mask: Optional[torch.Tensor] = None,
    target01: Optional[torch.Tensor] = None,
    prompt: str = "",
    num_inference_steps: int = 4,
    guidance_scale: float = 7.5,
    seed: int = 0,
    **_,
) -> PhotoGuardContext:
    """建立目標張量與文字嵌入。

    `prompt`、`num_inference_steps`、`guidance_scale` 的預設值取自 cell 10
    實際執行的呼叫：`prompt = ""`、`num_inference_steps = 4`、
    `guidance_scale = 7.5`。空 prompt 獨立佐證了 `DESIGN` §2.1 把全部條件
    改為 prompt-free 的決定。

    `strength` **無預設且必填**，理由見模組 docstring。
    """
    if strength is None and mask is None:
        raise NotImplementedError(
            "PhotoGuard-c 的 img2img 版缺 strength：官方只有 inpainting 版，"
            "其 attack_forward 由 torch.randn 起跑、不吃 strength；"
            "cell 10 的 strength=0.7 被賦值但未傳入 super_l2，"
            "因此原始碼中沒有『攻擊時用哪個 strength』這個數。"
            "請由呼叫端依本專案的威脅模型明確指定，"
            "並在報表標註為我方設定（見 docs/SOURCE_AUDIT §3.3）"
        )

    vr = spec.value_range
    if target01 is None:
        # cell 10：`target_image_tensor = 0*target_image_tensor`，
        # 即 [-1,1] 的零張量。轉回 [0,1] 是 0.5（中性灰）。
        target = torch.zeros_like(x01)
    else:
        target = vr.from01(target01.to(x01.device).detach())

    emb = sd.encode_text(prompt).detach()
    # 無條件分支取 `uncond_prompt()`：stock SDXL base 的
    # `force_zeros_for_empty_prompt=true`，其無條件分支是零張量而非
    # `encode_text("")`。SD v1.x 上兩者恆等，故此改動不影響原作的重現。
    #
    # 附帶後果（SDXL 上才有）：原作的 prompt 為空字串，在 SD v1.x 上
    # 條件與無條件兩支的嵌入相同，`_eps_cfg` 的兩次前向數值相同、CFG 恆等於
    # 無條件預測；在 SDXL 上 `encode_text("")` 與零張量不同，CFG 確實起作用。
    # 這是「攻擊方是 stock 模型」的直接後果，不是我方選擇。
    emb_uncond = sd.uncond_prompt().detach()
    latent_shape = sd.latent_shape(x01.shape[-2], x01.shape[-1])
    return PhotoGuardContext(
        sd, spec, target, emb, emb_uncond, latent_shape,
        strength, num_inference_steps, guidance_scale, seed, mask=mask,
    )


def loss_fn(sd, x_adv: torch.Tensor, ctx: PhotoGuardContext) -> torch.Tensor:
    """`L = ‖f(x_adv) − x_targ‖₂`（cell 8 的 `compute_grad`，未平方）。

    `f` 是完整 img2img pipeline 的 VAE 解碼輸出，值域 `[-1,1]`。
    `SDWrapper.sdedit` 回傳 `[0,1]`，故先換回該篇的值域再比較——
    差一個常數倍會直接改變 `loss` 的量級，報表上的曲線就對不上。
    """
    vr = ctx.spec.value_range
    x01_adv = vr.to01(x_adv)
    # 走 `edit`：遮罩給定時即回到原作的 9 通道 `attack_forward`（由純噪聲
    # 起跑、不吃 strength），否則維持本專案移植的 img2img 版。分派的規則
    # 只有 `SDWrapper.edit` 一處，此處不重寫。
    y01 = sd.edit(
        x01_adv,
        ctx.emb,
        ctx.next_noise(),
        ctx.num_inference_steps,
        mask=ctx.mask,
        strength=(None if ctx.mask is not None else ctx.strength),
        use_ckpt=True,
        vae_ckpt=True,
        guidance_scale=ctx.guidance_scale,
        emb_uncond=ctx.emb_uncond,
    )
    y = vr.from01(y01)
    return (y - ctx.target).norm(p=2)


SPEC = BaselineSpec(
    name="photoguard_c",
    source="arXiv:2302.06588（ICML 2023）；MadryLab/photoguard@686bea75c786cb46c88fc396a0cd0ee3d7d28c2e",
    value_range=PHOTOGUARD_RANGE,
    # cell 10：`eps=16` 傳給 `torch.renorm(..., maxnorm=eps)`，是整張
    # [1,3,512,512] 影像在 [-1,1] 下展平後的 L2 半徑，**不是** 16/255。
    eps=16.0,
    eps_pixel01=8.0,          # L2 半徑同樣線性縮放：16 / 2
    norm="l2",
    steps=200,                # cell 10 `iters=200`
    step_size=1.0,            # cell 10 `step_size=1`
    step_schedule="constant",  # cell 8 的衰減式被註解掉，`actual_step_size = step_size`
    update_rule="normalized_grad",
    objective="minimize",     # cell 8 `X_adv = X_adv - grad_normalized * step`
    init_rule="none",         # cell 8 `X_adv = X.clone()`
    grad_reps=10,             # cell 10 `grad_reps=10`
    needs_target_image=True,  # 但預設即零張量，不需外部檔案（cell 10 §3.4）
    needs_mask=False,
    # `attack_forward`：`grad = grad * (1 - cur_mask)`。inpainting 威脅
    # 模型下由 `run_pgd` 施加；img2img 下呼叫端不傳遮罩，本旗標無作用。
    grad_outside_mask=True,
    modified_from_paper=True,
    modification_note=(
        "官方只有 inpainting 版的 complex attack，img2img 版由本專案移植："
        "(1) latent 改由 E(x_adv) 加噪到 strength 對應的 t，非 torch.randn 起跑；"
        "(2) 去掉 9 通道的 mask 與 masked-image 拼接；"
        "(3) 擾動不再乘 (1 - mask)，作用於全圖；"
        "(4) DDIM 為決定性（原用 eta=1），grad_reps=10 改為每 rep 換一個編輯噪聲種子；"
        "(5) strength 由呼叫端指定，原始碼無此數；"
        "(6) SDWrapper.decode_latent 對輸出 clamp(0,1)，原 attack_forward 未裁切；"
        "(7) SDWrapper.encode_image 取 latent_dist.mean，原 attack_forward 取 "
        ".sample()，故本移植少了 VAE 後驗抽樣這一重隨機性（理由見模組 docstring）。"
    ),
    discrepancy_note=(
        "原始碼實跑 L2 renorm maxnorm=16、歸一化梯度更新、step 1、200 步、grad_reps 10；"
        "論文 Table 9 記為 ℓ∞ 16/255、step 2/255、200 步。repo 中不存在同時滿足 "
        "Table 9 四欄的程式碼。引用其失真預算時不得使用 Table 9 的數字。"
        "另：損失論文寫 ‖·‖₂²、原始碼為 .norm(p=2) 未平方。"
    ),
    prepare=prepare,
    loss_fn=loss_fn,
    extras={
        "prompt": "",                 # cell 10
        "num_inference_steps": 4,     # cell 10（攻擊內部，非 50）
        "guidance_scale": 7.5,        # cell 10
        "eta_original": 1,            # cell 10；本移植的 DDIM 為決定性
        "seed_original": 786349,      # cell 10 的 torch.manual_seed
        "target": "zero tensor in [-1,1]",
    },
)

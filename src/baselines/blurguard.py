"""BlurGuard（Kim et al., NeurIPS 2025）— 頻譜整形的加性 baseline。

**為什麼這一篇是本輪最該做的對照組**
────────────────────────────────────────────────────────────────────
FND-060 量到紋理重相位的擾動只有 22.6% 能量在半 Nyquist 以上，加性方法是
42–65%；FND-061 量到它在高斯模糊下勝加性 7.3 倍。最自然的解釋是「我們比較
低頻，所以比較耐磨」。

BlurGuard 證明**光是把普通的對抗雜訊做一次自適應模糊、把頻譜壓低，就能換到
抗淨化能力**。若本方法的優勢只來自能量落在較低頻，BlurGuard 會與之打平——
那樣「切塊、轉相位、保留幅度」這整套構造就沒有貢獻。**這是唯一能證否
「我們只是比較低頻」的實驗。**

出處
────────────────────────────────────────────────────────────────────
arXiv:2511.00143。程式碼 https://github.com/jsu-kim/BlurGuard ，
本檔逐行對應 `BlurGuard/code/attacks.py` 的 `LF_PGD.pgd_freq`
（2026-08-19 由 raw 檔核對）。

演算法（原始碼的實際運算）
────────────────────────────────────────────────────────────────────
1. 擾動 `pert` 初始化為 0；**每個遮罩區域各有一個模糊強度** `σ_i`，以
   `log σ_i = 0`（即 σ=1）起始。
2. `filter_delta`：把**整張** `pert` 用 `σ_i` 做高斯模糊，乘上第 i 個遮罩，
   全部相加。遮罩是互斥分割，故每個像素只取一個 σ。
3. 模糊後的擾動夾到 `±ε`，加到影像上再夾到 `[-1,1]`。
4. **對抗損失**：`(VAE latent)² 的平均`——把 latent 壓到零，與 DCT-Shield
   的 `‖E(x')‖₂` 同一族目標。
5. **σ 的損失不是防禦效果，而是「頻譜要看起來自然」**：另抽一組**隨機**雜訊
   （每個 iteration 重抽），過同一套模糊、同樣夾到 ±ε、加到原圖上，算它的
   徑向功率譜；要求它與**原圖**的徑向功率譜在 log10 尺度上，**每一個半徑
   分箱的差都不超過 `eps_sigma`**，取最大值後過 relu。這就是該篇說的
   「除了不可見，還要不可逆」——防禦圖的頻譜統計要像自然影像。
6. **兩階段**（`i < 50` 為界，硬寫在原始碼裡）：
   - 前 50 步只有 σ 的損失，Adam 更新 σ；此時 `pert.grad` 是 `None`，
     擾動不動。
   - 之後 σ **凍結**（不再呼叫 `optimizer.step()`），改以歸一化梯度的 PGD
     更新擾動，步長由 1 線性衰減到 1/100。

定案超參數（來自 repo 自己的 `configs/attack/base.yaml`，不是函式簽章的預設）
────────────────────────────────────────────────────────────────────
`epsilon = 16`、`steps = 150`、`learning_rate = 0.06`（Adam 對 σ）、
**`eps_sigma = 0`**（沒有寬容度，直接最小化最大頻譜偏差）、
`sigma_weighting = 10`、`input_size = 512`。SAM 用 `points_per_side = 2`
（`blurguard.py` 第 261 行）。模糊核寬 33、暖身 50 步寫死在 `attacks.py`。

**注意 `sigma_weighting` 有兩個不同的預設**：`blurguard.py:init()` 是 10、
`LF_PGD.__init__` 是 10000。實際生效的是 hydra config 的 **10**。

值域
────────────────────────────────────────────────────────────────────
原始碼在 `[-1,1]` 上工作，且 `epsilon/255.0 * (1-(-1))` 有乘 2——與 Mist 同
一種寫法（`SOURCE_AUDIT.md` §10）。故 `epsilon = 16` 在本專案的 `[0,1]`
介面上等於 **16/255**，與 Mist 的原生預算相同。本檔對外一律 `[0,1]`，
轉換只發生在頭尾各一次。

遮罩必須由呼叫端提供
────────────────────────────────────────────────────────────────────
原文用 SAM（Segment Anything）的自動分割，依面積由小到大排序後**做成互斥
分割**（每個像素只歸給第一個蓋到它的遮罩），最後補一張「沒被蓋到」的。
本檔**不提供任何替代的分割方式**——用格點或隨機分割去頂替會靜默改變方法，
且事後看不出來。`sam_masks()` 在相依不齊時直接拋出並寫明缺什麼。

一個必須寫下來的觀察：徑向分箱的半徑是從**陣列中心**量的
────────────────────────────────────────────────────────────────────
`compute_histogram_fft` 以 `((H−1)/2, (W−1)/2)` 為圓心算半徑，但
`fft_fps` 的 `torch.fft.fft2` **沒有 fftshift**，零頻在角落。因此該篇的
「半徑 0」對應的是 Nyquist、「半徑最大」對應的是直流，**分箱與頻率大小的
對應是反過來且會混箱的**（例如索引 (0,0) 與 (0,W−1) 頻率都很低，但到陣列
中心的距離不同）。

這不會讓約束失效——兩邊用同一套分箱，「分箱後的功率剖面要一致」這件事仍然
成立，只是頻率解析度變差。本檔**照原樣實作**（重現就是重現原樣），並在此
記下該性質，`FND-` 引用時不可寫成「它約束的是徑向頻譜」。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import torch
import torch.nn.functional as F

# 來自 repo 的 configs/attack/base.yaml
PAPER_EPSILON_255 = 16          # `[0,1]` 上即 16/255
PAPER_STEPS = 150
PAPER_LR = 0.06                 # Adam，只作用在 log σ 上
PAPER_EPS_SIGMA = 0.0           # 沒有寬容度
PAPER_SIGMA_WEIGHTING = 10.0
PAPER_INPUT_SIZE = 512
PAPER_SAM_POINTS_PER_SIDE = 2   # blurguard.py:261
# 寫死在 attacks.py 裡
PAPER_BLUR_WIDTH = 33
PAPER_WARMUP_ITERS = 50
PAPER_STEP_SIZE = 1.0


def gaussian_blur(x: torch.Tensor, sigma: torch.Tensor,
                  width: int = PAPER_BLUR_WIDTH) -> torch.Tensor:
    """`attacks.py` 的 `gaussian_blur`，逐行對應。

    核是 `exp(-(dy²+dx²)/(2σ²))` 正規化後的 (width, width) 二維高斯，邊界用
    **反射填補**，以 `groups=channels` 的卷積逐通道作用。`width` 會被調成
    奇數（原始碼的 `width + (width+1)%2`；33 已是奇數，故不變）。

    `sigma` 是張量而非 float——它是被最佳化的變數，核必須落在計算圖上。
    """
    if x.dim() != 4:
        raise ValueError(f"需要 (N,C,H,W)，收到 {tuple(x.shape)}")
    width = width + (width + 1) % 2
    half = width // 2
    d = torch.arange(-half, half + 1, dtype=x.dtype, device=x.device)
    g = torch.exp(-(d[:, None] ** 2 + d[None, :] ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    c = x.shape[1]
    kernel = g[None, None].expand(c, -1, -1, -1)
    return F.conv2d(F.pad(x, (half,) * 4, mode="reflect"), kernel, groups=c)


def filter_delta(log_sigmas: torch.Tensor, x: torch.Tensor,
                 masks: Dict[str, torch.Tensor]) -> torch.Tensor:
    """`attacks.py` 的 `filter_delta`：逐區域模糊再相加。

    `masks` 的鍵是 `mask1`…`maskN`，順序即 `log_sigmas` 的索引順序。
    每個遮罩是 (1,1,H,W) 的 0/1 張量，且**彼此互斥**——原文由 SAM 分割
    去重後保證，本檔的測試釘住這個前提。
    """
    n = len(masks)
    if log_sigmas.numel() != n:
        raise ValueError(f"log_sigmas 有 {log_sigmas.numel()} 個，遮罩有 {n} 個")
    out = None
    for i in range(n):
        m = masks[f"mask{i + 1}"].to(device=x.device, dtype=x.dtype)
        part = gaussian_blur(x, log_sigmas[i].exp()) * m
        out = part if out is None else out + part
    return out


def fft_power(x_pm1: torch.Tensor) -> torch.Tensor:
    """`attacks.py` 的 `fft_fps`：`[-1,1]` → `[0,1]`，二維 FFT，功率沿通道相加。

    回傳 (H,W)，**未經 fftshift**（零頻在角落）。見模組 docstring 的觀察。
    """
    if x_pm1.dim() == 4:
        x_pm1 = x_pm1[0]
    img = x_pm1 / 2 + 0.5
    return (torch.fft.fft2(img).abs() ** 2).sum(dim=0)


def radial_histogram(power: torch.Tensor) -> torch.Tensor:
    """`attacks.py` 的 `compute_histogram_fft`：以**陣列中心**為圓心分箱求和。

    原始碼堆出 (r_max, H, W) 的 one-hot 遮罩再相乘求和；512² 時那是約
    380 MB 的中間張量，而且在 150 步的迴圈裡每步都要建一次。本檔改用
    `scatter_add`——**同一個運算**，只是不materialise 那個張量。分箱邊界與
    原始碼逐格相同（`r.round().long()`、`r_max = r_int.max() + 1`）。
    """
    h, w = power.shape[-2:]
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    y = torch.arange(h, device=power.device, dtype=power.dtype)
    x = torch.arange(w, device=power.device, dtype=power.dtype)
    r = ((x[None, :] - cx) ** 2 + (y[:, None] - cy) ** 2).sqrt()
    idx = r.round().long()
    out = torch.zeros(int(idx.max()) + 1, device=power.device, dtype=power.dtype)
    return out.scatter_add(0, idx.reshape(-1), power.reshape(-1))


def spectrum_deviation(x_src_pm1: torch.Tensor, x_adv_pm1: torch.Tensor
                       ) -> torch.Tensor:
    """兩張圖的分箱功率剖面在 log10 上的**最大**絕對差（`attacks.py` 的
    `torch.max(torch.abs(lfs - lfa))`）。`+1e-8` 的下限照原樣。"""
    a = torch.log10(radial_histogram(fft_power(x_src_pm1)) + 1e-8)
    b = torch.log10(radial_histogram(fft_power(x_adv_pm1)) + 1e-8)
    return (a - b).abs().max()


@dataclass(frozen=True)
class BlurGuardSpec:
    """一組 BlurGuard 設定。預設即 repo 的 `configs/attack/base.yaml`。"""

    name: str = "blurguard"
    eps_pixel01: float = PAPER_EPSILON_255 / 255.0
    steps: int = PAPER_STEPS
    lr: float = PAPER_LR
    eps_sigma: float = PAPER_EPS_SIGMA
    sigma_weighting: float = PAPER_SIGMA_WEIGHTING
    warmup: int = PAPER_WARMUP_ITERS
    step_size: float = PAPER_STEP_SIZE
    blur_width: int = PAPER_BLUR_WIDTH
    modified_from_paper: bool = False
    modification_note: str = ""
    source: str = ("arXiv:2511.00143；超參數取自 repo 的 "
                   "configs/attack/base.yaml，非函式簽章的預設")

    def __post_init__(self):
        if self.modified_from_paper and not self.modification_note:
            raise ValueError(f"{self.name} 標了 modified_from_paper 卻沒寫改了什麼")
        if self.warmup >= self.steps:
            raise ValueError(
                f"warmup={self.warmup} 不小於 steps={self.steps}，"
                "那樣擾動一步都不會被更新")


SPEC_PAPER = BlurGuardSpec()


@dataclass
class BlurGuardResult:
    x_def: torch.Tensor
    spec: BlurGuardSpec
    sigmas: List[float]
    history: List[Dict] = field(default_factory=list)


def sam_masks(x01: torch.Tensor, ckpt=None,
              points_per_side: int = PAPER_SAM_POINTS_PER_SIDE) -> Dict[str, torch.Tensor]:
    """用 SAM 產生互斥的區域分割，照 `utils.generate_and_process_masks`。

    **相依不齊時直接拋出，不提供替代分割。** 用格點或隨機分割去頂替會靜默
    改變方法：BlurGuard 的每個 σ 綁在一個語意區域上，換成任意分割之後量到
    的就不是那一篇。
    """
    try:
        from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
    except ImportError as e:
        raise NotImplementedError(
            "BlurGuard 的遮罩需要 segment_anything（pip install segment-anything）"
            "與 SAM 檢查點。本檔不提供替代分割——見模組 docstring。"
        ) from e
    if ckpt is None:
        raise NotImplementedError("需要 SAM 檢查點路徑（ckpt=...）")

    import numpy as np

    img = (x01[0].permute(1, 2, 0).clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()
    model = sam_model_registry["vit_h"](checkpoint=str(ckpt)).to(x01.device)
    gen = SamAutomaticMaskGenerator(
        model=model, points_per_side=points_per_side, pred_iou_thresh=0.86,
        stability_score_thresh=0.92, crop_n_layers=1,
        crop_n_points_downscale_factor=2, min_mask_region_area=100)
    raw = gen.generate(img)

    out: Dict[str, torch.Tensor] = {}
    used = np.zeros_like(raw[0]["segmentation"], dtype=np.uint8)
    for i, m in enumerate(sorted(raw, key=lambda d: d["area"])):
        arr = np.where(m["segmentation"], 1, 0).astype(np.uint8)
        arr = np.where(used == 0, arr, 0)          # 互斥：先到先得
        used = np.where(arr > 0, 1, used)
        out[f"mask{i + 1}"] = torch.tensor(arr, dtype=torch.float32,
                                           device=x01.device)[None, None]
    left = np.where(used == 0, 1, 0).astype(np.uint8)
    if left.any():
        out[f"mask{len(out) + 1}"] = torch.tensor(
            left, dtype=torch.float32, device=x01.device)[None, None]
    return out


def check_partition(masks: Dict[str, torch.Tensor], atol: float = 1e-6) -> None:
    """遮罩必須是互斥且覆蓋全圖的分割。不成立時 `filter_delta` 會靜默地
    對某些像素套用兩個 σ 或一個都不套用。"""
    total = None
    for i in range(len(masks)):
        m = masks[f"mask{i + 1}"]
        total = m.clone() if total is None else total + m
    if total is None:
        raise ValueError("遮罩集合是空的")
    if float((total - 1.0).abs().max()) > atol:
        raise ValueError(
            f"遮罩不是分割：逐像素總和落在 [{float(total.min())}, "
            f"{float(total.max())}]，應恆為 1")


def run_blurguard(
    sd,
    x01: torch.Tensor,
    masks: Dict[str, torch.Tensor],
    spec: BlurGuardSpec = SPEC_PAPER,
    *,
    loss_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    seed: int = 0,
    log_every: int = 0,
) -> BlurGuardResult:
    """`LF_PGD.pgd_freq`，逐行對應。輸入輸出皆 `[0,1]`。

    `loss_fn` 未給時用該篇的目標：`mean(latent²)`。
    """
    check_partition(masks)
    if loss_fn is None:
        def loss_fn(x):
            return sd.encode_image(x).pow(2).mean()

    dev = x01.device
    x_pm1 = (x01 * 2.0 - 1.0).detach()
    eps = spec.eps_pixel01 * 2.0                     # `[-1,1]` 上的半徑
    n = len(masks)

    pert = torch.zeros_like(x_pm1, requires_grad=True)
    log_sigmas = torch.zeros(n, device=dev, dtype=x_pm1.dtype, requires_grad=True)
    opt = torch.optim.Adam([log_sigmas], lr=spec.lr)
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    history: List[Dict] = []

    for i in range(spec.steps):
        step = spec.step_size - (spec.step_size - spec.step_size / 100.0) \
            / spec.steps * i
        opt.zero_grad()
        pert.requires_grad_(True)

        # ---- 對抗項：σ 在此處是 detach 的（原始碼同此）----
        pert_lf = filter_delta(log_sigmas.detach(), pert, masks).clamp(-eps, eps)
        x_adv = (x_pm1 + pert_lf).clamp(-1.0, 1.0)
        adv_loss = loss_fn((x_adv + 1.0) / 2.0)

        # ---- σ 項：用**每步重抽**的隨機雜訊，不是 pert ----
        rnd = torch.randn(x_pm1.shape, generator=gen,
                          dtype=x_pm1.dtype).to(dev).clamp(-eps, eps)
        rnd_lf = filter_delta(log_sigmas, rnd, masks).clamp(-eps, eps)
        x_rob = (x_pm1 + rnd_lf).clamp(-1.0, 1.0)
        sigma_loss = torch.relu(spectrum_deviation(x_pm1, x_rob) - spec.eps_sigma)

        if i < spec.warmup:
            total = spec.sigma_weighting * sigma_loss
        else:
            total = adv_loss + spec.sigma_weighting * sigma_loss
        total.backward()

        if i < spec.warmup:
            opt.step()                                # 只有暖身期更新 σ
        if pert.grad is not None:
            with torch.no_grad():
                g = pert.grad
                nrm = g.reshape(g.shape[0], -1).norm(p=2, dim=1) + 1e-10
                g = g / nrm.view(-1, 1, 1, 1) * 2.0    # 原始碼的 ×2
                pert = (pert - step * g).detach().requires_grad_(True)

        if log_every and (i % log_every == 0 or i == spec.steps - 1):
            history.append({"step": i, "adv": float(adv_loss.detach()),
                            "sigma": float(sigma_loss.detach()),
                            "sigmas": log_sigmas.exp().detach().tolist()})
            print(f"    [blurguard] step {i:4d} adv {float(adv_loss):.4f} "
                  f"spec {float(sigma_loss):.4f}", flush=True)

    with torch.no_grad():
        pert_lf = filter_delta(log_sigmas, pert, masks).clamp(-eps, eps)
        x_def = ((x_pm1 + pert_lf).clamp(-1.0, 1.0) + 1.0) / 2.0
    return BlurGuardResult(x_def.detach(), spec,
                           log_sigmas.exp().detach().tolist(), history)


class BlurGuardParam:
    """`param_pgd.Parameterization` 的實作 —— 消融用，**不是** baseline 用。

    σ **不最佳化**：本專案的共用迴圈只有一條 sign 更新式，硬把 Adam 的 σ 塞
    進去就不再是「唯一變因是參數化」。故 σ 由 `sigmas` 明給（實務上取
    `run_blurguard` 跑出來的值），此處只最佳化擾動本身。

    `radius` 即 ε（`[0,1]` 尺度）。`set_radius` 讓 `fit_to_budget` 可以把它
    對齊到與相位臂相同的 DISTS。
    """

    name = "blurguard"

    def __init__(self, masks: Dict[str, torch.Tensor],
                 sigmas: Optional[List[float]] = None,
                 radius: float = PAPER_EPSILON_255 / 255.0):
        check_partition(masks)
        self.masks = masks
        self.radius = radius
        self.sigmas = sigmas if sigmas is not None else [1.0] * len(masks)
        if len(self.sigmas) != len(masks):
            raise ValueError(
                f"sigmas 有 {len(self.sigmas)} 個，遮罩有 {len(masks)} 個")
        self.delta: Optional[torch.Tensor] = None
        self._log_sigmas: Optional[torch.Tensor] = None

    def reset(self, x01: torch.Tensor, seed: int) -> None:
        self.delta = torch.zeros_like(x01, requires_grad=True)
        self._log_sigmas = torch.log(torch.tensor(
            self.sigmas, device=x01.device, dtype=x01.dtype))

    def render(self, x01: torch.Tensor) -> torch.Tensor:
        d = filter_delta(self._log_sigmas, self.delta, self.masks)
        return (x01 + d.clamp(-self.radius, self.radius)).clamp(0.0, 1.0)

    def params(self) -> List[torch.Tensor]:
        return [self.delta]

    @torch.no_grad()
    def project(self) -> None:
        self.delta.clamp_(-self.radius, self.radius)

    def set_radius(self, r: float) -> None:
        self.radius = r

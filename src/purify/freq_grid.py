"""兩個針對性淨化算子：GrIDPure 與頻域淨化（FD-Pure）。

DEC-025 把本輪的淨化算子縮到 `blur`／`crop`／`jpeg`／`freqpure`／`gridpure`。
前三個 `ops.py` 已有，本檔補後兩個。兩者都建立在 `diffpure._load_guided`
載入的同一個 guided-diffusion 256² 無條件檢查點上，不新增權重相依。

**「freqpure」這個名字有陷阱，本檔用的不是那一篇。**
────────────────────────────────────────────────────────────────────
FreqPure（Ju, Xue, Lyu；ICCV 2025 Workshop APAI）**不是**逆向過程中的頻域
介入，而是一條**兩階段的訓練式管線**：一個重建模組先移除保護擾動造成的
瑕疵，再由一個以低頻影像為條件的擴散模型合成高頻。它需要在 FFHQ 人臉資料
上訓練，且無公開程式碼（2026-08-19 查證），**不可能在無訓練的前提下重現**。

`docs/reference/SURVEY_2026-08-18_frequency.md` §2.3 把「逐時間步替換低頻
幅度、投影低頻相位」寫成 FreqPure 的作法，那是錯的——該條目自己註明內容
「由檢索摘要確認」，即未讀原文。正確的歸屬是下面這一篇。

本檔實作的是 **FD-Pure**（Pei, Ma, Sun, Xu, Huang, arXiv:2505.01267,
"Diffusion-based Adversarial Purification from the Perspective of the
Frequency Domain"）。它訓練自由、Algorithm 1 完整可實作，機制正是 survey
誤記在 FreqPure 名下的那一套，也正是本專案最需要測的那一套：**它刻意保留
輸入影像的低頻相位**，而紋理重相位把訊息編碼在相位上。
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F

from src.purify.diffpure import (
    DIFFPURE_RESOLUTION, _load_guided, resize_roundtrip,
)

# ---------------------------------------------------------------------------
# GrIDPure（Zhao et al., CVPR 2024, arXiv:2312.00084）
# ---------------------------------------------------------------------------
#
# 論文正文完整規定的部分（圖 9 說明文字與 §5.2「Implementation of Iteration」）：
#
#   * 512×512 的影像切成**九個 256×256 網格、重疊 128 像素**（stride 128），
#     四個角落的 128×128 另組成第十個 256×256 網格，**共十格**，使每一塊
#     像素至少落在兩個網格裡
#   * 每格以「小步 SDEdit」淨化，格的邊長等於無條件擴散模型的原生解析度
#     （256²）——**故 GrIDPure 不需要 resize**，這正是它相對 DiffPure 的賣點
#   * 重疊處取所有覆蓋它的網格的**平均**
#   * 每次迭代之後與前一輪混合：`x_{i+1} = (1 − γ)·x̃_i + γ·x_i`
#
# **論文正文未載的部分（本專案指定，報表必須標註）**：小步的 `t`、混合權重
# `γ`、迭代次數 `N`。CVF 版正文的表格只標 DiffPure 的 t=50／100，沒有標
# GrIDPure 自己的三個值。處置方式與 `diffpure.py` 對 resize 的處置相同：
# 設為必填參數、不給預設，由呼叫端明給並記錄。
GRIDPURE_GRID = DIFFPURE_RESOLUTION      # 256，等於檢查點原生解析度（論文規定）
GRIDPURE_STRIDE = 128                    # 論文圖 9 明載
GRIDPURE_UNSPECIFIED = ("t", "gamma", "iters")


def grid_specs(size: int, grid: int = GRIDPURE_GRID,
               stride: int = GRIDPURE_STRIDE) -> list:
    """回傳每一格的取法 `(shift_t, shift_l, top, left)`。

    取法是「先把影像環狀平移 `(shift_t, shift_l)`，再裁 `[top:top+grid,
    left:left+grid]`」；`shift = 0` 就是一般的裁切。

    對 512² 回傳 **十格**，與論文圖 9 的格數一致：

    * 九格是規則格點（起點 0／128／256，`shift = 0`）
    * 第十格是論文說的「四個 128×128 角落合併成一個 256×256」。

    **第十格為什麼用環狀平移**：把影像平移半個邊長之後，四個角落恰好拼成
    正中央一塊連續的 `grid × grid`——`x[-128:128, -128:128]` 平移後即
    `rolled[128:384, 128:384]`。這不是近似，是那四塊角落的精確重排，且
    相鄰關係與論文「合併成一個 256×256」的描述一致。論文沒有寫出拼接的
    座標，此處的具體拼法是本專案的實作選擇。

    沒有第十格時，四個 128×128 的角落只被一格覆蓋，違反論文 §5.2 步驟 (1)
    「每一塊像素至少落在兩個網格裡」——`tests/test_freq_grid.py` 釘住這件事。
    """
    if size < grid:
        raise ValueError(f"影像邊長 {size} 小於網格 {grid}")
    if size % 2:
        raise ValueError(f"角落格需要偶數邊長，收到 {size}")
    xs = list(range(0, size - grid + 1, stride))
    if xs[-1] != size - grid:
        xs.append(size - grid)
    specs = [(0, 0, t, l) for t in xs for l in xs]
    half = size // 2
    corner = half - grid // 2
    if corner < 0:
        raise ValueError(f"影像邊長 {size} 不足以容納角落格 {grid}")
    specs.append((half, half, corner, corner))
    return specs


def _small_step_sdedit(x01: torch.Tensor, t: int, model, diffusion,
                       gen=None) -> torch.Tensor:
    """一格的小步 SDEdit：加噪到 `t` 再逐步去噪回 0。值域 `[0,1]` 進出。

    與 `diffpure.diffpure_real` 的內圈同一條路徑，只是 `t` 小很多且不做
    resize——輸入本來就是 256²。
    """
    a = (1.0 - torch.from_numpy(diffusion.betas).float().to(x01.device)).cumprod(0)
    x0 = x01 * 2.0 - 1.0
    e = torch.randn(x0.shape, generator=gen, dtype=x0.dtype).to(x0.device)
    x = x0 * a[t - 1].sqrt() + e * (1.0 - a[t - 1]).sqrt()
    with torch.no_grad():
        for i in reversed(range(t)):
            tt = torch.full((x.shape[0],), i, device=x.device, dtype=torch.long)
            x = diffusion.p_sample(model, x, tt, clip_denoised=True)["sample"]
    return ((x + 1.0) / 2.0).clamp(0, 1)


def gridpure_real(x01: torch.Tensor, *, t: int, gamma: float, iters: int,
                  ckpt=None, seed: Optional[int] = None,
                  grid: int = GRIDPURE_GRID,
                  stride: int = GRIDPURE_STRIDE) -> torch.Tensor:
    """GrIDPure。`t`／`gamma`／`iters` **必填**——論文正文未載，見上方說明。

    每一輪：切格 → 各格小步 SDEdit → 重疊處取平均合回 → 與上一輪混合。
    """
    if x01.dim() != 4:
        raise ValueError(f"需要 (B,C,H,W)，收到 {tuple(x01.shape)}")
    h, w = x01.shape[-2:]
    if h != w:
        raise ValueError(f"目前只支援正方形影像，收到 {h}×{w}")
    model, diffusion = _load_guided(ckpt, device=x01.device)
    gen = (torch.Generator(device="cpu").manual_seed(int(seed))
           if seed is not None else None)
    specs = grid_specs(h, grid, stride)

    cur = x01.clamp(0, 1)
    for _ in range(iters):
        acc = torch.zeros_like(cur)
        cnt = torch.zeros_like(cur)
        for (st, sl, top, left) in specs:
            src = (cur if (st == 0 and sl == 0)
                   else torch.roll(cur, shifts=(st, sl), dims=(-2, -1)))
            out = _small_step_sdedit(
                src[..., top:top + grid, left:left + grid], t, model, diffusion, gen)
            a_pad = torch.zeros_like(cur)
            c_pad = torch.zeros_like(cur)
            a_pad[..., top:top + grid, left:left + grid] = out
            c_pad[..., top:top + grid, left:left + grid] = 1.0
            if st or sl:                       # 平移回原座標
                a_pad = torch.roll(a_pad, shifts=(-st, -sl), dims=(-2, -1))
                c_pad = torch.roll(c_pad, shifts=(-st, -sl), dims=(-2, -1))
            acc += a_pad
            cnt += c_pad
        if float(cnt.min()) < 2.0:
            raise RuntimeError(
                "有像素只被一個網格覆蓋，違反論文『每一塊至少落在兩格』的規定")
        merged = acc / cnt
        cur = ((1.0 - gamma) * merged + gamma * cur).clamp(0, 1)
    return cur


# ---------------------------------------------------------------------------
# FD-Pure（Pei et al., arXiv:2505.01267）
# ---------------------------------------------------------------------------
#
# Algorithm 1（逐行）：
#     x_t ~ N(0, I)
#     for t = t*, …, 1:
#         x_{0|t} = (x_t − √(1−ᾱ_t)·ε_θ(x_t,t)) / √ᾱ_t
#         (A_{0|t}, P_{0|t}) = DFT(x_{0|t});  (A_adv, P_adv) = DFT(x_adv)
#         Â = A_{0|t}·(1−H_A) + A_adv·H_A                      # 式 12
#         P̂ = Π_{P_L ± δ}( P_adv·H_P ) + P_{0|t}·(1−H_P)       # 式 13
#         x̂_{0|t} = iDFT(Â, P̂)                                 # 式 14
#         x_{t−1} ~ p(x_{t−1} | x_t, x̂_{0|t})                   # 式 15＝DDPM 後驗
#
# 濾波器 H(u,v) = 1 當 D(u,v) < D_thresh，否則 0；D 是到頻譜中心的距離（式 10–11）。
#
# 超參數：論文附錄 A.1 給 **D_A = 3、D_P = 2、δ = 0.2**，量在 CIFAR-10 的
# 32×32 上。本專案在 512² 上工作（經 resize 到 256² 進模型），故半徑必須換算。
# **換算方式是本專案指定的**：按 Nyquist 比例縮放，`D_A/16` 與 `D_P/16` 乘上
# 目標邊長的一半。t* 論文未在正文載明，設為必填。
FDPURE_DA_CIFAR = 3.0
FDPURE_DP_CIFAR = 2.0
FDPURE_DELTA = 0.2
FDPURE_CIFAR_NYQUIST = 16.0     # 32×32 的最大半徑


def scale_radius(d_cifar: float, size: int) -> float:
    """把 CIFAR-10 32² 上的半徑按 Nyquist 比例換算到 `size²`。**本專案指定。**"""
    return d_cifar / FDPURE_CIFAR_NYQUIST * (size / 2.0)


def lowpass_mask(size: int, radius: float, device=None,
                 dtype=torch.float32) -> torch.Tensor:
    """式 10–11 的理想低通遮罩，(1,1,size,size)，**中心在頻譜中央**。

    與 `torch.fft.fft2` 的輸出對齊：後者把零頻放在角落，故此處建好中心版本
    之後用 `ifftshift` 搬回去，而不是對頻譜做 `fftshift`——後者會多兩次
    搬移且容易在奇偶邊界上錯一格。
    """
    u = torch.arange(size, device=device, dtype=dtype).view(size, 1)
    v = torch.arange(size, device=device, dtype=dtype).view(1, size)
    d = ((u - size / 2.0) ** 2 + (v - size / 2.0) ** 2).sqrt()
    m = (d < radius).to(dtype)
    return torch.fft.ifftshift(m).view(1, 1, size, size)


def _project_phase(p_ref: torch.Tensor, p_est: torch.Tensor,
                   delta: float) -> torch.Tensor:
    """把 `p_est` 投影到 `p_ref ± δ` 內。相位是週期量，差值先繞回 (−π, π]。"""
    diff = torch.remainder(p_est - p_ref + math.pi, 2 * math.pi) - math.pi
    return p_ref + diff.clamp(-delta, delta)


def fdpure_real(x01: torch.Tensor, *, t_star: int,
                d_a: Optional[float] = None, d_p: Optional[float] = None,
                delta: float = FDPURE_DELTA, ckpt=None,
                seed: Optional[int] = None) -> torch.Tensor:
    """FD-Pure（arXiv:2505.01267）Algorithm 1。輸入輸出 `(B,3,H,W)`、`[0,1]`。

    `d_a`／`d_p` 未給時由 `scale_radius` 從論文的 CIFAR-10 值換算到模型的
    原生解析度（256²）。`t_star` 必填——論文正文未載。

    解析度沿用 `diffpure.resize_roundtrip`（降到 256 → 淨化 → 升回原尺寸），
    與既有的 `diffpure` 算子一致，故兩者的差異純粹來自演算法而不是取樣。
    """
    if x01.dim() != 4:
        raise ValueError(f"需要 (B,C,H,W)，收到 {tuple(x01.shape)}")
    model, diffusion = _load_guided(ckpt, device=x01.device)
    gen = (torch.Generator(device="cpu").manual_seed(int(seed))
           if seed is not None else None)
    n = DIFFPURE_RESOLUTION
    ra = d_a if d_a is not None else scale_radius(FDPURE_DA_CIFAR, n)
    rp = d_p if d_p is not None else scale_radius(FDPURE_DP_CIFAR, n)

    def inner(small01: torch.Tensor) -> torch.Tensor:
        x_adv = small01 * 2.0 - 1.0
        z_adv = torch.fft.fft2(x_adv, dim=(-2, -1))
        a_adv, p_adv = z_adv.abs(), z_adv.angle()
        h_a = lowpass_mask(n, ra, x_adv.device, x_adv.dtype)
        h_p = lowpass_mask(n, rp, x_adv.device, x_adv.dtype)

        x = torch.randn(x_adv.shape, generator=gen,
                        dtype=x_adv.dtype).to(x_adv.device)      # 行 1
        with torch.no_grad():
            for i in reversed(range(t_star)):
                tt = torch.full((x.shape[0],), i, device=x.device,
                                dtype=torch.long)
                out = diffusion.p_mean_variance(model, x, tt, clip_denoised=True)
                x0t = out["pred_xstart"]                          # 行 3

                z = torch.fft.fft2(x0t, dim=(-2, -1))             # 行 4
                amp = z.abs() * (1.0 - h_a) + a_adv * h_a         # 行 7（式 12）
                pha = (_project_phase(p_adv, z.angle(), delta) * h_p
                       + z.angle() * (1.0 - h_p))                 # 行 9（式 13）
                x0t = torch.fft.ifft2(
                    amp * torch.exp(1j * pha), dim=(-2, -1)).real.clamp(-1, 1)

                mean, _, log_var = diffusion.q_posterior_mean_variance(
                    x_start=x0t, x_t=x, t=tt)                     # 行 12（式 15）
                if i > 0:
                    noise = torch.randn(x.shape, generator=gen,
                                        dtype=x.dtype).to(x.device)
                    x = mean + (0.5 * log_var).exp() * noise
                else:
                    x = mean
        return ((x + 1.0) / 2.0).clamp(0, 1)

    return resize_roundtrip(x01, inner=inner)

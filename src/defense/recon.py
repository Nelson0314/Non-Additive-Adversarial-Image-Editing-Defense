"""重建下限的壓縮：latent 對齊（A1）與解碼器逐圖微調（A2）。

DEC-016 A 段。走生成路徑的條件（`apa`／`Ra`／`N3`）產生防禦圖的方式是
`decode(BDIA 反演(encode(x)) 之後注入 φ)`，即使 φ=0，`decode(encode(x))`
也不等於 `x`。那個差就是**重建下限**，它把整條失真預算軸的原點往上推。

為什麼是這兩件事而不是現行的階段一（FND-016）
──────────────────────────────────────────────────────────────────────
`optimize.align` 訓練的是 UNet 上的 LoRA，而下限來自 **VAE 的編解碼來回**，
LoRA 碰不到它；再者 APA 原文的階段一補的是 DDIM 的反演誤差，本專案用 BDIA
精確反演，那一半誤差本來就不存在。實測 200 步只讓 LPIPS 由 0.15806 走到
0.15768。故階段一要搬到真正產生誤差的兩個位置：

- **A1 latent 對齊**：起點不再是 `encode(x)`，改為解 `z*` 使 `decode(z*) ≈ x`。
  參數量與既有 latent 相同（SD v1.4／512² 為 4×64×64），不新增模組。
- **A2 解碼器逐圖微調**：在 VAE 解碼器上開一組小參數對這一張影像過擬合。

A2 為什麼必須有硬停止條件
──────────────────────────────────────────────────────────────────────
解碼器若把原圖背得太熟，會對 latent 的擾動變得遲鈍——在 latent 上加東西它
照樣吐原圖，防禦就失去表達管道。故 `descend` 以「達到目標即停」為停止規則，
**不跑到收斂**，且可訓練的參數限於 GroupNorm 的 affine 與各層 conv 的 bias
（數萬個，而非整個解碼器的 5×10⁷）。停止步數與當時的值一律落盤。
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from src.metrics.acutance import gradient_energy


def decoder_tunable(decoder: nn.Module) -> List[Tuple[str, nn.Parameter]]:
    """逐圖微調要開的參數：GroupNorm 的 affine 與各層 conv 的 bias。

    為什麼不是全參數微調：整個解碼器約 5×10⁷ 個參數，逐圖存下來是 198 MB／張，
    `runs/` 是唯一的證據來源而該體積無法逐圖入庫。更重要的是容量本身就是
    停止條件的一部分——參數愈多，「把原圖背起來」愈快發生（見模組 docstring）。

    只認 `nn.GroupNorm` 與 `nn.Conv2d`，不比對層的名稱：名稱在 diffusers
    改版時會變，而「哪一種層」不會。解碼器裡的 attention 走 `nn.Linear`，
    其 bias **不在此列**——挑的是逐通道的仿射自由度（GroupNorm 的
    weight/bias 與 conv 的 bias），那正是不改變感受野也不改變空間結構的
    那一組。
    """
    picked: List[Tuple[str, nn.Parameter]] = []
    for name, mod in decoder.named_modules():
        if isinstance(mod, nn.GroupNorm):
            for suffix in ("weight", "bias"):
                p = getattr(mod, suffix, None)
                if p is not None:
                    picked.append((f"{name}.{suffix}", p))
        elif isinstance(mod, nn.Conv2d) and mod.bias is not None:
            picked.append((f"{name}.bias", mod.bias))
    if not picked:
        raise ValueError(
            "解碼器裡找不到任何 GroupNorm affine 或 conv bias，"
            "逐圖微調沒有東西可以更新")
    return picked


@contextmanager
def restored(params: Sequence[nn.Parameter], trainable: bool = True):
    """在區塊內讓 `params` 可訓練，離開時把數值原樣還原。

    逐圖微調是**針對這一張影像**的，下一張影像必須從 stock 權重重新開始。
    不還原的話第二張圖的起點是第一張圖的過擬合結果，而那在輸出上沒有症狀
    ——下限一樣會降，只是降的原因變成兩張圖的混合，跨影像的比較就毀了。

    `SDWrapper` 把 VAE 整個 `requires_grad_(False)`，故此處要顯式打開；
    離開時一併關回去。
    """
    saved = [(p, p.detach().clone(), p.requires_grad) for p in params]
    if trainable:
        for p in params:
            p.requires_grad_(True)
    try:
        yield
    finally:
        with torch.no_grad():
            for p, value, flag in saved:
                p.copy_(value)
                p.requires_grad_(flag)
                p.grad = None


def blunting_penalty(y: torch.Tensor, x01: torch.Tensor, tau_ratio: float
                     ) -> torch.Tensor:
    """只罰「比 `tau_ratio` 更鈍」：`max(0, τ − 銳利度比)`。

    量與 `MetricSuite.pairwise` 的 `acutance_ratio` 是同一個（`gradient_energy`
    的比值），故報表上看到的數字就是被約束的數字。

    **只罰鈍化、不罰過銳**：本階段唯一要防的是「拿變模糊換低下限」；比原圖
    更銳在重建對齊裡不會發生（下降方向是往原圖走），為它加一道對稱的懲罰
    只會多一個沒有作用卻會影響梯度尺度的項。

    為什麼不是專案在防禦階段用的 `local_acutance_dev`：那一項的門檻在此處
    必須取起點自己的值（判準是「不可以比 VAE 來回更差」），於是約束**從第 0
    步就是緊的**，而 γ=100 的 hinge 在起點就是一道陡壁。2026-08-10 於
    horse_00 實測：加上它之後 LPIPS 由 0.1283 **上升**到 0.1418 再也沒回來，
    120 步全在壁上，等於整段對齊被關掉。此處的量級（比值 ≈ 1，可動範圍
    ~0.2）與係數搭得起來，`local_acutance_dev`（≈0.096，可動範圍 1e-4）搭不起來。
    """
    r = (gradient_energy(y.float().clamp(0, 1))
         / gradient_energy(x01.float().clamp(0, 1)).clamp_min(1e-12))
    return torch.clamp(tau_ratio - r, min=0.0).mean()


def reconstruction_loss(y: torch.Tensor, x01: torch.Tensor,
                        lpips_fn: Callable, w_lpips: float, w_pixel: float,
                        gamma_acut: float = 0.0, tau_acut: float = 0.0
                        ) -> torch.Tensor:
    """重建損失：感知項、逐像素項，加一道鈍化 hinge。

    前兩項都要。只有 LPIPS 時 PSNR 會自由漂移（E9 實測 car_01 的 PSNR 在對齊
    後反而掉 1.84 dB）；只有逐像素項時影像會變鈍。

    第三項不是可選的裝飾。2026-08-10 於 horse_00 實測：不加它時 A1 把 LPIPS
    由 0.1283 壓到 0.0829，**銳利度比同時由 0.9935 掉到 0.7887**——那個下限
    有一部分是拿變鈍換來的，而使用者在 0.664 的銳利度比上判過「看得出來」。
    先驗紀錄的同一件事：高頻保留率 84.2%，調高感知項權重後才回到 93.2%。

    門檻 `tau_acut` 由呼叫端取**該影像自己的舊下限**的銳利度比（判準與
    `optimize.recon_floor_thresholds` 同一條線：不可以比 VAE 來回更差）。
    `gamma_acut=0` 關閉本項。
    """
    loss = (w_lpips * lpips_fn(y, x01).mean()
            + w_pixel * (y - x01).abs().mean())
    if gamma_acut:
        loss = loss + gamma_acut * blunting_penalty(y, x01, tau_acut)
    return loss


def descend(
    params: Sequence[nn.Parameter],
    forward: Callable[[], torch.Tensor],
    loss_fn: Callable[[torch.Tensor], torch.Tensor],
    measure: Callable[[torch.Tensor], Dict[str, float]],
    *,
    steps: int,
    lr: float,
    key: str,
    target: Optional[float] = None,
    log_every: int = 10,
    tag: str = "",
) -> Tuple[List[Dict], Dict]:
    """Adam 下降，每 `log_every` 步量一次指標，`measure()[key]` 達標即停。

    回傳 `(history, summary)`。`summary` 記 `reached`（有沒有達到 `target`）、
    `stop_step`（實際停在第幾步）、`best_step` 與 `best`（`key` 的最佳值）。

    **達不到 target 不是例外，是結果**：本函式如實記 `reached=False` 並回傳
    最佳步的參數，由呼叫端決定怎麼呈現。把它變成 raise 會讓一張圖的容量上限
    毀掉整批的量測，而那個上限本身就是要報的東西（E9 實測 car_01 在第 45 步
    停在 LPIPS 0.1634，其後 155 步無改善）。

    `target=None` 時跑滿 `steps`，仍回傳最佳步——`key` 在逐步之間會震盪，
    取末步等於把震盪的相位當成結果。
    """
    opt = torch.optim.Adam(params, lr=lr)
    history: List[Dict] = []
    best, best_step = float("inf"), -1
    best_state: List[torch.Tensor] = []
    stop_step, reached = steps, False

    for step in range(steps + 1):
        y = forward()
        loss = loss_fn(y)
        if step % log_every == 0 or step == steps:
            with torch.no_grad():
                m = measure(y.detach())
            history.append({"step": step, "loss": float(loss.detach()), **m})
            print(f"  [{tag}] step {step:>4d}  loss={float(loss.detach()):.5f}  "
                  + "  ".join(f"{k}={v:.4f}" for k, v in m.items()), flush=True)
            if m[key] < best:
                best, best_step = m[key], step
                best_state = [p.detach().clone() for p in params]
            if target is not None and m[key] <= target:
                stop_step, reached = step, True
                print(f"  [{tag}] 第 {step} 步達到目標 {key}≤{target:.4f}"
                      f"（{m[key]:.4f}），停止", flush=True)
                break
        if step == steps:
            stop_step = step
            break
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    if best_step < 0:
        raise RuntimeError("下降迴圈一次也沒有量到指標，log_every 或 steps 有誤")
    with torch.no_grad():
        for p, value in zip(params, best_state):
            p.copy_(value)
    return history, {"reached": reached, "stop_step": stop_step,
                     "best_step": best_step, "best": best,
                     "target": target, "lr": lr, "steps": steps}


def align_latent(
    sd,
    x01: torch.Tensor,
    lpips_fn: Callable,
    measure: Callable[[torch.Tensor], Dict[str, float]],
    *,
    steps: int,
    lr: float,
    key: str = "lpips",
    target: Optional[float] = None,
    w_lpips: float = 1.0,
    w_pixel: float = 0.5,
    gamma_acut: float = 0.0,
    tau_acut: float = 0.0,
    log_every: int = 10,
) -> Tuple[torch.Tensor, List[Dict], Dict]:
    """A1：解 `z*` 使 `decode(z*) ≈ x`。回傳 `(z*, history, summary)`。

    起點取 `encode(x)`——那正是現行路徑用的值，故第 0 步量到的就是舊下限，
    整條曲線因此可以直接讀成「比現行好多少」。

    `decode_latent` 的輸出有 `clamp(0, 1)`，飽和處梯度為零。這裡刻意不繞過它：
    量測與訓練必須用同一個 `decode_latent`，否則壓下來的下限與實際生成路徑
    產生的影像對不起來。
    """
    with torch.no_grad():
        z0 = sd.encode_image(x01).detach()
    z = z0.clone().requires_grad_(True)
    history, summary = descend(
        [z],
        forward=lambda: sd.decode_latent(z),
        loss_fn=lambda y: reconstruction_loss(y, x01, lpips_fn, w_lpips,
                                              w_pixel, gamma_acut, tau_acut),
        measure=measure, steps=steps, lr=lr, key=key, target=target,
        log_every=log_every, tag="A1")
    summary.update({"w_lpips": w_lpips, "w_pixel": w_pixel,
                    "gamma_acut": gamma_acut, "tau_acut": tau_acut,
                    "n_params": int(z.numel())})
    return z.detach(), history, summary


def finetune_decoder(
    sd,
    x01: torch.Tensor,
    z: torch.Tensor,
    params: Sequence[nn.Parameter],
    lpips_fn: Callable,
    measure: Callable[[torch.Tensor], Dict[str, float]],
    *,
    steps: int,
    lr: float,
    target: float,
    key: str = "lpips",
    w_lpips: float = 1.0,
    w_pixel: float = 0.5,
    gamma_acut: float = 0.0,
    tau_acut: float = 0.0,
    log_every: int = 5,
) -> Tuple[List[Dict], Dict]:
    """A2：固定 `z`，只更新 `params` 對這一張影像過擬合。

    `z` 固定的理由是歸因：第 3 欄（A1）與第 4 欄（A1+A2）之間只差解碼器這
    一個變因。兩者一起動會得到更低的下限，但分不出哪一半在起作用。

    `target` 是**必填**。這一段的停止規則就是它（見模組 docstring 的
    「A2 為什麼必須有硬停止條件」），沒有預設值可以沿用。

    呼叫端負責用 `restored` 包住本函式，本函式不自行還原權重——它回傳之後
    呼叫端還要用微調過的解碼器去量指標與存圖。
    """
    return descend(
        list(params),
        forward=lambda: sd.decode_latent(z),
        loss_fn=lambda y: reconstruction_loss(y, x01, lpips_fn, w_lpips,
                                              w_pixel, gamma_acut, tau_acut),
        measure=measure, steps=steps, lr=lr, key=key, target=target,
        log_every=log_every, tag="A2")


def latent_response(sd, z: torch.Tensor, pairwise: Callable,
                    *, seed: int, scale: float) -> Dict[str, float]:
    """解碼器對 latent 擾動的反應強度：`pairwise(decode(z), decode(z+δ))`。

    存在的理由是 A2 的風險本身（模組 docstring）：微調過的解碼器若把原圖背
    起來，防禦在 latent 上的擾動就傳不到輸出。這個量在微調前後各測一次，
    比值掉太多即代表管道被關掉了——那不是靠停止步數「應該還好」推斷，是量
    出來的。

    δ 由固定 seed 產生並縮放到 `scale × ‖z‖`，故微調前後餵的是同一個方向、
    同一個能量，兩次的差只來自解碼器。
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    d = torch.randn(z.shape, generator=g).to(z.device, z.dtype)
    d = d * (scale * z.norm() / d.norm())
    with torch.no_grad():
        y0 = sd.decode_latent(z)
        y1 = sd.decode_latent(z + d)
    return pairwise(y0, y1)

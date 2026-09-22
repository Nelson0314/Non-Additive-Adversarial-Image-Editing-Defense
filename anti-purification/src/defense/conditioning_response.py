"""UNet 對**文字條件輸入**的局部敏感度，量在固定的空文字點上。

為什麼既有代理量不到這件事
────────────────────────────────────────────────────────────────────
`instruction_free.FreeObjective.null_edit` 在空文字下只跑兩個分支：文字分支與
影像分支的條件相同，`s_t·(ε_text − ε_image)` 恆等於零。所以 `enc`／`cond`／`id`
三項**結構上**只看得到影像導引，不論防禦把文字分支壓成什麼樣，它們都不會變。

本模組換一個量：固定 `z`、`t` 與影像條件 `c`，只對**文字嵌入**做中央差分

    D(c; v, h) = [ ε(z, t, c, e₀ + hσ_e v) − ε(z, t, c, e₀ − hσ_e v) ] / (2h)

`e₀` 是空字串的嵌入，`v` 是固定種子抽出的隨機方向（RMS 正規化為 1），
`σ_e = RMS(e₀)`。防禦圖那一側的 `‖D‖²` 除以原圖那一側同一個量，得到反應比值。

**`v` 不是指令。** 它由隨機數產生器產出，不查 tokenizer 詞表、不抽 token、
不做最近詞句搜尋，也不由任何評估結果挑選。整個模組唯一碰到 text encoder 的
地方是取空字串的 `e₀`。

這個量能支持什麼、不能支持什麼
────────────────────────────────────────────────────────────────────
它量的是 `e₀` 這一點、`v` 這一個方向上的**局部**敏感度。三件它**不**保證：

- 單一方向的下降不蘊含其餘方向下降。驗證要用沒有參與更新的方向。
- 局部導數不控制有限位移：`ε(e₀+d) − ε(e₀) = ∫₀¹ J_e ε(e₀+sd) d ds`，
  壓低一點的導數沒有控制整條積分路徑，真實指令走的是那條路徑。
- 隨機方向未必落在自然語言嵌入所佔的區域。

所以比值下降**不等於**指令被擋掉；反過來，比值推不動則是強的否定證據——
那表示影像側對文字分支沒有控制權。

數值
────────────────────────────────────────────────────────────────────
模型跑 bf16，差分的兩端先轉 fp32 再相減、平方、累加；這去掉累加的誤差，
但不會去掉 UNet 內部的 bf16 捨入。因此**一律跑兩個差分尺度**：真正的一階
反應在兩個尺度上應該接近，只有捨入主導時兩者才會差很多。
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import torch

from .instruction_free import null_text_embedding, pick_timesteps


def probe_directions(e0: torch.Tensor, count: int, seed: int) -> torch.Tensor:
    """(count, …) 個與 `e0` 同形狀、RMS 正規化為 1 的隨機方向。

    逐元素獨立抽樣，**不**把一個通道向量廣播到所有 token：那種方向在
    cross-attention 的 key 上會被 softmax 的共同位移抵銷，量到的成分與
    影像側能不能控制它是兩回事。
    """
    if count < 1:
        raise ValueError(f'方向數至少為 1，收到 {count}')
    g = torch.Generator(device='cpu').manual_seed(int(seed))
    v = torch.randn((count,) + tuple(e0.shape[1:]), generator=g)
    rms = v.flatten(1).pow(2).mean(1).sqrt().view(-1, *([1] * (v.ndim - 1)))
    return (v / rms).to(device=e0.device, dtype=e0.dtype)


@torch.no_grad()
def null_trajectory(ip2p, x01: torch.Tensor, steps: int, seed: int,
                    s_i: float = 1.5) -> List[Tuple[torch.Tensor, torch.Tensor,
                                                    torch.Tensor]]:
    """空文字下的取樣軌跡，回傳每一步**進 UNet 之前**的 `(t, z, σ_t)`。

    空文字時三分支退化成兩分支（`ε_text ≡ ε_image`），導引是
    `ε_uncond + s_i·(ε_image − ε_uncond)`，與 `null_edit` 同一式。
    軌跡在**原圖**上跑一次就凍結，兩側共用，所以比值的分子分母站在同一個
    latent 狀態上。
    """
    from copy import deepcopy

    sch = deepcopy(ip2p.pipe.scheduler)
    sch.set_timesteps(steps, device=ip2p.device)
    text = null_text_embedding(ip2p)
    cond = ip2p.image_latents(x01).to(text.dtype)
    gen = torch.Generator(device=ip2p.device).manual_seed(int(seed))
    z = torch.randn(cond.shape, device=ip2p.device, dtype=text.dtype,
                    generator=gen) * sch.init_noise_sigma
    extra = ip2p.pipe.prepare_extra_step_kwargs(gen, 0.0)
    out = []
    for i, t in enumerate(sch.timesteps):
        out.append((t, z.clone(), sch.sigmas[i].clone()))
        image_batch = torch.cat([cond, torch.zeros_like(cond)])
        noise_batch = sch.scale_model_input(torch.cat([z] * 2), t)
        eps = ip2p.unet(torch.cat([noise_batch, image_batch], dim=1), t,
                        encoder_hidden_states=torch.cat([text, text]),
                        return_dict=False)[0]
        image, uncond = eps.chunk(2)
        guided = uncond + s_i * (image - uncond)
        z = sch.step(guided, t, z, **extra, return_dict=False)[0]
    return out


def scale_latent(z: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """把軌跡上的 `z` 縮成該時刻的 UNet 輸入：`z / √(σ_t² + 1)`。

    不呼叫 `scheduler.scale_model_input`：`EulerAncestralDiscreteScheduler`
    的 `step_index` 只在第一次呼叫時依 timestep 初始化，之後要靠 `step()`
    前進，逐時刻獨立呼叫會全部套用**第一個**時刻的 σ。σ 由軌跡逐步帶出來，
    所以這裡是無狀態的。
    """
    s = sigma.to(z)
    return z / (s * s + 1).sqrt()


@torch.no_grad()
def finite_response(ip2p, scaled: torch.Tensor, t: torch.Tensor,
                    cond: torch.Tensor, e0: torch.Tensor, v: torch.Tensor,
                    h: float, sigma_e: float) -> torch.Tensor:
    """文字嵌入方向 `v` 上的中央差分，回傳 fp32 的 `D`。

    `scaled` 是已經過 `scale_latent` 的 latent。兩個條件張量在送進 UNet 之前
    會檢查**確實不同**——bf16 下 `h` 太小時 `e₀ ± hσ_e v` 會捨入成同一個張量，
    那時比值是零除零而不是零反應。不相符就拋錯，不靜默回傳零。
    """
    step = float(h) * float(sigma_e)
    plus = (e0 + step * v).to(e0.dtype)
    minus = (e0 - step * v).to(e0.dtype)
    if torch.equal(plus, minus):
        raise ValueError(
            f'h={h} 在 {e0.dtype} 下把 e₀±hσ_e·v 捨入成同一個張量；'
            f'差分沒有定義，換一個較大的 h 或較高的精度')
    model_input = torch.cat([scaled, cond.to(scaled.dtype)], dim=1)
    both = ip2p.unet(torch.cat([model_input, model_input]), t,
                     encoder_hidden_states=torch.cat([plus, minus]),
                     return_dict=False)[0]
    a, b = both.chunk(2)
    return (a.float() - b.float()) / (2.0 * float(h))


def response_energy(d: torch.Tensor) -> float:
    """`‖D‖²`。非有限就拋錯，不用 nan 當成零反應。"""
    e = float(d.pow(2).sum())
    if not torch.isfinite(torch.tensor(e)):
        raise ValueError('反應能量非有限，數值路徑有問題')
    return e


def timestep_probe(scheduler, count: int, steps: int) -> Sequence[int]:
    """沿完整時間表取 `count` 個時刻，涵蓋高噪到低噪。"""
    return pick_timesteps(scheduler, count, steps)

"""SDS 類梯度：不反傳穿過 UNet 的擴散損失。**不含文字，也不含任何編輯指令。**

`mainstream_terms.MainstreamTerms.diffusion`（AdvDM／Mist 的無目標項）要把
`∂/∂x ‖ε_θ(z_t) − ε‖²` 一路反傳穿過 UNet，記憶體與時間都由 UNet 主導。
Score Distillation Sampling 的作法是把 `∂ε_θ/∂z_t` 整項丟掉，只留

    ∂L/∂z₀ ≈ (2/N) · (ε_θ(z_t) − ε) · ∂z_t/∂z₀

`ε_θ` 因此只需要一次**無梯度**前向，梯度只反傳穿過 VAE encoder。文獻上丟掉
的那一項被認為是 U-Net 的 Jacobian，在小時刻上接近病態；這裡不主張近似更好，
只主張它更便宜——**回報的損失值與 `diffusion` 逐位元同一個函數**，所以兩者的
差別只在梯度，可以直接對照。

為什麼值與梯度要分開
────────────────────────────────────────────────────────────────────
選 checkpoint 與收斂判定走 `eval_terms`，那裡要的是**損失本身**；最佳化要的
是梯度。把兩者綁在同一個張量上的寫法是

    value = mse + (surrogate − surrogate.detach())

前向值等於 `mse`，反向梯度等於 `∂surrogate/∂z₀`。這樣一來 CSV 裡的
`free_term_sds` 與 `free_term_diffusion` 是同一個量綱，可以並排讀。

**括號不能拿掉。** `mse + surrogate − surrogate.detach()` 在 float32 下是先加
再減，而 `surrogate` 的量級與 `mse` 無關，可以大上好幾個數量級——相加再相減
會把 `mse` 的低位吃掉，回報的損失因此帶著一個與梯度大小相關的誤差。
先算括號裡的差，它逐位元是零，前向值才等於 `mse` 本身。

`w(t)` 一律取 1
────────────────────────────────────────────────────────────────────
文獻上的 SDS 帶一個時刻權重 `w(t)`。這裡**不加**：`diffusion` 對它的 k 個時刻
是等權平均，加上權重就不再是同一個損失，比較會同時變兩件事。權重寫錯的失效
是靜默的——符號或縮放弄反等於在幫攻擊者把圖修回去——所以先把可比性釘住，
要掃權重是另一批的事。`sds_sign_check` 是上卡前的守門。

加噪與 `σ` 的形式沿用 `mainstream_terms`：
`z_t = (z₀ + σ_t·ε) / √(σ_t² + 1)`，所以 `∂z_t/∂z₀ = 1/√(σ_t² + 1)`。
"""
from __future__ import annotations

from typing import Dict

import torch

from .instruction_free import _sigma_at


class SDSDiffusion:
    """`diffusion` 的值，SDS 的梯度。**越大越好**，與 `diffusion` 同方向。"""

    name = 'sds'

    def __init__(self, objective):
        self.obj = objective
        self.ip2p = objective.ip2p

    def __call__(self, x_def: torch.Tensor) -> torch.Tensor:
        obj = self.obj
        z0 = self.ip2p.encode_image(x_def)
        z0f = z0.float()
        cond = self.ip2p.image_latents(x_def).detach().to(obj.text.dtype)
        noise = obj.noise
        values, surrogates = [], []
        z0n = z0.to(noise.dtype)
        for t in obj.timesteps:
            s = _sigma_at(obj.scheduler, t).to(z0n)
            denom = (s * s + 1).sqrt()
            scale = 1.0 / float(denom)
            zt = ((z0n + s * noise) / denom).detach()
            with torch.no_grad():
                model_input = torch.cat([zt, cond.to(zt.dtype)], dim=1)
                eps = obj._run_unet(model_input, t, obj.text).float()
            resid = eps - noise.float()
            grad = resid * (2.0 * scale / resid.numel())
            values.append(resid.pow(2).mean())
            surrogates.append((z0f * grad).sum())
        value = torch.stack(values).mean()
        surrogate = torch.stack(surrogates).mean()
        return value + (surrogate - surrogate.detach())

    def terms(self, x_def: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {'sds': self(x_def)}


def directional_check(term, x01: torch.Tensor, direction: torch.Tensor, *,
                      steps=(0.005, 0.02, 0.05)) -> Dict:
    """沿 `direction` 兩側各走幾步，回報損失怎麼變。

    方向以 **RMS 正規化**，不是以最大值正規化：曲線載體解出來的梯度是重尾的，
    用 `grad.abs().amax()` 正規化之後絕大多數像素的位移遠小於 1/255，實際移動
    的只有少數幾個尖峰，量到的差值落在 bf16 前向的數值雜訊裡，判不出方向。
    RMS 正規化讓 `step` 就是每像素位移的均方根。

    多個步長一起走，是要看差值隨步長單調放大——單一步長上的一個小差值有可能
    只是雜訊。
    """
    unit = direction / direction.pow(2).mean().sqrt().clamp_min(1e-12)
    out = {'base': float(term(x01))}
    with torch.no_grad():
        for h in steps:
            up = (x01 + float(h) * unit).clamp(0, 1)
            down = (x01 - float(h) * unit).clamp(0, 1)
            out[f'ascend_{h}'] = float(term(up))
            out[f'descend_{h}'] = float(term(down))
            out[f'gain_{h}'] = out[f'ascend_{h}'] - out['base']
            out[f'loss_{h}'] = out[f'descend_{h}'] - out['base']
    return out


def sds_sign_check(term, x01: torch.Tensor,
                   steps=(0.005, 0.02, 0.05)) -> Dict:
    """沿 SDS 梯度走，損失要往上；反向走，要往下。

    這一項是要被**最大化**的（`CompositeObjective` 以 `−w·tanh` 收它），所以
    梯度的正方向必須讓損失上升。只檢查上升那一側會被「任何擾動都讓損失上升」
    蒙混過去，所以兩側都走。

    回傳原始數字，不回傳通過與否——判定留給呼叫端與使用者。
    """
    y = x01.clone().detach().requires_grad_(True)
    base = term(y)
    base.backward()
    grad = y.grad.detach()
    out = directional_check(term, x01, grad, steps=steps)
    out['grad_absmax'] = float(grad.abs().amax())
    out['grad_rms'] = float(grad.pow(2).mean().sqrt())
    return out

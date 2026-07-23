"""PhotoGuard（加性，校準錨點）— SPEC.md §3.2、§3.3 含 v2.1 修正註記。

依官方 notebook（external/photoguard/notebooks/）核對後實作；凡官方實作與
SPEC 偽碼衝突處一律以官方為準（NOTES.md 決策紀錄 2026-07-23）：

- 投影：標準 PGD，每步將 x_adv 投影回以「原圖」為中心之 ε-ball（L1.3 已確認）
- encoder attack：預設 target_latent="zeros"（官方最小化 ‖ℰ(x')‖）、
  隨機初始化、step size 線性衰減至 1/100
- diffusion attack：梯度以 grad_reps 次平均（EOT）、無隨機初始化、step 為常數
- norm（linf/l2）與 epsilon_scale（pm1/01）為 SPEC §8 第 7–8 項待確認參數，
  兩者皆可跑，T2 校準階段以 DAYN Table 1 定奪

內部一律於 [-1,1] 像素空間運作（SD VAE 之輸入值域）；對外介面
（ProtectionMethod.protect）維持 [0,1]。
"""

import torch

from src.protect.base import ProtectionMethod
from src.utils.device import get_device


def _pgd_loop(
    x0: torch.Tensor,
    loss_fn,
    *,
    norm: str,
    eps: float,
    step: float,
    n_iter: int,
    random_init: bool,
    step_decay: bool,
    grad_reps: int = 1,
) -> torch.Tensor:
    """最小化 loss_fn 的 PGD。x0 與回傳值均於 [-1,1] 空間。

    投影一律以原圖 x0 為中心（官方實作，L1.3 確認），非累積扣減。
    """
    x_adv = x0.clone().detach()
    if random_init and norm == "linf":
        # 官方 encoder attack：於 ε-ball 內均勻隨機初始化（l2 版官方無此步驟）
        x_adv = (x_adv + torch.empty_like(x_adv).uniform_(-eps, eps)).clamp(-1.0, 1.0)

    for i in range(n_iter):
        # 官方 encoder attack：step size 線性衰減至 1/100
        cur_step = step - (step - step / 100) / n_iter * i if step_decay else step

        grads = []
        for _ in range(grad_reps):  # EOT：對 SD 採樣隨機性取期望（diffusion attack）
            x_in = x_adv.detach().requires_grad_(True)
            loss = loss_fn(x_in)
            grads.append(torch.autograd.grad(loss, x_in)[0])
        grad = torch.stack(grads).mean(dim=0)

        if norm == "linf":
            x_adv = x_adv.detach() - cur_step * grad.sign()
            x_adv = torch.minimum(torch.maximum(x_adv, x0 - eps), x0 + eps)
        elif norm == "l2":
            g_norm = grad.reshape(grad.shape[0], -1).norm(dim=1).view(-1, 1, 1, 1)
            x_adv = x_adv.detach() - cur_step * grad / (g_norm + 1e-10)
            x_adv = x0 + torch.renorm(x_adv - x0, p=2, dim=0, maxnorm=eps)
        else:
            raise ValueError(f"未知 norm: {norm}")
        x_adv = x_adv.clamp(-1.0, 1.0)

    return x_adv.detach()


class _PhotoGuardBase(ProtectionMethod):
    """兩種 attack 之共用邏輯：值域轉換、ε 尺度、峰值記憶體。"""

    @property
    def is_additive(self) -> bool:
        return True

    def _scaled_eps_step(self) -> tuple[float, float]:
        """依 epsilon_scale 將 ε 與 step 換算至內部 [-1,1] 空間。

        "pm1"：config 值即為 [-1,1] 空間之值（官方 notebook 用法），原樣使用。
        "01" ：config 值定義於 [0,1] 空間；[0,1] → [-1,1] 為 ×2 映射，故 ×2。
        """
        scale = self.cfg.get("epsilon_scale", "pm1")
        factor = {"pm1": 1.0, "01": 2.0}[scale]
        return self.cfg["epsilon"] * factor, self.cfg["step_size"] * factor

    def _run(self, image: torch.Tensor, loss_fn_builder, **pgd_kwargs) -> torch.Tensor:
        device = get_device()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        x0 = (image.to(device) * 2.0 - 1.0)  # [0,1] → [-1,1]
        eps, step = self._scaled_eps_step()
        x_adv = _pgd_loop(
            x0,
            loss_fn_builder(x0),
            norm=self.cfg.get("norm", "linf"),
            eps=eps,
            step=step,
            n_iter=self.cfg["n_iter"],
            **pgd_kwargs,
        )
        self._peak_mb = (
            torch.cuda.max_memory_allocated() / 1e6
            if torch.cuda.is_available()
            else float("nan")  # CPU 無對應之峰值統計
        )
        return ((x_adv + 1.0) / 2.0).clamp(0.0, 1.0)  # [-1,1] → [0,1]

    def peak_memory_mb(self) -> float:
        return getattr(self, "_peak_mb", float("nan"))


class PhotoGuardEncoder(_PhotoGuardBase):
    """encoder attack：min ‖ℰ(x') − z_targ‖（SPEC §3.2）。concept 不使用。"""

    @property
    def name(self) -> str:
        return "photoguard_encoder"

    def protect(self, image: torch.Tensor, concept: str) -> torch.Tensor:
        def build(x0):
            target = self.cfg.get("target_latent", "zeros")
            if target == "zeros":
                z_targ = None  # 官方 demo：loss = ‖z‖，隱含 z_targ = 0
            elif target == "gray":
                # 灰階 0.5（[0,1]）在 [-1,1] 空間為零「像素」，其 latent 不為零
                gray = torch.zeros_like(x0)
                z_targ = self.sd.vae.encode(gray).latent_dist.mean.detach()
            else:
                raise ValueError(f"未知 target_latent: {target}")

            def loss_fn(x_in):
                z = self.sd.vae.encode(x_in).latent_dist.mean
                return z.norm() if z_targ is None else (z - z_targ).norm()

            return loss_fn

        return self._run(
            image,
            build,
            random_init=self.cfg.get("random_init", True),
            step_decay=self.cfg.get("step_decay", True),
            grad_reps=1,
        )


class PhotoGuardDiffusion(_PhotoGuardBase):
    """diffusion attack：min ‖f(x') − x_targ‖₂，僅反傳 T 步（SPEC §3.3）。

    需 sd_wrapper 提供可微分 img2img：
        sd.img2img_differentiable(x_pm1, prompt, num_steps) -> 影像（[-1,1]，可反傳）
    """

    @property
    def name(self) -> str:
        return "photoguard_diffusion"

    def protect(self, image: torch.Tensor, concept: str) -> torch.Tensor:
        def build(x0):
            target = self.cfg.get("diffusion_target", "gray")
            if target == "gray":
                x_targ = torch.zeros_like(x0)  # [-1,1] 空間之零張量 = 灰階影像（官方 demo）
            elif target == "noise":
                x_targ = torch.randn_like(x0).clamp(-1.0, 1.0)
            else:
                raise ValueError(f"未知 diffusion_target: {target}")

            def loss_fn(x_in):
                out = self.sd.img2img_differentiable(
                    x_in, prompt="", num_steps=self.cfg["diffusion_T"]
                )
                return (out - x_targ).norm(p=2)

            return loss_fn

        # 官方 diffusion attack：無隨機初始化、step 為常數（衰減式被註解掉）
        return self._run(
            image,
            build,
            random_init=False,
            step_decay=False,
            grad_reps=self.cfg.get("grad_reps", 10),
        )

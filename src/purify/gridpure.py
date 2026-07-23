"""GrIDPure — SPEC.md §5.3（Zhao et al., CVPR 2024；官方 repo external/GrIDPure 核對）。

四步驟（official gridpure.py 之 grid_pure 忠實移植為張量運算）：
1. grid 切分：512×512 → stride 128 切九個 256×256；四個 128×128 角落
   另拼成第十個 grid（確保每一部分至少與兩個 grid 重疊）
2. 每 grid 以小步 DDPM 淨化（pixel-space 無條件 guided diffusion，
   即 DiffPure 機制：前向加噪至 t=pure_steps 後 p_sample 反向去噪）
3. 合併回原解析度，重疊區取平均
4. 與前一輪影像混合 x_{i+1} = (1−γ)·x̃_i + γ·x_i

purifier 以 callable 注入（(1,3,g,g)[0,1] → 同形狀），使 grid 機制可於
CPU/tiny 環境獨立驗證；正式 purifier 由 load_guided_diffusion_purifier()
載入官方 checkpoint（256x256_diffusion_uncond.pt，約 2GB，TWCC 下載）。
成本警告：單張 512×512 於 V100 約 2 分鐘，執行前須估算總時數。
"""

import sys
from pathlib import Path

import torch

from src.utils.device import get_device


def grid_boxes(width: int, height: int, grid: int = 256, stride: int = 128) -> list[tuple]:
    """回傳 (x0, y0, x1, y1) 清單（official get_crop_box 一般化）。"""
    def _axis(res):
        coords = [c for c in range(0, res - grid + 1, stride)]
        if (res - grid) not in coords:
            coords.append(res - grid)
        return coords

    return [(x, y, x + grid, y + grid) for y in _axis(height) for x in _axis(width)]


def corner_boxes(width: int, height: int, corner: int = 128) -> list[tuple]:
    """四角落 (x0, y0, x1, y1)：左上、右上、左下、右下（official get_corner_box）。"""
    return [
        (0, 0, corner, corner),
        (width - corner, 0, width, corner),
        (0, height - corner, corner, height),
        (width - corner, height - corner, width, height),
    ]


def _corner_slots(corner: int) -> list[tuple]:
    """四角落於拼合 grid 內的位置：與 corner_boxes 同序（official rearrange_positions）。"""
    return [
        (0, 0, corner, corner),
        (corner, 0, 2 * corner, corner),
        (0, corner, corner, 2 * corner),
        (corner, corner, 2 * corner, 2 * corner),
    ]


def grid_pure(
    image: torch.Tensor,
    purifier,
    *,
    grid_size: int = 256,
    stride: int = 128,
    gamma: float = 0.1,
    iterations: int = 20,
) -> torch.Tensor:
    """(1,3,H,W) [0,1] → 淨化結果。purifier: (1,3,grid,grid)[0,1] → 同形狀。"""
    h, w = image.shape[-2:]
    if h < grid_size or w < grid_size:
        raise ValueError(f"影像 {h}x{w} 小於 grid_size {grid_size}")
    corner = grid_size // 2
    boxes = grid_boxes(w, h, grid_size, stride)
    corners = corner_boxes(w, h, corner)
    slots = _corner_slots(corner)

    x = image.clamp(0, 1)
    for _ in range(iterations):
        acc = torch.zeros_like(x)
        cnt = torch.zeros_like(x)
        for (x0, y0, x1, y1) in boxes:
            patch = purifier(x[..., y0:y1, x0:x1])
            acc[..., y0:y1, x0:x1] += patch
            cnt[..., y0:y1, x0:x1] += 1.0
        # 四角落拼成第十個 grid，淨化後拆回原位
        assembled = torch.zeros(*x.shape[:2], grid_size, grid_size, device=x.device, dtype=x.dtype)
        for (cx0, cy0, cx1, cy1), (sx0, sy0, sx1, sy1) in zip(corners, slots):
            assembled[..., sy0:sy1, sx0:sx1] = x[..., cy0:cy1, cx0:cx1]
        assembled = purifier(assembled)
        for (cx0, cy0, cx1, cy1), (sx0, sy0, sx1, sy1) in zip(corners, slots):
            acc[..., cy0:cy1, cx0:cx1] += assembled[..., sy0:sy1, sx0:sx1]
            cnt[..., cy0:cy1, cx0:cx1] += 1.0
        merged = acc / cnt
        x = ((1.0 - gamma) * merged + gamma * x).clamp(0, 1)
    return x


class GuidedDiffusionPurifier:
    """DiffPure 機制之 grid 淨化器（official runners/diffpure_guided.py）。

    前向：x_t = √ᾱ_t·x_0 + √(1−ᾱ_t)·ε；反向：p_sample 逐步去噪 t−1 → 0。
    輸入輸出 [0,1]，內部於 [-1,1] 運作。pure_steps 尺度為 1000 步 DDPM 原始 timestep。
    """

    def __init__(self, model, diffusion, pure_steps: int, device=None):
        self.model = model
        self.diffusion = diffusion
        self.pure_steps = int(pure_steps)
        self.device = device or get_device()
        self.alphas_cumprod = torch.from_numpy(
            (1.0 - diffusion.betas).cumprod(axis=0)
        ).float().to(self.device)

    def set_pure_steps(self, pure_steps: int) -> None:
        self.pure_steps = int(pure_steps)

    @torch.no_grad()
    def __call__(self, x01: torch.Tensor) -> torch.Tensor:
        x = (x01.to(self.device) - 0.5) * 2.0
        t = self.pure_steps
        a_t = self.alphas_cumprod[t - 1]
        x = x * a_t.sqrt() + torch.randn_like(x) * (1.0 - a_t).sqrt()
        for i in reversed(range(t)):
            ts = torch.tensor([i] * x.shape[0], device=self.device)
            x = self.diffusion.p_sample(
                self.model, x, ts, clip_denoised=True, denoised_fn=None,
                cond_fn=None, model_kwargs=None,
            )["sample"]
        return ((x + 1.0) * 0.5).clamp(0, 1).to(x01.device, x01.dtype)


def load_guided_diffusion_purifier(
    checkpoint_path: str,
    pure_steps: int,
    repo_dir: str = "external/GrIDPure",
    use_fp16: bool = None,
) -> GuidedDiffusionPurifier:
    """自官方 repo 載入 pixel-space 無條件 guided diffusion（imagenet.yml 設定）。

    checkpoint_path: 256x256_diffusion_uncond.pt（約 2GB，見 TWCC_CHECKLIST.md）。
    use_fp16 預設依裝置：cuda 用 fp16（官方設定），cpu 用 fp32。
    """
    repo = Path(repo_dir).resolve()
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from guided_diffusion.script_util import (  # noqa: E402（官方 repo 內套件）
        create_model_and_diffusion,
        model_and_diffusion_defaults,
    )

    device = get_device()
    if use_fp16 is None:
        use_fp16 = device.type == "cuda"
    model_config = model_and_diffusion_defaults()
    model_config.update(  # official imagenet.yml 之 model 區塊
        {
            "attention_resolutions": "32,16,8",
            "class_cond": False,
            "diffusion_steps": 1000,
            "rescale_timesteps": True,
            "timestep_respacing": "1000",
            "image_size": 256,
            "learn_sigma": True,
            "noise_schedule": "linear",
            "num_channels": 256,
            "num_head_channels": 64,
            "num_res_blocks": 2,
            "resblock_updown": True,
            "use_fp16": use_fp16,
            "use_scale_shift_norm": True,
        }
    )
    model, diffusion = create_model_and_diffusion(**model_config)
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model.requires_grad_(False).eval().to(device)
    if use_fp16:
        model.convert_to_fp16()
    return GuidedDiffusionPurifier(model, diffusion, pure_steps, device)

"""PhotoGuard 單元測試（CPU、tiny SD VAE）。

驗證項目：輸出形狀與值域、ε 約束（linf/l2、pm1/01 兩種尺度）、
loss 確實下降、diffusion attack 之 PGD/EOT 迴圈可反傳（以 mock 可微分編輯器）。
"""

from types import SimpleNamespace

import pytest
import torch

from src.protect.photoguard import PhotoGuardDiffusion, PhotoGuardEncoder
from src.utils.device import get_device

TINY_MODEL = "hf-internal-testing/tiny-stable-diffusion-pipe"

BASE_CFG = {
    "norm": "linf",
    "epsilon": 0.06,
    "epsilon_scale": "pm1",
    "step_size": 0.006,
    "n_iter": 8,
    "random_init": True,
    "step_decay": True,
    "target_latent": "zeros",
    "diffusion_T": 2,
    "diffusion_target": "gray",
    "grad_reps": 2,
    "projection": "from_original",
}


@pytest.fixture(scope="module")
def tiny_vae():
    from diffusers import AutoencoderKL

    return AutoencoderKL.from_pretrained(TINY_MODEL, subfolder="vae").to(get_device())


@pytest.fixture()
def x():
    torch.manual_seed(0)
    return torch.rand(1, 3, 64, 64).to(get_device())


def _encoder(vae, **overrides):
    cfg = {**BASE_CFG, **overrides}
    return PhotoGuardEncoder(SimpleNamespace(vae=vae), cfg)


def test_encoder_linf_pm1(tiny_vae, x):
    out = _encoder(tiny_vae).protect(x, concept="unused")
    assert out.shape == x.shape
    assert out.min() >= 0.0 and out.max() <= 1.0
    # pm1 尺度 eps=0.06，於 [0,1] 空間對應 0.03
    assert (out - x).abs().max() <= 0.06 / 2 + 1e-6


def test_encoder_linf_scale01(tiny_vae, x):
    out = _encoder(tiny_vae, epsilon_scale="01").protect(x, concept="unused")
    assert (out - x).abs().max() <= 0.06 + 1e-6


def test_encoder_l2(tiny_vae, x):
    out = _encoder(tiny_vae, norm="l2", epsilon=16.0, step_size=1.0).protect(
        x, concept="unused"
    )
    # pm1 尺度 eps=16，於 [0,1] 空間對應 8
    assert (out - x).reshape(-1).norm() <= 16.0 / 2 + 1e-4


def test_encoder_loss_decreases(tiny_vae, x):
    torch.manual_seed(0)
    out = _encoder(tiny_vae, n_iter=15).protect(x, concept="unused")
    with torch.no_grad():
        z_before = tiny_vae.encode(x * 2 - 1).latent_dist.mean.norm()
        z_after = tiny_vae.encode(out * 2 - 1).latent_dist.mean.norm()
    assert z_after < z_before


def test_encoder_target_gray(tiny_vae, x):
    out = _encoder(tiny_vae, target_latent="gray").protect(x, concept="unused")
    assert (out - x).abs().max() <= 0.06 / 2 + 1e-6


def test_diffusion_attack_mock(x):
    """以 mock 可微分編輯器驗證 PGD/EOT 迴圈：可反傳、約束成立、輸出趨近目標。"""
    conv = torch.nn.Conv2d(3, 3, 3, padding=1).to(get_device())
    torch.manual_seed(0)

    def img2img_differentiable(x_in, prompt, num_steps):
        return torch.tanh(conv(x_in) + 0.05 * torch.randn_like(x_in))

    sd = SimpleNamespace(img2img_differentiable=img2img_differentiable)
    method = PhotoGuardDiffusion(sd, {**BASE_CFG, "n_iter": 5})
    out = method.protect(x, concept="unused")
    assert out.shape == x.shape
    assert (out - x).abs().max() <= 0.06 / 2 + 1e-6
    assert method.is_additive and method.name == "photoguard_diffusion"

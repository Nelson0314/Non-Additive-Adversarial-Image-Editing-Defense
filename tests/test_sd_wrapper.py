"""SDWrapper 單元測試（CPU、tiny SD）。

驗證：可微分 img2img 之梯度可反傳至輸入影像、cross-attention 擷取、
編解碼 round-trip 形狀。
"""

import pytest
import torch

from src.models.sd_wrapper import SDWrapper

TINY_MODEL = "hf-internal-testing/tiny-stable-diffusion-pipe"


@pytest.fixture(scope="session")
def sd():
    return SDWrapper(TINY_MODEL)


def _image_size(sd):
    scale = 2 ** (len(sd.vae.config.block_out_channels) - 1)
    return sd.unet.config.sample_size * scale


def test_encode_decode_shapes(sd):
    n = _image_size(sd)
    x = torch.rand(1, 3, n, n) * 2 - 1
    z = sd.encode_image(x)
    x_rec = sd.decode_latents(z)
    assert x_rec.shape == x.shape


def test_img2img_differentiable_grad_flows(sd):
    n = _image_size(sd)
    torch.manual_seed(0)
    x = (torch.rand(1, 3, n, n) * 2 - 1).requires_grad_(True)
    out = sd.img2img_differentiable(x, prompt="", num_steps=2, strength=0.5)
    assert out.shape == x.shape
    loss = out.norm()
    (grad,) = torch.autograd.grad(loss, x)
    assert grad.shape == x.shape
    assert torch.isfinite(grad).all()
    assert grad.abs().sum() > 0  # 梯度確實流回輸入


def test_photoguard_diffusion_with_wrapper(sd):
    """PhotoGuardDiffusion 串接真實（tiny）pipeline：EOT 迴圈可跑、約束成立。"""
    from src.protect.photoguard import PhotoGuardDiffusion

    n = _image_size(sd)
    torch.manual_seed(0)
    x = torch.rand(1, 3, n, n)
    cfg = {
        "norm": "linf",
        "epsilon": 0.06,
        "epsilon_scale": "pm1",
        "step_size": 0.006,
        "n_iter": 2,
        "diffusion_T": 2,
        "diffusion_target": "gray",
        "grad_reps": 2,
    }
    out = PhotoGuardDiffusion(sd, cfg).protect(x, concept="unused")
    assert out.shape == x.shape
    assert (out - x).abs().max() <= 0.06 / 2 + 1e-6


def test_capture_cross_attention(sd):
    torch.manual_seed(0)
    s = sd.unet.config.sample_size
    z = torch.randn(1, sd.unet.config.in_channels, s, s)
    emb = sd.encode_text("a dog")
    with sd.capture_cross_attention() as maps:
        sd.unet(z, 1, encoder_hidden_states=emb)
    assert len(maps) > 0
    for m in maps:
        assert m["probs"].ndim == 3
        assert m["probs"].shape[-1] == emb.shape[1]  # key 維 = text token 數
    # 離開 context 後恢復原 processor：再跑一次不再累積
    n_maps = len(maps)
    sd.unet(z, 1, encoder_hidden_states=emb)
    assert len(maps) == n_maps

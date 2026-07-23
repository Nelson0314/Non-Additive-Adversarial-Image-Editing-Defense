"""非加性方法單元測試（CPU、tiny SD）— 指令 D 之小規模驗證。

驗證：reward 梯度可算且方向正確（沿梯度上升 R 增加）、DDIM inversion 決定性、
AdvDiff/APA/Hybrid 三方法可跑通且輸出合法、LoRA 注入零初始化不改變輸出且
可精確還原。超參數為縮減值，非正式設定。
"""

import pytest
import torch

from src.models.sd_wrapper import SDWrapper
from src.protect.advdiff_based import AdvDiffProtection
from src.protect.apa_based import APAProtection, inject_lora
from src.protect.hybrid import HybridProtection
from src.protect.rewards import attention_reward_latent

TINY_MODEL = "hf-internal-testing/tiny-stable-diffusion-pipe"

APA_CFG = {
    "variant": "gc",
    "T": 3,
    "T_a": 2,
    "N": 2,
    "eps_a": 0.4,
    "mu": 0.04,
    "lora_rank": 2,
    "lora_lr": 1.0e-4,
    "lora_steps": 3,
    "reward": "attention",
}

ADVDIFF_CFG = {
    "sampler": "ddim",
    "T": 5,
    "N": 2,
    "s": 0.7,
    "a": 0.5,
    "guidance_range": [0.0, 0.5],  # 縮小 T 後放寬區間，確保涵蓋 guided 步
    "eps_latent": 0.2,
    "reward": "attention",
}


@pytest.fixture(scope="session")
def sd():
    return SDWrapper(TINY_MODEL)


def _image(sd):
    scale = 2 ** (len(sd.vae.config.block_out_channels) - 1)
    n = sd.unet.config.sample_size * scale
    torch.manual_seed(0)
    return torch.rand(1, 3, n, n)


def _latent(sd):
    torch.manual_seed(0)
    s = sd.unet.config.sample_size
    return torch.randn(1, sd.unet.config.in_channels, s, s)


def test_reward_grad_computable(sd):
    z = _latent(sd).requires_grad_(True)
    r = attention_reward_latent(z, 1, sd, "dog")
    assert torch.isfinite(r) and r <= 0  # R = −‖A‖₁
    (g,) = torch.autograd.grad(r, z)
    assert torch.isfinite(g).all() and g.abs().sum() > 0


def test_reward_ascent_direction(sd):
    """沿 ∇R 上升一步後 R 須增加（方向正確）。"""
    z = _latent(sd).requires_grad_(True)
    r0 = attention_reward_latent(z, 1, sd, "dog")
    (g,) = torch.autograd.grad(r0, z)
    improved = False
    for step in (1e-3, 1e-2, 1e-1):
        z2 = (z + step * g / g.abs().max()).detach()
        r2 = attention_reward_latent(z2, 1, sd, "dog")
        if r2.item() > r0.item():
            improved = True
            break
    assert improved


def test_ddim_inversion_deterministic(sd):
    with torch.no_grad():
        z0 = sd.encode_image(_image(sd) * 2 - 1)
        emb = sd.encode_text("")
        a = sd.ddim_inversion(z0, emb, 4)
        b = sd.ddim_inversion(z0, emb, 4)
    assert torch.equal(a, b)
    assert a.shape == z0.shape and torch.isfinite(a).all()


def test_lora_inject_restore(sd):
    z, emb = _latent(sd), sd.encode_text("dog")
    with torch.no_grad():
        y_ref = sd.unet(z, 1, encoder_hidden_states=emb).sample
    params, restore = inject_lora(sd.unet, rank=2)
    assert len(params) > 0
    with torch.no_grad():
        y_injected = sd.unet(z, 1, encoder_hidden_states=emb).sample
    assert torch.allclose(y_ref, y_injected, atol=1e-6)  # up 零初始化
    restore()
    with torch.no_grad():
        y_restored = sd.unet(z, 1, encoder_hidden_states=emb).sample
    assert torch.equal(y_ref, y_restored)


def _assert_valid_output(out, x):
    assert out.shape == x.shape
    assert 0.0 <= out.min() and out.max() <= 1.0
    assert torch.isfinite(out).all()
    assert not torch.equal(out, x)  # 非加性：輸出由生成過程產出，必與原圖不同


def test_advdiff_protect(sd):
    x = _image(sd)
    out = AdvDiffProtection(sd, ADVDIFF_CFG).protect(x, "dog")
    _assert_valid_output(out, x)


def test_apa_protect_and_unet_restored(sd):
    x = _image(sd)
    z, emb = _latent(sd), sd.encode_text("dog")
    with torch.no_grad():
        y_before = sd.unet(z, 1, encoder_hidden_states=emb).sample
    out = APAProtection(sd, APA_CFG).protect(x, "dog")
    _assert_valid_output(out, x)
    with torch.no_grad():
        y_after = sd.unet(z, 1, encoder_hidden_states=emb).sample
    assert torch.equal(y_before, y_after)  # LoRA 已移除，unet 復原


def test_hybrid_protect(sd):
    x = _image(sd)
    cfg = {**APA_CFG, "s": 0.7, "a": 0.5}
    out = HybridProtection(sd, cfg).protect(x, "dog")
    _assert_valid_output(out, x)
    assert HybridProtection(sd, cfg).name == "hybrid"

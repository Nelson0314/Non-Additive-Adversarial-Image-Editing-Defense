"""非加性方法單元測試（CPU、tiny SD）— v4 對齊版。

驗證：reward 梯度可算且方向正確、DDIM inversion 決定性、AdvDiff（後置加法 +
skip-gradient）、APA-SG（完整 inversion + skip）、APA-GC（部分 inversion +
checkpoint）、Hybrid 可跑通且輸出合法、peft LoRA 於 protect 後精確還原。
超參數為縮減值，非正式設定。
"""

import pytest
import torch

from src.models.sd_wrapper import SDWrapper
from src.protect.advdiff_based import AdvDiffProtection
from src.protect.apa_based import APAProtection
from src.protect.hybrid import HybridProtection
from src.protect.rewards import attention_reward_latent

TINY_MODEL = "hf-internal-testing/tiny-stable-diffusion-pipe"

APA_CFG = {
    "variant": "gc",
    "grid_steps": 10,            # 測試縮減（官方 50）
    "T_a": 3,
    "N": 2,
    "eps_a": 0.4,
    "mu": 0.04,
    "guidance_scale": 1.0,
    "reward": "attention",
    "lora_impl": "peft",
    "lora_rank": 2,
    "lora_alpha": 2,
    "lora_target_modules": ["to_k", "to_q", "to_v", "to_out.0"],
    "lora_init": "gaussian",
    "lora_lr": 1.0e-4,
    "lora_weight_decay": 1.0e-2,
    "lora_grad_clip": 1.0,
    "lora_steps": 3,             # 測試縮減（官方 200）
    "noise_offset": 0.1,
    "sg": {"trajectory_grad": "skip", "scale_constant": 14.58, "inversion": "full"},
    "gc": {
        "trajectory_grad": "checkpoint",
        "inversion": "partial",
        "inversion_steps": 3,    # 測試縮減（官方 10）
        "sampling_steps": 4,
        "mse_reg_weight": 10.0,
    },
    "aug": {"base": "final_generated", "resize_range": [0.9, 1.0], "brightness": False},
}

ADVDIFF_CFG = {
    "T": 5,
    "N": 2,
    "a": 0.5,
    "guidance_range": [0.0, 0.5],  # 縮小 T 後放寬區間，確保涵蓋 guided 步
    "eta": 0.0,
    "reward": "attention",
    "injection_form": "post_hoc",
    "injection_coeff": None,
    "ztT_update": "skip_gradient",
    "s": 0.7,
    "eps_latent": 0.2,
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


def _assert_valid_output(out, x):
    assert out.shape == x.shape
    assert 0.0 <= out.min() and out.max() <= 1.0
    assert torch.isfinite(out).all()
    assert not torch.equal(out, x)  # 非加性：輸出由生成過程產出，必與原圖不同


def test_advdiff_protect(sd):
    x = _image(sd)
    out = AdvDiffProtection(sd, ADVDIFF_CFG).protect(x, "dog")
    _assert_valid_output(out, x)


@pytest.mark.parametrize("variant", ["sg", "gc"])
def test_apa_protect_and_unet_restored(sd, variant):
    x = _image(sd)
    z, emb = _latent(sd), sd.encode_text("dog")
    with torch.no_grad():
        y_before = sd.unet(z, 1, encoder_hidden_states=emb).sample
    cfg = {**APA_CFG, "variant": variant}
    method = APAProtection(sd, cfg)
    assert method.name == f"apa_{variant}"
    out = method.protect(x, "dog")
    _assert_valid_output(out, x)
    with torch.no_grad():
        y_after = sd.unet(z, 1, encoder_hidden_states=emb).sample
    assert torch.equal(y_before, y_after)  # peft adapter 已刪除，unet 復原


def test_hybrid_protect(sd):
    x = _image(sd)
    cfg = {**APA_CFG, "variant": "sg", "s": 0.7, "a": 0.5}
    method = HybridProtection(sd, cfg)
    out = method.protect(x, "dog")
    _assert_valid_output(out, x)
    assert method.name == "hybrid"

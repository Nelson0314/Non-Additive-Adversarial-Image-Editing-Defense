"""固定評估的 prompt EOT 必須重用同一組條件；只用 CPU 與假的 UNet。"""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import src.defense.image_guidance_loss as mod  # noqa: E402


@pytest.mark.parametrize("zt_mode", ["noise", "diffuse_src"])
def test_fixed_prompt_eot_reuses_unet_conditions(monkeypatch, zt_mode):
    calls = []

    def unet(z, t, encoder_hidden_states):
        calls.append((z.detach().clone(), t.clone(),
                      encoder_hidden_states.clone()))
        scale = encoder_hidden_states.mean()
        return SimpleNamespace(sample=z[:, :4] + z[:, 4:] * scale)

    def encode(x):
        return x.mean(dim=1, keepdim=True).repeat(1, 4, 1, 1)

    ip = SimpleNamespace(
        device=torch.device("cpu"), unet=unet,
        scheduler=SimpleNamespace(alphas_cumprod=torch.linspace(0.99, 0.01, 10)),
        encode_image=encode, image_latents=encode,
    )
    monkeypatch.setattr(mod, "_null_embedding", lambda ip: torch.zeros(1, 2, 3))
    monkeypatch.setattr(mod, "_scheduler_of", lambda ip: ip.scheduler)
    prompts = torch.arange(1, 9, dtype=torch.float32).view(8, 1, 1).expand(8, 2, 3)
    x = torch.linspace(0, 1, 48).reshape(1, 3, 4, 4)
    loss = mod.make_image_guidance_loss(
        ip, zt_mode=zt_mode, x_clean=x, t_max=10, text_embeds=prompts,
    )
    fixed = loss.make_fixed(4, 99991)
    first_value = fixed(x)
    first = calls[:]
    calls.clear()
    second_value = fixed(x)

    assert len(first) == len(calls) == 8
    for before, after in zip(first, calls):
        for a, b in zip(before, after):
            assert a.device.type == b.device.type == "cpu"
            assert torch.equal(a, b)
        assert any(torch.equal(before[2], p.unsqueeze(0)) for p in prompts)
    for k in range(0, 8, 2):
        assert torch.equal(first[k][2], first[k + 1][2])
    assert torch.equal(first_value, second_value)

    # 同種子重新建立評估器，即使中間呼叫訓練損失也必須重現。
    loss(x)
    calls.clear()
    recreated_value = loss.make_fixed(4, 99991)(x)
    assert torch.equal(first_value, recreated_value)
    for before, after in zip(first, calls):
        assert all(torch.equal(a, b) for a, b in zip(before, after))

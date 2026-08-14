"""`SDWrapper._latent_in` 在 CFG 下必須把 9 通道的條件複製到 latent 的 batch。

2026-08-14 的 inpainting 預檢撞到這一點：PromptFlare 的 token 探測走
`torch.cat([z] * 2)`（CFG 的兩支），而 `inpaint_conditioning` 逐影像只建一
份條件，`torch.cat(..., dim=1)` 於是在 batch 軸上對不起來。
"""

import types

import pytest
import torch

from src.models.sd import SDWrapper


@pytest.fixture
def sd(monkeypatch):
    """不載入任何權重：`_latent_in` 只用到三個屬性與 `_require_inpaint_cond`。"""
    monkeypatch.setattr(SDWrapper, "is_inpainting", property(lambda self: True))
    monkeypatch.setattr(SDWrapper, "latent_channels", property(lambda self: 4))
    monkeypatch.setattr(SDWrapper, "unet", property(lambda self: types.SimpleNamespace(
        config=types.SimpleNamespace(in_channels=9))))
    s = SDWrapper.__new__(SDWrapper)
    s.model_name = "fake-inpaint"
    s._inpaint_cond = None
    return s


def test_batch相同時原樣串接(sd):
    sd._inpaint_cond = (torch.zeros(1, 1, 8, 8), torch.zeros(1, 4, 8, 8))
    out = sd._latent_in(torch.zeros(1, 4, 8, 8))
    assert out.shape == (1, 9, 8, 8)


def test_CFG的兩倍batch把條件複製而不是報錯(sd):
    m = torch.full((1, 1, 8, 8), 0.5)
    zm = torch.full((1, 4, 8, 8), 0.25)
    sd._inpaint_cond = (m, zm)
    out = sd._latent_in(torch.zeros(2, 4, 8, 8))
    assert out.shape == (2, 9, 8, 8)
    # 兩支拿到的是同一份條件
    assert torch.equal(out[0, 4:], out[1, 4:])
    assert torch.allclose(out[:, 4], torch.full((2, 8, 8), 0.5))


def test_不整除時報錯而不是靜默廣播(sd):
    sd._inpaint_cond = (torch.zeros(2, 1, 8, 8), torch.zeros(2, 4, 8, 8))
    with pytest.raises(ValueError, match="整數倍"):
        sd._latent_in(torch.zeros(3, 4, 8, 8))


def test_已是9通道時不再補(sd):
    sd._inpaint_cond = (torch.zeros(1, 1, 8, 8), torch.zeros(1, 4, 8, 8))
    z = torch.zeros(2, 9, 8, 8)
    assert sd._latent_in(z) is z


def test_沒有作用中的條件時拋出(sd):
    with pytest.raises(RuntimeError, match="inpaint_conditioning"):
        sd._latent_in(torch.zeros(1, 4, 8, 8))

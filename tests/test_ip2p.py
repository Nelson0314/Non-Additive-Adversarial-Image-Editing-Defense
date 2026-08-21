"""`src/models/ip2p.py` 的驗收 — 用替身管線，不下載 4 GB 權重。

主線的攻擊模型由 SDEdit 換成 InstructPix2Pix（使用者 2026-08-19 裁定）。
這一支要釘的是**三個補錯了也不會拋錯的地方**：

1. 載成一般 SD 1.5（UNet 4 通道）時影像條件會靜默失效，編輯退化成純文生圖；
2. 拼進 UNet 的影像 latent **不乘** `scaling_factor`，與 `encode_image` 差
   一個 0.18 的倍率；
3. 兩個導引尺度不能對調——`guidance_scale` 是文字、`image_guidance_scale`
   是影像。

三者都會產出「一張合理的圖」，靠看輸出是抓不到的。
"""

import pytest
import torch
from torch import nn

from src.models.ip2p import (
    IP2P_IMAGE_GUIDANCE, IP2P_STEPS, IP2P_TEXT_GUIDANCE, IP2PWrapper,
)


class _Dist:
    def __init__(self, mean):
        self.mean = mean

    def mode(self):
        return self.mean


class _Enc:
    def __init__(self, mean):
        self.latent_dist = _Dist(mean)


class _VAE(nn.Module):
    """把 latent 定義成每 8×8 區塊的平均，形狀與真的一致（512² → 64²）。"""

    def __init__(self, scaling_factor=0.18215):
        super().__init__()
        self.config = type("C", (), {"scaling_factor": scaling_factor})()
        self.register_buffer("_w", torch.zeros(1))   # 讓 .to(dtype) 有東西可轉

    @property
    def dtype(self):
        return self._w.dtype

    def encode(self, x):
        return _Enc(nn.functional.avg_pool2d(x, 8)[:, :4] if x.shape[1] >= 4
                    else nn.functional.avg_pool2d(x, 8).repeat(1, 2, 1, 1)[:, :4])

    def decode(self, z):
        up = nn.functional.interpolate(z[:, :3], scale_factor=8, mode="nearest")
        return type("O", (), {"sample": up})()


class _UNet(nn.Module):
    def __init__(self, in_channels=8):
        super().__init__()
        self.config = type("C", (), {"in_channels": in_channels})()


class _Pipe(nn.Module):
    """記下 `__call__` 收到的每一個參數，供測試檢查。"""

    def __init__(self, in_channels=8):
        super().__init__()
        self.unet = _UNet(in_channels)
        self.vae = _VAE()
        self.text_encoder = nn.Linear(2, 2)
        self.calls = []

    def set_progress_bar_config(self, **kw):
        pass

    def __call__(self, **kw):
        self.calls.append(kw)
        img = kw["image"]
        return type("R", (), {"images": (img.float() + 1.0) / 2.0})()


def _wrap(in_channels=8):
    return IP2PWrapper(model_name="stub", pipe=_Pipe(in_channels))


def _img(n=1, size=64):
    g = torch.Generator().manual_seed(0)
    return torch.rand(n, 3, size, size, generator=g)


def test_載錯checkpoint時拒絕繼續():
    # 一般的 SD 1.5 是 4 通道；載錯時影像條件靜默消失而不報錯。
    with pytest.raises(RuntimeError, match="in_channels"):
        _wrap(in_channels=4)


def test_影像條件不乘scaling_factor():
    w = _wrap()
    x = _img()
    cond = w.image_latents(x)
    enc = w.encode_image(x)
    assert torch.allclose(cond * w.scaling_factor, enc, atol=1e-6)
    # 兩者確實不同——差的就是那個 0.18 的倍率。
    assert not torch.allclose(cond, enc)


def test_encode_image與SDWrapper同語意():
    # 取 mean（非 sample），故同一輸入必得同一輸出。
    w = _wrap()
    x = _img()
    assert torch.allclose(w.encode_image(x), w.encode_image(x))


def test_edit把兩個導引尺度接到正確的參數():
    w = _wrap()
    w.edit(_img(), "add a hat", seed=1, steps=7, s_t=9.0, s_i=1.25)
    kw = w.pipe.calls[-1]
    assert kw["guidance_scale"] == 9.0            # s_T：文字
    assert kw["image_guidance_scale"] == 1.25     # s_I：影像
    assert kw["num_inference_steps"] == 7
    assert kw["prompt"] == "add a hat"


def test_edit不繞PIL且影像送進去是負一到一():
    w = _wrap()
    x = _img()
    out = w.edit(x, "instr", seed=0)
    kw = w.pipe.calls[-1]
    assert kw["output_type"] == "pt"              # 繞 PIL 會經過 uint8 量化
    assert kw["image"].min() >= -1.0 and kw["image"].max() <= 1.0
    assert out.shape == x.shape


def test_edit的seed進到generator():
    w = _wrap()
    w.edit(_img(), "instr", seed=1234)
    assert w.pipe.calls[-1]["generator"].initial_seed() == 1234


def test_edit拒絕非四維輸入():
    w = _wrap()
    with pytest.raises(ValueError, match=r"\(N,3,H,W\)"):
        w.edit(_img()[0], "instr")


def test_推論常數是本專案指定的那三個值():
    """論文（arXiv:2504.17894 §5.3）沒有給步數與兩個導引尺度。

    這三個值是本專案取 diffusers 預設後定下的，改動會讓新舊批次不可比，
    故釘住——要改必須連同這個測試一起改，改不掉就是誤動。
    """
    assert (IP2P_STEPS, IP2P_TEXT_GUIDANCE, IP2P_IMAGE_GUIDANCE) == (100, 7.5, 1.5)

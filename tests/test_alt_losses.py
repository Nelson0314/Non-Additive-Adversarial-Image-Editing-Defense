"""`src.baselines.alt_losses`：兩個 untargeted 損失的介面與方向。

方向錯了不會有症狀——PGD 照樣跑完、照樣產出防禦圖，只是把影像推向原圖而不是
推離它，而效果欄會安靜地變小。故「離得越遠、損失越小」這件事必須被測到。
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.baselines.alt_losses import make_latent_untargeted_loss  # noqa: E402


class _FakeSD:
    """把影像當成 latent 用，只為測損失的形狀與方向。"""

    def encode_image(self, x01):
        return x01.mean(dim=1, keepdim=True)


def test_latent_損失在原圖上取到最大值():
    sd = _FakeSD()
    x = torch.rand(1, 3, 8, 8)
    loss = make_latent_untargeted_loss(sd, x)
    assert float(loss(x)) == 0.0


def test_latent_離原圖越遠損失越小():
    sd = _FakeSD()
    x = torch.rand(1, 3, 8, 8)
    loss = make_latent_untargeted_loss(sd, x)
    near = float(loss(x + 0.01))
    far = float(loss(x + 0.20))
    assert far < near < 0.0


def test_latent_損失可微且梯度把影像推離原圖():
    sd = _FakeSD()
    x = torch.rand(1, 3, 8, 8)
    loss = make_latent_untargeted_loss(sd, x)
    y = (x + 0.05).clone().requires_grad_(True)
    loss(y).backward()
    # 沿負梯度走一步應該離原圖更遠
    step = (y - 0.1 * y.grad).detach()
    assert (step - x).pow(2).mean() > (y - x).pow(2).mean()


def test_參考_latent_不帶梯度():
    sd = _FakeSD()
    x = torch.rand(1, 3, 8, 8, requires_grad=True)
    loss = make_latent_untargeted_loss(sd, x)
    y = torch.rand(1, 3, 8, 8, requires_grad=True)
    loss(y).backward()
    assert x.grad is None


class _StubOut:
    """本版 transformers 的 `get_image_features` 回傳形狀：特徵在 pooler_output。"""

    def __init__(self, t):
        self.pooler_output = t


class _StubClip:
    """記下收到的 pixel_values，回傳可預測的特徵。"""

    def __init__(self):
        self.seen = None

    def get_image_features(self, pixel_values):
        self.seen = pixel_values
        return _StubOut(pixel_values.flatten(1)[:, :8])


def _embed_with_stub():
    from src.baselines.alt_losses import _ClipEmbed, CLIP_MEAN, CLIP_STD
    e = _ClipEmbed.__new__(_ClipEmbed)
    e.device = "cpu"
    e.model = _StubClip()
    e.mean = torch.tensor(CLIP_MEAN).view(1, 3, 1, 1)
    e.std = torch.tensor(CLIP_STD).view(1, 3, 1, 1)
    return e


def test_clip_前處理縮到_224_並做正規化():
    e = _embed_with_stub()
    out = e(torch.rand(1, 3, 512, 512))
    assert e.model.seen.shape == (1, 3, 224, 224)
    assert out.shape == (1, 8)


def test_clip_取的是_pooler_output_而不是回傳物件本身():
    """壞掉時的症狀是 AttributeError，不是靜默錯誤——但仍要釘住這條路徑。"""
    e = _embed_with_stub()
    out = e(torch.rand(1, 3, 64, 64))
    assert isinstance(out, torch.Tensor)


def test_clip_輸出已正規化為單位向量():
    e = _embed_with_stub()
    out = e(torch.rand(1, 3, 64, 64))
    assert torch.allclose(out.norm(dim=-1), torch.ones(1), atol=1e-5)

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

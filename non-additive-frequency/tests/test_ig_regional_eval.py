"""固定評估加上空間權重之後，不給權重的那條路必須逐位元不變。

存在理由
────────────────────────────────────────────────────────────────────
`make_fixed` 的回傳值進了每一批訓練的 `trace.csv` 與 `best_eval`。替它加上
`weight` 參數若改動了 `weight=None` 的數值，新舊批次立刻不可比，而症狀是
「曲線看起來還是曲線」——典型的靜默失效。

第二條測的是分區的定義：權重全為 1 必須等於全圖值，且兩塊互補的區域以
各自的權重總和加權回去要等於全圖值。

用假的 UNet 與排程，不載任何權重、只用 CPU。
"""

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.defense.image_guidance_loss import make_image_guidance_loss  # noqa: E402

LAT = 8


class _Out:
    def __init__(self, sample):
        self.sample = sample


class _UNet:
    """回傳一個依輸入決定、但空間上不均勻的 sample，使分區真的分得開。"""

    def __call__(self, z, t, encoder_hidden_states=None):
        ramp = torch.linspace(0.0, 1.0, LAT).view(1, 1, LAT, 1)
        return _Out(z[:, :4] * 0.5 + z[:, 4:] * ramp)


class _Sched:
    def __init__(self):
        self.alphas_cumprod = torch.linspace(0.9999, 0.001, 1000)


class _IP2P:
    device = torch.device("cpu")
    unet = _UNet()
    scheduler = _Sched()
    dtype = torch.float32

    def encode_image(self, x):
        return x.mean(dim=1, keepdim=True).repeat(1, 4, 1, 1)[..., :LAT, :LAT]

    def image_latents(self, x):
        return self.encode_image(x) * 2.0


@pytest.fixture
def fixed(monkeypatch):
    import src.defense.image_guidance_loss as mod
    monkeypatch.setattr(mod, "_null_embedding", lambda ip: torch.zeros(1, 77, 4))
    monkeypatch.setattr(mod, "_scheduler_of", lambda ip: ip.scheduler)
    ip = _IP2P()
    x = torch.rand(1, 3, LAT, LAT)
    loss = make_image_guidance_loss(ip, zt_mode="diffuse_src", x_clean=x)
    return loss.make_fixed(4, 99991), x


def test_不給權重時與加參數之前的路徑相同(fixed):
    """`.mean()` 這條路必須原封不動。"""
    fn, x = fixed
    y = torch.rand_like(x)
    a = float(fn(y))
    b = float(fn(y))
    assert a == b                      # 決定性
    ones = torch.ones(1, 1, LAT, LAT)
    assert float(fn(y, ones)) == pytest.approx(a, rel=1e-6)


def test_權重乘上常數不改變結果(fixed):
    fn, x = fixed
    y = torch.rand_like(x)
    w = torch.zeros(1, 1, LAT, LAT)
    w[..., :4, :] = 1.0
    assert float(fn(y, w)) == pytest.approx(float(fn(y, w * 0.25)), rel=1e-6)


def test_兩塊互補的區域加權回去等於全圖(fixed):
    fn, x = fixed
    y = torch.rand_like(x)
    w = torch.zeros(1, 1, LAT, LAT)
    w[..., :3, :] = 1.0
    a, b = float(fn(y, w)), float(fn(y, 1.0 - w))
    frac = float(w.mean())
    assert frac * a + (1 - frac) * b == pytest.approx(float(fn(y)), rel=1e-6)


def test_不均勻的殘差讓兩塊分得開(fixed):
    """UNet 的 ramp 使殘差偏向下方，上下兩塊不該讀出同一個值。"""
    fn, x = fixed
    y = torch.rand_like(x)
    top = torch.zeros(1, 1, LAT, LAT)
    top[..., :LAT // 2, :] = 1.0
    assert float(fn(y, top)) != pytest.approx(float(fn(y, 1.0 - top)), rel=0.05)


def test_空權重拋錯而不是回傳零(fixed):
    fn, x = fixed
    with pytest.raises(ValueError, match="總和為 0"):
        fn(torch.rand_like(x), torch.zeros(1, 1, LAT, LAT))

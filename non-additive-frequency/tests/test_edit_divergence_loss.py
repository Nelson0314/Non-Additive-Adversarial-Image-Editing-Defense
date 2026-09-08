"""直接以防禦效果為目標的損失：符號、常數支、以及攻擊指令不得進入。

存在理由
────────────────────────────────────────────────────────────────────
三個容易寫錯而且**不會有症狀**的地方：

1. **符號。** `run_param_pgd` 一律最小化。回傳正的散度只會安靜地把防禦圖推回
   原圖，而 `trace.csv` 的曲線看起來仍然在收斂。
2. **原圖那一支必須是常數。** 它不依賴 `x_def`，忘了 detach 只會多一次反傳、
   數值不變，於是永遠不會被發現。
3. **文字條件必須是空字串。** 威脅模型的前提是攻擊指令未知；把 prompt 寫進
   損失會變成另一個威脅模型，而數字看起來完全正常。

用假的 UNet 與排程，不載任何權重、只用 CPU。
"""

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import src.defense.edit_divergence_loss as mod  # noqa: E402
from src.defense.edit_divergence_loss import make_edit_divergence_loss  # noqa: E402

LAT = 8


class _Out:
    def __init__(self, sample): self.sample = sample


class _UNet:
    """噪聲預測與拼進來的影像條件成正比，故影像條件差越多、x̂₀ 差越多。"""
    def __init__(self): self.calls = []
    def __call__(self, z, t, encoder_hidden_states=None):
        self.calls.append(encoder_hidden_states)
        return _Out(z[:, :4] * 0.2 + z[:, 4:] * 0.8)


class _Sched:
    def __init__(self): self.alphas_cumprod = torch.linspace(0.9999, 0.02, 1000)


class _IP2P:
    device = torch.device("cpu"); dtype = torch.float32
    def __init__(self): self.unet = _UNet(); self.scheduler = _Sched()
    def encode_image(self, x):
        return x.mean(dim=1, keepdim=True).repeat(1, 4, 1, 1)[..., :LAT, :LAT]
    def image_latents(self, x):
        return self.encode_image(x) * 2.0


@pytest.fixture
def built(monkeypatch):
    monkeypatch.setattr(mod, "_null_embedding", lambda ip: torch.zeros(1, 77, 4))
    monkeypatch.setattr(mod, "_scheduler_of", lambda ip: ip.scheduler)
    ip = _IP2P()
    x = torch.rand(1, 3, LAT, LAT)
    fn = make_edit_divergence_loss(ip, zt_mode="diffuse_src", x_clean=x, seed=3)
    return ip, x, fn


def test_原圖自己的散度為零(built):
    """`x_def == x` 時兩支完全相同，散度是 0，損失也是 0。"""
    _, x, fn = built
    assert float(fn(x)) == pytest.approx(0.0, abs=1e-6)


def test_符號是負的_推得越開損失越小(built):
    """最小化的方向必須是「推開」。寫成正號會安靜地把圖推回原圖。"""
    _, x, fn = built
    near = (x + 0.02).clamp(0, 1)
    far = (1.0 - x)
    assert float(fn(near)) < 0.0
    assert float(fn(far)) < float(fn(near))


def test_文字條件一律是空字串(built):
    """攻擊指令未知是威脅模型的前提。"""
    ip, x, fn = built
    ip.unet.calls.clear()
    fn(torch.rand_like(x))
    assert ip.unet.calls, "UNet 沒有被呼叫"
    for emb in ip.unet.calls:
        assert torch.equal(emb, torch.zeros(1, 77, 4))


def test_沒有原圖就拋錯(monkeypatch):
    monkeypatch.setattr(mod, "_null_embedding", lambda ip: torch.zeros(1, 77, 4))
    monkeypatch.setattr(mod, "_scheduler_of", lambda ip: ip.scheduler)
    with pytest.raises(ValueError, match="x_clean"):
        make_edit_divergence_loss(_IP2P(), zt_mode="diffuse_src", x_clean=None)


def test_未知的zt模式拋錯(monkeypatch):
    monkeypatch.setattr(mod, "_null_embedding", lambda ip: torch.zeros(1, 77, 4))
    monkeypatch.setattr(mod, "_scheduler_of", lambda ip: ip.scheduler)
    with pytest.raises(ValueError, match="zt_mode"):
        make_edit_divergence_loss(_IP2P(), zt_mode="亂填",
                                  x_clean=torch.rand(1, 3, LAT, LAT))


def test_固定評估是決定性的且同樣是負的(built):
    _, x, fn = built
    fixed = fn.make_fixed(4, 99991)
    y = torch.rand_like(x)
    a, b = float(fixed(y)), float(fixed(y))
    assert a == b
    assert a < 0.0
    assert float(fixed(x)) == pytest.approx(0.0, abs=1e-6)


def test_空間權重只算指定區域(built):
    """設成主體遮罩就是「只要求主體那一塊被推開」。"""
    _, x, fn = built
    fixed = fn.make_fixed(3, 7)
    y = x.clone()
    y[..., :LAT // 2, :] = 1.0 - y[..., :LAT // 2, :]   # 只改上半
    top = torch.zeros(1, 1, LAT, LAT); top[..., :LAT // 2, :] = 1.0
    assert float(fixed(y, top)) < float(fixed(y, 1.0 - top))


def test_梯度回得到防禦圖但原圖那一支是常數(built):
    _, x, fn = built
    y = torch.rand_like(x).requires_grad_(True)
    fn(y).backward()
    assert y.grad is not None and float(y.grad.abs().sum()) > 0

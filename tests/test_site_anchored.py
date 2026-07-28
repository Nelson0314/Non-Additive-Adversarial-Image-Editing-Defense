"""site LA（錨定變體）的正確性測試。**分支上的提案，未進 main。**

核心不變量只有一條：φ=0 時 `x_def = x` **逐元素相等**。這正是 site L
做不到而 site P 做得到的事，也是本變體存在的唯一理由；若此測試不通過，
變體就沒有意義。
"""

import pytest
import torch

from src.defense.generator import DefenseGenerator
from src.models.sd import SDWrapper
from src.residual.site_latent import LatentResidual
from src.residual.site_latent_anchored import AnchoredLatentResidual
from src.utils.device import get_device

TINY = "hf-internal-testing/tiny-stable-diffusion-pipe"
SIZE, SEED, K = 64, 20260728, 2
DEV = get_device()


@pytest.fixture(scope="module")
def sd():
    return SDWrapper(TINY)


@pytest.fixture(scope="module")
def x01():
    g = torch.Generator().manual_seed(SEED)
    return torch.rand(1, 3, SIZE, SIZE, generator=g).to(DEV)


def _mod(sd, rank=4):
    lat = sd.latent_shape(SIZE, SIZE)
    return AnchoredLatentResidual(
        steps=K, channels=lat[1], size=lat[-1],
        max_rank=rank, const_rank=rank, seed=SEED,
    ).to(DEV)


def _baseline(sd, mod, x01):
    """G(x; φ=0)：停用模塊跑一次生成。"""
    gen = DefenseGenerator(sd, mod, k_inv=K, t_max=500)
    was = mod.enabled
    mod.disable()
    try:
        with torch.no_grad():
            return gen.generate(x01, gen.prepare(x01)).detach()
    finally:
        if was:
            mod.enable()


def test_錨定後phi為零時防禦圖與原圖逐元素相等(sd, x01):
    """本變體存在的唯一理由。site L 在此處必然失敗，見下一個測試。"""
    mod = _mod(sd)
    mod.set_baseline(_baseline(sd, mod, x01))
    gen = DefenseGenerator(sd, mod, k_inv=K, t_max=500)
    with torch.no_grad():
        x_def = gen.generate(x01, gen.prepare(x01))
    assert torch.equal(x_def, x01), "V 初始為零，錨定後必須與原圖完全相同"


def test_未錨定的site_L在phi為零時不等於原圖(sd, x01):
    """對照組：確認上一個測試檢驗的是錨定的效果，而非 V=0 的平凡結果。"""
    lat = sd.latent_shape(SIZE, SIZE)
    mod = LatentResidual(
        steps=K, channels=lat[1], size=lat[-1], max_rank=4, const_rank=4, seed=SEED
    ).to(DEV)
    gen = DefenseGenerator(sd, mod, k_inv=K, t_max=500)
    with torch.no_grad():
        x_def = gen.generate(x01, gen.prepare(x01))
    assert not torch.equal(x_def, x01), (
        "site L 若在此相等，代表重建誤差為零，錨定變體就沒有存在必要"
    )


def test_參數非零時錨定後仍會改變(sd, x01):
    """錨定不得把防禦一起消掉。"""
    mod = _mod(sd)
    mod.set_baseline(_baseline(sd, mod, x01))
    with torch.no_grad():
        mod.tensor.V.normal_(0, 0.5, generator=torch.Generator(DEV).manual_seed(SEED))
    gen = DefenseGenerator(sd, mod, k_inv=K, t_max=500)
    with torch.no_grad():
        x_def = gen.generate(x01, gen.prepare(x01))
    assert not torch.allclose(x_def, x01, atol=1e-6)


def test_梯度仍抵達phi(sd, x01):
    """錨定是常數平移，不得阻斷梯度。"""
    mod = _mod(sd)
    mod.set_baseline(_baseline(sd, mod, x01))
    gen = DefenseGenerator(sd, mod, k_inv=K, t_max=500)
    x_def = gen.generate(x01, gen.prepare(x01))
    x_def.pow(2).mean().backward()
    assert mod.tensor.V.grad is not None
    assert mod.tensor.V.grad.abs().sum() > 0


def test_缺少baseline時必須報錯而非退化為site_L(sd, x01):
    """靜默退化會讓兩個方法的結果混在一起，是最難察覺的錯誤來源。"""
    mod = _mod(sd)
    assert not mod.has_baseline
    with pytest.raises(RuntimeError, match="baseline"):
        mod.anchor(x01, x01)


def test_baseline必須是常數(sd, x01):
    mod = _mod(sd)
    with pytest.raises(ValueError, match="detach"):
        mod.set_baseline(x01.clone().requires_grad_(True))

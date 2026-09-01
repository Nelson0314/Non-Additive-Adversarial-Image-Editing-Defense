"""加法項的價目分配。**三個變體的總預算相同，改的是花在哪裡。**

存在理由
────────────────────────────────────────────────────────────────────
DCT-Shield 的預算是逐係數的 ±eps·Q(w)，**跨區塊是一個常數**。本方法的加法
下限（`--spectral-floor`）預設也是這個形狀，兩者因此在構造敘述上靠得很近。
`floor_gate` 提供兩個內容相依的替代分配：

    complement  只花在紋理閘的補集上，即乘法那一半動不了的區塊
    watson      亮度遮蔽 × 對比遮蔽（Watson 1993 / Podilchuk & Zeng 1998）

比較必須在**等預算**上做，否則只是強度比較。這裡把「等預算」這條性質釘死。
"""

import math

import pytest
import torch

from src.residual.texture_rephase import (
    FLOOR_GATES, PhaseResidual, radial_gate,
)
from src.residual.perceptual_weight import freq_weight

BLOCK = 32
DT = torch.float64
DEV = torch.device("cpu")


def _module(floor_gate: str, spectral_floor: float = 0.04) -> PhaseResidual:
    m = PhaseResidual(size=128, block=BLOCK, hop=8, r_min=0.12, theta_max=1.0,
                      gain_max=1.0, energy_quantile=0.0,
                      freq_weight="jpeg_luma", freq_weight_power=0.25,
                      spectral_floor=spectral_floor,
                      floor_gate=floor_gate).to(DT)
    return m


def _image(seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    x = torch.rand(1, 3, 128, 128, generator=g, dtype=DT)
    # 左半平坦、右半有紋理，讓紋理閘的補集不是全零也不是全一
    x[:, :, :, :64] = 0.45
    return x


def test_未知的名字直接拋錯而不是靜默退回預設():
    with pytest.raises(ValueError, match="floor_gate"):
        PhaseResidual(size=64, block=16, spectral_floor=0.04,
                      floor_gate="not_a_gate")


def test_uniform_逐位元等於加這個旗標之前():
    """加旗標之前的價目就是 徑向帶通 × jpeg_luma（一次方）。"""
    x = _image()
    m = _module("uniform")
    m.prepare_gates(x)
    want = (radial_gate(BLOCK, 0.12, DEV, DT)
            * freq_weight("jpeg_luma", BLOCK, DEV, DT))
    assert torch.equal(m.floor_price(), want)


@pytest.mark.parametrize("gate", ["complement", "watson"])
def test_總預算與uniform相同(gate):
    x = _image()
    ref = _module("uniform")
    ref.prepare_gates(x)
    m = _module(gate)
    m.prepare_gates(x)
    assert float(m.floor_price().mean()) == pytest.approx(
        float(ref.floor_price().mean()), rel=1e-9)


@pytest.mark.parametrize("gate", ["complement", "watson"])
def test_通帶外的零格不會復活(gate):
    """乘上任何逐區塊的因子都不得讓 fx=0 與 fx=N/2 兩行復活——rfft2 的共軛
    對稱依賴它們，破壞了輸出仍是實數，只是幅度不再保留，而且沒有症狀。"""
    x = _image()
    m = _module(gate)
    m.prepare_gates(x)
    price = m.floor_price()
    assert float(price[..., :, 0].abs().max()) == 0.0
    assert float(price[..., :, -1].abs().max()) == 0.0
    band = radial_gate(BLOCK, 0.12, DEV, DT) > 0
    assert float(price[..., ~band].abs().max()) == 0.0


def test_complement_在乘法可達量最大的區塊上是零():
    """定義即 `1 - reach_b / max reach_b`，最大的那一格必須恰好是零。"""
    x = _image()
    m = _module("complement")
    m.prepare_gates(x)
    spec = m.analyze(x)
    reach = (spec.abs().mean(dim=1) * m.gate()).flatten(2).norm(dim=2)[0]
    top = int(torch.argmax(reach))
    price = m.floor_price()
    assert float(price[0, top].abs().max()) == pytest.approx(0.0, abs=1e-12)


def test_complement_平坦區拿到的預算高於紋理區():
    x = _image()                                  # 左半平坦、右半隨機紋理
    m = _module("complement")
    m.prepare_gates(x)
    per_block = m.floor_price().sum(dim=(-1, -2))[0]
    n = int(round(per_block.numel() ** 0.5))
    grid = per_block.reshape(n, n)
    flat_half = float(grid[:, : n // 2].mean())
    tex_half = float(grid[:, n // 2:].mean())
    assert flat_half > tex_half


def test_complement_逐區塊正比於一減正規化可達量():
    x = _image()
    m = _module("complement")
    m.prepare_gates(x)
    per_block = m.floor_price().sum(dim=(-1, -2))[0]
    spec = m.analyze(x)
    reach = (spec.abs().mean(dim=1) * m.gate()).flatten(2).norm(dim=2)[0]
    want = 1.0 - reach / reach.max()
    k = float(per_block.sum() / want.sum())
    assert torch.allclose(per_block, want * k, rtol=1e-8, atol=1e-12)


def test_watson_價目隨區塊變而uniform不隨區塊變():
    x = _image()
    m = _module("watson")
    m.prepare_gates(x)
    per_block = m.floor_price().sum(dim=(-1, -2))[0]
    assert float(per_block.std()) > 0.0


def test_theta為零時三個變體都仍然恆等():
    """價目表只決定加法項的上限，參數初始化為 0 時它不動任何東西。"""
    x = _image()
    for gate in FLOOR_GATES:
        m = _module(gate)
        m.prepare_gates(x)
        y = m.pixel_residual(x)
        assert float((y - x).abs().max()) < 1e-12, gate


def test_spectral_floor為零時不建價目表():
    x = _image()
    m = _module("complement", spectral_floor=0.0)
    m.prepare_gates(x)
    with pytest.raises(RuntimeError, match="沒有價目表"):
        m.floor_price()


def test_complement_rank_低可達量的一半恰好拿到四分之三的預算():
    """名次轉換的目的就是把重尾分布拉平：`1 − rank/(L−1)` 的前半段面積是
    後半段的三倍，與影像內容無關。"""
    x = _image()
    m = _module("complement_rank")
    m.prepare_gates(x)
    per_block = m.floor_price().sum(dim=(-1, -2))[0]
    spec = m.analyze(x)
    reach = (spec.abs().mean(dim=1) * m.gate()).flatten(2).norm(dim=2)[0]
    order = torch.argsort(reach)
    half = len(order) // 2
    lo = float(per_block[order[:half]].sum() / per_block.sum())
    assert lo == pytest.approx(0.75, abs=0.02)


def test_complement_rank_可達量最大的區塊拿到零():
    x = _image()
    m = _module("complement_rank")
    m.prepare_gates(x)
    spec = m.analyze(x)
    reach = (spec.abs().mean(dim=1) * m.gate()).flatten(2).norm(dim=2)[0]
    top = int(torch.argmax(reach))
    assert float(m.floor_price()[0, top].abs().max()) == pytest.approx(0.0, abs=1e-12)


def test_complement_rank_的總預算也與uniform相同():
    x = _image()
    ref = _module("uniform")
    ref.prepare_gates(x)
    m = _module("complement_rank")
    m.prepare_gates(x)
    assert float(m.floor_price().mean()) == pytest.approx(
        float(ref.floor_price().mean()), rel=1e-9)

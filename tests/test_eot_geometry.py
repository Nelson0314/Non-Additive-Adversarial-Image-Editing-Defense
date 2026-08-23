"""隨機化的幾何 EOT：對**一族**裁切取期望值，而不是對一個固定的幾何。

`make_eot_ops_transform` 裡本來就有 `crop_resize`，但那是**固定**的中心裁切、
比例恆為 0.10。固定的變換可以被 co-adapt——最佳化只要學會那一個位移即可，
不會產生對一族裁切的不變性。

量測上的理由（`runs/ip2p_residual_signature/band_transfer.csv`）：裁切縮放留下
51–99% 的殘差能量，對原網格的餘弦是 0.000，對算子自己搬過的同一擾動卻是
0.995–0.996。擾動原封不動地通過了，只是被搬走——要對付的是**對位**。
"""

import math

import pytest
import torch

from src.defense.purify_aware import (
    GEOMETRY_FRACTIONS, make_eot_geometry_transform, random_crop_resize,
)
from src.purify.ops import CROP_FRACTION_DIA, crop_resize


def _x(n: int = 64) -> torch.Tensor:
    g = torch.Generator().manual_seed(0)
    return torch.rand(1, 3, n, n, generator=g)


def test_比例為零時逐位元恆等():
    x = _x()
    g = torch.Generator().manual_seed(0)
    assert torch.equal(random_crop_resize(x, 0.0, g), x)


def test_輸出尺寸不變():
    x = _x()
    g = torch.Generator().manual_seed(0)
    for frac in (0.05, 0.10, 0.15):
        assert random_crop_resize(x, frac, g).shape == x.shape


def test_同一個比例下位置每次不同():
    """這正是它與 ops.crop_resize 的唯一差別。"""
    x = _x()
    g = torch.Generator().manual_seed(1)
    a, b = random_crop_resize(x, 0.1, g), random_crop_resize(x, 0.1, g)
    assert not torch.equal(a, b)


def test_偏移抽到正中央時與中心裁切一致():
    """位置是唯一的差別，所以抽到中央就必須逐位元等於既有的算子——否則兩支
    的插值設定已經悄悄分岔了。"""
    x = _x()

    class _Centre:
        def __init__(self, mid):
            self.mid = mid
        def __call__(self, *a, **k):
            return torch.tensor([self.mid])

    n = x.shape[-1]
    dh = int(round(n * CROP_FRACTION_DIA))
    orig = torch.randint
    try:
        torch.randint = _Centre(dh)          # 讓 top = left = dh，即置中
        g = torch.Generator().manual_seed(0)
        got = random_crop_resize(x, CROP_FRACTION_DIA, g)
    finally:
        torch.randint = orig
    assert torch.equal(got, crop_resize(x, CROP_FRACTION_DIA))


def test_族裡含identity也含非identity():
    x = _x()
    t = make_eot_geometry_transform(seed=0)
    outs = [t(x, i) for i in range(24)]
    assert any(torch.equal(o, x) for o in outs)
    assert any(not torch.equal(o, x) for o in outs)


def test_評測用的比例在族內():
    """EOT 只有在評測點落在訓練分布裡時才成立。"""
    assert CROP_FRACTION_DIA in GEOMETRY_FRACTIONS


def test_梯度通得過():
    x = _x().requires_grad_(True)
    t = make_eot_geometry_transform(seed=3)
    t(x, 0).sum().backward()
    assert x.grad is not None and float(x.grad.abs().sum()) > 0


def test_空的族與越界的比例都拋錯():
    with pytest.raises(ValueError, match="不可為空"):
        make_eot_geometry_transform(())
    with pytest.raises(ValueError, match="0, 0.5"):
        make_eot_geometry_transform((0.6,))


def test_種子相同時序列可重跑():
    x = _x()
    a = [make_eot_geometry_transform(seed=7)(x, i) for i in range(6)]
    b = [make_eot_geometry_transform(seed=7)(x, i) for i in range(6)]
    assert all(torch.equal(p, q) for p, q in zip(a, b))


def test_驅動的旗標認得這一支():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    from ip2p_run import build_parser
    args = build_parser().parse_args(["--out", "x", "--purify-aware",
                                      "eot_geometry"])
    assert args.purify_aware == "eot_geometry"

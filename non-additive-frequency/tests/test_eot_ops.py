"""多算子 EOT（2026-08-21 的改動三第二段）。

由來：縮小版階段 C 量到紋理重相位在 JPEG 上淨增益 +0.1704（勝 DCT-Shield
2.77 倍、逐圖 5/5），但模糊上打平、裁切縮放上輸（+0.0360 對 +0.1083、0/5）。
把整組算子放進期望值是文獻上處理這種情形的標準解。

要釘的是四件會靜默失效的事：

1. 兩個新算子必須**可微**——不可微的話梯度是零，最佳化看起來在跑但什麼都
   沒學到；
2. 它們的參數必須與事後量測用的 `src/purify/ops.py` 對得上，否則最佳化的
   對象和評測的對象是兩個東西；
3. `identity` 必須在候選裡，否則會為了抗淨化而放棄未淨化時的效果；
4. 抽樣要真的在動——固定回同一個算子等於沒有 EOT。
"""

import torch

from src.defense.purify_aware import make_eot_ops_transform
from src.purify import ops as purify_ops


def _img(n=1, size=64):
    g = torch.Generator().manual_seed(0)
    return torch.rand(n, 3, size, size, generator=g)


def test_算子直接取自ops而不是另寫一份():
    """複製一份會讓最佳化的對象與評測的對象悄悄不同。這裡要求是**同一個函式**。"""
    import src.defense.purify_aware as pa

    assert pa.gaussian_blur is purify_ops.gaussian_blur
    assert pa.crop_resize is purify_ops.crop_resize


def test_兩個算子都可微且形狀不變():
    for fn, arg in ((purify_ops.gaussian_blur, 1.0),
                    (purify_ops.crop_resize, purify_ops.CROP_FRACTION_DIA)):
        x = _img().requires_grad_(True)
        y = fn(x, arg)
        assert y.shape == x.shape
        y.sum().backward()
        assert x.grad is not None and float(x.grad.abs().sum()) > 0, fn.__name__


def test_identity在候選裡():
    """抽很多次，必須有一次輸出逐位等於輸入。"""
    x = _img()
    t = make_eot_ops_transform(seed=0)
    assert any(torch.equal(t(x, i), x) for i in range(60)), \
        "identity 不在候選裡，最佳化會放棄未淨化時的效果"


def test_抽樣真的在動():
    x = _img()
    t = make_eot_ops_transform(seed=0)
    outs = [t(x, i) for i in range(40)]
    distinct = 0
    for i in range(1, len(outs)):
        if not torch.allclose(outs[i], outs[0], atol=1e-6):
            distinct += 1
    assert distinct > 5, "抽樣沒有在動，等於沒有 EOT"


def test_可微且梯度傳得回輸入():
    x = _img().requires_grad_(True)
    t = make_eot_ops_transform(seed=1)
    total = sum(t(x, i).sum() for i in range(8))
    total.backward()
    assert float(x.grad.abs().sum()) > 0


def test_空品質清單被拒絕():
    import pytest

    with pytest.raises(ValueError, match="qualities"):
        make_eot_ops_transform(qualities=())

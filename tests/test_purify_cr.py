"""C&R 串接淨化算子（arXiv:2604.23688 §1 的協定）。

該協定的重點是**順序**與**串接**：只測單獨變換會嚴重高估 robustness。
故這裡除了值域與代理行為之外，也釘住「串接比單獨 JPEG 破壞更多」。
"""

import torch

from src.purify.ops import CR_JPEG_QUALITY, Purifier, jpeg_then_resize


def _image(seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.rand(2, 3, 64, 64, generator=g)


def test_shape_dtype_and_range_are_preserved():
    x = _image()
    y = jpeg_then_resize(x)
    assert y.shape == x.shape and y.dtype == x.dtype
    assert float(y.min()) >= 0.0 and float(y.max()) <= 1.0


def test_registered_and_not_differentiable():
    p = Purifier("jpeg_then_resize", CR_JPEG_QUALITY)
    assert p.available
    assert not p.differentiable


def test_forward_is_straight_through():
    """前向為真實輸出、反向視為恆等——與 `jpeg` 相同的限制。"""
    x = _image().requires_grad_()
    p = Purifier("jpeg_then_resize", CR_JPEG_QUALITY)
    p.forward(x).sum().backward()
    assert torch.allclose(x.grad, torch.ones_like(x))


def test_evaluate_matches_the_real_operator():
    x = _image()
    p = Purifier("jpeg_then_resize", CR_JPEG_QUALITY)
    assert torch.equal(p.evaluate(x), p._run(x))


def test_chain_destroys_more_than_jpeg_alone():
    x = _image()
    cr = Purifier("jpeg_then_resize", CR_JPEG_QUALITY).evaluate(x)
    j = Purifier("jpeg", CR_JPEG_QUALITY).evaluate(x)
    assert float((cr - x).pow(2).mean()) > float((j - x).pow(2).mean())


def test_without_upsample_back_the_size_changes():
    """升回原尺寸是我方指定的一步，關掉時必須真的停在 0.5×。"""
    x = _image()
    y = jpeg_then_resize(x, upsample_back=False)
    assert y.shape[-2:] == (32, 32)

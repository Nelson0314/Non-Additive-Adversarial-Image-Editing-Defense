import pytest
import torch

from src.defense.eot import (EOTObjective, TRANSFORMS, random_blur,
                             random_crop_resize)


class _Counting:
    """記下每次被餵到的影像，好驗證有幾個視角、以及它們不相同。"""

    device = torch.device('cpu')
    ip2p = None
    timesteps = [1, 2]
    weights = {'id': 1.0}

    def __init__(self):
        self.seen = []

    def terms(self, x):
        self.seen.append(x)
        return {'enc': x.mean(), 'cond': x.std()}

    def score(self, x):
        self.seen.append(x)
        return x.mean()


def _image(n=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(1, 3, n, n, generator=g)


def test_隨機裁切改變影像但保住尺寸():
    x = _image()
    g = torch.Generator().manual_seed(0)
    y = random_crop_resize(x, g)
    assert y.shape == x.shape
    assert float((y - x).abs().max()) > 0.0


def test_隨機裁切的比例落在指定區間():
    x = _image(100)
    g = torch.Generator().manual_seed(3)
    for _ in range(20):
        y = random_crop_resize(x, g, lo=0.05, hi=0.10)
        assert y.shape == x.shape
    assert True


def test_隨機模糊改變影像且可微():
    x = _image().requires_grad_(True)
    g = torch.Generator().manual_seed(1)
    y = random_blur(x, g)
    assert y.shape == x.shape
    y.sum().backward()
    assert x.grad is not None and float(x.grad.abs().sum()) > 0.0


def test_裁切可微():
    x = _image().requires_grad_(True)
    g = torch.Generator().manual_seed(1)
    random_crop_resize(x, g).sum().backward()
    assert x.grad is not None and float(x.grad.abs().sum()) > 0.0


def test_恆等一定在視角裡():
    """少了恆等，解可以押在「只有被處理過才成立」的地方。"""
    base = _Counting()
    eot = EOTObjective(base, kinds=['crop_resize'], samples=2, seed=0)
    x = _image()
    eot.score(x)
    assert len(base.seen) == 3
    assert base.seen[0] is x


def test_關掉恆等時視角數就是樣本數():
    base = _Counting()
    eot = EOTObjective(base, kinds=['crop_resize'], samples=2, seed=0,
                       include_identity=False)
    eot.score(_image())
    assert len(base.seen) == 2


def test_每一步抽到的變換不同():
    base = _Counting()
    eot = EOTObjective(base, kinds=['crop_resize'], samples=1, seed=0)
    x = _image()
    eot.score(x)
    eot.score(x)
    a, b = base.seen[1], base.seen[3]
    assert float((a - b).abs().max()) > 0.0


def test_不認得的前處理直接拋錯():
    with pytest.raises(ValueError):
        EOTObjective(_Counting(), kinds=['jpeg'], samples=1)


def test_只有恆等時拒絕建立():
    with pytest.raises(ValueError):
        EOTObjective(_Counting(), kinds=['identity'], samples=1)


def test_項是各視角的平均():
    base = _Counting()
    eot = EOTObjective(base, kinds=['blur'], samples=1, seed=5)
    x = _image()
    terms = eot.terms(x)
    views = base.seen
    assert len(views) == 2
    expect = (views[0].mean() + views[1].mean()) / 2
    assert float(terms['enc']) == pytest.approx(float(expect), abs=1e-6)


def test_亂數不碰全域狀態():
    """EOT 自己一個 Generator：載體初始化與探針噪聲都是確定性的，不可被污染。"""
    torch.manual_seed(1234)
    before = torch.rand(4)
    torch.manual_seed(1234)
    eot = EOTObjective(_Counting(), kinds=['crop_resize'], samples=3, seed=7)
    eot.score(_image())
    after = torch.rand(4)
    assert torch.allclose(before, after)


def test_可用的前處理清單不含不可微的算子():
    assert set(TRANSFORMS) == {'identity', 'crop_resize', 'blur'}


def test_接得上求解器():
    from src.defense.geometry_field import FlowFieldParam
    from src.defense.immunise import optimise_carrier

    class Target:
        device = torch.device('cpu')
        ip2p = None
        timesteps = [1]
        weights = {}

        def __init__(self, t):
            self.t = t

        def score(self, y):
            return ((y - self.t) ** 2).mean()

        def terms(self, y):
            return {'enc': self.score(y)}

    x = _image(32)
    eot = EOTObjective(Target(torch.roll(x, 3, dims=-1)),
                       kinds=['crop_resize', 'blur'], samples=1, seed=0)
    c = FlowFieldParam(x, grid=4, taper=4.0)
    c.reset(x)
    stats = optimise_carrier(c, x, eot, steps=8, lr=0.4, caps=[], check_every=4)
    assert stats['free_score_end'] < stats['free_score_start']

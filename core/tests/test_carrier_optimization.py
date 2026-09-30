"""不含指令目標與受約束載體最佳化；來源為 anti-purification/tests/test_instruction_free.py。

歷史載體 `carrier_search.build_carrier` 不屬於 core，以 `carrier_stub.OffsetCarrier` 代替；
指令設定檔守衛屬於舊入口腳本，未移植。
"""
import pytest
import torch

from immunization_core.optimization.carrier import (Cap, cap_violations, fit_caps,
                                                    optimize_carrier, randomize_carrier)
from immunization_core.optimization.instruction_free import (FreeObjective, _scaled_sample,
                                                             _sigma_at, pick_timesteps)

from carrier_stub import OffsetCarrier



class FakeScheduler:
    init_noise_sigma = 1.0

    def __init__(self):
        self.timesteps = torch.tensor([])
        self.sigmas = None

    def set_timesteps(self, n, device=None):
        self.timesteps = torch.arange(n - 1, -1, -1, dtype=torch.long) * 10
        self.sigmas = torch.linspace(1.0, 0.05, n, dtype=torch.float64)

    def scale_model_input(self, z, t):
        return z

    def step(self, eps, t, z, generator=None, return_dict=True):
        out = z - 0.1 * eps
        return (out,) if not return_dict else out


class FakePipe:
    def __init__(self):
        self.scheduler = FakeScheduler()

    def _encode_prompt(self, prompt, device, num, cfg, negative=None):
        if prompt != '':
            raise AssertionError('不含指令的目標只能用空字串')
        return torch.zeros(1, 4, 8, dtype=torch.float64)


class FakeVae:
    dtype = torch.float64


class FakeIP2P:
    """只保留 `FreeObjective` 會碰到的那幾個介面，全部保持可微。"""

    def __init__(self, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.w = torch.randn(4, 3, generator=g, dtype=torch.float64)
        self.pipe = FakePipe()
        self.device = torch.device('cpu')
        self.vae = FakeVae()
        self.scaling_factor = 0.18215

    def encode_image(self, x01, use_ckpt=False):
        return torch.einsum('ij,bjhw->bihw', self.w, x01.double())

    def image_latents(self, x01):
        return self.encode_image(x01) / self.scaling_factor

    def decode_latent(self, z, use_ckpt=False):
        return torch.einsum('ji,bjhw->bihw', self.w, z.double()).clamp(0, 1)

    def unet(self, model_input, t, encoder_hidden_states=None, return_dict=True):
        eps = model_input[:, :4] * 0.5 + model_input[:, 4:] * 0.25
        return (eps,) if not return_dict else eps


def _image(seed=0, size=32):
    g = torch.Generator().manual_seed(seed)
    return (0.3 + 0.4 * torch.rand(1, 3, size, size, generator=g)).clamp(0, 1)


def _carrier(x, frame_amplitude=0.3, **_):
    return OffsetCarrier(x, amplitude=frame_amplitude)


def test_取的時刻不重複且最後一個最低():
    ts = pick_timesteps(FakeScheduler(), 4, 20)
    assert len(ts) == 4
    assert len(set(int(t) for t in ts)) == 4
    assert int(ts[-1]) == min(int(t) for t in ts)


def test_時刻數超過時間表長度就拋錯():
    with pytest.raises(ValueError):
        pick_timesteps(FakeScheduler(), 30, 20)


def test_原圖自己代進去時兩個位移項都是零():
    x = _image()
    obj = FreeObjective(FakeIP2P(), x, box=None, k=3, steps=20)
    t = obj.terms(x)
    assert float(t['enc']) == pytest.approx(0.0, abs=1e-12)
    assert float(t['cond']) == pytest.approx(0.0, abs=1e-12)
    assert 'id' not in t


def test_位移越大分數越低():
    x = _image(1)
    obj = FreeObjective(FakeIP2P(), x, box=None, k=3, steps=20)
    near = (x + 0.02).clamp(0, 1)
    far = (x + 0.20).clamp(0, 1)
    assert float(obj.score(far)) < float(obj.score(near)) < float(obj.score(x))


def test_梯度會傳到載體的參數():
    x = _image(2)
    obj = FreeObjective(FakeIP2P(), x, box=None, k=2, steps=20)
    c = _carrier(x, frame_amplitude=0.5)
    obj.score(c.render(x)).backward()
    g = c.stages[0].delta.grad
    assert g is not None and float(g.abs().max()) > 0


def test_噪聲會依時刻的_sigma_縮放():
    s = FakeScheduler()
    s.set_timesteps(4)
    noise = torch.ones(1, 4, 2, 2, dtype=torch.float64)
    hi = _scaled_sample(s, noise, s.timesteps[0])
    lo = _scaled_sample(s, noise, s.timesteps[-1])
    assert float(hi.abs().mean()) > float(lo.abs().mean())
    sg = s.sigmas[-1]
    assert torch.allclose(lo, noise * sg / (sg * sg + 1).sqrt())


def test_探針縮放與呼叫次序無關():
    """真的 EulerAncestral 的 scale_model_input 有狀態，連呼四次會全用第一個 sigma。"""
    from diffusers import EulerAncestralDiscreteScheduler
    sch = EulerAncestralDiscreteScheduler(beta_start=0.00085, beta_end=0.012,
                                          beta_schedule='scaled_linear',
                                          num_train_timesteps=1000)
    sch.set_timesteps(20)
    noise = torch.ones(1, 4, 2, 2)
    ts = [sch.timesteps[i] for i in (0, 7, 14, 19)]
    first = [float(_scaled_sample(sch, noise, t).abs().mean()) for t in ts]
    again = [float(_scaled_sample(sch, noise, t).abs().mean())
             for t in reversed(ts)][::-1]
    assert first == again
    assert len(set(round(v, 6) for v in first)) == len(ts)


def test_沒有_sigmas_的排程器會拋錯():
    class Bare:
        timesteps = torch.tensor([1])
        sigmas = None
    with pytest.raises(ValueError):
        _sigma_at(Bare(), 1)


def test_最佳化會把分數壓低():
    x = _image(3)
    obj = FreeObjective(FakeIP2P(), x, box=None, k=2, steps=20)
    c = _carrier(x, frame_amplitude=0.5)
    out = optimize_carrier(c, x, obj, steps=20, lr=0.08)
    assert out['free_score_end'] < out['free_score_start']
    assert out['free_steps'] == 20
    assert 'free_term_enc' in out and 'free_term_cond' in out


def test_多條上限同時二分且回報實際縮放():
    x = _image(7)
    c = _carrier(x, frame_amplitude=1.0)
    base = [s.amplitude for s in c.stages]

    def tight(y):
        return float((y - x).abs().mean() * 100)

    def loose(y):
        return float((y - x).abs().max() * 10)

    caps = [Cap('loose', None, loose, tight(c.render(x)) * 10),
            Cap('tight', None, tight, tight(c.render(x)) * 0.3)]
    m = fit_caps(c, x, caps)
    assert 0.0 < m < 1.0
    assert cap_violations(c, x, caps) == 0
    assert c.stages[0].amplitude == pytest.approx(base[0] * m)


def test_受約束最佳化會回到可行的_checkpoint():
    x = _image(4)
    obj = FreeObjective(FakeIP2P(), x, box=None, k=2, steps=20)
    c = _carrier(x, frame_amplitude=1.0)

    def measure(y):
        return float((y - x).abs().mean() * 100)

    cap = Cap('mean', lambda y: (y - x).abs().mean() * 100, measure, 0.1)
    out = optimize_carrier(c, x, obj, steps=10, lr=0.05, caps=[cap],
                           check_every=2)
    assert measure(c.render(x)) <= 0.1 + 1e-6
    assert out['free_cap_violations'] == 0
    assert out['free_feasible_step'] > 0 or out['free_amplitude_shrink'] < 1.0


def test_沒有可學參數時拋錯():
    x = _image(5)
    obj = FreeObjective(FakeIP2P(), x, box=None, k=2, steps=20)
    c = _carrier(x)
    for p in c.params():
        p.requires_grad_(False)
    with pytest.raises(ValueError):
        optimize_carrier(c, x, obj, steps=2)


def test_隨機對照與最佳化回傳相同的鍵():
    x = _image(6)
    obj = FreeObjective(FakeIP2P(), x, box=None, k=2, steps=20)
    c = _carrier(x, frame_amplitude=1.0)

    def measure(y):
        return float((y - x).abs().mean() * 100)

    cap = Cap('mean', lambda y: (y - x).abs().mean() * 100, measure, 0.5)
    rand = randomize_carrier(c, x, obj, seed=1, caps=[cap])
    opt = optimize_carrier(_carrier(x, frame_amplitude=1.0), x, obj, steps=4, caps=[cap])
    assert set(rand) == set(opt)
    assert rand['free_cap_violations'] == 0
    assert measure(c.render(x)) <= 0.5 + 1e-6

import pytest
import torch

from src.defense.instruction_free import FreeObjective
from src.defense.mainstream_terms import MainstreamTerms
from src.defense.readout_terms import CompositeObjective
from src.defense.sds_terms import SDSDiffusion, sds_sign_check
from tests.test_instruction_free import FakeIP2P, _image


def _trio(seed=0, size=16, **kw):
    x = _image(seed, size)
    obj = FreeObjective(FakeIP2P(), x, box=None, k=3, steps=20, **kw)
    return x, obj, SDSDiffusion(obj), MainstreamTerms(obj, x, use_ckpt=False)


def test_回報的損失與完整反傳的擴散項逐位元相同():
    """值必須是同一個函數，否則 CSV 裡兩欄不同量綱，比較不成立。"""
    x, obj, sds, extra = _trio()
    assert float(sds(x)) == pytest.approx(float(extra.diffusion(x)), abs=1e-12)


def test_梯度不穿過_unet():
    """UNet 一律在 `no_grad` 下呼叫；這是這一項唯一的省法。"""
    x, obj, sds, extra = _trio()
    seen = []
    original = obj._run_unet

    def watched(model_input, t, text):
        seen.append(bool(torch.is_grad_enabled()))
        return original(model_input, t, text)

    obj._run_unet = watched
    y = x.clone().requires_grad_(True)
    sds(y).backward()
    assert seen and not any(seen)
    assert y.grad is not None and torch.isfinite(y.grad).all()
    assert float(y.grad.abs().sum()) > 0.0


def test_梯度等於丟掉_unet_雅可比之後的解析式():
    """`∂L/∂z₀ = (2/N)·(ε_θ − ε)/√(σ²+1)` 逐時刻平均，再經 VAE 的線性映射。"""
    from src.defense.instruction_free import _sigma_at

    x, obj, sds, extra = _trio()
    cond = obj.ip2p.image_latents(x).to(obj.text.dtype)
    manual = None
    with torch.no_grad():
        z0 = obj.ip2p.encode_image(x)
        for t in obj.timesteps:
            sig = _sigma_at(obj.scheduler, t).to(z0)
            scale = 1.0 / float((sig * sig + 1).sqrt())
            zt = (z0.to(obj.noise.dtype)
                  + sig.to(obj.noise.dtype) * obj.noise) * scale
            eps = obj._run_unet(torch.cat([zt, cond.to(zt.dtype)], dim=1), t,
                                obj.text).float()
            resid = eps - obj.noise.float()
            g = resid * (2.0 * scale / resid.numel())
            manual = g if manual is None else manual + g
        manual = manual / len(obj.timesteps)
        expect = torch.einsum('ij,bihw->bjhw', obj.ip2p.w, manual.double())

    y = x.clone().requires_grad_(True)
    sds(y).backward()
    assert torch.allclose(y.grad.double(), expect, atol=1e-10)


def test_符號檢查兩側都動而且方向正確():
    """升方向要讓損失變大、降方向要變小；只看單側會被「怎麼動都變大」蒙混。"""
    x, obj, sds, extra = _trio(size=24)
    r = sds_sign_check(sds, x, steps=(0.02, 0.05))
    for h in (0.02, 0.05):
        assert r[f'descend_{h}'] < r['base'] < r[f'ascend_{h}']
    assert r['gain_0.05'] > r['gain_0.02'] > 0.0
    assert r['grad_rms'] > 0.0


def test_在合成目標裡與_diffusion_同符號同縮放():
    x, obj, sds, extra = _trio()
    base_w = {'id': 0.0, 'enc': 0.0, 'cond': 0.0}
    a = CompositeObjective(obj, sds_term=sds, weights=dict(base_w, sds=1.0))
    b = CompositeObjective(obj, mainstream=extra,
                           weights=dict(base_w, diffusion=1.0))
    assert float(a.score(x)) == pytest.approx(float(b.score(x)), abs=1e-9)
    assert float(a.score(x)) < 0.0


def test_沒有掛上_sds_就不會出現在逐項裡():
    x, obj, sds, extra = _trio()
    c = CompositeObjective(obj, sds_term=None,
                           weights={'id': 0.0, 'enc': 0.5, 'cond': 1.0})
    assert 'sds' not in c.terms(x)


def test_接得上求解器並寫出逐項欄位():
    from src.defense.color_param import ColorCurveParam
    from src.defense.immunise import optimise_carrier

    x, obj, sds, extra = _trio(size=24)
    carrier = ColorCurveParam(radius=1.0, pieces=8)
    carrier.reset(x, seed=0)
    c = CompositeObjective(obj, carrier=carrier, sds_term=sds,
                           weights={'id': 0.0, 'enc': 0.5, 'cond': 1.0,
                                    'sds': 1.0})
    stats = optimise_carrier(carrier, x, c, steps=6, lr=0.3, caps=[],
                             check_every=3)
    assert stats['free_score_end'] < stats['free_score_start']
    assert 'free_term_sds' in stats

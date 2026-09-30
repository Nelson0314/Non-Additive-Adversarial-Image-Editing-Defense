import pytest
import torch

from src.defense.conditioning_response import (finite_response, null_trajectory,
                                               probe_directions,
                                               response_energy, scale_latent)

NULL = torch.tensor([0.3, -0.7, 1.1, 0.5, -0.2, 0.9, -1.3, 0.4])


class FakeScheduler:
    init_noise_sigma = 4.0

    def __init__(self):
        self.timesteps = torch.tensor([])
        self.sigmas = None

    def set_timesteps(self, n, device=None):
        self.timesteps = torch.arange(n - 1, -1, -1, dtype=torch.long) * 10
        self.sigmas = torch.linspace(4.0, 0.05, n, dtype=torch.float32)

    def scale_model_input(self, z, t):
        return z

    def step(self, eps, t, z, generator=None, return_dict=True):
        out = z - 0.05 * eps
        return (out,) if not return_dict else out


class FakePipe:
    def __init__(self):
        self.scheduler = FakeScheduler()

    def _encode_prompt(self, prompt, device, num, cfg, negative=None):
        if prompt != '':
            raise AssertionError('這個探針只能用空字串')
        return NULL.view(1, 2, 4).clone()

    def prepare_extra_step_kwargs(self, generator, eta):
        return {'generator': generator}


class FakeIP2P:
    """`cross=True` 時文字與影像有乘性交互，`False` 時只有加性。"""

    def __init__(self, cross=True, alpha=0.75):
        self.pipe = FakePipe()
        self.device = torch.device('cpu')
        self.scaling_factor = 0.18215
        self.cross = cross
        self.alpha = alpha

    def image_latents(self, x01):
        return x01.mean(1, keepdim=True).repeat(1, 4, 1, 1).float()

    def unet(self, model_input, t, encoder_hidden_states=None, return_dict=True):
        img = model_input[:, 4:]
        s = encoder_hidden_states.flatten(1).mean(1).view(-1, 1, 1, 1)
        eps = img + s if not self.cross else img * (1.0 + self.alpha * s)
        return (eps,) if not return_dict else eps


def _image(seed=0, size=16):
    g = torch.Generator().manual_seed(seed)
    return (0.2 + 0.6 * torch.rand(1, 3, size, size, generator=g)).clamp(0, 1)


def _setup(cross=True, size=16):
    ip2p = FakeIP2P(cross=cross)
    e0 = ip2p.pipe._encode_prompt('', ip2p.device, 1, False)
    sigma_e = float(e0.pow(2).mean().sqrt())
    z = torch.zeros(1, 4, size, size)
    sigma = torch.tensor(1.0)
    return ip2p, e0, sigma_e, scale_latent(z, sigma)


def test_directions_are_rms_one_and_seed_reproducible():
    e0 = NULL.view(1, 2, 4)
    a = probe_directions(e0, 5, 11)
    b = probe_directions(e0, 5, 11)
    assert a.shape == (5, 2, 4)
    assert torch.equal(a, b)
    rms = a.flatten(1).pow(2).mean(1).sqrt()
    assert rms.tolist() == pytest.approx([1.0] * 5, abs=1e-5)
    assert not torch.allclose(a[0], a[1])


def test_central_difference_is_exact_on_a_text_linear_model():
    ip2p, e0, sigma_e, scaled = _setup(cross=True)
    v = probe_directions(e0, 1, 3)[0:1]
    cond = ip2p.image_latents(_image())
    d = finite_response(ip2p, scaled, torch.tensor(10), cond, e0, v,
                        0.0625, sigma_e)
    want = cond * ip2p.alpha * float(v.flatten().mean()) * sigma_e
    assert torch.allclose(d, want, atol=1e-5)


def test_two_step_sizes_agree_when_the_response_is_first_order():
    """兩個尺度差很多，代表是捨入主導而不是真的一階反應。"""
    ip2p, e0, sigma_e, scaled = _setup(cross=True)
    v = probe_directions(e0, 1, 3)[0:1]
    cond = ip2p.image_latents(_image())
    a, b = (response_energy(finite_response(ip2p, scaled, torch.tensor(10),
                                            cond, e0, v, h, sigma_e))
            for h in (0.0625, 0.125))
    assert a == pytest.approx(b, rel=1e-6)


def test_an_additive_text_branch_gives_ratio_exactly_one():
    """Codex 的結構性反對意見寫成測試。

    若局部行為是 `f(c,e) = a(c) + b(e)`，文字 Jacobian 非零但**完全不受影像
    控制**，任何載體都推不動比值。這一題釘住那個情況會長什麼樣，
    好在真實模型上認得出來。
    """
    ip2p, e0, sigma_e, scaled = _setup(cross=False)
    v = probe_directions(e0, 1, 3)[0:1]
    t = torch.tensor(10)
    ref = ip2p.image_latents(_image(0))
    alt = ip2p.image_latents(_image(1) * 0.25)
    ea = response_energy(finite_response(ip2p, scaled, t, ref, e0, v, 0.0625, sigma_e))
    eb = response_energy(finite_response(ip2p, scaled, t, alt, e0, v, 0.0625, sigma_e))
    assert eb / ea == pytest.approx(1.0, rel=1e-6)


def test_a_multiplicative_cross_term_makes_the_ratio_track_the_image():
    ip2p, e0, sigma_e, scaled = _setup(cross=True)
    v = probe_directions(e0, 1, 3)[0:1]
    t = torch.tensor(10)
    ref = ip2p.image_latents(_image(0))
    alt = ref * 0.5
    ea = response_energy(finite_response(ip2p, scaled, t, ref, e0, v, 0.0625, sigma_e))
    eb = response_energy(finite_response(ip2p, scaled, t, alt, e0, v, 0.0625, sigma_e))
    assert eb / ea == pytest.approx(0.25, rel=1e-5)


def test_a_step_that_rounds_away_raises_instead_of_reporting_zero():
    ip2p, _, _, scaled = _setup(cross=True)
    e0 = NULL.view(1, 2, 4).to(torch.bfloat16)
    v = probe_directions(e0, 1, 3)[0:1]
    cond = ip2p.image_latents(_image()).to(torch.bfloat16)
    with pytest.raises(ValueError):
        finite_response(ip2p, scaled, torch.tensor(10), cond, e0, v,
                        1e-9, 1.0)


def test_scale_latent_divides_by_sqrt_sigma_squared_plus_one():
    z = torch.randn(1, 4, 8, 8)
    s = torch.tensor(3.0)
    assert torch.allclose(scale_latent(z, s), z / (s * s + 1).sqrt())


def test_trajectory_carries_its_own_decreasing_sigma():
    ip2p = FakeIP2P()
    traj = null_trajectory(ip2p, _image(), steps=6, seed=5)
    assert len(traj) == 6
    sig = [float(s) for _, _, s in traj]
    assert sig == sorted(sig, reverse=True)
    assert all(z.shape == traj[0][1].shape for _, z, _ in traj)


def test_trajectory_is_reproducible_and_seed_dependent():
    ip2p = FakeIP2P()
    a = null_trajectory(ip2p, _image(), steps=4, seed=5)
    b = null_trajectory(ip2p, _image(), steps=4, seed=5)
    c = null_trajectory(ip2p, _image(), steps=4, seed=6)
    assert torch.equal(a[-1][1], b[-1][1])
    assert not torch.equal(a[-1][1], c[-1][1])


def test_non_finite_energy_raises():
    with pytest.raises(ValueError):
        response_energy(torch.tensor([float('inf')]))


def test_bad_direction_count_raises():
    with pytest.raises(ValueError):
        probe_directions(NULL.view(1, 2, 4), 0, 1)

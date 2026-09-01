"""latent 臂：APA 階段二換上紋理重相位的相位參數化。

不載入 Stable Diffusion——`_attack_phase` 只透過四個方法用到 sd
（`alphas_cumprod`／`_eps_cfg`／`decode_latent`），故可以用一個假的 sd
把「只換這一個位置」這件事釘住：更新規則、迭代數、reward 形式都不動。
"""

import math

import pytest
import torch
import torch.nn.functional as F

from src.defense.apa_native_stage2 import (
    NativeStage2Config,
    _attack_phase,
    attack_native,
)


class _FakeSD:
    """只實作 `_trajectory_pass` 與 `_step_guidance` 真正呼叫到的東西。"""

    def __init__(self, device=torch.device("cpu")):
        self.device = device
        t = torch.linspace(0.999, 0.02, 1000)
        self._abar = t

    def alphas_cumprod(self, device=None):
        return self._abar.to(device or self.device)

    def _eps_cfg(self, z, t, emb, gs, emb_uncond, use_ckpt=False):
        # 對 z 可微、且與 t 有關，足以讓軌跡不是常數。
        return 0.1 * z * (1.0 + 0.001 * float(t))

    def decode_latent(self, z, use_ckpt=False):
        return torch.sigmoid(F.interpolate(z[:, :3], scale_factor=8,
                                           mode="nearest"))


def _fixture(size=16, steps=4):
    torch.manual_seed(0)
    sd = _FakeSD()
    la_0 = torch.randn(1, 4, size, size)
    ori = torch.randn(1, 4, size, size)
    y = torch.rand(1, 3, size * 8, size * 8)
    emb = torch.zeros(1, 77, 768)
    ts = torch.arange(0, 1000, 1000 // (steps + 2))[: steps + 2]
    return sd, la_0, ori, y, emb, ts


def _cfg(**kw):
    base = dict(parameterization="phase", steps=3, guidance_steps=1, niters=3,
                phase_block=8, phase_r_min=0.12, use_ckpt=False)
    base.update(kw)
    return NativeStage2Config(**base)


def test_phase_branch_moves_theta_within_bound():
    sd, la_0, ori, y, emb, ts = _fixture()
    cfg = _cfg(phase_theta_max=1.0, phase_mu=0.1)
    x_def, hist = _attack_phase(sd, la_0, emb, emb, ori, y, cfg, ts, log_every=99)

    assert len(hist) == cfg.niters
    assert x_def.shape == (1, 3, 128, 128)
    assert all(h["theta_linf"] <= 1.0 + 1e-6 for h in hist)
    assert hist[-1]["theta_linf"] > 0.0


def test_phase_branch_logs_latent_linf_not_theta():
    """同名欄位必須是同一個量：`linf` 是 latent 空間的，相位另記 `theta_linf`。"""
    sd, la_0, ori, y, emb, ts = _fixture()
    cfg = _cfg(phase_theta_max=1.0, phase_mu=0.5)
    _, hist = _attack_phase(sd, la_0, emb, emb, ori, y, cfg, ts, log_every=99)
    assert hist[-1]["linf"] != pytest.approx(hist[-1]["theta_linf"])
    assert hist[-1]["linf"] > 0.0


def test_phase_reward_is_recorded_and_normalized_once():
    """DEC-021 的正規化常數跨迭代共用，故第一筆 reward_main 恰為 −1。"""
    sd, la_0, ori, y, emb, ts = _fixture()
    cfg = _cfg(phase_theta_max=1.0, phase_mu=0.1, normalize_reward=True)
    _, hist = _attack_phase(sd, la_0, emb, emb, ori, y, cfg, ts, log_every=99)
    assert hist[0]["reward_main"] == pytest.approx(-1.0, abs=1e-6)


def test_zero_theta_max_reduces_to_no_perturbation():
    """半徑趨近零時 latent 不動——構造上的恆等在階段二裡也成立。"""
    sd, la_0, ori, y, emb, ts = _fixture()
    cfg = _cfg(phase_theta_max=1e-9, phase_mu=1e-9)
    _, hist = _attack_phase(sd, la_0, emb, emb, ori, y, cfg, ts, log_every=99)
    assert hist[-1]["linf"] < 1e-6


def test_bdia_is_rejected_rather_than_silently_wrong():
    sd, la_0, ori, y, emb, ts = _fixture()
    cfg = _cfg(use_bdia=True)
    with pytest.raises(ValueError, match="BDIA"):
        _attack_phase(sd, la_0, emb, emb, ori, y, cfg, ts, log_every=99)


def test_unknown_parameterization_raises():
    sd, la_0, ori, y, emb, ts = _fixture()
    cfg = NativeStage2Config(parameterization="nope", steps=3, niters=1)
    sd.encode_text = lambda s: emb
    sd.uncond_prompt = lambda: emb
    sd.timesteps = lambda n, t_max=None: ts
    with pytest.raises(ValueError, match="parameterization"):
        attack_native(sd, la_0, None, ori, "cat", y, cfg)


def test_default_parameterization_is_the_weak_baseline():
    """預設必須仍是 DEC-023 的 linf，否則既有批次會靜默換掉方法。"""
    assert NativeStage2Config().parameterization == "linf"
    assert NativeStage2Config().phase_theta_max == pytest.approx(math.pi)

"""`src/baselines/diffusionguard.py` 的驗收 — 用替身 SD，不載入真模型。

這一篇的實作幾乎全在「損失怎麼組」上，而組錯不會有症狀：輸出仍是一張合理的
防禦圖。故這裡用一個替身把 UNet 與 VAE 換成可預測的線性運算，驗證**起始
時間步的算式、噪聲每次重抽、PGD 的球投影**三件事。
"""

import pytest
import torch

from src.baselines.diffusionguard import (
    PAPER_EPS, PAPER_ITERS, PAPER_STEP_SIZE, DiffusionGuardSpec,
    make_early_step_loss, run_diffusionguard,
)


class _StubSD:
    """把 VAE 與 UNet 換成線性運算的替身。

    `encode_image` 取每 8×8 區塊的平均（形狀對得上真的 latent），
    `unet_forward` 回傳輸入乘一個常數——如此噪聲預測的範數就與輸入的範數
    成正比，可以手算。
    """

    num_train_timesteps = 1000

    def __init__(self, gain=2.0, device="cpu", dtype=torch.float64):
        self.gain = gain
        self.device = torch.device(device)
        self.dtype = dtype
        self.calls = []

    def alphas_cumprod(self, device=None):
        # 與真實排程同型：由 1 單調遞減到接近 0
        t = torch.arange(self.num_train_timesteps, dtype=self.dtype)
        return (1.0 - t / self.num_train_timesteps).clamp(min=1e-6).to(
            device or self.device)

    def encode_text(self, prompt):
        return torch.zeros(1, 77, 8, dtype=self.dtype, device=self.device)

    def encode_image(self, x01):
        return torch.nn.functional.avg_pool2d(x01, 8)

    def unet_forward(self, z, t, emb):
        self.calls.append((tuple(z.shape), int(t)))
        return z * self.gain


def _img(h=32, w=32, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(1, 3, h, w, generator=g, dtype=torch.float64)


# ---- 起始時間步 ----

@pytest.mark.parametrize("strength,want", [(0.7, 700), (0.55, 550), (1.0, 999)])
def test_start_timestep_matches_sdedit(strength, want):
    """必須與 `sd.sdedit` 的 `t0 = min(int(1000·strength), 999)` 逐字相同——
    對不上就等於在攻擊一個攻擊方不會走的時間點。"""
    sd = _StubSD()
    assert make_early_step_loss(sd, "", strength).t0 == want


def test_loss_uses_that_timestep_in_the_unet_call():
    sd = _StubSD()
    make_early_step_loss(sd, "", 0.7)(_img())
    assert sd.calls[-1][1] == 700


def test_strength_out_of_range_is_rejected():
    for bad in (0.0, -0.1, 1.5):
        with pytest.raises(ValueError):
            make_early_step_loss(_StubSD(), "", bad)


# ---- 損失的方向與噪聲 ----

def test_loss_is_negative_so_minimising_maximises_the_norm():
    """本專案所有 `loss_fn` 都是要最小化的，故此處必須是負範數。"""
    assert float(make_early_step_loss(_StubSD(), "", 0.7, seed=0)(_img())) < 0


def test_noise_is_redrawn_on_every_call():
    """原始碼每個 iteration 重抽 `latents`；固定噪聲會讓攻擊過擬合單一實現。"""
    f = make_early_step_loss(_StubSD(), "", 0.7, seed=0)
    x = _img()
    assert float(f(x)) != float(f(x))


def test_seed_makes_the_draw_reproducible():
    x = _img()
    a = [float(make_early_step_loss(_StubSD(), "", 0.7, seed=7)(x))
         for _ in range(1)]
    b = [float(make_early_step_loss(_StubSD(), "", 0.7, seed=7)(x))
         for _ in range(1)]
    assert a == b


def test_batch_replicates_the_image_and_divides_by_batch():
    sd = _StubSD()
    make_early_step_loss(sd, "", 0.7, batch=4, seed=0)(_img())
    assert sd.calls[-1][0][0] == 4, "latent 應被複製成 batch 列"


def test_loss_is_differentiable_wrt_the_image():
    x = _img().requires_grad_(True)
    make_early_step_loss(_StubSD(), "", 0.7, seed=0)(x).backward()
    assert x.grad is not None and float(x.grad.abs().sum()) > 0


# ---- PGD ----

def test_pgd_stays_inside_the_epsilon_ball_and_the_value_range():
    x = _img()
    spec = DiffusionGuardSpec(iters=12)
    res = run_diffusionguard(_StubSD(), x, "", 0.7, spec, seed=0)
    d = (res.x_def - x).abs()
    assert float(d.max()) <= spec.eps_pixel01 + 1e-12
    assert 0.0 <= float(res.x_def.min()) and float(res.x_def.max()) <= 1.0


def test_pgd_actually_moves_the_image():
    x = _img()
    res = run_diffusionguard(_StubSD(), x, "", 0.7,
                             DiffusionGuardSpec(iters=8), seed=0)
    assert float((res.x_def - x).abs().max()) > 1e-6


def test_result_records_the_timestep_used():
    res = run_diffusionguard(_StubSD(), _img(), "", 0.55,
                             DiffusionGuardSpec(iters=2), seed=0)
    assert res.t0 == 550


# ---- spec ----

def test_spec_cannot_claim_to_be_the_paper():
    """本檔是 img2img 的移植，不是原文的重現。"""
    with pytest.raises(ValueError):
        DiffusionGuardSpec(modified_from_paper=False)


def test_spec_rejects_a_step_larger_than_the_ball():
    with pytest.raises(ValueError):
        DiffusionGuardSpec(eps_pixel01=0.01, step_size=0.05)


def test_paper_hyperparameters_match_the_repo_config():
    assert PAPER_ITERS == 800
    assert abs(PAPER_EPS - 0.06274509803921569) < 1e-15
    assert abs(PAPER_STEP_SIZE - 0.00392156862745098) < 1e-15
    assert abs(PAPER_EPS - 16 / 255) < 1e-15
    assert abs(PAPER_STEP_SIZE - 1 / 255) < 1e-15


def test_modification_note_is_present_and_mentions_the_two_changes():
    note = DiffusionGuardSpec().modification_note
    assert "img2img" in note and "mask" in note.lower()

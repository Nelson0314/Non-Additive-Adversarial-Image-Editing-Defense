"""極小規模全流程煙霧測試 — STRUCTURE.md §1。

以 hf-internal-testing/tiny-stable-diffusion-pipe（CPU 可跑）驗證：
資料載入 → 保護（PhotoGuard encoder）→ 編輯（sdedit / inpaint，同 seed）
→ 指標（piq 五項），各介面可串接、張量形狀與值域正確、同 seed 可重現。
超參數為煙霧測試用縮減值，非 SPEC §2.2 正式設定。
"""

import pytest
import torch

from src.data.dataset import load_dataset
from src.edit import edit_image
from src.edit.seed_protocol import search_valid_seeds
from src.metrics.quality import METRIC_HIGHER_IS_BETTER, compute_all
from src.models.sd_wrapper import SDWrapper
from src.protect.photoguard import PhotoGuardEncoder

TINY_MODEL = "hf-internal-testing/tiny-stable-diffusion-pipe"


@pytest.fixture(scope="session")
def sd():
    return SDWrapper(TINY_MODEL)


@pytest.fixture(scope="module")
def config(sd):
    scale = 2 ** (len(sd.vae.config.block_out_channels) - 1)
    n = sd.unet.config.sample_size * scale
    return {
        "generation": {
            "height": n,
            "width": n,
            "guidance_scale": 7.5,
            "num_inference_steps": 4,   # 煙霧測試縮減值（正式為 100）
            "eta": 1,
        },
        "edit": {"sdedit_strength": 0.5, "n_seeds": 2, "seed_clip_threshold": None},
        "data": {"is_placeholder": True, "n_images": 3},
        "runtime": {"batch_size": 1, "seed": 42},
    }


@pytest.fixture(scope="module")
def protect_cfg():
    return {
        "norm": "linf",
        "epsilon": 0.06,
        "epsilon_scale": "pm1",
        "step_size": 0.006,
        "n_iter": 3,                    # 煙霧測試縮減值（正式為 100）
        "random_init": True,
        "step_decay": True,
        "target_latent": "zeros",
    }


def test_dataset_placeholder(config):
    data = load_dataset(config, max_images=3)
    assert len(data) == 3
    for s in data:
        img = s["image"]
        assert img.shape == (1, 3, config["generation"]["height"], config["generation"]["width"])
        assert 0.0 <= img.min() and img.max() <= 1.0
        assert len(s["edit_prompts"]) == 2 and s["concept"]


def test_smoke_full_pipeline(sd, config, protect_cfg):
    """3 張圖：保護 → 同 seed 編輯原圖與保護圖 → 指標。"""
    data = load_dataset(config, max_images=3)
    method = PhotoGuardEncoder(sd, protect_cfg)

    for sample in data:
        x = sample["image"]
        prompt = sample["edit_prompts"][0]
        protected = method.protect(x, sample["concept"])
        assert protected.shape == x.shape
        assert (protected - x).abs().max() <= protect_cfg["epsilon"] / 2 + 1e-6

        seeds = search_valid_seeds(x, prompt, n_seeds=2, config=config)
        assert len(seeds) == 2

        edited_orig = edit_image(x, prompt, seeds[0], "sdedit", config=config, sd=sd)
        edited_prot = edit_image(protected, prompt, seeds[0], "sdedit", config=config, sd=sd)
        assert edited_orig.shape == x.shape
        assert 0.0 <= edited_orig.min() and edited_orig.max() <= 1.0

        m = compute_all(edited_prot, edited_orig)
        assert set(m) == {"psnr", "ssim", "vifp", "fsim", "lpips"}
        assert all(k in METRIC_HIGHER_IS_BETTER for k in m)
        assert all(torch.isfinite(torch.tensor(v)) for v in m.values())


def test_edit_seed_reproducible(sd, config):
    """同 seed 之編輯結果須完全一致（seed 協定之前提）。"""
    data = load_dataset(config, max_images=1)
    x = data[0]["image"]
    prompt = data[0]["edit_prompts"][0]
    a = edit_image(x, prompt, 123, "sdedit", config=config, sd=sd)
    b = edit_image(x, prompt, 123, "sdedit", config=config, sd=sd)
    c = edit_image(x, prompt, 456, "sdedit", config=config, sd=sd)
    assert torch.equal(a, b)
    assert not torch.equal(a, c)


def test_inpaint_smoke(sd, config):
    """inpaint（4-channel UNet 之 latent 混合路徑）可跑通。"""
    data = load_dataset(config, max_images=1)
    x = data[0]["image"]
    h, w = x.shape[-2:]
    mask = torch.zeros(1, 1, h, w)
    mask[..., : h // 2, :] = 1.0  # 重繪上半部
    out = edit_image(x, "a cat", 7, "inpaint", mask=mask, config=config, sd=sd)
    assert out.shape == x.shape
    assert 0.0 <= out.min() and out.max() <= 1.0

"""批次化編輯等價性測試 — v5（STRUCTURE.md §3 runtime 註記）。

v5 啟用判準：4 張影像分別以 batch=1 與 batch=4 執行，輸出逐位元相同；
不通過則不可啟用批次化（edit_batch_size 維持 1）。

實測結論（本地 CPU、tiny 模型；記入 NOTES.md）：
1. 噪聲協定**位元級一致**——批次中每樣本各自 generator+seed，diffusers 對
   generator list 逐樣本取樣，與 batch=1 完全相同（test_noise_protocol）。
   seed 協定（SPEC §2.3）未被破壞。
2. 網路前傳輸出存在 ~5e-6 之 float32 差異——批次矩陣運算之歸約順序隨 batch
   尺寸而異，屬 kernel 數值性質，**位元級等價原理上不可達**；8-bit 量化
  （存檔格式）後仍有 ~0.02% 像素差 ±1 LSB。
3. 故位元級判準之測試設計為「啟用閘門」：edit_batch_size==1（未啟用）時跳過；
   一旦設 >1，此測試將執行且（依實測）失敗，阻止未經裁定的啟用。
   判準若經指導者放寬（噪聲協定位元級一致＋輸出容差 ≤1e-4），修改本檔閘門
   測試後方可啟用。
"""

import pytest
import torch
import yaml
from diffusers.utils.torch_utils import randn_tensor

from src.edit import edit_image, edit_image_batch
from src.models.sd_wrapper import SDWrapper

TINY_MODEL = "hf-internal-testing/tiny-stable-diffusion-pipe"
FLOAT_TOL = 1e-4   # 實測 kernel 數值差 ~5e-6，容差取 20 倍餘裕

PROMPTS = ["a photo of a dog", "a photo of a cat", "an oil painting of a tree",
           "a photo of a dog"]
SEEDS = [11, 22, 33, 44]


@pytest.fixture(scope="module")
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
            "num_inference_steps": 4,   # 縮減值（正式為 100）；等價性與步數無關
            "eta": 1,                   # 覆蓋 DDIM 變異噪聲之 generator 路徑
        },
        "edit": {"sdedit_strength": 0.5},
        "runtime": {"protect_batch_size": 1, "edit_batch_size": 4, "seed": 42},
    }


@pytest.fixture(scope="module")
def images(config):
    g = torch.Generator().manual_seed(0)
    n = config["generation"]["height"]
    return [torch.rand(1, 3, n, n, generator=g) for _ in range(4)]


@pytest.fixture(scope="module")
def mask(config):
    n = config["generation"]["height"]
    m = torch.zeros(1, 1, n, n)
    m[..., n // 4 : 3 * n // 4, n // 4 : 3 * n // 4] = 1.0
    return m


def test_noise_protocol_bitwise_equal():
    """seed 協定核心：generator list（batch=4）之噪聲須與各樣本單獨取樣
    （batch=1）位元級一致。此為批次化不破壞 SPEC §2.3 的必要條件。"""
    gens4 = [torch.Generator().manual_seed(s) for s in SEEDS]
    noise_batch = randn_tensor((4, 4, 8, 8), generator=gens4)
    gens1 = [torch.Generator().manual_seed(s) for s in SEEDS]
    noise_single = torch.cat([randn_tensor((1, 4, 8, 8), generator=[g]) for g in gens1])
    assert torch.equal(noise_batch, noise_single)


def test_batch_call_deterministic(sd, config, images):
    a = edit_image_batch(images, PROMPTS, SEEDS, "sdedit", config=config, sd=sd)
    b = edit_image_batch(images, PROMPTS, SEEDS, "sdedit", config=config, sd=sd)
    for x, y in zip(a, b):
        assert torch.equal(x, y)


def test_sdedit_batch4_matches_batch1_within_tolerance(sd, config, images):
    """批次 vs 單張之差異須僅為 kernel 數值層級（≤1e-4；實測 ~5e-6）。
    若此測試失敗（差異大），代表隨機性協定被破壞，屬嚴重錯誤。"""
    singles = [edit_image(img, p, s, "sdedit", config=config, sd=sd)
               for img, p, s in zip(images, PROMPTS, SEEDS)]
    batched = edit_image_batch(images, PROMPTS, SEEDS, "sdedit", config=config, sd=sd)
    assert len(batched) == 4
    for i, (a, b) in enumerate(zip(singles, batched)):
        d = (a - b).abs().max().item()
        assert d <= FLOAT_TOL, f"sdedit 樣本 {i}: max|Δ|={d:.2e} 超出 kernel 數值層級"


def test_sdedit_chunking_matches_batch1_within_tolerance(sd, config, images):
    """batch_size=3 時 4 樣本分為 3+1 兩塊，分塊不得引入額外差異。"""
    cfg = {**config, "runtime": {**config["runtime"], "edit_batch_size": 3}}
    singles = [edit_image(img, p, s, "sdedit", config=config, sd=sd)
               for img, p, s in zip(images, PROMPTS, SEEDS)]
    chunked = edit_image_batch(images, PROMPTS, SEEDS, "sdedit", config=cfg, sd=sd)
    for a, b in zip(singles, chunked):
        assert (a - b).abs().max().item() <= FLOAT_TOL


def test_inpaint_batch4_matches_batch1_within_tolerance(sd, config, images, mask):
    singles = [edit_image(img, p, s, "inpaint", mask=mask, config=config, sd=sd)
               for img, p, s in zip(images, PROMPTS, SEEDS)]
    batched = edit_image_batch(images, PROMPTS, SEEDS, "inpaint",
                               masks=[mask] * 4, config=config, sd=sd)
    for i, (a, b) in enumerate(zip(singles, batched)):
        d = (a - b).abs().max().item()
        assert d <= FLOAT_TOL, f"inpaint 樣本 {i}: max|Δ|={d:.2e} 超出 kernel 數值層級"


def test_bitwise_gate_blocks_enabling_batching(sd, config, images):
    """啟用閘門（v5 判準原文）：configs/base.yaml 之 edit_batch_size > 1 時，
    要求 batch=4 與 batch=1 逐位元相同；未啟用（=1）時跳過。
    依實測此判準無法通過，故 config 須維持 1，除非判準經指導者放寬。"""
    base = yaml.safe_load(
        (__import__("pathlib").Path(__file__).parents[1] / "configs" / "base.yaml")
        .read_text(encoding="utf-8"))
    if int(base["runtime"].get("edit_batch_size", 1)) <= 1:
        pytest.skip("批次化未啟用（edit_batch_size=1）；位元級判準實測未通過，"
                    "啟用須先經指導者放寬判準")
    singles = [edit_image(img, p, s, "sdedit", config=config, sd=sd)
               for img, p, s in zip(images, PROMPTS, SEEDS)]
    batched = edit_image_batch(images, PROMPTS, SEEDS, "sdedit", config=config, sd=sd)
    for i, (a, b) in enumerate(zip(singles, batched)):
        assert torch.equal(a, b), f"樣本 {i}: batch=4 與 batch=1 非位元級一致"

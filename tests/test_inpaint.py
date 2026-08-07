"""inpainting 威脅模型的路徑 —— `SDWrapper.inpaint` 與 `src/data/masks.py`。

用一個結構正確但極小的 9 通道 SD（隨機初始化）驗**契約**，不驗數值品質：
數值要真實權重與 GPU。這裡要釘住的是那些「錯了也不會報錯」的地方——

- 9 通道輸入的三段拼接順序與各段內容；
- 遮罩外的區域每一步都被貼回，而不是只在最後貼一次；
- `latent_shape` 的通道數取自 VAE 而非 UNet 的 `in_channels`（inpainting
  權重上後者是 9，取錯會讓噪聲與 latent 對不起來）；
- 梯度確實從防禦圖流到輸出（防禦訓練整個建立在這條路上）。
"""

import pytest
import torch
from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    StableDiffusionInpaintPipeline,
    UNet2DConditionModel,
)
from transformers import CLIPTextConfig, CLIPTextModel, CLIPTokenizer

from src.models.sd import SDInpaintWrapper, SDWrapper
from src.utils.device import get_device

IMG = 64
DIM = 32
DEV = get_device()


def _tiny_pipe(in_channels: int = 9):
    """結構正確的極小 SD。`in_channels=9` 即 inpainting 權重的形態。"""
    tok = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")
    cfg = CLIPTextConfig(
        vocab_size=len(tok), hidden_size=DIM, intermediate_size=DIM * 2,
        num_hidden_layers=2, num_attention_heads=2,
        max_position_embeddings=tok.model_max_length,
        bos_token_id=tok.bos_token_id, eos_token_id=tok.eos_token_id,
    )
    torch.manual_seed(0)
    unet = UNet2DConditionModel(
        sample_size=IMG // 8, in_channels=in_channels, out_channels=4,
        down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
        up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
        block_out_channels=(16, 32), layers_per_block=1,
        cross_attention_dim=DIM, attention_head_dim=4, norm_num_groups=8,
    )
    vae = AutoencoderKL(
        in_channels=3, out_channels=3,
        down_block_types=("DownEncoderBlock2D",) * 4,
        up_block_types=("UpDecoderBlock2D",) * 4,
        block_out_channels=(8, 8, 8, 8), layers_per_block=1,
        latent_channels=4, norm_num_groups=8, sample_size=IMG,
        scaling_factor=0.18215,
    )
    sched = DDIMScheduler(num_train_timesteps=1000, beta_start=0.00085,
                          beta_end=0.012, beta_schedule="scaled_linear")
    return StableDiffusionInpaintPipeline(
        vae=vae, text_encoder=CLIPTextModel(cfg), tokenizer=tok, unet=unet,
        scheduler=sched, safety_checker=None, feature_extractor=None,
        requires_safety_checker=False,
    )


@pytest.fixture(scope="module")
def sd9():
    return SDInpaintWrapper("tiny-inpaint", pipe=_tiny_pipe(9))


@pytest.fixture(scope="module")
def sd4():
    return SDWrapper("tiny-plain", pipe=_tiny_pipe(4))


@pytest.fixture(scope="module")
def x01():
    g = torch.Generator().manual_seed(20260807)
    return torch.rand(1, 3, IMG, IMG, generator=g).to(DEV)


@pytest.fixture(scope="module")
def mask():
    m = torch.zeros(1, 1, IMG, IMG, device=DEV)
    m[..., IMG // 4:IMG * 3 // 4, IMG // 4:IMG * 3 // 4] = 1.0
    return m


# ---------------------------------------------------------------------------
# latent 的通道數
# ---------------------------------------------------------------------------

def test_latent通道數取自vae而非unet(sd9, sd4):
    """inpainting 權重的 `unet.in_channels` 是 9，而 latent 仍是 4。

    2026-08-07 修正。before：`latent_shape` 直接取 `unet.config.in_channels`。
    一般權重上兩者都是 4 故無症狀；inpainting 權重上會回傳 9 通道的形狀，
    `sample_edit_noise` 產生的噪聲於是與 latent 對不起來。
    """
    assert sd9.unet.config.in_channels == 9
    assert sd9.latent_channels == 4
    assert sd9.latent_shape(IMG, IMG) == (1, 4, IMG // 8, IMG // 8)
    # 一般權重不受影響
    assert sd4.latent_shape(IMG, IMG) == (1, 4, IMG // 8, IMG // 8)


def test_是否為inpainting由模型自己回答(sd9, sd4):
    """宣告錯的症狀是形狀不符中止；恰好對上時會安靜地算錯。故由 config 讀。"""
    assert sd9.is_inpainting is True
    assert sd4.is_inpainting is False


def test_一般權重呼叫inpaint立刻拋出(sd4, x01, mask):
    emb = sd4.encode_text("a cat")
    noise = sd4.sample_edit_noise(
        torch.empty(sd4.latent_shape(IMG, IMG), device=DEV), seed=0)
    with pytest.raises(RuntimeError, match="不是 inpainting 權重"):
        sd4.inpaint(x01, mask, emb, noise, num_steps=2)


# ---------------------------------------------------------------------------
# 9 通道輸入的三段
# ---------------------------------------------------------------------------

def test_遮罩後影像先遮罩再編碼(sd9, x01, mask):
    """VAE 是非線性的，`E(x·(1−m))` 與 `E(x)·(1−m)` 不等價。

    diffusers 的 pipeline 做的是前者；做成後者不會報錯，只會讓後 4 個通道
    的內容與攻擊方實際餵入的不同。
    """
    m, z_masked = sd9.mask_latents(x01, mask)
    expect = sd9.encode_image(x01 * (1.0 - mask))
    assert torch.equal(z_masked, expect)
    # 先編碼再遮罩會得到不同的東西——這一條是為了讓上面那個等式有對照
    wrong = sd9.encode_image(x01) * (1.0 - m)
    assert not torch.allclose(z_masked, wrong, atol=1e-3)


def test_遮罩以最近鄰下採樣且維持二值(sd9, x01, mask):
    """插值會在邊界產生 0 與 1 之間的值，那些格子既不算保留也不算重畫。"""
    m, _ = sd9.mask_latents(x01, mask)
    assert m.shape == (1, 1, IMG // 8, IMG // 8)
    assert torch.isin(m, torch.tensor([0.0, 1.0], device=m.device)).all()


def test_九通道的拼接順序(sd9, x01, mask, monkeypatch):
    """順序錯了 UNet 照樣跑得動（通道數對得上），只是算的是別的東西。"""
    seen = {}

    orig = sd9._eps_cfg

    def spy(z, t, *a, **kw):
        seen.setdefault("z", z)
        return orig(z, t, *a, **kw)

    monkeypatch.setattr(sd9, "_eps_cfg", spy)
    emb = sd9.encode_text("a cat")
    noise = sd9.sample_edit_noise(
        torch.empty(sd9.latent_shape(IMG, IMG), device=DEV), seed=0)
    sd9.inpaint(x01, mask, emb, noise, num_steps=1)

    z = seen["z"]
    assert z.shape[1] == 9, "UNet 輸入不是 9 通道"
    m, z_masked = sd9.mask_latents(x01, mask)
    # 前 4：帶噪 latent（第 0 步即 noise 本身）；中 1：遮罩；後 4：遮罩後 latent
    assert torch.allclose(z[:, :4], noise.to(z.dtype), atol=1e-5)
    assert torch.allclose(z[:, 4:5], m.to(z.dtype), atol=1e-5)
    assert torch.allclose(z[:, 5:], z_masked.to(z.dtype), atol=1e-4)


# ---------------------------------------------------------------------------
# 遮罩外的處置
# ---------------------------------------------------------------------------

def test_遮罩外每一步都貼回原圖(sd9, x01, mask):
    """只在最後貼一次，模型在生成過程中會看見自己改動過的脈絡。

    那與攻擊方實際跑的不是同一條鏈，而輸出看起來仍然合理，故此處以
    「遮罩外的輸出必須非常接近原圖的 VAE 重建」來釘住。
    """
    emb = sd9.encode_text("a cat")
    noise = sd9.sample_edit_noise(
        torch.empty(sd9.latent_shape(IMG, IMG), device=DEV), seed=0)
    y = sd9.inpaint(x01, mask, emb, noise, num_steps=3)
    recon = sd9.decode_latent(sd9.encode_image(x01))

    outside = (1.0 - mask)
    d_out = ((y - recon) * outside).abs().sum() / outside.sum()
    d_in = ((y - recon) * mask).abs().sum() / mask.sum()
    assert d_out < d_in, (
        f"遮罩外的偏差 {float(d_out):.4f} 不小於遮罩內的 {float(d_in):.4f}；"
        "遮罩外沒有被貼回")


def test_遮罩尺寸不符立刻拋出(sd9, x01):
    """呼叫端自行縮放會出現兩種插值方式並存而無從得知用了哪一種。"""
    emb = sd9.encode_text("a cat")
    noise = sd9.sample_edit_noise(
        torch.empty(sd9.latent_shape(IMG, IMG), device=DEV), seed=0)
    small = torch.ones(1, 1, IMG // 8, IMG // 8, device=DEV)
    with pytest.raises(ValueError, match="空間"):
        sd9.inpaint(x01, small, emb, noise, num_steps=1)


# ---------------------------------------------------------------------------
# 梯度
# ---------------------------------------------------------------------------

def test_梯度由防禦圖流到輸出(sd9, x01, mask):
    """防禦訓練整條路都建立在這上面。斷了的話損失照樣算得出來、梯度為零。"""
    xd = x01.clone().requires_grad_(True)
    emb = sd9.encode_text("a cat")
    noise = sd9.sample_edit_noise(
        torch.empty(sd9.latent_shape(IMG, IMG), device=DEV), seed=0)
    y = sd9.inpaint(xd, mask, emb, noise, num_steps=2)
    y.sum().backward()
    assert xd.grad is not None and float(xd.grad.abs().sum()) > 0


def test_遮罩外的像素也收得到梯度(sd9, x01, mask):
    """防禦方的著力點正是**未遮罩的脈絡**——inpainting 的生成內容由它條件化。

    這是本威脅模型與全圖 img2img 最大的差別；若遮罩外收不到梯度，防禦就只能
    改攻擊方無論如何都會覆寫掉的那一塊，等於沒有著力點。
    """
    xd = x01.clone().requires_grad_(True)
    emb = sd9.encode_text("a cat")
    noise = sd9.sample_edit_noise(
        torch.empty(sd9.latent_shape(IMG, IMG), device=DEV), seed=0)
    y = sd9.inpaint(xd, mask, emb, noise, num_steps=2)
    # 只對遮罩**內**的輸出取損失，看梯度會不會傳到遮罩外的輸入
    (y * mask).sum().backward()
    g_out = (xd.grad * (1.0 - mask)).abs().sum()
    assert float(g_out) > 0, "遮罩外的像素收不到梯度，防禦沒有著力點"


# ---------------------------------------------------------------------------
# 遮罩產生
# ---------------------------------------------------------------------------

def test_置中方框只是對照且與內容無關(x01):
    from src.data.masks import center_box

    m = center_box(x01, frac=0.5)
    assert m.shape == (1, 1, IMG, IMG)
    assert 0.2 < float(m.mean()) < 0.3          # 0.5² = 0.25
    assert torch.isin(m, torch.tensor([0.0, 1.0], device=m.device)).all()


def test_未知的遮罩模式立刻拋出(x01):
    from src.data.masks import content_mask

    with pytest.raises(ValueError, match="mask_mode"):
        content_mask(None, x01, "cat", mode="nope")


def test_content為空時不落回置中方框(sd9, x01):
    """落回去會讓兩種來源不同的遮罩混在同一張表上而看不出差別。"""
    from src.data.masks import content_mask

    with pytest.raises(ValueError, match="content"):
        content_mask(sd9, x01, "", mode="attention")


def test_注意力遮罩涵蓋影像且為二值(sd9, x01):
    from src.data.masks import content_mask

    out = content_mask(sd9, x01, "cat", mode="attention_box", tau=0.5)
    m = out["mask"]
    assert m.shape == (1, 1, IMG, IMG)
    assert torch.isin(m, torch.tensor([0.0, 1.0], device=m.device)).all()
    # `attention_region_mask` 保證遮罩至少含峰值，故不可能為空
    assert float(m.sum()) > 0
    assert out["coverage"] == pytest.approx(float(m.mean()))
    assert out["mode"] == "attention_box" and out["content"] == "cat"


def test_外接矩形是實心的(sd9, x01):
    """人工畫的遮罩不會有孔洞；孔洞造成的斑點是假陽性的來源。"""
    from src.data.masks import content_mask

    m = content_mask(sd9, x01, "cat", mode="attention_box")["mask"]
    idx = (m[0, 0] > 0.5).nonzero()
    y0, x0 = idx.min(dim=0).values.tolist()
    y1, x1 = idx.max(dim=0).values.tolist()
    assert float(m[0, 0, y0:y1 + 1, x0:x1 + 1].min()) == 1.0


# ---------------------------------------------------------------------------
# 唯一的分派點
# ---------------------------------------------------------------------------

def test_edit依權重分派(sd9, sd4, x01, mask):
    """呼叫端有九處。在每一處各加一個 if，漏掉任何一處都不會報錯。"""
    emb9 = sd9.encode_text("a cat")
    n9 = sd9.sample_edit_noise(
        torch.empty(sd9.latent_shape(IMG, IMG), device=DEV), seed=0)
    y9 = sd9.edit(x01, emb9, n9, num_steps=2, mask=mask)
    assert y9.shape == x01.shape

    emb4 = sd4.encode_text("a cat")
    n4 = sd4.sample_edit_noise(
        torch.empty(sd4.latent_shape(IMG, IMG), device=DEV), seed=0)
    y4 = sd4.edit(x01, emb4, n4, num_steps=2, strength=0.6)
    assert y4.shape == x01.shape


def test_inpainting缺遮罩立刻拋出(sd9, x01):
    emb = sd9.encode_text("a cat")
    n = sd9.sample_edit_noise(
        torch.empty(sd9.latent_shape(IMG, IMG), device=DEV), seed=0)
    with pytest.raises(ValueError, match="需要遮罩"):
        sd9.edit(x01, emb, n, num_steps=1)


def test_inpainting不接受strength(sd9, x01, mask):
    """沿用一個不起作用的值會讓紀錄看起來像是設定過。"""
    emb = sd9.encode_text("a cat")
    n = sd9.sample_edit_noise(
        torch.empty(sd9.latent_shape(IMG, IMG), device=DEV), seed=0)
    with pytest.raises(ValueError, match="沒有 strength"):
        sd9.edit(x01, emb, n, num_steps=1, mask=mask, strength=0.6)


def test_img2img不接受遮罩(sd4, x01, mask):
    """傳了遮罩表示呼叫端以為在跑 inpainting，而載入的是一般權重。"""
    emb = sd4.encode_text("a cat")
    n = sd4.sample_edit_noise(
        torch.empty(sd4.latent_shape(IMG, IMG), device=DEV), seed=0)
    with pytest.raises(ValueError, match="不吃遮罩"):
        sd4.edit(x01, emb, n, num_steps=1, mask=mask, strength=0.6)


def test_img2img缺strength立刻拋出(sd4, x01):
    """五篇 baseline 的原始碼都沒有這個數，故無預設值。"""
    emb = sd4.encode_text("a cat")
    n = sd4.sample_edit_noise(
        torch.empty(sd4.latent_shape(IMG, IMG), device=DEV), seed=0)
    with pytest.raises(ValueError, match="需要 strength"):
        sd4.edit(x01, emb, n, num_steps=1)

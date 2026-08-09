"""inpainting 威脅模型的路徑 —— `SDWrapper.inpaint` 與 `src/data/masks.py`。

用一個結構正確但極小的 9 通道 SD（隨機初始化）驗**契約**，不驗數值品質：
數值要真實權重與 GPU。這裡要釘住的是那些「錯了也不會報錯」的地方——

- 9 通道輸入的三段拼接順序與各段內容；
- 遮罩外的區域每一步都被貼回，而不是只在最後貼一次；
- `latent_shape` 的通道數取自 VAE 而非 UNet 的 `in_channels`（inpainting
  權重上後者是 9，取錯會讓噪聲與 latent 對不起來）；
- 梯度確實從防禦圖流到輸出（防禦訓練整個建立在這條路上）。
"""

from unittest.mock import MagicMock, patch

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
# Lo 配置的遮罩：c_a 必須在遮罩外（DEF-011）
# ---------------------------------------------------------------------------
#
# 這一組釘住的是「錯了也不會報錯」的那個型態。ip1／ip2／ip3 三批的遮罩與
# 式 (4) 的 M 由同一個詞產生而完全重疊，遮罩產得出來、涵蓋率印得出數字、
# 格點全部 done——缺的就是這道檢查。
#
# 極小模型的注意力圖只有 4×4，預設 guard_k=5 會把整張圖吃掉，故此處明給
# guard_k=1；保護帶本身以合成張量單獨驗（見下方兩條）。


def test_遮罩內的格點不計入式5():
    """核心不變量。用合成張量驗，不賭任何模型的注意力落在哪裡。

    M 是左上 4×4，攻擊方的遮罩蓋住其中右半。式 (5) 只該算剩下的左半。
    """
    from src.models.attention import restrict_outside_mask

    M = torch.zeros(1, 1, 8, 8)
    M[..., 0:4, 0:4] = 1.0                   # 16 格
    edit = torch.zeros(1, 1, 64, 64)
    edit[..., 0:32, 16:32] = 1.0             # 對到 8×8 的 (0:4, 2:4)，8 格
    out, kept = restrict_outside_mask(M, edit, where="test")
    assert float(out.sum()) == 8.0
    assert kept == pytest.approx(0.5)
    # 被排除的那 8 格確實是遮罩內的那一半
    assert float(out[..., 0:4, 0:2].sum()) == 8.0
    assert float(out[..., 0:4, 2:4].sum()) == 0.0


def test_遮罩與M完全不相交時M原樣保留():
    from src.models.attention import restrict_outside_mask

    M = torch.zeros(1, 1, 8, 8)
    M[..., 0:3, 0:3] = 1.0
    edit = torch.zeros(1, 1, 64, 64)
    edit[..., 48:64, 48:64] = 1.0            # 對到 (6:8, 6:8)
    out, kept = restrict_outside_mask(M, edit, where="test")
    assert kept == 1.0
    assert torch.equal(out, M)


def test_限制以max_pool保守排除():
    """影像解析度上只要有**一個像素**落進某格，整格就要排除。

    取 mean 或 nearest 會讓那一個像素被平均掉／取樣掉，於是一個實際上會被
    攻擊方覆寫的格點仍留在損失裡——那不會有症狀，只是白花失真預算。
    """
    from src.models.attention import restrict_outside_mask

    M = torch.zeros(1, 1, 8, 8)
    M[..., 4, 4] = 1.0
    M[..., 0, 0] = 1.0
    edit = torch.zeros(1, 1, 64, 64)
    edit[..., 32, 32] = 1.0                  # 8×8 上恰為 (4,4) 這一格
    out, kept = restrict_outside_mask(M, edit, where="test")
    assert float(out[..., 4, 4]) == 0.0
    assert float(out[..., 0, 0]) == 1.0
    assert kept == pytest.approx(0.5)


def test_遮罩外幾乎沒有M時拋出():
    """c_a 的注意力幾乎整片落在會被重畫的區域，該影像沒有可施力的地方。

    此時要拋出而不是讓最佳化在幾個格點上空轉——後者跑得完、也寫得出
    metrics，只是量到的不是防禦。
    """
    from src.models.attention import restrict_outside_mask

    M = torch.zeros(1, 1, 8, 8)
    M[..., 0:4, 0:4] = 1.0                   # 16 格
    edit = torch.zeros(1, 1, 64, 64)
    edit[..., 0:32, 0:24] = 1.0              # 對到 (0:4, 0:3)，只剩 4 格 = 0.25
    restrict_outside_mask(M, edit, where="t", min_kept=0.2)      # 0.25 ≥ 0.2
    with pytest.raises(ValueError, match="低於下限"):
        restrict_outside_mask(M, edit, where="t", min_kept=0.3)


def test_人工遮罩載入為二值且尺寸對齊(tmp_path):
    from PIL import Image

    from src.data.masks import load_drawn_mask

    m = Image.new("L", (64, 64), 0)
    m.paste(255, (0, 0, 32, 64))                 # 左半邊要重畫
    m.save(tmp_path / "a.png")

    t = load_drawn_mask(tmp_path / "a.png", 64, torch.device("cpu"))
    assert t.shape == (1, 1, 64, 64)
    assert torch.isin(t, torch.tensor([0.0, 1.0])).all()
    assert float(t.mean()) == pytest.approx(0.5)


def test_人工遮罩尺寸不符時以nearest縮放而不插值(tmp_path):
    """任何插值都會在邊界造出中間值，而遮罩是二值的。"""
    from PIL import Image

    from src.data.masks import load_drawn_mask

    m = Image.new("L", (16, 16), 0)
    m.paste(255, (0, 0, 8, 16))
    m.save(tmp_path / "a.png")

    t = load_drawn_mask(tmp_path / "a.png", 64, torch.device("cpu"))
    assert t.shape == (1, 1, 64, 64)
    assert torch.isin(t, torch.tensor([0.0, 1.0])).all()
    assert float(t.mean()) == pytest.approx(0.5)


def test_缺遮罩檔時拋出且不落回自動產生(tmp_path):
    """混用人工與自動產生的遮罩會讓同一張表上的各列不可比，且看不出症狀。"""
    from src.data.masks import load_drawn_mask

    with pytest.raises(FileNotFoundError, match="draw_masks"):
        load_drawn_mask(tmp_path / "nope.png", 64, torch.device("cpu"))


def test_空遮罩與填滿的遮罩都拒收(tmp_path):
    from PIL import Image

    from src.data.masks import load_drawn_mask

    Image.new("L", (64, 64), 0).save(tmp_path / "empty.png")
    Image.new("L", (64, 64), 255).save(tmp_path / "full.png")

    with pytest.raises(ValueError, match="為空"):
        load_drawn_mask(tmp_path / "empty.png", 64, torch.device("cpu"))
    # 填滿整張時 c_a 不可能在遮罩外，即 DEF-011 的配置
    with pytest.raises(ValueError, match="DEF-011"):
        load_drawn_mask(tmp_path / "full.png", 64, torch.device("cpu"))


def test_總覽圖不得進遮罩雜湊(tmp_path):
    """`overview.png` 是給人看的衍生物，重產一次就變。

    放進 digest 的話，「重新產一張總覽圖」會靜默改掉每一格的 config_hash，
    續跑時把已完成的格全部判為未完成——而那看不出症狀。
    """
    from PIL import Image

    from src.data.masks import mask_files, masks_digest

    m = Image.new("L", (8, 8), 0)
    m.paste(255, (0, 0, 4, 8))
    m.save(tmp_path / "horse_00.png")
    before = masks_digest(mask_files(tmp_path))

    Image.new("RGB", (64, 64), (9, 9, 9)).save(tmp_path / "overview.png")
    assert mask_files(tmp_path) == [tmp_path / "horse_00.png"]
    assert masks_digest(mask_files(tmp_path)) == before


def test_遮罩內容進雜湊而不只是目錄名(tmp_path):
    """換一張遮罩就是換一個攻擊，舊結果不可沿用。"""
    from PIL import Image

    from src.data.masks import masks_digest

    p = tmp_path / "a.png"
    Image.new("L", (8, 8), 0).save(p)
    before = masks_digest([p])
    m = Image.new("L", (8, 8), 0)
    m.paste(255, (0, 0, 4, 8))
    m.save(p)
    assert masks_digest([p]) != before


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


# ---------------------------------------------------------------------------
# 遮罩設定不得污染 img2img 的雜湊
# ---------------------------------------------------------------------------

def test_遮罩鍵只在inpainting批次出現(tmp_path):
    """`config_hash` 吃整個 dict，多一個鍵就改變**每一格**的雜湊。

    無條件加入的話，img2img 的既有批次一旦續跑就會把已完成的格全部判為
    未完成。2026-08-07 當下正有三個分片在跑，其中一個還剩約五小時，故此處
    以「img2img 的雜湊逐位不變」釘住。
    """
    import sys
    import types
    from pathlib import Path as _P

    sys.path.insert(0, str(_P(__file__).resolve().parent.parent / "scripts"))
    import run_stage as rs
    from src.utils.cellid import config_hash

    def mk(**over):
        a = types.SimpleNamespace(
            spec_version=1, model="m", resolution=512, guidance=7.5, steps=50,
            strength=0.6, gpu_tag="g", precision="fp32",
            masks=None, prompt_index=0,
        )
        for k, v in over.items():
            setattr(a, k, v)
        return a

    real = rs.run_config

    class _Cfg:
        def loss_params(self):
            return {}

        def module_params(self):
            return {}

        def optim_params(self):
            return {}

    from PIL import Image

    mdir = tmp_path / "masks"
    mdir.mkdir()
    m = Image.new("L", (8, 8), 0)
    m.paste(255, (0, 0, 4, 8))
    m.save(mdir / "horse_00.png")

    rs.run_config = lambda a: _Cfg()
    try:
        plain = rs.base_config(mk())
        inpaint = rs.base_config(mk(masks=mdir))
        second = rs.base_config(mk(prompt_index=1))
    finally:
        rs.run_config = real

    assert "masks" not in plain, "img2img 的 base_config 不該有 masks 鍵"
    assert inpaint["masks"]["dir"] == "masks"
    assert len(inpaint["masks"]["digest"]) == 16

    def h(cfg):
        return config_hash(dict(cfg, condition="N2", image_id="a", tau=0.2,
                                purify=None, seed=0, lr=None))

    assert h(plain) != h(inpaint), "換威脅模型必須換雜湊"
    # 攻擊 prompt 換一個就是換一個攻擊，續跑不可把舊格判為完成。
    assert "prompt_index" not in plain
    assert second["prompt_index"] == 1
    assert h(plain) != h(second), "換攻擊 prompt 必須換雜湊"


# ---------------------------------------------------------------------------
# 訓練期與評測期必須是同一種攻擊
# ---------------------------------------------------------------------------

def test_訓練期的代理編輯鏈也走分派點():
    """訓練與評測用不同的攻擊，症狀是防禦對著另一個威脅模型最佳化。

    報表上兩者都叫「編輯」，數字看起來也合理，故此處以原始碼檢查釘住
    `optimize` 不得再直接呼叫 `sd.sdedit`。
    """
    import inspect

    from src.defense import optimize as op

    src = inspect.getsource(op._build_output_step)
    body = src.replace(op._build_output_step.__doc__ or "\0", "")
    assert "sd.sdedit(" not in body, "代理編輯鏈仍直接呼叫 sdedit"
    assert body.count("sd.edit(") == 2, (
        "y_orig 與 y_def 兩條鏈都必須走 edit")
    assert "mask=cfg.edit_mask" in body


def test_optim_config把遮罩與strength一起切換():
    """inpainting 下 strength 必須是 None，否則 `edit` 會拒絕。

    兩者是同一個開關的兩面：留一個沿用下來的 strength 會讓紀錄看起來像是
    設定過，而 `edit` 會在第一格就拋出——那時段 0 已經跑掉一部分。
    """
    import inspect

    from src.experiment import executors as ex

    src = inspect.getsource(ex.optim_config)
    assert "edit_mask=(entry.mask if entry is not None else None)" in src
    assert "None if (entry is not None and entry.mask is not None)" in src


# ---------------------------------------------------------------------------
# 三篇原生 inpainting 的 baseline
# ---------------------------------------------------------------------------

def test_photoguard的梯度只落在遮罩外():
    """`attack_forward` 逐字為 `grad = grad * (1 - cur_mask)`。

    遮罩內的像素攻擊方會整片覆寫掉，往那裡放擾動等於把預算丟進去換不到
    東西。這是該篇在 inpainting 下唯一有意義的放置處。
    """
    import inspect

    from src.baselines import pgd, photoguard

    assert photoguard.SPEC.grad_outside_mask is True
    # mist／dia／advpaint 的原始碼沒有這一步（promptflare 有，見下一條）
    from src.baselines import advpaint, dia, mist
    for m in (mist, dia, advpaint):
        specs = [v for v in vars(m).values()
                 if isinstance(v, pgd.BaselineSpec)]
        for sp in specs:
            assert sp.grad_outside_mask is False, f"{sp.name} 不該遮罩梯度"

    body = inspect.getsource(pgd.run_pgd)
    assert "grad = grad * (1.0 - grad_mask.to(grad))" in body


def test_needs_mask與grad_outside_mask是兩件事():
    """兩者問的不是同一件事，用錯一個不會報錯只會算錯。

    `needs_mask` 問「原作的攻擊需不需要遮罩」，`grad_outside_mask` 問
    「原始碼有沒有把梯度乘 (1 − mask)」。PhotoGuard-c 在前者為 False、
    後者為 True；AdvPaint 恰好相反。若兩者可互相代替，這兩篇就會各自
    被套上對方的處置。

    2026-08-07 修正：先前此處斷言 PromptFlare 的 `grad_outside_mask` 為
    False。模組 docstring 記載 `promptflare.py:82-94` 同樣把梯度乘
    `(1 - cur_mask)`，故應為 True。
    """
    from src.baselines import advpaint, photoguard, promptflare

    assert photoguard.SPEC.needs_mask is False
    assert photoguard.SPEC.grad_outside_mask is True
    assert advpaint.SPEC.needs_mask is True
    assert advpaint.SPEC.grad_outside_mask is False
    assert promptflare.SPEC.needs_mask is True
    assert promptflare.SPEC.grad_outside_mask is True


def test_三篇的strength與遮罩一起切換(tmp_path):
    """遮罩給定時它們回到原生形態，strength 隨之消失。

    留一個不起作用的 strength 會讓紀錄看起來像是設定過，而 `SDWrapper.edit`
    會在第一格就拋出——那時段 0 已經跑掉一部分。
    """
    import inspect

    from src.experiment import executors as ex

    src = inspect.getsource(ex.baseline_kwargs)
    assert 'kw["mask"] = entry.mask' in src
    assert 'None if entry.mask is not None else res.cfg.strength' in src


def test_photoguard在有遮罩時不再要求strength():
    """img2img 版缺 strength 要拋出，inpainting 版**本來就沒有**這個數。"""
    import inspect

    from src.baselines import photoguard

    src = inspect.getsource(photoguard.prepare)
    assert "if strength is None and mask is None:" in src
    assert "mask" in inspect.signature(photoguard.prepare).parameters


def test_advpaint的GT與迭代走同一條前向():
    """GT 與迭代若走不同路徑，相減得到的距離量的是路徑差異而不是擾動。

    `_forward_and_record` 原本由 `ctx_like` 讀遮罩，而 GT 那次呼叫傳的是
    `None`——遮罩讀不到，GT 會走 img2img 而迭代走 inpainting。改成顯式
    參數之後兩處都必須明寫。
    """
    import inspect

    from src.baselines import advpaint

    sig = inspect.signature(advpaint._forward_and_record).parameters
    assert "mask" in sig and "ctx_like" not in sig

    prep = inspect.getsource(advpaint.prepare)
    assert "sd, mask, x_paper, emb2, t0, generator, recorder" in prep
    loss = inspect.getsource(advpaint.loss_fn)
    assert "sd, ctx.mask, x_adv" in loss


def test_advpaint在遮罩下讓梯度只走masked_image():
    """`AdvPaint.py:206-214` 把 encode→add_noise 包在 no_grad 內。

    img2img 沒有 masked-image 那一路，故移植版讓梯度經加噪後的 latent；
    遮罩給定時必須切回原作，否則 9 通道跑起來了但梯度仍走錯路。
    """
    import inspect

    from src.baselines import advpaint

    body = inspect.getsource(advpaint._forward_and_record)
    assert "z_ctx = torch.no_grad() if mask is not None else ctx" in body
    assert "x_paper * (1.0 - mask.to(x_paper))" in body
    assert "torch.cat([z, m, z_masked.to(z.dtype)], dim=1)" in body


def test_advpaint在遮罩下取整條排程的第一個timestep():
    """原作取 inpainting pipeline 預設排程的 timesteps[0]（50 步、strength=1）。"""
    import inspect

    from src.baselines import advpaint

    src = inspect.getsource(advpaint.prepare)
    assert "sd.num_train_timesteps - 1 if mask is not None else" in src


def test_promptflare在遮罩下latent與x_adv無關():
    """原作的 `latents` 是隨機噪聲，`x_adv` 只經後 4 個通道進入。

    照搬到 img2img 會讓損失對 `x_adv` 的梯度**恆為零**——那正是本專案改由
    加噪 latent 進入的理由。遮罩給定時必須切回原作，否則 9 通道跑起來了但
    梯度仍走移植版那一路。
    """
    import inspect

    from src.baselines import promptflare

    body = inspect.getsource(promptflare.loss_fn)
    assert "if ctx.mask is not None:" in body
    assert "z = torch.randn(lat, generator=ctx.generator" in body
    assert "x_adv * (1.0 - ctx.mask.to(x_adv))" in body
    assert "torch.cat([z, m, z_masked.to(z.dtype)], dim=1)" in body
    # 移植版那一路必須留著，img2img 批次仍要走它
    assert "abar.sqrt() * z0 + (1 - abar).sqrt() * noise" in body


def test_三篇的prepare都接遮罩且遮罩下不要求strength():
    import inspect

    from src.baselines import advpaint, photoguard, promptflare

    for m in (photoguard, advpaint, promptflare):
        sig = inspect.signature(m.prepare).parameters
        assert "mask" in sig, f"{m.__name__}.prepare 不接遮罩"
        src = inspect.getsource(m.prepare)
        assert "if strength is None and mask is None:" in src, m.__name__


def test_遮罩落盤存證():
    """遮罩決定攻擊方能改哪一塊，是**結果的一部分**而不是中間狀態。

    涵蓋率不同，同一個防禦的效果就不同。只印涵蓋率而不存圖，事後無從判斷
    某一格的遮罩是不是落在對的物件上——而 `runs/` 是唯一的證據來源
    （`CLAUDE.md`）。疊圖也必須存：純遮罩看不出對位。
    """
    import inspect
    import sys
    from pathlib import Path as _P

    sys.path.insert(0, str(_P(__file__).resolve().parent.parent / "scripts"))
    import run_stage as rs

    src = inspect.getsource(rs.build_resources)
    assert 'mask_dir = batch_dir / "masks"' in src
    assert '_mask.png' in src and '_overlay.png' in src
    assert 'write_csv(mask_dir / "masks.csv", rows)' in src
    # CSV 不得夾帶張量：`write_csv` 會把它轉成字串塞進一欄。逐鍵明寫，
    # 且每個鍵都是純量或路徑。
    assert '"coverage": cov' in src and '"source": str(src)' in src
    # 遮罩只能載入，不得在此產生——缺檔要拋出而不是落回自動版本
    assert "load_drawn_mask" in src


def test_inpainting批次的strength被清成None():
    """`--strength` 的預設是 0.6，不清掉會進 config_hash 再被 `edit` 拋出。

    那時模型已經載完、段 0 也跑掉一部分。清成 None 之後「這個威脅模型沒有
    strength」出現在雜湊與每一格的紀錄裡，而不是一個沿用下來卻不起作用的數。
    """
    import inspect
    import sys
    from pathlib import Path as _P

    sys.path.insert(0, str(_P(__file__).resolve().parent.parent / "scripts"))
    import run_stage as rs

    src = inspect.getsource(rs.main)
    assert "args.strength = None" in src
    # 兩者並用是矛盾的指令，要擋掉而不是靜默忽略其中一個
    assert "--masks 與 --strength 不可並用" in src


def test_shard的三個profile():
    """b3 的命令列必須逐字不變——段 0 要兩小時，差一項就是 CalibrationMismatch。"""
    from pathlib import Path as _P

    s = (_P(__file__).resolve().parent.parent
         / "scripts" / "shard.sh").read_text(encoding="utf-8")
    assert "ip*)" in s and "runwayml/stable-diffusion-inpainting" in s
    assert "--wrapper sd_inpaint" in s
    assert "--masks data/lo_masks" in s
    # inpainting 用裁切後的資料集，不可與 img2img 共用目錄
    assert "--data data/lo_inpaint" in s
    # 遮罩目錄不可在資料集裡：`load_lo_aligned` 拒絕未宣告卻含 PNG 的子目錄
    assert "--masks data/lo_aligned" not in s
    # inpainting profile 不得帶 strength
    ip = s.split("ip*)")[1].split(";;")[0]
    assert "--strength" not in ip


# ---------------------------------------------------------------------------
# N1（targeted_attn）的注意力前向在 inpainting 下
# ---------------------------------------------------------------------------

def _attn_cfg(mask, strength):
    """`_build_attn_step` 實際會讀到的欄位，其餘留預設。"""
    from src.defense.optimize import OptimConfig, StageSpec

    return OptimConfig(
        stages=(StageSpec(group="default", lr_key="lr.N1", max_steps=2),),
        attn_timesteps=2, seed=0, strength=strength, edit_mask=mask,
        unet_ckpt=False, vae_ckpt=False)


def test_注意力前向在inpainting下取滿整個timestep區間(sd9, x01, mask):
    """before：`t_edit = int(T · cfg.strength)`，而 inpainting 下 strength 是
    None，於是 ip1 段 0 的第一次上機以 TypeError 中止。

    after：inpainting 取滿 [0, T−1]。那個形態由純噪聲起跑並跑滿自己的排程，
    整個區間都是攻擊者會經過的地方，不存在「超出區間就白費」的那一段。
    """
    from src.defense import optimize as op

    seen = {}
    real = torch.linspace

    def spy(a, b, n, **kw):
        seen.setdefault("hi", float(b))
        return real(a, b, n, **kw)

    obj = MagicMock(return_value=(torch.zeros(()), {"shared_mass": 0.5}))
    gen = MagicMock()
    gen.generate.return_value = x01.clone()
    with patch.object(torch, "linspace", spy):
        op._build_attn_step(sd9, gen, obj, _attn_cfg(mask, None), x01,
                            torch.zeros(1, 4, DIM, device=DEV), None, [], x01,
                            MagicMock())
    assert seen["hi"] == float(sd9.num_train_timesteps - 1)


def test_注意力前向在inpainting下餵九通道(sd9, x01, mask):
    """inpainting 權重的 `in_channels` 是 9。只餵 4 通道會在 `_eps` 內以形狀
    不符中止——那是第一個缺陷修好之後接著出現的第二個。
    """
    from src.defense import optimize as op
    from src.purify.ops import Purifier

    got = {}
    real_eps = sd9._eps

    def spy(zin, t, emb, **kw):
        got["ch"] = int(zin.shape[1])
        return real_eps(zin, t, emb, **kw)

    gen = MagicMock()
    gen.generate.return_value = x01.clone()
    gen.prepare.return_value = None
    obj = MagicMock(return_value=(torch.zeros(()), {"shared_mass": 0.5}))
    step = op._build_attn_step(
        sd9, gen, obj, _attn_cfg(mask, None), x01,
        sd9.encode_text(""), None, [Purifier("identity")], x01, MagicMock())
    with patch.object(sd9, "_eps", spy):
        step("default", 0, 0)
    assert got["ch"] == sd9.inpaint_in_channels


def test_img2img下缺strength立刻拋出(sd4, x01):
    """兩個威脅模型各自拒絕對方缺的參數，不互相沿用預設值。"""
    from src.defense import optimize as op

    with pytest.raises(ValueError, match="strength"):
        op._build_attn_step(sd4, MagicMock(), MagicMock(),
                            _attn_cfg(None, None), x01,
                            torch.zeros(1, 4, DIM, device=DEV), None, [], x01,
                            MagicMock())


def test_inpainting下缺遮罩立刻拋出注意力前向(sd9, x01):
    from src.defense import optimize as op

    with pytest.raises(ValueError, match="edit_mask"):
        op._build_attn_step(sd9, MagicMock(), MagicMock(),
                            _attn_cfg(None, None), x01,
                            torch.zeros(1, 4, DIM, device=DEV), None, [], x01,
                            MagicMock())


# ---------------------------------------------------------------------------
# 9 通道權重下的 conditioning context
# ---------------------------------------------------------------------------

def test_四通道latent缺conditioning立刻拋出(sd9, x01):
    """ip3 段 0 跑了 2.5 小時才死在 N3 的階段一，錯誤是 torch 的
    `expected input[1, 4, 64, 64] to have 9 channels`——看不出後 5 個通道
    該放什麼。改由本專案自己的訊息在同一點拋出。

    **不預設補零**：對重建路徑「不重畫任何區域」恰好是對的，對模擬攻擊方的
    路徑卻是錯的，而兩者都會產出一張合理的圖。
    """
    z = torch.randn(1, 4, IMG // 8, IMG // 8, device=DEV)
    with pytest.raises(RuntimeError, match="inpaint_conditioning"):
        sd9._eps(z, torch.tensor(10, device=DEV), sd9.encode_text(""))


def test_conditioning下四通道被補成九通道(sd9, x01):
    got = {}
    real = sd9._unet_call

    def spy(zc, t, *cond, **kw):
        got["ch"] = int(zc.shape[1])
        return real(zc, t, *cond, **kw)

    z = torch.randn(1, 4, IMG // 8, IMG // 8, device=DEV)
    with patch.object(sd9, "_unet_call", spy), sd9.conditioning_for(x01):
        sd9._eps(z, torch.tensor(10, device=DEV), sd9.encode_text(""))
    assert got["ch"] == sd9.inpaint_in_channels


def test_預設的conditioning是不重畫任何區域(sd9, x01):
    """重建路徑要 G(x; φ=0) 盡量等於 x，整個保真預算的下限建立在那件事上。
    全 1 遮罩等於叫模型從噪聲重畫整張。"""
    with sd9.conditioning_for(x01):
        m, z_masked = sd9._inpaint_cond
    assert float(m.abs().max()) == 0.0
    assert torch.allclose(z_masked, sd9.encode_image(x01), atol=1e-5)


def test_全一遮罩下沒有影像條件(sd9, x01):
    """Mist 與 DIA 用的那一種：`masked_image_latents` 是 `encode(0)`，
    UNet 退化為純文字條件的去噪器，才對得上它們原作的 ε_θ。"""
    with sd9.conditioning_for(x01, mask=torch.ones_like(x01[:, :1])):
        m, z_masked = sd9._inpaint_cond
    assert float(m.min()) == 1.0
    assert torch.allclose(z_masked, sd9.encode_image(torch.zeros_like(x01)),
                          atol=1e-5)


def test_conditioning離開後還原(sd9, x01):
    assert sd9._inpaint_cond is None
    with sd9.conditioning_for(x01):
        assert sd9._inpaint_cond is not None
    assert sd9._inpaint_cond is None


def test_四通道權重不接受這個context(sd4, x01):
    """靜默接受會讓「這批到底是不是 inpainting」在呼叫端看不出來。"""
    with pytest.raises(RuntimeError, match="不是 inpainting 權重"):
        with sd4.inpaint_conditioning(x01):
            pass
    # 與權重無關的入口在 4 通道下是空操作
    with sd4.conditioning_for(x01):
        pass


def test_防禦生成路徑在九通道下跑得完(sd9, x01):
    """`DefenseGenerator` 的反演與去噪都要在同一組後 5 通道下跑，否則兩半
    走不同的條件而 G(x; φ=0) 不再逼近 x——偏差只表現為「重建差一點」。"""
    from src.defense.generator import DefenseGenerator
    from src.residual.site_apa import build_apa

    mod = build_apa(sd9.unet, steps=2, latent_size=IMG // 8,
                    latent_channels=4, lora_rank=2, latent_max_rank=2,
                    latent_const_rank=1, seed=0).to(DEV)
    try:
        gen = DefenseGenerator(sd9, mod, k_inv=2, t_max=200)
        out = gen.generate(x01, gen.prepare(x01))
        assert out.shape == x01.shape
    finally:
        mod.remove()

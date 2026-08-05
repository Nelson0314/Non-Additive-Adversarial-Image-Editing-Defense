"""SDXL 移植的結構與分派測試 — 全部在 CPU／小模型上執行，不下載 SDXL 權重。

本檔驗的是**結構推導與分派邏輯**，不是數值。理由是 SDXL base 的權重約
6.9 GB，而本輪要釘住的三件事（層數推導、精度規則、兩個 text encoder 與
`added_cond_kwargs`）都不依賴權重的數值：

1. 層數與解析度可由 `unet/config.json` 的欄位純算術推出，連模型都不必建。
2. 精度規則是一張 dtype 對照表，`resolve_precision` 是純函式。
3. `added_cond_kwargs` 的鍵與維度是介面約定，用一個縮小的 SDXL 結構
   （`block_out_channels` 與 `transformer_layers_per_block` 都縮小）即可完整驗證。

**不能在此驗、必須到 GPU 上跑真實 SDXL 才算數的項目**，集中列在
`test_GPU上待驗項目清單` 的 docstring 內，避免它們只存在於報告裡。
"""

import inspect
from pathlib import Path

import pytest
import torch

from src.models.attention import (
    CrossAttentionRecorder,
    aggregate_token_attention,
    cross_attention_layer_count,
    cross_attention_resolutions,
)
from src.models.sd import SDXLPrompt, SDXLWrapper
from src.utils.device import bf16_supported, get_device, resolve_precision

DEV = get_device()

# stabilityai/stable-diffusion-xl-base-1.0 的 unet/config.json 實際欄位。
# 這裡只抄下推導層數與解析度會用到的那幾項，故不需要下載模型。
SDXL_UNET_CONFIG = {
    "down_block_types": [
        "DownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D",
    ],
    "up_block_types": [
        "CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "UpBlock2D",
    ],
    "mid_block_type": "UNetMidBlock2DCrossAttn",
    "transformer_layers_per_block": [1, 2, 10],
    "block_out_channels": [320, 640, 1280],
    "layers_per_block": 2,
    "cross_attention_dim": 2048,
}

# runwayml/stable-diffusion-v1-5 的對應欄位，作為「同一條公式也算得對」的對照
SD15_UNET_CONFIG = {
    "down_block_types": [
        "CrossAttnDownBlock2D", "CrossAttnDownBlock2D",
        "CrossAttnDownBlock2D", "DownBlock2D",
    ],
    "up_block_types": [
        "UpBlock2D", "CrossAttnUpBlock2D",
        "CrossAttnUpBlock2D", "CrossAttnUpBlock2D",
    ],
    "mid_block_type": "UNetMidBlock2DCrossAttn",
    "transformer_layers_per_block": 1,
    "layers_per_block": 2,
    "cross_attention_dim": 768,
}


# ===========================================================================
# 1. 層數與解析度：純算術，不建模型
# ===========================================================================


def test_由SDXL的config推導出的attn2層數為70():
    """down 4+20、mid 10、up 30+6，合計 70。SD v1.5 是 16，相差 4.4 倍。

    這個數字決定三件事：`CrossAttentionRecorder` 應該記到幾張圖、Lo 式 (3)
    的求和上界、以及每步注意力目標的記憶體。寫死 16 的地方全部作廢。
    """
    got = cross_attention_layer_count(SDXL_UNET_CONFIG)
    assert got["down"] == 24, "down 兩個 CrossAttn block：2×2 + 2×10"
    assert got["mid"] == 10, "mid 只有一個 Transformer2DModel，層數取最後一項"
    assert got["up"] == 36, "up 每個 block 有 layers_per_block+1 個：3×10 + 3×2"
    assert got["total"] == 70


def test_同一條公式在SD_v15上得到16():
    got = cross_attention_layer_count(SD15_UNET_CONFIG)
    assert (got["down"], got["mid"], got["up"]) == (6, 1, 9)
    assert got["total"] == 16


def test_SDXL有attention的latent解析度只有64與32():
    """1024² 的 latent 是 128²，但最細的那一層是 DownBlock2D／UpBlock2D，
    沒有 cross-attention。故聚合圖的最高解析度是 latent 的一半。

    任何「注意力圖與 latent 同尺寸」的假設在 SDXL 上都會錯，且錯法是遮罩
    整體縮放到別的位置，不會有錯誤訊息。
    """
    sides = cross_attention_resolutions(SDXL_UNET_CONFIG, latent_size=128)
    assert sides == [64, 32]
    assert 128 not in sides, "最細的 128² 那一層沒有 attention"


def test_SD_v15的attention解析度含最細的latent層():
    assert cross_attention_resolutions(SD15_UNET_CONFIG, 64) == [64, 32, 16, 8]


def test_未知的mid_block_type必須報錯():
    """安靜地當成 0 或 1 會讓層數核對失去意義。"""
    cfg = dict(SDXL_UNET_CONFIG, mid_block_type="UNetMidBlock2DSimpleCrossAttn")
    with pytest.raises(ValueError, match="mid_block_type"):
        cross_attention_layer_count(cfg)


def test_上下行block數不等時必須報錯():
    cfg = dict(SDXL_UNET_CONFIG, up_block_types=["UpBlock2D"])
    with pytest.raises(ValueError, match="等長"):
        cross_attention_layer_count(cfg)


def test_per_block清單長度不符時必須報錯():
    cfg = dict(SDXL_UNET_CONFIG, transformer_layers_per_block=[1, 2])
    with pytest.raises(ValueError, match="transformer_layers_per_block"):
        cross_attention_layer_count(cfg)


# ===========================================================================
# 2. 精度規則
# ===========================================================================


def test_fp16時VAE必須留在fp32():
    """SDXL 原生 VAE 在 fp16 下中間激活會超過 65504 而變成 inf，解碼出全黑圖。

    這條是規則不是註解：由 `resolve_precision` 回傳，`SDWrapper` 一律照它施加。
    """
    backbone, vae = resolve_precision(torch.float16)
    assert backbone == torch.float16
    assert vae == torch.float32


def test_bf16時可全程半精度():
    """bf16 的指數位寬與 fp32 相同（8 bit），動態範圍一致，不會溢位。
    這正是 RTX 5090（sm_120）相對 V100（sm_70，無 bf16）的實質好處。"""
    backbone, vae = resolve_precision(torch.bfloat16)
    assert backbone == torch.bfloat16
    assert vae == torch.bfloat16


def test_fp32時三者相同():
    assert resolve_precision(torch.float32) == (torch.float32, torch.float32)


def test_不支援的精度必須報錯而非落回預設():
    """靜默落回 fp32 會讓「這批資料是什麼精度」無從查證，而精度正是
    V100 與 5090 之間唯一的差異來源。"""
    with pytest.raises(ValueError, match="不支援的計算精度"):
        resolve_precision(torch.float64)


def _code_without_docstrings(path: Path) -> str:
    """回傳剝除 docstring 後的原始碼。

    掃描必須只看**會執行的程式碼**。直接 grep 原文會命中
    「為什麼不用 sdxl-vae-fp16-fix」這類解釋性 docstring，
    使測試在文件寫得越清楚時越容易誤報——那會逼人刪掉正確的說明來讓測試通過。

    真正的載入程式碼（字串常數作為函式引數）不是 docstring，仍會留在輸出中。
    """
    import ast

    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                             ast.AsyncFunctionDef)):
            body = node.body
            if (body and isinstance(body[0], ast.Expr)
                    and isinstance(body[0].value, ast.Constant)
                    and isinstance(body[0].value.value, str)):
                node.body = body[1:] or [ast.Pass()]
    return ast.unparse(tree)


def test_不存在換VAE權重的路徑():
    """威脅模型要求攻擊方使用 stock SDXL，換 VAE 權重就是換模型。

    以原始碼掃描釘住「這條路徑不存在」而非「預設沒有開」：後者擋不住
    有人為了省事把 `sdxl-vae-fp16-fix` 加成一個選項。掃 `sd.py` 與
    `device.py` 兩個檔，任何替代 VAE 的來源字串都不得出現於**可執行的程式碼**中。
    docstring 不在掃描範圍內——說明為何不採用某個作法，本身不是採用它。
    """
    root = Path(__file__).resolve().parent.parent
    forbidden = ["fp16-fix", "fp16_fix", "madebyollin", "AutoencoderKL.from_pretrained"]
    for rel in ("src/models/sd.py", "src/utils/device.py"):
        code = _code_without_docstrings(root / rel)
        for token in forbidden:
            assert token not in code, f"{rel} 的程式碼出現替代 VAE 的來源 {token!r}"

    # 建構子也不得提供「VAE 從哪裡來」的參數：只有 dtype 可以調
    params = set(inspect.signature(SDXLWrapper.__init__).parameters)
    assert params == {"self", "model_name", "dtype", "pipe"}, (
        f"SDXLWrapper 的建構參數多了 {params}，精度以外的模型來源不得可調"
    )


# ===========================================================================
# 3. 極小 SDXL 結構：兩個 text encoder、added_cond_kwargs、注意力擷取
# ===========================================================================

TOKENIZER_REPO = "hf-internal-testing/tiny-stable-diffusion-pipe"
DIM_1, DIM_2 = 32, 64          # 兩個 text encoder 的 hidden size
POOLED_DIM = 32                # bigG 的 projection_dim；SDXL 真值為 1280
TIME_EMBED_DIM = 8             # SDXL 真值為 256
N_TIME_IDS = 6                 # base 的 micro-conditioning 維數（refiner 為 5）
IMG = 128                      # latent 為 16²


def _tiny_sdxl_pipe():
    """用 diffusers 的 config 建一個極小但**結構與 SDXL 相同**的 pipeline。

    縮小的只有寬度：`block_out_channels`、`transformer_layers_per_block`、
    兩個 text encoder 的 hidden size。block 的種類與順序、mid block 的型別、
    `addition_embed_type="text_time"`、兩個 encoder 串接後等於
    `cross_attention_dim`——這些決定分派邏輯的欄位全部保持 SDXL 的形態。

    tokenizer 借用 tiny-SD 的 CLIP tokenizer（既有測試已在用，非 SDXL 權重）。
    SDXL 的兩個 tokenizer 用的是同一套 CLIP BPE 詞彙與同樣的 77 格上限，
    故對「串接與 pooled 的形狀」這個待驗性質而言是等價的。
    """
    from diffusers import (
        AutoencoderKL, EulerDiscreteScheduler,
        StableDiffusionXLImg2ImgPipeline, UNet2DConditionModel,
    )
    from transformers import (
        CLIPTextConfig, CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer,
    )

    tok = CLIPTokenizer.from_pretrained(TOKENIZER_REPO, subfolder="tokenizer")
    vocab = len(tok)

    def _text_cfg(hidden):
        # `eos_token_id` 必須取自 tokenizer 本身。CLIPTextConfig 的預設值是
        # 真實 CLIP 的 49407，而此處的詞彙表只有 len(tok) 個 token；
        # 預設值落在詞彙表之外時，transformers 的
        # `(input_ids == eos_token_id).int().argmax(-1)` 全為 False，argmax
        # 回傳 0，pooled 因而取自 BOS 位置。BOS 在因果注意力下只看得到自己，
        # 於是**任何 prompt 都得到同一個 pooled**——
        # `test_CFG的兩支各用自己的pooled` 會因此失敗，但失敗的原因與
        # `sd.py` 無關（實測 `out[0]` 與 `out.text_embeds` 逐位元相同，
        # 取的位置與 diffusers 的 `encode_prompt` 一致）。
        return CLIPTextConfig(
            vocab_size=vocab, hidden_size=hidden, intermediate_size=hidden * 2,
            num_hidden_layers=2, num_attention_heads=2,
            max_position_embeddings=tok.model_max_length,
            projection_dim=POOLED_DIM,
            bos_token_id=tok.bos_token_id, eos_token_id=tok.eos_token_id,
        )

    torch.manual_seed(0)
    unet = UNet2DConditionModel(
        sample_size=IMG // 8,
        in_channels=4, out_channels=4,
        down_block_types=(
            "DownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D",
        ),
        up_block_types=(
            "CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "UpBlock2D",
        ),
        block_out_channels=(32, 64, 64),
        layers_per_block=1,
        transformer_layers_per_block=(1, 2, 3),
        cross_attention_dim=DIM_1 + DIM_2,
        attention_head_dim=(4, 8, 8),
        norm_num_groups=8,
        addition_embed_type="text_time",
        addition_time_embed_dim=TIME_EMBED_DIM,
        projection_class_embeddings_input_dim=(
            TIME_EMBED_DIM * N_TIME_IDS + POOLED_DIM
        ),
    )
    vae = AutoencoderKL(
        in_channels=3, out_channels=3,
        down_block_types=("DownEncoderBlock2D",) * 4,
        up_block_types=("UpDecoderBlock2D",) * 4,
        block_out_channels=(8, 8, 8, 8),
        layers_per_block=1, latent_channels=4, norm_num_groups=8,
        sample_size=IMG, scaling_factor=0.13025,
    )
    scheduler = EulerDiscreteScheduler(
        num_train_timesteps=1000, beta_start=0.00085, beta_end=0.012,
        beta_schedule="scaled_linear",
    )
    return StableDiffusionXLImg2ImgPipeline(
        vae=vae,
        text_encoder=CLIPTextModel(_text_cfg(DIM_1)),
        text_encoder_2=CLIPTextModelWithProjection(_text_cfg(DIM_2)),
        tokenizer=tok, tokenizer_2=tok, unet=unet, scheduler=scheduler,
        force_zeros_for_empty_prompt=True, add_watermarker=False,
    )


@pytest.fixture(scope="module")
def sdxl():
    return SDXLWrapper("tiny-sdxl-structure", pipe=_tiny_sdxl_pipe())


@pytest.fixture(scope="module")
def x01():
    g = torch.Generator().manual_seed(20260805)
    return torch.rand(1, 3, IMG, IMG, generator=g).to(DEV)


def test_極小結構的層數推導與實際掃描一致(sdxl):
    """公式若只在真實 SDXL 上對、在別的形狀上錯，就不能拿來當核對依據。"""
    scanned = [n for n, _ in sdxl.unet.named_modules() if n.endswith("attn2")]
    derived = cross_attention_layer_count(sdxl.unet.config)
    assert derived["total"] == len(scanned)
    # layers_per_block=1、transformer_layers=(1,2,3) ⟹ down 1+2·? 由公式算
    assert derived["down"] == 1 * 2 + 1 * 3
    assert derived["mid"] == 3
    assert derived["up"] == (1 + 1) * 3 + (1 + 1) * 2


def test_兩個text_encoder串接後等於cross_attention_dim(sdxl):
    """SDXL 的 2048 = CLIP-L 的 768 + OpenCLIP-bigG 的 1280。串接的維度對不上
    UNet 就無法前向，但若有人只取其中一個 encoder，維度仍可能碰巧能跑。"""
    p = sdxl.encode_text("a wrecked car")
    assert isinstance(p, SDXLPrompt)
    assert p.embeds.shape == (1, sdxl.tokenizer.model_max_length, DIM_1 + DIM_2)
    assert p.embeds.shape[-1] == sdxl.unet.config.cross_attention_dim
    assert p.pooled.shape == (1, POOLED_DIM), "pooled 只來自第二個 encoder"
    assert p.shape == p.embeds.shape, "對外的 shape 是序列嵌入的，供 site E 建模塊用"


def test_pooled與序列嵌入是兩個不同的東西(sdxl):
    """pooled 不是序列嵌入的某一格，取錯會得到能跑但語意不同的條件。"""
    p = sdxl.encode_text("a wrecked car")
    assert p.pooled.dim() == 2 and p.embeds.dim() == 3
    assert p.pooled.shape[-1] != p.embeds.shape[-1]


def test_added_cond_kwargs的鍵與形狀正確(sdxl, x01):
    """`text_embeds`（pooled）與 `time_ids`（6 維 micro-conditioning）兩個鍵
    缺一不可。diffusers 在 `addition_embed_type="text_time"` 下會直接 KeyError，
    但那個錯誤發生在 UNet 深處，看不出是條件組錯。"""
    seen = {}

    def hook(module, args, kwargs):
        seen.update(kwargs)
        return None

    h = sdxl.unet.register_forward_pre_hook(hook, with_kwargs=True)
    try:
        with torch.no_grad():
            z = sdxl.encode_image(x01)
            sdxl._eps(z, torch.tensor(200, device=DEV), sdxl.encode_text("a car"))
    finally:
        h.remove()

    assert set(seen["added_cond_kwargs"]) == {"text_embeds", "time_ids"}
    assert seen["added_cond_kwargs"]["text_embeds"].shape == (1, POOLED_DIM)
    assert seen["added_cond_kwargs"]["time_ids"].shape == (1, N_TIME_IDS)
    assert seen["encoder_hidden_states"].shape[-1] == DIM_1 + DIM_2


def test_time_ids由latent尺寸反推(sdxl, x01):
    """(original_h, original_w, crop_top, crop_left, target_h, target_w)。
    由 latent 乘以 VAE 的下採樣倍率反推，故不可能與實際輸入對不上；
    crop 取 (0,0)，本專案不做隨機裁切。"""
    with torch.no_grad():
        z = sdxl.encode_image(x01)
    ids = sdxl._time_ids(z, torch.float32)
    assert ids.tolist() == [[IMG, IMG, 0, 0, IMG, IMG]]


def test_time_ids維數錯誤時當場報錯(sdxl, x01):
    """refiner 用 5 維（含 aesthetic score）。配到 base 上總寬會少 256，
    diffusers 會在 add_embedding 內以形狀不符中止，訊息看不出原因。"""
    with torch.no_grad():
        z = sdxl.encode_image(x01)
    p = sdxl.encode_text("a car")
    bad = torch.zeros(1, 5, device=z.device)
    with pytest.raises(ValueError, match="projection_class_embeddings_input_dim"):
        sdxl._check_added_cond_dim(p.pooled, bad.shape[-1])


def test_條件是裸張量時必須報錯(sdxl, x01):
    """只帶 cross-attention 那一半，`added_cond_kwargs` 就會缺 text_embeds。
    這在 SD v1.x 上是合法呼叫，故必須在型別上擋掉而非靠呼叫端記得。"""
    with torch.no_grad():
        z = sdxl.encode_image(x01)
    with pytest.raises(TypeError, match="SDXLPrompt"):
        sdxl._eps(z, torch.tensor(200, device=DEV), torch.zeros(1, 77, 96, device=DEV))


# ---- site E：殘差只作用在序列嵌入 ----


def test_嵌入殘差只加在序列上不碰pooled(sdxl):
    """`emb_residual` 的作用對象是那條 2048 維的序列，不是 pooled——後者
    另經 `added_cond_kwargs` 進入 UNet 的 additive embedding，不走
    cross-attention。加錯地方會把 micro-conditioning 一併擾動，那不是
    「破壞文字與影像的綁定」。

    `generator.py` 寫的是 `emb = emb + d_emb`，故此語意由 `__add__` 承擔。
    """
    p = sdxl.encode_text("a wrecked car")
    delta = torch.full_like(p.embeds, 0.1)
    q = p + delta
    assert torch.allclose(q.embeds, p.embeds + delta)
    assert torch.equal(q.pooled, p.pooled), "pooled 不得被 site E 的殘差改動"


def test_殘差維度對到pooled時報錯(sdxl):
    p = sdxl.encode_text("a car")
    with pytest.raises(ValueError, match="pooled"):
        p + torch.zeros(1, 77, POOLED_DIM, device=p.embeds.device)


def test_detach同時作用於兩個張量(sdxl):
    """`generator.prepare()` 呼叫 `encode_text(...).detach()`。只 detach 一半
    會讓 pooled 留在計算圖上，症狀是反向時多出一條沒人要的路徑。"""
    p = sdxl.encode_text("a car")
    d = p.detach()
    assert not d.embeds.requires_grad and not d.pooled.requires_grad


# ---- 擴散迴圈：完全沿用基底類別 ----


def test_去噪與反演迴圈不需為SDXL改動(sdxl, x01):
    """`ddim_inversion`／`denoise`／`bdia_*` 只把 emb 原樣轉交 `_eps`，
    條件的內部結構由 `_cond_tensors` 與 `_unet_call` 吸收。這條測試釘住
    那個設計：迴圈一行都沒改也必須跑得完。"""
    k = 2
    emb = sdxl.encode_text("").detach()
    ts = sdxl.timesteps(k, t_max=500)
    with torch.no_grad():
        z0 = sdxl.encode_image(x01)
        z_inv = sdxl.ddim_inversion(z0, emb, ts, k)
        z_out, _ = sdxl.denoise(z_inv, emb, ts, k)
        pair = sdxl.bdia_inversion(z0, emb, ts, k)
        z_bdia, _ = sdxl.bdia_denoise(pair, emb, ts, k)
    assert z_out.shape == z0.shape
    assert torch.isfinite(z_out).all() and torch.isfinite(z_bdia).all()


def test_CFG的兩支各用自己的pooled(sdxl, x01):
    """條件與無條件的 pooled 不同。若 pooled 改由 wrapper 的狀態提供，
    無條件那一支會拿到條件的 pooled，而症狀只是「CFG 效果怪怪的」。
    綁在 SDXLPrompt 裡使這種錯配在型別上不可能發生。"""
    pooled_seen = []

    def hook(module, args, kwargs):
        pooled_seen.append(kwargs["added_cond_kwargs"]["text_embeds"].clone())
        return None

    emb = sdxl.encode_text("a wrecked car").detach()
    emb_u = sdxl.encode_text("").detach()
    assert not torch.allclose(emb.pooled, emb_u.pooled), "兩個 prompt 的 pooled 應不同"

    h = sdxl.unet.register_forward_pre_hook(hook, with_kwargs=True)
    try:
        with torch.no_grad():
            z = sdxl.encode_image(x01)
            sdxl._eps_cfg(z, torch.tensor(200, device=DEV), emb, 7.5, emb_u)
    finally:
        h.remove()

    assert len(pooled_seen) == 2
    assert torch.allclose(pooled_seen[0], emb_u.pooled), "第一支必須是無條件"
    assert torch.allclose(pooled_seen[1], emb.pooled)


def test_梯度經pooled路徑仍抵達序列嵌入(sdxl, x01):
    """site E 的 φ 只動 `embeds`。pooled 被攤平成 checkpoint 的獨立引數，
    正是為了避免張量藏在 dataclass 裡而在重算時silently 斷梯度。"""
    with torch.no_grad():
        z = sdxl.encode_image(x01)
    p = sdxl.encode_text("a car")
    delta = torch.zeros_like(p.embeds).requires_grad_(True)
    eps = sdxl._eps(z, torch.tensor(200, device=DEV), p.detach() + delta,
                    use_ckpt=True)
    eps.pow(2).sum().backward()
    assert delta.grad is not None and delta.grad.abs().sum() > 0


# ---- attention.py 在 SDXL 上 ----


def test_注意力擷取的層數等於推導值(sdxl, x01):
    """掃描靠的是名稱後綴這條字串規則，換模型時可能多掃或漏掉，而漏掉的
    症狀只是目標值偏小，沒有錯誤訊息。故 recorder 一律與 config 核對。"""
    rec = CrossAttentionRecorder(sdxl.unet)
    expect = cross_attention_layer_count(sdxl.unet.config)["total"]
    assert rec.n_layers == expect
    with torch.no_grad(), rec:
        sdxl._eps(sdxl.encode_image(x01), torch.tensor(200, device=DEV),
                  sdxl.encode_text("a car"))
    assert len(rec.maps) == expect
    for m in rec.maps:
        assert m.shape[-1] == sdxl.tokenizer.model_max_length
        s = m.sum(dim=-1)
        assert torch.allclose(s, torch.ones_like(s), atol=1e-3)


def test_找不到attn2時明確報錯():
    """cross-attention 目標套在非 SD 架構上必須當場講，不是回傳空清單。"""
    class _NotAUNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(2, 2)

    with pytest.raises(RuntimeError, match="attn2"):
        CrossAttentionRecorder(_NotAUNet())


def test_掃到的層數與config不符時報錯():
    """名稱規則掃到的層數少於 config 推導值，代表擷取沒有完整覆蓋。"""
    class _Fake(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.attn2 = torch.nn.Linear(2, 2)
            self.config = SDXL_UNET_CONFIG

    with pytest.raises(RuntimeError, match="70"):
        CrossAttentionRecorder(_Fake())


def test_聚合的sum與mean只差實際層數這個因子():
    """Lo 式 (3) 的 Σ 值域上界隨層數而變（16 → 70），故 `reduce="sum"` 的
    絕對值跨模型不可比。`reduce="mean"` 由實際掃到的層數推導，回到 [0,1]。"""
    maps = [torch.rand(1, 16, 8) for _ in range(5)]
    s = aggregate_token_attention(maps, span=(1, 3), reduce="sum")
    m = aggregate_token_attention(maps, span=(1, 3), reduce="mean")
    assert torch.allclose(s / len(maps), m, atol=1e-6)


def test_未知的reduce必須報錯():
    with pytest.raises(ValueError, match="reduce"):
        aggregate_token_attention([torch.rand(1, 4, 4)], span=(0, 2), reduce="max")


# ===========================================================================
# 4. 精度規則實際施加到子模組上
# ===========================================================================


def test_fp16下VAE的權重確實是fp32():
    """規則寫在 `resolve_precision`，但真正要成立的是「模型上的 dtype」。
    兩者之間隔著 `_apply_precision`，那一段也必須有測試。"""
    sd = SDXLWrapper("tiny-sdxl-structure", dtype=torch.float16,
                     pipe=_tiny_sdxl_pipe())
    assert sd.unet.dtype == torch.float16
    assert sd.text_encoder.dtype == torch.float16
    assert sd.text_encoder_2.dtype == torch.float16
    assert sd.vae.dtype == torch.float32, "SDXL VAE 在 fp16 下會溢位成全黑圖"


def test_fp16下解碼路徑不因dtype不符而中止():
    """UNet 產出 fp16 的 latent、VAE 是 fp32，兩者相接處必須明確轉換。
    不轉會是 "expected scalar type" 的硬錯誤；用 autocast 蓋掉則會讓 VAE
    實際跑在哪個精度看不出來。"""
    sd = SDXLWrapper("tiny-sdxl-structure", dtype=torch.float16,
                     pipe=_tiny_sdxl_pipe())
    z = torch.randn(1, 4, IMG // 8, IMG // 8, dtype=torch.float16, device=sd.device)
    with torch.no_grad():
        x = sd.decode_latent(z)
    assert x.dtype == torch.float32 and torch.isfinite(x).all()


@pytest.mark.skipif(not bf16_supported(), reason="此裝置沒有 bf16（如 V100 sm_70）")
def test_bf16下VAE也是bf16():
    sd = SDXLWrapper("tiny-sdxl-structure", dtype=torch.bfloat16,
                     pipe=_tiny_sdxl_pipe())
    assert sd.unet.dtype == torch.bfloat16
    assert sd.vae.dtype == torch.bfloat16, "bf16 的指數位與 fp32 相同，不會溢位"


def test_GPU上待驗項目清單():
    """本檔驗不到、必須在 GPU 上以真實 SDXL 權重跑過才算數的項目。

    列成測試而非只寫在報告裡，是為了讓它跟著程式走：某一項在 GPU 上完成
    驗證之後，應把它從這裡刪掉並補上真正的測試。

    1. **BDIA 的來回誤差**。代數與模型無關，且 SDXL base 的排程參數與
       SD v1.x 相同（beta 0.00085–0.012、scaled_linear、1000 步），故
       `alphas_cumprod` 逐元素相同、`_ddim_step` 不需改動。但「比 DDIM 小
       2 個數量級以上」取決於 ε 預測本身，必須在 1024²、真實權重下重測。
    2. **fp16 與 fp32 的一致性**（`ARCH` §7.1 新增的驗證要求）。PGD 的梯度
       品質對精度敏感；兩者結果不一致就得退回 fp32 並重估耗時。BDIA 尤其
       要各跑一次比較——ε 在 fp16 下只有 10 bit 尾數。
    3. **SDXL VAE 在 fp16 下確實溢位**。本檔的極小 VAE 是隨機權重、通道數
       只有 8，不會重現真實 VAE 的激活量級，故只能釘住規則，釘不住現象。
    4. **70 層注意力擷取的記憶體**。64² 那 24 層每層是 4096×77 的分佈，
       逐 head 保留會再乘上 head 數。V100 32 GB 下的上限須實測。
    5. **平台停止門檻**。先驗值於 SD v1.4、16 層、512²、site PF 上實測，
       層數變 4.4 倍後不可沿用。**已結構性處置**：`optimize.MONITOR_TOL` 已
       更名為 `LEGACY_MONITOR_TOL`（僅作量級紀錄、無任何讀取路徑），
       `resolve_stop_tol` 取消全部回退，未校準即拋 `KeyError`。
       實際數值仍須由段 0 的 `calibrate_lr` 在 GPU 上產出。
    6. **baseline 在真實 SDXL 權重上的數值**。§7 的實跑用的是極小結構
       （隨機權重、`IMG=128`），驗的是「路徑能不能跑、梯度是否非零」，
       不是攻擊強度。五篇的損失量級、`grad_reps` 的收斂，以及 1024² 下
       PhotoGuard-c（200 步 × 10 reps × 4 去噪步）與 PromptFlare
       （400 步、全 activation 保留）的 peak VRAM 都必須在 GPU 上實測。
    """
    from src.defense import optimize as O

    # 這個檢查表是「列成測試讓它跟著程式走」的，清單被清空或門檻回退被
    # 悄悄加回來時必須失敗，否則本測試退化成一行 `assert True`。
    items = [ln for ln in test_GPU上待驗項目清單.__doc__.splitlines()
             if ln.strip().startswith(("1.", "2.", "3.", "4.", "5.", "6."))]
    assert len(items) == 6, "待驗清單的項目數與文件不符"
    assert not hasattr(O, "MONITOR_TOL"), "第 5 項的結構性處置被回退了"
    assert hasattr(O, "LEGACY_MONITOR_TOL"), "先驗門檻的量級紀錄不應消失"


# ===========================================================================
# 6. 無條件分支的構造（審查發現，2026-08-05）
# ===========================================================================

def test_無條件分支不得寫死():
    """`force_zeros_for_empty_prompt` 必須由 pipeline 的 config 讀取。

    stock SDXL base 記為 `true`，即其 CFG 無條件分支是零張量而非
    `encode_text("")`。refiner 與其他檢查點的值可能不同，寫死會在換模型時
    靜默給出錯誤的無條件分支——影像仍然生得出來，只是攻擊方不再是 stock 模型。
    """
    import inspect

    from src.models import sd as sd_mod

    src = inspect.getsource(sd_mod.SDXLWrapper.force_zeros_for_empty_prompt.fget)
    assert "config" in src, "必須由 config 讀取"
    assert "return True" not in src and "return False" not in src, "不得寫死"


def test_config缺該欄位時拋出而非猜一邊(sdxl):
    """猜錯的差異沒有症狀，故必須拋出。"""
    w = sdxl

    class _NoFlag:
        """沒有該欄位的 config。diffusers 的 FrozenDict 不可就地賦值，
        故以整個物件替換而非刪一個鍵。"""

    monkey = type(w.pipe)
    original = monkey.config
    try:
        monkey.config = property(lambda self: _NoFlag())
        with pytest.raises(RuntimeError, match="force_zeros_for_empty_prompt"):
            _ = w.force_zeros_for_empty_prompt
    finally:
        monkey.config = original


def test_旗標為真時無條件分支為零張量(sdxl):
    w = sdxl
    if not w.force_zeros_for_empty_prompt:
        pytest.skip("此 pipeline 的旗標為 False")
    u = w.uncond_prompt()
    assert float(u.embeds.abs().max()) == 0.0
    assert float(u.pooled.abs().max()) == 0.0


def test_無條件分支的形狀與條件分支一致(sdxl):
    """形狀不符會在 UNet 前向才炸，且訊息指不到真正的原因。"""
    w = sdxl
    c = w.encode_text("a photo")
    u = w.uncond_prompt()
    assert u.embeds.shape == c.embeds.shape
    assert u.pooled.shape == c.pooled.shape


def test_空prompt的編碼不等於無條件分支(sdxl):
    """兩者在 stock SDXL base 上是不同的東西。混用沒有症狀，故要有測試。"""
    w = sdxl
    if not w.force_zeros_for_empty_prompt:
        pytest.skip("此 pipeline 的旗標為 False，兩者本就相同")
    assert not torch.equal(w.encode_text("").embeds, w.uncond_prompt().embeds)


def test_CFG的兩支不得是同一個張量(sdxl):
    """條件與無條件分支若逐元素相同，`ε_u + w·(ε_c − ε_u)` 的括號恆為零，
    guidance_scale 設多少都等同 w=1。

    w=1 下 SD 幾乎不服從 prompt、SDEdit 退化成「加噪再去噪」，
    防禦訓練會跑在一個不會發生的攻擊上（`LOGIC_CHECK` A6）。
    """
    import inspect

    from src.defense import optimize as opt

    src = inspect.getsource(opt.optimize)
    assert "sd.uncond_prompt()" in src, "無條件分支必須由模型的規則產生"
    assert "emb_uncond=emb_cond" not in src, (
        "不得把條件分支同時當成無條件分支——CFG 會退化成 w=1")


def test_variant是官方權重檔而非換模型():
    """`variant="fp16"` 取的是官方 repo 內的 fp16 權重檔，不是換模型。

    官方 `stabilityai/stable-diffusion-xl-base-1.0` 同時提供 fp32 與 fp16
    兩組權重檔，兩者都是同一個模型的官方發布，差別只在存檔精度。

    這與 `sdxl-vae-fp16-fix` 有本質差別——後者是第三方**重新訓練**的 VAE，
    換它就是換模型。`test_不存在換VAE權重的路徑` 仍然把那條路擋死；
    本測試確認兩者沒有被混為一談。
    """
    import inspect

    from src.models.sd import SDXLWrapper

    sig = inspect.signature(SDXLWrapper._load_pipeline)
    assert "variant" in sig.parameters
    assert sig.parameters["variant"].default == "fp16"

    src = inspect.getsource(SDXLWrapper._load_pipeline)
    for forbidden in ("fp16-fix", "madebyollin"):
        assert forbidden not in src.split('"""')[2], (
            f"可執行的程式碼不得出現 {forbidden}")


def test_promptflare的白名單在SDXL結構上實測(sdxl, x01):
    """`test_baselines.py` 的白名單測試以 stub 餵 token 數，驗的是規則本身；
    這裡驗前向真的能收集到 token 數，且 SDXL 的結構確實只有兩級。

    SDXL 的 `down_blocks[0]` 是 `DownBlock2D`（無 attention），故 attn2 只
    出現在 latent/2 與 latent/4 兩級——與 SD1.x 的四級不同，這正是原作寫死的
    `[1024, 256, 64]` 不能沿用的原因。
    """
    from src.baselines import promptflare as PF
    from src.baselines.pgd import BaselineSpec

    spec = PF.SPEC
    ctx = PF.prepare(sdxl, x01, spec, strength=0.6, seed=0)
    try:
        counts = PF.observed_token_counts(sdxl, ctx, IMG, IMG)
        assert len(counts) == 2, f"SDXL 的 attn2 應只有兩級，實得 {counts}"
        assert counts == sorted(counts)
        # latent 邊長 IMG/8，兩級分別為 (IMG/16)² 與 (IMG/32)²。
        assert counts == [(IMG // 32) ** 2, (IMG // 16) ** 2]
        assert ctx.loss_depth == counts[:-1], "白名單應為去掉最外一級"
    finally:
        ctx.close()


# ===========================================================================
# 7. baseline 在 SDXL 上實跑（審查點名的最大缺口）
# ===========================================================================

# 2026-08-05 的程式審查（`docs/INDEX.md` §3）：五篇的 `prepare` 與 `loss_fn` 一行都沒有被
# 執行過，`run_pgd` 的骨幹迴圈也沒有。既有 34 項測試全部只驗 `BaselineSpec`
# 的欄位值與四個純函式。
#
# 這個缺口在換到 SDXL 之後由「沒被驗到」升級為「一定是壞的」：五篇原本都照
# SD v1.x 的 API 寫，`sd.encode_text()` 在 SDXL 上回傳 `SDXLPrompt` 而非張量，
# 而 `.repeat()`／`.expand()`／`torch.cat()` 對它全部不適用。以極小 SDXL
# 結構實跑即可在 CPU 上攔下這類錯誤，不需要 6.9 GB 權重。

# 各篇 `prepare` 需要、而本專案的威脅模型明確決定的項目。
# 與 `src/experiment/executors.py::baseline_kwargs` 同一組來源。
BASELINE_KW = {
    "photoguard_c": dict(strength=0.6),
    "advpaint": dict(strength=0.6, prompt="", guidance_scale=7.5),
    "promptflare": dict(strength=0.6),
    "mist": dict(),          # target01 於測試中另給，見下
    "dia_r": dict(),
    "dia_pt": dict(),
}


@pytest.mark.parametrize("name", sorted(BASELINE_KW))
def test_baseline在SDXL上可實跑且梯度非零(sdxl, x01, name):
    """`prepare` → `loss_fn` → `autograd.grad`，三步都必須真的走過。

    `loss` 有限且梯度非零是最低要求：梯度為零時 `sign(0) = 0`，PGD 會原地
    踏步跑完全部步數、log 正常列印、輸出一張未改動的圖——沒有任何症狀。

    Mist 在此傳入合成的 target 張量。那**不是** MIST.png，故本測試不宣稱
    重現 Mist 的數值，只驗其計算路徑在 SDXL 上可執行。
    """
    from src.baselines import REGISTRY
    from src.baselines.pgd import BaselineSpec

    spec: BaselineSpec = REGISTRY[name]
    kw = dict(BASELINE_KW[name])
    if spec.needs_target_image:
        g = torch.Generator().manual_seed(7)
        kw["target01"] = torch.rand(1, 3, IMG, IMG, generator=g).to(DEV)

    ctx = spec.prepare(sdxl, x01, spec, seed=0, **kw)
    try:
        vr = spec.value_range
        x_adv = vr.from01(x01).clone().requires_grad_(True)
        loss = spec.loss_fn(sdxl, x_adv, ctx)

        assert loss.dim() == 0, f"{name} 的損失必須是純量，實得 shape {tuple(loss.shape)}"
        assert torch.isfinite(loss), f"{name} 的損失為 {loss.item()}"

        (grad,) = torch.autograd.grad(loss, x_adv)
        assert torch.isfinite(grad).all(), f"{name} 的梯度含 inf/nan"
        assert float(grad.abs().sum()) > 0, (
            f"{name} 的梯度全為零：sign(0)=0 會讓 PGD 原地踏步而毫無症狀"
        )
    finally:
        if hasattr(ctx, "close"):
            ctx.close()


@pytest.mark.parametrize("name", ["photoguard_c", "advpaint", "promptflare"])
def test_run_pgd的骨幹在SDXL上跑得完(sdxl, x01, name):
    """骨幹迴圈（初始化→梯度→更新→投影→夾限）從未被執行過。

    步數以 `dataclasses.replace` 降到 2：這裡驗的是**流程能不能跑完**，
    不是攻擊強度。失真預算仍以原 spec 的 eps 判定——投影寫錯時
    `delta_linf01` 會超出預算，那是骨幹最容易出錯而事後看不出來的地方。
    """
    import dataclasses

    from src.baselines import REGISTRY
    from src.baselines.pgd import run_pgd

    spec = dataclasses.replace(REGISTRY[name], steps=2, grad_reps=1)
    result = run_pgd(
        sdxl, x01, spec, seed=0, verbose=False, **BASELINE_KW[name]
    )

    assert result.x_adv01.shape == x01.shape
    assert torch.isfinite(result.x_adv01).all()
    assert float(result.x_adv01.min()) >= 0.0 and float(result.x_adv01.max()) <= 1.0
    assert len(result.history) == 2, "每一步都要留一筆紀錄"

    if spec.norm == "linf":
        got = float((result.x_adv01 - x01).abs().max())
        assert got <= spec.eps_pixel01 + 1e-5, (
            f"{name} 的 ℓ∞ 失真 {got:.6f} 超出預算 {spec.eps_pixel01:.6f}"
        )


# ===========================================================================
# 8. attention 擷取（CODE §4.2）
# ===========================================================================

def test_取樣步固定含第零步與最後一步():
    """最後一步不可省：x̂₀ 在末段才收斂，只看前段會把「注意力早期被打散、
    後期又長回來」誤讀成防禦成功。"""
    from src.experiment.attn_capture import sampled_steps

    assert sampled_steps(50, every=5) == [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 49]
    assert sampled_steps(1) == [0]
    assert sampled_steps(3, every=5) == [0, 2]
    with pytest.raises(ValueError):
        sampled_steps(0)


def test_擷取只在取樣步開啟(sdxl, x01):
    """recorder 會實體化 (Q, 77) 的注意力矩陣——那正是 SDPA 融合核所避免的。
    全程開著等於讓 UNet 前向成本大幅增加。"""
    from src.experiment.attn_capture import AttnCapture, capture_span

    steps = 6
    cap = AttnCapture(sdxl, steps, capture_span(sdxl, ""), every=5)
    seen = []
    with cap:
        assert cap.recorder.enabled is False, "進入時預設關閉"
        for i in range(steps):
            cap.step_hook(i, torch.tensor(1), None)
            seen.append(cap.recorder.enabled)
    assert seen == [True, False, False, False, False, True]


def test_擷取產生聚合圖逐層圖與統計表(sdxl, x01, tmp_path):
    """`attn_stats.csv` 的列數必須等於「取樣步數 × 層數」——少了就是
    某些層或某些步沒被記到，而圖仍然畫得出來。"""
    from src.experiment.attn_capture import AttnCapture, capture_span

    steps = 4
    cap = AttnCapture(sdxl, steps, capture_span(sdxl, ""), every=5)
    lat = sdxl.latent_shape(IMG, IMG)
    noise = sdxl.sample_edit_noise(torch.empty(lat, device=DEV), seed=0)
    emb = sdxl.encode_text("a wrecked car")
    with cap, torch.no_grad():
        sdxl.sdedit(x01, emb, noise, steps, strength=0.6,
                    guidance_scale=7.5, emb_uncond=sdxl.uncond_prompt(),
                    step_hook=cap.step_hook)

    n_layers = cap.n_layers
    assert n_layers == len([n for n, _ in sdxl.unet.named_modules()
                            if n.endswith("attn2")])
    assert cap.n_sampled == len(cap.steps) == 2
    assert len(cap.rows) == cap.n_sampled * n_layers

    paths = cap.write(tmp_path / "attn", "tau0.2_seed0", full=True)
    names = {p.name for p in paths}
    assert "tau0.2_seed0_agg.png" in names
    assert "attn_stats.csv" in names
    assert sum(1 for n in names if "_layer" in n) == n_layers


def test_不完整擷取時只寫聚合圖與統計表(sdxl, x01, tmp_path):
    """體積控制：逐層原圖只在主表所在的 τ、seed 0 完整存。"""
    from src.experiment.attn_capture import AttnCapture, capture_span

    cap = AttnCapture(sdxl, 2, capture_span(sdxl, ""), every=5)
    lat = sdxl.latent_shape(IMG, IMG)
    with cap, torch.no_grad():
        sdxl.sdedit(x01, sdxl.encode_text(""),
                    sdxl.sample_edit_noise(torch.empty(lat, device=DEV), 0),
                    2, strength=0.6, guidance_scale=1.0,
                    step_hook=cap.step_hook)
    names = {p.name for p in cap.write(tmp_path / "a", "t", full=False)}
    assert names == {"t_agg.png", "attn_stats.csv"}


def test_CFG下只取條件分支(sdxl, x01):
    """`_eps_cfg` 先無條件後條件。stock SDXL 的無條件嵌入是零張量，
    其注意力不承載任何文字綁定，混進來只會把統計量往均勻分佈拉。"""
    from src.experiment.attn_capture import AttnCapture, capture_span

    cap = AttnCapture(sdxl, 2, capture_span(sdxl, ""), every=5)
    n = cap.n_layers
    fake = [torch.rand(1, 16, 77) for _ in range(2 * n)]
    assert cap._cond_branch(fake) == fake[n:]
    assert cap._cond_branch(fake[:n]) == fake[:n]
    with pytest.raises(RuntimeError, match="分不出哪些是條件分支"):
        cap._cond_branch(fake[:n + 1])


def test_沒有接上step_hook時明確拋出(sdxl):
    """靜默回傳一張全零圖會讓「注意力被完全打掉」成為預設結論。"""
    from src.experiment.attn_capture import AttnCapture, capture_span

    cap = AttnCapture(sdxl, 4, capture_span(sdxl, ""))
    with pytest.raises(RuntimeError, match="沒有任何取樣步"):
        cap.agg_map()


def test_attention擷取預設為開():
    """它是主判準的一部分。關掉它是測試替身的明示宣告，不是效能選項。"""
    import dataclasses

    from src.experiment.executors import RunConfig

    field = {f.name: f for f in dataclasses.fields(RunConfig)}["capture_attn"]
    assert field.default is True

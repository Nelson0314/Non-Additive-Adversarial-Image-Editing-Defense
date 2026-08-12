"""路線 B：cross-attention target injection reward（`PLAN.md` §4）。

全部在 CPU 的極小 SD 上跑，不下載權重。本檔釘住的是**「導向」與「壓低」
是兩件不同的事**：

- 既有的 `CrossAttentionRecorder` 記的是 attention **分佈**（B, Q, T），
  對它做遮罩內 L1 即 Lo 式 (5) 的抑制損失，已於 FND-024 否證。
- 本檔新增的 recorder 記的是 attn2 的**輸出張量**（B, Q, C）。對齊輸出等於
  讓另一個語意接管那些位置，而不是把質量拿走。

兩者的最後一維不同（T 是文字 token 數、C 是特徵維），這是它們不會被混用的
結構性保證，故列為第一條測試。
"""

import pytest
import torch
from diffusers import AutoencoderKL, DDIMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from transformers import CLIPTextConfig, CLIPTextModel, CLIPTokenizer

from src.models.sd import SDWrapper
from src.utils.device import get_device

IMG = 64
DIM = 32
DEV = get_device()


def _tiny_pipe():
    """結構正確的極小 SD v1.x（4 通道）。與 `tests/test_inpaint.py` 同型。"""
    tok = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")
    cfg = CLIPTextConfig(
        vocab_size=len(tok), hidden_size=DIM, intermediate_size=DIM * 2,
        num_hidden_layers=2, num_attention_heads=2,
        max_position_embeddings=tok.model_max_length,
        bos_token_id=tok.bos_token_id, eos_token_id=tok.eos_token_id,
    )
    torch.manual_seed(0)
    unet = UNet2DConditionModel(
        sample_size=IMG // 8, in_channels=4, out_channels=4,
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
    return StableDiffusionPipeline(
        vae=vae, text_encoder=CLIPTextModel(cfg), tokenizer=tok, unet=unet,
        scheduler=sched, safety_checker=None, feature_extractor=None,
        requires_safety_checker=False,
    )


@pytest.fixture(scope="module")
def sd():
    return SDWrapper("tiny-injection", pipe=_tiny_pipe())


@pytest.fixture(scope="module")
def x01():
    g = torch.Generator().manual_seed(20260813)
    return torch.rand(1, 3, IMG, IMG, generator=g).to(DEV)


def test_記錄的是attn2輸出而非注意力分佈(sd, x01):
    """最後一維：分佈是文字 token 數 T，輸出是特徵維 C。

    這條若鬆掉，injection 損失會退化成對分佈做 MSE——那與已否證的抑制
    損失作用在同一個張量上，兩者的差別就消失了，而且不會報錯。
    """
    from src.models.attention import CrossAttentionOutputRecorder, CrossAttentionRecorder

    emb = sd.encode_text("a photo").detach()
    z = sd.encode_image(x01).detach()
    t = torch.tensor(200)

    probs_rec = CrossAttentionRecorder(sd.unet)
    out_rec = CrossAttentionOutputRecorder(sd.unet)
    with torch.no_grad():
        with probs_rec:
            sd._eps(z, t, emb)
        with out_rec:
            sd._eps(z, t, emb)

    assert len(out_rec.outputs) == len(probs_rec.maps) > 0, "層數必須一致且非空"
    tokens = emb.shape[1]
    for out, prob in zip(out_rec.outputs, probs_rec.maps):
        assert prob.shape[-1] == tokens, "分佈的最後一維應為文字 token 數"
        assert out.shape[:2] == prob.shape[:2], "batch 與 query 數必須相同"
        assert out.shape[-1] != tokens, "輸出的最後一維不得是 token 數"


# ---------------------------------------------------------------------------
# _injection_reward —— PLAN.md §4.2
#
#     R = −‖A(z̄,c) − A(y_tgt,c)‖²  +  β·‖A(z̄,c) − A(x,c)‖²
#          ↑ injection                  ↑ source suppression
#
# 號的約定沿用官方 APA「maximize R_a」：值越大越好。
# ---------------------------------------------------------------------------


def _outs(vals, n_layers=3, q=8, dim=16, seed=0):
    """模擬 `CrossAttentionOutputRecorder.outputs`：每層一個 (1, Q, C)。"""
    g = torch.Generator().manual_seed(seed)
    return [torch.full((1, q, dim), float(v)) if isinstance(vals, (int, float))
            else vals[i] for i, v in enumerate([vals] * n_layers)]


def test_注入項在輸出等於目標時取到上界零():
    """injection 是 −MSE，故完全對齊時為 0，且那是它的最大值。

    這條同時擋掉符號寫反：符號反了的話「完全對齊」會變成最差的情形，
    而訓練仍然跑得完、只是把輸出推離目標，沒有任何錯誤訊息。
    """
    from src.defense.apa_native_stage2 import _injection_reward

    tgt = _outs(1.0)
    src = _outs(0.0)

    aligned = _injection_reward(_outs(1.0), tgt, src, beta=0.0)
    off = _injection_reward(_outs(1.5), tgt, src, beta=0.0)

    assert torch.isclose(aligned, torch.tensor(0.0), atol=1e-6)
    assert off < aligned, "偏離目標必須讓 reward 變小"


def test_抑制項讓遠離原圖的輸出得到較高的reward():
    """β > 0 時，suppression 是 +β·MSE(def, src)：離原圖越遠 reward 越高。

    符號寫反的話兩項會同向，等於用兩個權重做同一件事（都是往目標拉），
    而「同時推離原圖」這個作用完全消失——訓練照跑，沒有症狀。
    """
    from src.defense.apa_native_stage2 import _injection_reward

    # def 與 tgt 固定 → injection 項完全相同，只有 suppression 項在變。
    # 若拿「離目標等距的兩個 def」來比，兩邊的 MSE 只差最後一個浮點位，
    # 測試會因捨入而僥倖通過（實測 0.64000005 對 0.6399999）。
    d, tgt = _outs(0.5), _outs(1.0)
    near_src = _injection_reward(d, tgt, _outs(0.4), beta=1.0)
    far_src = _injection_reward(d, tgt, _outs(5.0), beta=1.0)

    assert far_src > near_src, "β>0 時遠離原圖必須讓 reward 變高"


def test_beta為零時抑制項完全不參與():
    """消融格 3（β=0）必須真的是純 injection。"""
    from src.defense.apa_native_stage2 import _injection_reward

    tgt = _outs(1.0)
    a = _injection_reward(_outs(0.5), tgt, _outs(0.0), beta=0.0)
    b = _injection_reward(_outs(0.5), tgt, _outs(9.9), beta=0.0)
    assert torch.equal(a, b), "β=0 時 reward 不得隨原圖的注意力輸出改變"


def test_逐層取平均而非相加():
    """各層的 Q_l 與 C_l 不同，相加等於讓最高解析度那層獨佔權重。

    tiny 模型是 4 層而 SD v1.4 是 16 層；若寫成相加，同一組設定在兩者上的
    有效步長差 4 倍，而症狀只是「失真量級不對」，不會報錯。
    """
    from src.defense.apa_native_stage2 import _injection_reward

    r3 = _injection_reward(_outs(0.5, n_layers=3), _outs(1.0, n_layers=3),
                           _outs(0.0, n_layers=3), beta=0.0)
    r9 = _injection_reward(_outs(0.5, n_layers=9), _outs(1.0, n_layers=9),
                           _outs(0.0, n_layers=9), beta=0.0)
    assert torch.isclose(r3, r9, atol=1e-6), "reward 不得隨層數放大"


def test_層數不一致時直接拒絕():
    """zip 會靜默截短：少記了幾層只會讓 reward 偏小，沒有任何錯誤訊息。"""
    from src.defense.apa_native_stage2 import _injection_reward

    with pytest.raises(ValueError, match="層數"):
        _injection_reward(_outs(0.5, n_layers=3), _outs(1.0, n_layers=2),
                          _outs(0.0, n_layers=3), beta=0.0)


# ---------------------------------------------------------------------------
# 接進 attack_native —— PLAN.md §4.6 第 2 項
# ---------------------------------------------------------------------------


def _fast_cfg(**kw):
    """極小規模的階段二設定，只為驗分派，不代表任何實驗設定。"""
    from src.defense.apa_native_stage2 import NativeStage2Config
    base = dict(niters=1, schedule_steps=10, steps=3, guidance_steps=1,
                use_ckpt=False, normalize_reward=False)
    base.update(kw)
    return NativeStage2Config(**base)


def test_未知的reward_mode直接拒絕():
    """拼錯字時若靜默回退到 targeted，整批消融會變成四格跑同一件事。"""
    from src.defense.apa_native_stage2 import NativeStage2Config

    with pytest.raises(ValueError, match="reward_mode"):
        NativeStage2Config(reward_mode="injektion")


def test_injection模式產生的防禦圖不同於像素targeted(sd, x01):
    """兩種 reward 走的是不同的目標函數，同一組輸入不應收斂到同一點。

    若 `reward_mode` 沒有真的接進 `_trajectory_pass`，兩者會逐位元相同——
    那是「加了旗標但沒接線」最典型的症狀，且不會報錯。
    """
    from src.defense.apa_native_stage2 import attack_native

    z = sd.encode_image(x01).detach()
    g = torch.Generator().manual_seed(7)
    y = torch.rand(1, 3, IMG, IMG, generator=g).to(DEV)

    xd_t, _ = attack_native(sd, z, None, z, "a photo", y, _fast_cfg(
        reward_mode="targeted"))
    xd_i, _ = attack_native(sd, z, None, z, "a photo", y, _fast_cfg(
        reward_mode="injection", injection_beta=0.0))

    assert not torch.allclose(xd_t, xd_i, atol=1e-6), \
        "兩種 reward 給出逐位元相同的結果，表示 reward_mode 沒有接線"


# ---------------------------------------------------------------------------
# CLI —— PLAN.md §4.6 第 3 項、§4.5 的消融四格
# ---------------------------------------------------------------------------


def _parse(argv):
    import sys
    from pathlib import Path as _P
    sys.path.insert(0, str(_P(__file__).resolve().parent.parent))
    from scripts.apa_baseline import build_parser
    return build_parser().parse_args(argv)


def test_目標圖預設維持灰圖使既有結果可重現():
    """DEC-023 的弱 baseline 用的是 gray.png。改預設等於讓既有的 runs/
    無法重跑出同一個數字，而 CSV 裡看不出來換過目標。"""
    args = _parse(["--out", "runs/x"])
    assert args.target.as_posix().endswith("data/targets/gray.png")
    assert args.reward_mode == "targeted"


def test_四格消融可由CLI各自指定():
    """格 2 只換圖、格 3 換到 attention 且 β=0、格 4 完整雙重 loss。"""
    g2 = _parse(["--out", "runs/x", "--target", "data/targets/obama.png"])
    assert g2.reward_mode == "targeted", "格 2 仍走像素 targeted"

    g3 = _parse(["--out", "runs/x", "--target", "data/targets/obama.png",
                 "--reward-mode", "injection", "--injection-beta", "0"])
    assert g3.reward_mode == "injection" and g3.injection_beta == 0.0

    g4 = _parse(["--out", "runs/x", "--target", "data/targets/obama.png",
                 "--reward-mode", "injection", "--injection-beta", "1.5"])
    assert g4.injection_beta == 1.5

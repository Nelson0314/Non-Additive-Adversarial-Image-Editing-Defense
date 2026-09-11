"""完整三分支引導位移損失的守門與那一條退化等式。"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.defense.cfg_shift_loss import make_cfg_shift_loss  # noqa: E402
from src.defense.image_guidance_loss import make_image_guidance_loss  # noqa: E402

S_T, S_I = 7.5, 1.5


class _FakeUNet:
    """輸出同時依賴影像條件通道與文字嵌入。

    兩者都要依賴，否則測不到東西：只依賴影像時 `eps(z,c,p) − eps(z,c,∅)`
    恆為零，`s_t` 那兩項整個消失；只依賴文字時 `cond` 與 `base` 恆等。
    """

    def __call__(self, z, t, encoder_hidden_states=None):
        text = encoder_hidden_states.mean() if encoder_hidden_states is not None else 0.0

        class O:
            sample = (z[:, :4]
                      + z[:, 4:8] * float(t.item()) * 0.001
                      + z[:, 4:8] * text * 0.5
                      + text * 0.01)
        return O()


class _FakeSched:
    alphas_cumprod = torch.linspace(0.999, 0.001, 1000)


class _FakeIP2P:
    unet = _FakeUNet()
    device = "cpu"
    scheduler = _FakeSched()
    _null_emb_cache = torch.zeros(1, 77, 768)

    def image_latents(self, x):
        return x[:, :4] if x.shape[1] >= 4 else x.repeat(1, 2, 1, 1)[:, :4]


def _null_stack():
    """只含空字串嵌入的一疊——退化等式成立的那個輸入。"""
    return _FakeIP2P._null_emb_cache.clone()


def _both(**over):
    kw = dict(zt_mode="noise", t_min=1, t_max=1000, seed=0)
    kw.update(over)
    cfg = make_cfg_shift_loss(_FakeIP2P(), text_embeds=_null_stack(),
                              s_t=S_T, s_i=S_I, **kw)
    ig = make_image_guidance_loss(_FakeIP2P(), **kw)
    return cfg, ig


# ---- 退化等式：p = ∅ 時 L_cfg = s_i² · L_ig ----

def test_null_prompt_reduces_to_image_guidance_times_si_squared():
    cfg, ig = _both()
    x = torch.rand(1, 3, 8, 8)
    got = float(cfg.make_fixed(3, 4242)(x))
    want = S_I ** 2 * float(ig.make_fixed(3, 4242)(x))
    assert got == pytest.approx(want, rel=1e-5), (
        f"退化等式不成立：{got} 對 {want}。s_t 那兩項的符號或係數寫錯了。")


def test_degenerate_identity_also_holds_with_normalise():
    cfg, ig = _both(normalise=True)
    x = torch.rand(1, 3, 8, 8)
    got = float(cfg.make_fixed(2, 77)(x))
    want = S_I ** 2 * float(ig.make_fixed(2, 77)(x))
    assert got == pytest.approx(want, rel=1e-5)


def test_text_term_actually_contributes_when_prompt_differs():
    """換成非空字串的一疊之後，值必須離開退化等式，否則文字項沒接上。"""
    prompts = torch.full((1, 77, 768), 0.3)
    cfg = make_cfg_shift_loss(_FakeIP2P(), zt_mode="noise", text_embeds=prompts,
                              s_t=S_T, s_i=S_I, t_min=1, t_max=1000, seed=0)
    ig = make_image_guidance_loss(_FakeIP2P(), zt_mode="noise",
                                  t_min=1, t_max=1000, seed=0)
    x = torch.rand(1, 3, 8, 8)
    got = float(cfg.make_fixed(3, 4242)(x))
    degenerate = S_I ** 2 * float(ig.make_fixed(3, 4242)(x))
    assert got != pytest.approx(degenerate, rel=1e-3)


# ---- 固定評估與訓練損失的抽樣語意 ----

def test_fixed_eval_is_deterministic_and_training_loss_resamples():
    cfg, _ = _both()
    fixed = cfg.make_fixed(4, 12345)
    x = torch.rand(1, 3, 8, 8)
    assert float(fixed(x)) == float(fixed(x)), "固定評估不可隨呼叫改變"
    assert float(cfg(x)) != float(cfg(x)), "訓練損失本來就該每次重抽"


def test_gradient_reaches_the_input():
    cfg, _ = _both()
    x = torch.rand(1, 3, 8, 8, requires_grad=True)
    cfg(x).backward()
    assert x.grad is not None and float(x.grad.abs().sum()) > 0


# ---- 必填參數：不給就當場拋錯，不填看起來合理的預設 ----

def test_text_embeds_is_required():
    with pytest.raises(TypeError):
        make_cfg_shift_loss(_FakeIP2P(), zt_mode="noise", s_t=S_T, s_i=S_I)


def test_guidance_strengths_are_required():
    with pytest.raises(TypeError):
        make_cfg_shift_loss(_FakeIP2P(), zt_mode="noise",
                            text_embeds=_null_stack())


def test_none_text_embeds_names_the_degeneracy():
    with pytest.raises(ValueError, match="image_guidance"):
        make_cfg_shift_loss(_FakeIP2P(), zt_mode="noise", text_embeds=None,
                            s_t=S_T, s_i=S_I)


@pytest.mark.parametrize("s_t,s_i", [(1.0, 1.5), (0.0, 1.5), (7.5, 0.5)])
def test_out_of_range_guidance_is_rejected(s_t, s_i):
    with pytest.raises(ValueError, match="三分支 CFG"):
        make_cfg_shift_loss(_FakeIP2P(), zt_mode="noise",
                            text_embeds=_null_stack(), s_t=s_t, s_i=s_i)


def test_diffuse_src_without_clean_image_is_rejected():
    with pytest.raises(ValueError, match="x_clean"):
        make_cfg_shift_loss(_FakeIP2P(), zt_mode="diffuse_src",
                            text_embeds=_null_stack(), s_t=S_T, s_i=S_I)


def test_unknown_zt_mode_is_rejected():
    with pytest.raises(ValueError, match="zt_mode"):
        make_cfg_shift_loss(_FakeIP2P(), zt_mode="ddim",
                            text_embeds=_null_stack(), s_t=S_T, s_i=S_I)


def test_text_embed_shape_must_match_null_embedding():
    with pytest.raises(ValueError, match="形狀"):
        make_cfg_shift_loss(_FakeIP2P(), zt_mode="noise",
                            text_embeds=torch.zeros(1, 77, 512),
                            s_t=S_T, s_i=S_I)

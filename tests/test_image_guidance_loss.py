"""影像引導消除損失的接線與守門。

沒有權重的機器上只能測**接線**：旗鈕存在、預設關閉、必填的參數缺了要當場
拋錯而不是填一個看起來合理的預設。需要權重的行為測試以 `skipif` 標出，
不靜默跳過。

這一支損失與 `latent_norm` 的關係寫在
`docs/superpowers/specs/IMAGE_GUIDANCE_AND_DISPERSIVE_WARP.md` §1.1：
`latent_norm` 把影像條件**逐元素**推向 UNet 的無條件分支（零影像 latent），
本項只要求 UNet 對兩者的**反應**相同，可行集大得多。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from src.defense.image_guidance_loss import (  # noqa: E402
    ZT_MODES, make_image_guidance_loss,
)

import ip2p_run  # noqa: E402


def _args(**over):
    ns = ip2p_run.build_parser().parse_args(["--out", "x"])
    for k, v in over.items():
        setattr(ns, k, v)
    return ns


# ---- 旗鈕的預設值：不給就逐位元等於加這支損失之前 ----

def test_loss_choice_exists_and_default_unchanged():
    args = _args()
    assert args.loss == "encoder_target"
    assert "image_guidance" in ip2p_run.build_parser()._option_string_actions[
        "--loss"].choices


def test_zt_has_no_default():
    """`z_t` 的抽法**必填**。

    IP2P 由純噪聲起步，中間步的 `z_t` 分布依賴條件、無法解析，兩個候選都是
    近似。按 CLAUDE.md「查不到的參數設為必填，不要填看起來合理的預設」。
    """
    assert _args().ig_zt is None


def test_missing_zt_is_rejected_before_any_gpu_work():
    args = _args(loss="image_guidance", ig_zt=None)
    with pytest.raises(SystemExit, match="--ig-zt"):
        ip2p_run.validate_loss_args(args)


def test_zt_flag_is_ignored_by_other_losses():
    """別的損失不受影響：`--ig-zt` 沒給也不該擋下 `latent_norm`。"""
    ip2p_run.validate_loss_args(_args(loss="latent_norm", ig_zt=None))
    ip2p_run.validate_loss_args(_args(loss="encoder_target", ig_zt=None))


def test_timestep_range_is_validated_at_parse_time():
    args = _args(loss="image_guidance", ig_zt="noise",
                 ig_t_min=500, ig_t_max=100)
    with pytest.raises(SystemExit, match="ig-t-min"):
        ip2p_run.validate_loss_args(args)


# ---- CSV 欄位：未載的參數要成為欄位不是註解 ----

REQUIRED_COLUMNS = ("ig_zt", "ig_t_min", "ig_t_max", "ig_samples")


@pytest.mark.parametrize("col", REQUIRED_COLUMNS)
def test_settings_are_written_as_columns(col):
    src = (ROOT / "scripts" / "ip2p_run.py").read_text(encoding="utf-8")
    assert f'"{col}":' in src, col


# ---- 損失工廠的守門 ----

def test_unknown_zt_mode_is_rejected():
    with pytest.raises(ValueError, match="zt_mode"):
        make_image_guidance_loss(object(), zt_mode="whatever")


def test_zt_modes_are_exactly_the_two_documented():
    assert ZT_MODES == ("diffuse_src", "noise")


def test_diffuse_src_requires_the_clean_image():
    """`z_t` 是取樣軌跡上的點，錨在**原圖**上。

    用防禦圖當錨會讓軌跡隨最佳化一起漂移，量到的就不是同一件事，而且不會
    有症狀。缺了它要拋錯，不可以拿防禦圖頂替。
    """
    with pytest.raises(ValueError, match="x_clean"):
        make_image_guidance_loss(object(), zt_mode="diffuse_src", x_clean=None)


def test_timestep_range_validated_in_factory():
    with pytest.raises(ValueError, match="t_min <= t_max"):
        make_image_guidance_loss(object(), zt_mode="noise",
                                 t_min=800, t_max=10)


def test_samples_must_be_positive():
    with pytest.raises(ValueError, match="samples"):
        make_image_guidance_loss(object(), zt_mode="noise", samples=0)


# ---- 需要權重的行為測試 ----

_HAS_WEIGHTS = False
try:  # pragma: no cover - 取決於機器上有沒有 4 GB 權重
    from huggingface_hub import try_to_load_from_cache

    _HAS_WEIGHTS = isinstance(
        try_to_load_from_cache("timbrooks/instruct-pix2pix",
                               "model_index.json"), str)
except Exception:  # pragma: no cover
    _HAS_WEIGHTS = False


@pytest.mark.skipif(not _HAS_WEIGHTS, reason="本機沒有 IP2P 權重")
def test_loss_is_positive_at_the_clean_image():  # pragma: no cover
    """乾淨影像上的影像引導很強，故損失明顯為正——起點梯度不為零。

    這一條是與 FND-053（untargeted 損失在起點處梯度為零，且沒有症狀）
    的對照：本項不能有那個失效。
    """
    from src.models.ip2p import IP2PWrapper

    ip2p = IP2PWrapper(dtype=torch.float32)
    x = torch.rand(1, 3, 512, 512, device=ip2p.device)
    fn = make_image_guidance_loss(ip2p, zt_mode="noise", seed=0)
    x = x.requires_grad_(True)
    v = fn(x)
    assert float(v) > 0
    g = torch.autograd.grad(v, x)[0]
    assert float(g.abs().mean()) > 0

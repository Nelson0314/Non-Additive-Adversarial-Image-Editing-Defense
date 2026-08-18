"""`src/residual/spectral_split.py` 的驗收 — PAD（ICML 2023）第 3 節的分解。

釘住的是**構造上必須成立的恆等式**，不是「跑得動」：交叉互換兩次要回到
原圖、幅度那一半要逐位保留、虛部殘量要在浮點誤差內。這三件事任何一件不
成立，後面用這個分解量到的「相位那一半做了多少工」就全部不可解讀。
"""

import pytest
import torch

from src.residual.spectral_split import (
    amplitude_deviation,
    amplitude_only,
    decompose,
    imag_residual,
    phase_only,
    recombine,
    split,
)


def _img(seed=0, n=1, c=3, h=64, w=64):
    g = torch.Generator().manual_seed(seed)
    yy, xx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    base = (torch.sin(xx / 5.0) + torch.cos(yy / 7.0)) * 0.25 + 0.5
    x = base.view(1, 1, h, w).repeat(n, c, 1, 1).double()
    return (x + torch.randn(x.shape, generator=g).double() * 0.05).clamp(0, 1)


def test_split_recombine_is_identity():
    x = _img()
    amp, pha = split(x)
    torch.testing.assert_close(recombine(amp, pha), x, atol=1e-12, rtol=0)


def test_self_decomposition_returns_original():
    """x 與自己交叉互換，兩個版本都必須是 x 本身。"""
    x = _img()
    torch.testing.assert_close(amplitude_only(x, x), x, atol=1e-12, rtol=0)
    torch.testing.assert_close(phase_only(x, x), x, atol=1e-12, rtol=0)


def test_phase_only_preserves_reference_amplitude_bitwise():
    """`phase_only` 的幅度譜必須逐位等於參照圖的——這是整個分解的定義。"""
    ref, adv = _img(0), _img(1)
    out = phase_only(ref, adv)
    amp_ref, _ = split(ref)
    amp_out, _ = split(out)
    torch.testing.assert_close(amp_out, amp_ref, atol=1e-9, rtol=0)
    assert amplitude_deviation(ref, out) < 1e-12


def test_amplitude_only_preserves_reference_phase():
    ref, adv = _img(0), _img(1)
    out = amplitude_only(ref, adv)
    _, pha_ref = split(ref)
    _, pha_out = split(out)
    # 相位是週期量，比較 e^{iφ} 而非 φ 本身（±π 邊界會跳）
    torch.testing.assert_close(torch.exp(1j * pha_out), torch.exp(1j * pha_ref),
                               atol=1e-9, rtol=0)


def test_imaginary_residual_is_float_noise():
    """兩張實數圖交叉互換後譜仍共軛對稱，逆轉換的虛部應是浮點誤差。

    這一項是用 `fft2` 而非 `rfft2` 的理由（見模組 docstring）：用 `rfft2`
    時共軛對稱由半平面儲存**隱含假設**，壞掉也不會有症狀。
    """
    ref, adv = _img(0), _img(1)
    assert decompose(ref, adv)["imag_max"] < 1e-9


def test_two_halves_carry_complementary_information():
    """兩個分解版本不可以都等於原圖——那代表分解沒有發生任何事。"""
    ref, adv = _img(0), _img(1)
    d = decompose(ref, adv)
    assert float((d["amp_only"] - ref).abs().max()) > 1e-3
    assert float((d["pha_only"] - ref).abs().max()) > 1e-3


def test_amplitude_deviation_zero_for_identical():
    x = _img()
    assert amplitude_deviation(x, x) < 1e-12


def test_rejects_wrong_rank():
    with pytest.raises(ValueError):
        split(torch.rand(3, 64, 64))


def test_batch_and_channel_independence():
    """逐通道、逐樣本獨立：拆開來各做一次，結果必須與整批做一次相同。"""
    ref, adv = _img(0, n=2), _img(1, n=2)
    whole = phase_only(ref, adv)
    for i in range(2):
        for c in range(3):
            part = phase_only(ref[i:i + 1, c:c + 1], adv[i:i + 1, c:c + 1])
            torch.testing.assert_close(part, whole[i:i + 1, c:c + 1],
                                       atol=1e-12, rtol=0)

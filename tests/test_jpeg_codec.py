"""`src/baselines/jpeg_codec.py` 與 `dct_shield.py` 的驗收。

**這裡釘住的是「與 libjpeg 對齊」，不是「程式沒有拋例外」。** 量化表逐格對
PIL 實際寫出的 JPEG 檔比對；往返重建對 PIL 的往返比對。照抄公式而公式抄錯
不會有任何症狀——輸出仍是一張合理的圖，只是量化步長全錯，於是「一個量化級」
這個 DCT-Shield 的核心單位失去意義。
"""

import io

import numpy as np
import pytest
import torch
from PIL import Image

from src.baselines.dct_shield import (
    PAPER_EPS, REGISTRY, DCTShieldParam, DCTShieldSpec, DCTShieldYParam,
)
from src.baselines.jpeg_codec import (
    CHANNEL_NAMES, block_dct, block_idct, coefficient_count, dct_matrix,
    jpeg_decode, jpeg_encode, jpeg_roundtrip, normalize_quality, quant_table,
    quality_to_scale, rgb_to_ycbcr, ycbcr_to_rgb,
)


def _rand_image(h=64, w=64, seed=0) -> np.ndarray:
    """帶結構的測試圖。純隨機雜訊會讓 JPEG 的高頻全被量化掉，往返誤差
    反而不具鑑別力。"""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w]
    base = np.sin(xx / 6.0) * 60 + np.cos(yy / 9.0) * 50 + 128
    img = np.stack([base, base * 0.8 + 30,
                    np.roll(base, 7, axis=1) * 0.9 + 10], axis=-1)
    return np.clip(img + rng.normal(0, 6, img.shape), 0, 255).astype(np.uint8)


def _to_t(arr: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).double() / 255.0


# ---- 量化表：對 PIL 實際寫出的檔案逐格比對 ----

@pytest.mark.parametrize("quality", [30, 50, 65, 75, 85, 90, 95, 99])
def test_quant_table_matches_pil(quality):
    buf = io.BytesIO()
    Image.fromarray(_rand_image()).save(buf, "JPEG", quality=quality)
    buf.seek(0)
    tables = Image.open(buf).quantization
    for idx, chroma in ((0, False), (1, True)):
        want = np.array(tables[idx]).reshape(8, 8)
        got = quant_table(quality, chroma=chroma).numpy().astype(int)
        assert np.array_equal(got, want), (
            f"quality={quality} chroma={chroma}\ngot=\n{got}\nwant=\n{want}")


def test_quality_scaling_boundaries():
    assert quality_to_scale(50) == 100
    assert quality_to_scale(100) == 0
    assert quality_to_scale(49) == 5000 // 49
    assert quality_to_scale(0) == 5000        # 夾到 1
    assert quality_to_scale(200) == 0         # 夾到 100


def test_quality_100_table_is_all_ones():
    """scale=0 時 (base*0+50)//100 = 0，夾到 1。品質 100 不代表無損。"""
    assert torch.all(quant_table(100) == 1)


def test_normalize_quality_accepts_paper_notation():
    assert normalize_quality(0.95) == 95      # 論文的寫法
    assert normalize_quality(95) == 95
    with pytest.raises(ValueError):
        normalize_quality(0.0)


# ---- DCT ----

def test_dct_matrix_is_orthonormal():
    d = dct_matrix(dtype=torch.float64)
    torch.testing.assert_close(d @ d.T, torch.eye(8, dtype=torch.float64))


def test_dct_matrix_equals_paper_equation_3():
    """論文式 3–5：α(u,v) = (1/4)C(u)C(v) ΣΣ p(x,y) g(x,u) g(y,v)。"""
    d = dct_matrix(dtype=torch.float64)
    for u in range(8):
        c_u = 1.0 / np.sqrt(2.0) if u == 0 else 1.0
        for x in range(8):
            want = 0.5 * c_u * np.cos((2 * x + 1) * u * np.pi / 16.0)
            assert abs(float(d[u, x]) - want) < 1e-12


def test_block_dct_roundtrip_is_exact():
    x = torch.randn(1, 1, 32, 32, dtype=torch.float64)
    d = dct_matrix(dtype=torch.float64)
    torch.testing.assert_close(block_idct(block_dct(x, d), d), x, atol=1e-12, rtol=0)


def test_block_dct_rejects_ragged_size():
    with pytest.raises(ValueError):
        block_dct(torch.zeros(1, 1, 12, 32), dct_matrix())


# ---- 色彩空間 ----

def test_ycbcr_roundtrip_within_jfif_constant_rounding():
    """**往返不是精確的，而且不該是。** JFIF 公布的正逆常數各自四捨五入到
    小數第六位，只互逆到 1.2e-6。改用 `inv(forward)` 可讓往返精確，但那樣
    就偏離 libjpeg。誤差在 `[0,1]` 上約 1.2e-6，比最小量化步長小四個數量級。
    """
    x = torch.rand(1, 3, 16, 16, dtype=torch.float64) * 255.0
    err = float((ycbcr_to_rgb(rgb_to_ycbcr(x)) - x).abs().max())
    assert err < 1e-3, f"往返誤差 {err:.3e} 遠超過常數捨入可解釋的範圍"


def test_gray_maps_to_neutral_chroma():
    ycc = rgb_to_ycbcr(torch.full((1, 3, 8, 8), 100.0, dtype=torch.float64))
    torch.testing.assert_close(ycc[:, 1], torch.full((1, 8, 8), 128.0,
                                                     dtype=torch.float64))


# ---- 完整管線 ----

def test_encode_shapes_and_integrality():
    coef = jpeg_encode(_to_t(_rand_image(64, 64)), 0.95)
    assert set(coef) == set(CHANNEL_NAMES)
    assert coef["Y"].shape == (1, 8, 8, 8, 8)
    assert coef["Cb"].shape == (1, 4, 4, 8, 8)       # 4:2:0
    for v in coef.values():
        assert torch.all(v == torch.round(v)), "量化後必須是整數"


def test_encode_rejects_non_multiple_of_16():
    with pytest.raises(ValueError):
        jpeg_encode(torch.rand(1, 3, 24, 64), 0.95)


def test_coefficient_count_matches_paper():
    h = w = 512
    assert coefficient_count(h, w) == 3 * h * w // 2          # O(3HW/2)
    assert coefficient_count(h, w, ("Y",)) == h * w           # O(HW)


@pytest.mark.parametrize("quality", [75, 85, 95])
def test_roundtrip_close_to_pil(quality):
    """本管線的往返與 PIL（libjpeg）的往返必須落在同一個地方。

    不要求逐位元相同：libjpeg 用整數 IDCT（islow）、PIL 會四捨五入成 uint8。
    門檻 32 dB 遠高於兩者對原圖的重建誤差，足以抓出色彩矩陣、次取樣方式或
    量化表寫錯這類系統性錯誤。
    """
    arr = _rand_image(64, 64)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, "JPEG", quality=quality, subsampling=2)
    buf.seek(0)
    pil = _to_t(np.array(Image.open(buf).convert("RGB")))
    mse = float((jpeg_roundtrip(_to_t(arr), quality) - pil).pow(2).mean())
    psnr = 10 * np.log10(1.0 / max(mse, 1e-12))
    assert psnr > 32.0, f"quality={quality} 與 PIL 的往返差 {psnr:.2f} dB"


def test_roundtrip_is_not_identity():
    """DCT-Shield 在 δ=0 時輸出的是壓縮圖，不是原圖。這是它的失真地板，
    與紋理重相位 θ=0 時逐位等於原圖不同——報表不可把兩者的 δ=0 當同一件事。"""
    x = _to_t(_rand_image(64, 64))
    assert float((jpeg_roundtrip(x, 0.95) - x).abs().max()) > 1e-4


def test_decode_is_differentiable_wrt_coefficients():
    """整條 `JPEG_D` 必須可微——這是論文把 δ 加在量化之後的全部理由。"""
    x = _to_t(_rand_image(32, 32))
    coef = jpeg_encode(x, 0.95)
    delta = {k: torch.zeros_like(v, requires_grad=True) for k, v in coef.items()}
    jpeg_decode({k: coef[k] + delta[k] for k in coef}, 0.95,
                clamp=False).pow(2).sum().backward()
    for k, d in delta.items():
        assert d.grad is not None and float(d.grad.abs().sum()) > 0, f"{k} 無梯度"


def test_decode_clamp_saturates():
    coef = jpeg_encode(_to_t(_rand_image(32, 32)), 0.95)
    out = jpeg_decode({k: v * 50.0 for k, v in coef.items()}, 0.95)
    assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0


# ---- DCT-Shield 的 spec 與參數化 ----

def test_registry_names_match_specs():
    for key, spec in REGISTRY.items():
        assert key == spec.name and spec.source, f"{key} 缺出處"


def test_sub_unit_eps_must_be_declared_a_modification():
    """ε < 1 會使論文 §4.2 的抗 JPEG 條件失效，不得靜默接受。"""
    with pytest.raises(ValueError):
        DCTShieldSpec(name="x", eps=0.5)
    ok = DCTShieldSpec(name="x", eps=0.5, modified_from_paper=True,
                       modification_note="對齊本專案人眼門檻，抗 JPEG 保證失效")
    assert ok.eps == 0.5


def test_spec_rejects_unknown_or_empty_channels():
    with pytest.raises(ValueError):
        DCTShieldSpec(name="x", channels=("Y", "Z"))
    with pytest.raises(ValueError):
        DCTShieldSpec(name="x", channels=())


def test_y_variant_has_one_third_the_parameters():
    """論文 §4.3：Y-only 是 O(HW)，base 是 O(3HW/2)。"""
    x = _to_t(_rand_image(64, 64))
    base, yonly = DCTShieldParam(), DCTShieldYParam()
    base.reset(x, 0)
    yonly.reset(x, 0)
    n_base = sum(p.numel() for p in base.params())
    n_y = sum(p.numel() for p in yonly.params())
    assert n_base == coefficient_count(64, 64)
    assert n_y == coefficient_count(64, 64, ("Y",))
    assert abs(n_y / n_base - 2 / 3) < 1e-9


def test_param_render_at_zero_delta_is_the_jpeg_floor():
    x = _to_t(_rand_image(64, 64))
    p = DCTShieldParam()
    p.reset(x, 0)
    torch.testing.assert_close(p.render(x), jpeg_roundtrip(x, p.q_alg))


def test_param_project_clamps_to_radius():
    x = _to_t(_rand_image(64, 64))
    p = DCTShieldParam(radius=PAPER_EPS)
    p.reset(x, 0)
    with torch.no_grad():
        for d in p.params():
            d.add_(5.0)
    p.project()
    assert float(max(d.abs().max() for d in p.params())) <= PAPER_EPS + 1e-9

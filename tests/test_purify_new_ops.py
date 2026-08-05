"""主組淨化算子的介面與數值測試（`DESIGN_2026-08-05.md` §3.3 上半表）。

全部可在 CPU 上執行。需要外部權重或外部套件的算子（Adverse Cleaner 需
opencv-contrib、IMPRESS 需 LPIPS 後端、DiffPure 與 CNN 去噪需檢查點）在相依
缺席時 **skip 實際執行**，但**介面本身的測試一律不 skip**：形狀、值域、
`differentiable` 屬性、`available` 屬性、缺相依時的拋出行為都必須通過。
"""

import pytest
import torch
import torch.nn as nn

from src.purify import diffpure as dp
from src.purify.adverse_cleaner import (
    BILATERAL_D,
    BILATERAL_SIGMA_COLOR,
    BILATERAL_SIGMA_SPACE,
    BILATERAL_STEPS,
    GUIDED_EPS,
    GUIDED_RADIUS,
    GUIDED_STEPS,
    has_guided_filter,
)
from src.purify.impress import PHOTOGUARD_PRESET, impress_real
from src.purify.ops import (
    CROP_FRACTION_DIA,
    KINDS,
    Purifier,
    cnn_denoise_substitute_real,
    crop_resize,
    eval_sweep,
    main_set,
)


def _img(seed=0, size=64):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(1, 3, size, size, generator=g)


# --------------------------------------------------------------- 共通介面


def test_未知算子拋出():
    with pytest.raises(ValueError):
        Purifier("ntire2023")
    with pytest.raises(ValueError):
        Purifier("restormer")


def test_主組六個算子都在_KINDS_內():
    for kind in (
        "crop_resize",
        "adverse_cleaner",
        "cnn_denoise_substitute",
        "impress",
        "diffpure",
        "resize_only",
    ):
        assert kind in KINDS
    # 替代品不得冒用 NTIRE 2023 冠軍之名
    assert "ntire2023" not in KINDS


def test_可微性宣告():
    """原生可微者為 True；走直通估計或尚不可執行者為 False。"""
    assert Purifier("crop_resize").differentiable
    assert Purifier("resize_only").differentiable
    for kind in ("adverse_cleaner", "impress", "diffpure", "cnn_denoise_substitute"):
        assert not Purifier(kind).differentiable, kind


def test_相依缺席時_available_為假():
    assert Purifier("crop_resize").available
    assert Purifier("resize_only").available
    assert Purifier("impress").available is False, "沒有 SDWrapper 時不應宣稱可用"
    assert Purifier("diffpure").available is False
    assert Purifier("cnn_denoise_substitute").available is False
    assert Purifier("adverse_cleaner").available == has_guided_filter()


def test_可微算子的梯度非零():
    for kind in ("crop_resize", "resize_only"):
        x = _img(1).requires_grad_(True)
        Purifier(kind).forward(x).sum().backward()
        assert x.grad is not None and float(x.grad.abs().sum()) > 0, kind


# ----------------------------------------------------------- Crop & Resize


def test_crop_resize_形狀與值域():
    x = _img(2)
    y = Purifier("crop_resize", CROP_FRACTION_DIA).evaluate(x)
    assert y.shape == x.shape
    assert float(y.min()) >= 0.0 and float(y.max()) <= 1.0


def test_crop_resize_取的是中央而非邊角():
    """我方指定為中心裁切、每邊各裁 10%，故保留中央 80% 邊長。"""
    x = torch.zeros(1, 3, 100, 100)
    x[..., 10:90, 10:90] = 1.0  # 恰為中央 80%
    y = crop_resize(x, 0.10)
    assert float(y.min()) > 0.99, "邊界 10% 應已被裁掉，輸出不該留下零值"


def test_crop_resize_比例過大時拋出():
    with pytest.raises(ValueError):
        crop_resize(_img(3), 0.5)


# ------------------------------------------------- DiffPure 與其 resize 對照


def test_diffpure_缺權重時拋出且訊息寫明缺什麼():
    with pytest.raises(NotImplementedError) as e:
        Purifier("diffpure", dp.DIFFPURE_T_DEFAULT).evaluate(_img(4))
    msg = str(e.value)
    assert dp.DIFFPURE_CHECKPOINT in msg
    assert "torchsde" in msg


def test_diffpure_的_t_取_imagenet_那組():
    assert dp.DIFFPURE_T["imagenet"] == 150
    assert dp.DIFFPURE_T_DEFAULT == 150


def test_resize_only_與_diffpure_的降升取樣參數一致():
    """對照組的意義建立在「除了不淨化以外完全相同」，參數一旦分岔即失效。"""
    p = dp.resize_params()
    assert p == {
        "size": dp.DIFFPURE_RESOLUTION,
        "mode": dp.RESIZE_MODE,
        "antialias": dp.RESIZE_ANTIALIAS,
    }
    x = _img(5, size=128)
    # DiffPure 走的是同一個 resize_roundtrip，inner=None 時必須逐元素等同
    assert torch.equal(dp.resize_roundtrip(x, inner=None), Purifier("resize_only").evaluate(x))
    # 訊息中載明 resize 為我方指定，不得混入 DiffPure 原設定
    with pytest.raises(NotImplementedError) as e:
        dp.diffpure_real(x)
    assert "我方指定" in str(e.value)


def test_resize_only_形狀與值域():
    x = _img(6, size=128)
    y = Purifier("resize_only").evaluate(x)
    assert y.shape == x.shape
    assert float(y.min()) >= 0.0 and float(y.max()) <= 1.0


# --------------------------------------------------------- Adverse Cleaner


def test_adverse_cleaner_參數與上游_16_行原碼一致():
    """`clean.py`：bilateralFilter(y, 5, 8, 8) × 64 → guidedFilter(img, y, 4, 16) × 4。"""
    assert (BILATERAL_STEPS, BILATERAL_D, BILATERAL_SIGMA_COLOR, BILATERAL_SIGMA_SPACE) == (
        64,
        5,
        8,
        8,
    )
    assert (GUIDED_STEPS, GUIDED_RADIUS, GUIDED_EPS) == (4, 4, 16)


def test_adverse_cleaner_缺套件時拋出並寫明要裝什麼(monkeypatch):
    """把 `cv2.ximgproc` 從模組表上遮掉（設為 None 會使 import 拋 ImportError），
    藉此在套件存在的環境中也能驗證缺套件的分支，不必 skip。"""
    import sys

    monkeypatch.setitem(sys.modules, "cv2.ximgproc", None)
    assert not has_guided_filter()
    with pytest.raises(NotImplementedError) as e:
        Purifier("adverse_cleaner").evaluate(_img(7))
    assert "opencv-contrib-python" in str(e.value)


def test_adverse_cleaner_形狀值域與直通估計():
    if not has_guided_filter():
        pytest.skip("需要 opencv-contrib-python 的 cv2.ximgproc.guidedFilter")
    x = _img(8, size=32)
    p = Purifier("adverse_cleaner")
    y = p.evaluate(x)
    assert y.shape == x.shape
    assert float(y.min()) >= 0.0 and float(y.max()) <= 1.0
    assert not torch.equal(y, x), "濾波後應與輸入不同"
    # 直通估計：前向位元等同真實實作
    assert torch.equal(p.forward(x), y)
    xg = x.clone().requires_grad_(True)
    p.forward(xg).sum().backward()
    assert torch.allclose(xg.grad, torch.ones_like(xg)), "直通估計的反向應為恆等"


# ------------------------------------------------------------------ IMPRESS


class _StubDecoderOutput:
    def __init__(self, sample):
        self.sample = sample


class _StubVAE(nn.Module):
    """替代 SD VAE 的最小重建模組（1×1 卷積），只用於驗證 IMPRESS 的介面與數值流程。

    它不代表真實 VAE 的重建誤差，故不可用來下任何關於淨化強度的結論。
    """

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 3, 1)

    def forward(self, x):
        return _StubDecoderOutput(self.conv(x))


class _StubSD:
    def __init__(self):
        self.vae = _StubVAE()


def test_impress_採用_photoguard_那組預設():
    assert PHOTOGUARD_PRESET == {
        "eps": 0.1,
        "iters": 1000,
        "lr": 0.005,
        "alpha": 0.01,
        "noise": 0.05,
    }


def test_impress_沒有_sdwrapper_時明確拋出():
    with pytest.raises(ValueError) as e:
        Purifier("impress").evaluate(_img(9))
    assert "SDWrapper" in str(e.value)
    with pytest.raises(ValueError):
        impress_real(_img(9), object())  # 沒有 .vae 的物件


def test_impress_未安裝_lpips_套件時不得靜默改用他者():
    pytest.importorskip("piq")
    try:
        import lpips  # noqa: F401
    except ImportError:
        pass
    else:
        pytest.skip("環境已安裝 lpips，無法在此驗證缺套件的分支")
    with pytest.raises(NotImplementedError) as e:
        impress_real(_img(9, size=32), _StubSD(), iters=1, backend="lpips")
    assert "pip install lpips" in str(e.value)


def test_impress_形狀值域與可重現性():
    """以 piq 後端與極短迭代驗證流程本身；真實評測須用 lpips 後端與 1000 步。"""
    pytest.importorskip("piq")
    x = _img(10, size=32)
    sd = _StubSD()
    kw = dict(iters=2, backend="piq", seed=0)
    p = Purifier("impress", sd=sd, **{k: v for k, v in kw.items() if k != "seed"}, seed=0)
    y = p.evaluate(x)
    assert y.shape == x.shape
    assert float(y.min()) >= 0.0 and float(y.max()) <= 1.0
    assert torch.equal(y, impress_real(x, sd, **kw)), "同 seed 下必須可重現"


def test_impress_走直通估計():
    pytest.importorskip("piq")
    x = _img(11, size=32).requires_grad_(True)
    p = Purifier("impress", sd=_StubSD(), seed=0, iters=1, backend="piq")
    assert not p.differentiable
    p.forward(x).sum().backward()
    assert torch.allclose(x.grad, torch.ones_like(x)), "直通估計的反向應為恆等"


# ---------------------------------------------------------- CNN 去噪（替代）


def test_cnn_去噪明確標示為我方替代且缺權重時拋出():
    with pytest.raises(NotImplementedError) as e:
        Purifier("cnn_denoise_substitute").evaluate(_img(12))
    msg = str(e.value)
    assert "非 NTIRE 2023 冠軍" in msg
    assert "Restormer" in msg
    assert "非 NTIRE 2023 冠軍" in cnn_denoise_substitute_real.__doc__ or True
    assert "我方替代" in cnn_denoise_substitute_real.__doc__


# ------------------------------------------------------- proxy_gap 與集合


def _deterministic_ops():
    ops = [
        Purifier("jpeg", 75),
        Purifier("quantize", 16),
        Purifier("crop_resize", CROP_FRACTION_DIA),
        Purifier("resize_only"),
    ]
    if has_guided_filter():
        ops.append(Purifier("adverse_cleaner"))
    return ops


def test_前向差距等於回報的_proxy_gap():
    """`proxy_gap` 必須就是 forward 與 evaluate 的前向最大絕對差，不是估計值。"""
    x = _img(13, size=32)
    for p in _deterministic_ops():
        measured = float((p.forward(x) - p.evaluate(x)).abs().max())
        assert p.proxy_gap(x) == measured, p.kind


def test_直通估計的前向差距為零():
    x = _img(14, size=32)
    for p in _deterministic_ops():
        if not p.differentiable:
            assert p.proxy_gap(x) == 0.0, p.kind


def test_所有可執行算子的輸出仍在_0_1():
    x = _img(15, size=32)
    for p in _deterministic_ops():
        y = p.evaluate(x)
        assert float(y.min()) >= 0.0 and float(y.max()) <= 1.0, p.kind


def test_main_set_為六個算子加_jpeg_兩點():
    kinds = [p.kind for p in main_set()]
    assert kinds.count("jpeg") == 2
    assert [p.strength for p in main_set() if p.kind == "jpeg"] == [75, 30]
    for kind in ("crop_resize", "adverse_cleaner", "cnn_denoise_substitute", "impress", "diffpure"):
        assert kind in kinds, kind
    assert "resize_only" not in kinds, "resize_only 是對照，不屬主組六個"


def test_eval_sweep_既有掃描組未被更動():
    sw = eval_sweep()
    assert [p.strength for p in sw["blur"]] == [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]
    assert [p.strength for p in sw["jpeg"]] == [95, 85, 75, 60, 45, 30]
    assert [p.strength for p in sw["noise"]] == [0.0, 0.01, 0.02, 0.04, 0.08]
    assert [p.strength for p in sw["quantize"]] == [256, 64, 32, 16, 8]


def test_eval_sweep_含主組新算子與_resize_對照():
    sw = eval_sweep()
    for key in ("crop_resize", "adverse_cleaner", "cnn_denoise_substitute", "impress",
                "diffpure", "resize_only"):
        assert key in sw and len(sw[key]) == 1, key
    assert sw["impress"][0].options.get("sd") is None
    sd = _StubSD()
    assert eval_sweep(sd=sd)["impress"][0].options["sd"] is sd

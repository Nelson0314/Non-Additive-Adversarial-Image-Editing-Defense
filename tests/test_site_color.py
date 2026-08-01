"""site C（色度矩陣場）的設計性質。

本檔釘住的是 site C 存在的理由與已知限制，不是通用的正確性檢查：

- 存在的理由：亮度不動，故鈍化約束 `local_acutance_dev` 結構上不啟動。
  site S 的死因（重取樣造成鈍化）在此不存在。
- 已知限制：色度變換是乘性的，無彩區域沒有容量。這不是缺陷而是參數化
  的性質，但它決定了「本位置對哪些影像有效」，必須有測試釘住。
"""

import pytest
import torch

from src.metrics.acutance import _luma, acutance
from src.metrics.local_acutance import local_acutance_dev
from src.residual.site_color import ColorResidual, rgb_to_yuv, yuv_to_rgb

SIZE = 64
SEED = 20260728
DEV = torch.device("cpu")


def _image(seed: int = SEED, gray: bool = False) -> torch.Tensor:
    """有彩／無彩的合成影像。含高頻結構，否則銳利度指標的分母趨近零。"""
    g = torch.Generator().manual_seed(seed)
    x = torch.rand(1, 3, SIZE, SIZE, generator=g)
    # 疊上條紋，製造明確的亮度梯度
    xs = torch.linspace(0, 12 * torch.pi, SIZE)
    x = (x * 0.4 + 0.3 + 0.2 * xs.sin().view(1, 1, 1, -1)).clamp(0.05, 0.95)
    if gray:
        y = _luma(x)
        x = y.expand(-1, 3, -1, -1).contiguous()
    return x


def _perturbed(std: float = 0.5, **kw) -> ColorResidual:
    mod = ColorResidual(size=SIZE, grid_size=8, init_std=0.0, **kw).to(DEV)
    g = torch.Generator().manual_seed(SEED)
    mod.delta.data = torch.randn(mod.delta.shape, generator=g) * std
    return mod


# ---- 存在的理由 ----

def test_色度變換不改變亮度():
    """這是 site C 存在的理由。鈍化約束量的是 Rec.601 亮度的逐區塊
    梯度能量比（`local_acutance.py` 的 `_grad_sq` → `_luma`），而本位置只動
    (U, V)。故亮度必須逐像素不變——除了 RGB 立方體外的夾回處。

    此處以未觸及夾回的溫和 ΔM 驗精確性，夾回的影響另有測試。
    """
    x = _image()
    mod = _perturbed(std=0.05)
    with torch.no_grad():
        out = mod.pixel_residual(x)
    assert not torch.equal(out, x), "ΔM 非零時必須改變影像"
    assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0
    err = (_luma(out) - _luma(x)).abs().max().item()
    assert err < 1e-5, f"亮度應逐像素不變，實測最大偏差 {err:.3e}"


def test_鈍化約束對色度變換結構上不啟動():
    """`local_acutance_dev` 的門檻是 0.04（`objective.py` 的 tau_acut）。
    site S 實測 0.15、純模糊 0.24；本位置應該低到與量測噪聲同級。
    """
    x = _image()
    mod = _perturbed(std=0.05)
    with torch.no_grad():
        out = mod.pixel_residual(x)
        dev = float(local_acutance_dev(x, out))
        ratio = acutance(x, out)["acutance_ratio"]
    assert dev < 1e-4, f"鈍化偏差應結構上為零，實測 {dev:.3e}"
    assert abs(ratio - 1.0) < 1e-4, f"銳利度保留率應為 1，實測 {ratio:.6f}"


def test_夾回造成的亮度偏差有界且可量測():
    """不隱藏這個例外。色度被重新混合後可能落到 RGB 立方體外，夾回會
    微幅改變亮度，故「不改變 Y」在飽和處並非精確成立。此處以刻意過大的 ΔM
    逼出夾回，量出偏差的量級，使報告中的宣稱可以帶著這個數字。
    """
    x = _image()
    mod = _perturbed(std=1.0, max_dev=0.6)
    with torch.no_grad():
        out = mod.pixel_residual(x)
        dev = float(local_acutance_dev(x, out))
    err = (_luma(out) - _luma(x)).abs().max().item()
    assert err > 0.0, "此組態應觸發夾回，否則本測試沒有在測它該測的東西"
    assert err < 0.1, f"夾回造成的亮度偏差 {err:.4f} 超出預期量級"
    # 即便在這種極端組態下，鈍化偏差仍應遠低於 site S 的 0.15
    assert dev < 0.02, f"夾回後的鈍化偏差 {dev:.4f} 過大"


# ---- 已知限制 ----

def test_無彩影像上色度變換沒有容量():
    """釘住已知限制，不是釘住正確行為。

    M 乘在 (U, V) 上，U = V = 0 的像素無論 M 為何都不動。灰階影像因此完全
    不受影響，梯度亦為零。這決定了本位置對哪些影像有效，必須寫進報告。
    加偏移項可以解掉，但偏移項是色度上的加性位移，會破壞本位置的定位。
    """
    x = _image(gray=True)
    mod = _perturbed(std=1.0)
    out = mod.pixel_residual(x)
    assert (out - x).abs().max().item() < 1e-5, "無彩影像不應被色度變換改動"

    out.pow(2).sum().backward()
    assert mod.delta.grad is not None
    assert mod.delta.grad.abs().max().item() < 1e-5, "無彩影像上梯度應為零"


# ---- 與其他位置一致的介面性質 ----

def test_零偏離時輸出逐位元等於原圖():
    """與 site S 同：φ=0 的正確結果就是原圖，經過 RGB↔YUV 來回反而引入
    純數值誤差。在 no_grad 下檢查——那正是短路生效的條件，也是所有評測、
    留存影像與 x_base0 = G(x;0) 的計算條件。
    """
    x = _image()
    mod = ColorResidual(size=SIZE, grid_size=8, init_std=0.0).to(DEV)
    with torch.no_grad():
        out = mod.pixel_residual(x)
    assert torch.equal(out, x), "ΔM 為零時必須逐位元相同"


def test_零偏離下梯度仍抵達phi():
    """短路若在建圖時也生效，訓練第一步就會以
    `element 0 of tensors does not require grad` 失敗（site S 實測過）。
    """
    x = _image()
    mod = ColorResidual(size=SIZE, grid_size=8, init_std=0.0).to(DEV)
    out = mod.pixel_residual(x)
    assert out.requires_grad, "建圖模式下輸出必須連著 φ"
    out.pow(2).sum().backward()
    assert mod.delta.grad is not None and mod.delta.grad.abs().sum() > 0


def test_色彩空間來回的數值誤差有界():
    """量測（不是假設）零偏離短路所迴避的那個底線。_RGB2YUV 與 _YUV2RGB
    互為反矩陣只到浮點精度，force_transform 讓它實際跑一次。
    """
    x = _image()
    mod = ColorResidual(size=SIZE, grid_size=8, init_std=0.0,
                        force_transform=True).to(DEV)
    with torch.no_grad():
        out = mod.pixel_residual(x)
    err = (out - x).abs().max().item()
    assert not torch.equal(out, x), "force_transform 必須真的跑一次變換"
    assert err < 1e-5, f"色彩空間來回誤差 {err:.3e} 超出預期量級"


def test_max_dev硬上界確實生效():
    """偏離量是本位置的保真度預算，必須確實不被超過（同 site S 的 max_disp）。"""
    mod = _perturbed(std=5.0, max_dev=0.1)
    d = mod.matrix_field(SIZE, SIZE)
    assert d.abs().max().item() <= 0.1 + 1e-6


def test_停用模塊不改變結果():
    x = _image()
    mod = _perturbed(std=0.5)
    mod.disable()
    assert torch.equal(mod.pixel_residual(x), x)


def test_raw_residual為None():
    """回傳 x_def − x 會讓報告誤以為這是加性方法。矩陣場以 color_stats 回報。"""
    assert _perturbed().raw_residual() is None


def test_color_stats回報矩陣偏離而非L無窮():
    mod = _perturbed(std=0.5, max_dev=0.15)
    s = mod.color_stats()
    assert set(s) == {"cdev_mean", "cdev_max", "cdev_p99", "matrix_tv", "grid_size"}
    assert 0.0 < s["cdev_mean"] <= s["cdev_p99"] <= s["cdev_max"]
    assert s["grid_size"] == 8


@pytest.mark.parametrize("gray", [False, True])
def test_yuv來回是恆等變換(gray):
    x = _image(gray=gray)
    assert (yuv_to_rgb(rgb_to_yuv(x)) - x).abs().max().item() < 1e-5

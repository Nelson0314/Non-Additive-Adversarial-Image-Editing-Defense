"""加法項自己的頻帶：`floor_r_min`／`floor_r_max`／`floor_survival`。

存在理由
────────────────────────────────────────────────────────────────────
本方法的擾動有兩半。乘性那一半（相位 `exp(iθ)`、增益 `exp(g)`）能改動的量
正比於原圖自己的振幅 `|S_b(ω)|`，所以平坦區與沒有能量的頻帶都動不了；加性
那一半（`spectral_floor`）不受該限制。高斯模糊在頻域乘的是實正數
`exp(-2π²σ²f²)`，把高頻的振幅整個拿走——「編碼器對哪一帶敏感」與「哪一帶
活得過模糊」方向相反。兩半原本共用同一個徑向帶通，於是壓低上界會連乘性那
一半的未淨化強度一起削掉。這三個旋鈕讓加性那一半有自己的頻帶。

本檔釘住四件事：

    1. 三個旗標預設關閉時，價目表與加它們之前**逐位元**相等。
    2. 新的帶與存活加權**只**動加法項，相位／增益的閘一位不改。
    3. `fx = 0` 與 `fx = block//2` 兩行在新的帶下仍然是零（rfft2 的共軛
       對稱依賴它，破壞了輸出仍是實數、只是幅度不再保留，而且沒有症狀）。
    4. 總預算不隨三個旋鈕改變——`_build_floor_price` 把價目表縮放回同一個
       參考平均值，改的是「花在哪裡」不是「花多少」。等失真對齊是對
       `radius` 二分搜尋，而 `radius` 只驅動 theta 與 gain、碰不到加法項；
       不補回總量的話，加法項掉下去、二分搜尋抬高 radius 來補，同一列就
       同時混進「加性變少乘性變多」與「加性換了頻帶」兩件事。
"""

import math

import pytest
import torch

from src.residual.perceptual_weight import freq_weight, survival_weight
from src.residual.texture_rephase import PhaseResidual, radial_gate

BLOCK = 32
DT = torch.float64
DEV = torch.device("cpu")


def _module(**kw) -> PhaseResidual:
    """主線工作點的縮小版。`r_max` 留在無窮大，才看得出加法項的帶被壓窄。"""
    base = dict(size=128, block=BLOCK, hop=8, r_min=0.12, theta_max=1.0,
                gain_max=1.0, energy_quantile=0.0,
                freq_weight="jpeg_luma", freq_weight_power=0.25,
                spectral_floor=0.04)
    base.update(kw)
    return PhaseResidual(**base).to(DT)


def _image(seed: int = 0) -> torch.Tensor:
    """左半平坦、右半有紋理，讓紋理閘既不是全零也不是全一。"""
    g = torch.Generator().manual_seed(seed)
    x = torch.rand(1, 3, 128, 128, generator=g, dtype=DT)
    x[:, :, :, :64] = 0.45
    return x


def _ready(**kw) -> PhaseResidual:
    m = _module(**kw)
    m.prepare_gates(_image())
    return m


# ---------------------------------------------------------------------------
# 一、預設關閉時逐位元等於加這三個旋鈕之前
# ---------------------------------------------------------------------------

def test_三個旗標的預設值是關閉():
    m = _module()
    assert (m.floor_r_min, m.floor_r_max, m.floor_survival) == (None, None, "none")


def test_預設關閉時價目表逐位元等於加這三個旋鈕之前():
    """加旗標之前的價目就是 徑向帶通（相位那一半的帶）× jpeg_luma（一次方）。"""
    want = (radial_gate(BLOCK, 0.12, DEV, DT)
            * freq_weight("jpeg_luma", BLOCK, DEV, DT))
    assert torch.equal(_ready().floor_price(), want)


@pytest.mark.parametrize("gate", ["uniform", "complement", "complement_rank",
                                  "watson"])
def test_預設關閉時每個floor_gate都逐位元不變(gate):
    """正規化的參考點由 base 換成一個獨立算出的 ref，四個變體都不得因此漂動。"""
    ref = _ready(floor_gate=gate).floor_price()
    got = _ready(floor_gate=gate, floor_r_min=None, floor_r_max=None,
                 floor_survival="none").floor_price()
    assert torch.equal(got, ref)


def test_顯式填入相位那一半的帶等於不填():
    """`floor_r_min=r_min`、`floor_r_max=r_max` 與 None 必須走到同一個結果。"""
    ref = _ready().floor_price()
    got = _ready(floor_r_min=0.12, floor_r_max=float("inf")).floor_price()
    assert torch.equal(got, ref)


def test_spectral_floor為零時三個旋鈕不建價目表也不改任何輸出():
    """`--spectral-floor 0` 時這三個旋鈕沒有作用，help 是這樣寫的。"""
    x = _image()
    ref = _ready(spectral_floor=0.0)
    m = _ready(spectral_floor=0.0, floor_r_max=0.4, floor_survival="blur12")
    assert m._floor_price is None and ref._floor_price is None
    with pytest.raises(RuntimeError, match="spectral_floor"):
        m.floor_price()
    assert torch.equal(m.gate(), ref.gate())
    with torch.no_grad():
        m.theta.zero_(); m.gain.zero_(); m.floor.zero_()
        ref.theta.zero_(); ref.gain.zero_(); ref.floor.zero_()
    assert torch.equal(m.pixel_residual(x), ref.pixel_residual(x))


def test_theta為零且spectral_floor為零時輸出等於原圖():
    """構造保證，不因為新旋鈕開著而失效。容差 1e-12 與
    `test_texture_rephase.py` 對同一條保證的釘法一致——`OLA(w^2)` 的那一次
    除法在 float64 上留下捨入誤差，恆等是代數上的而非位元上的。"""
    x = _image()
    m = _ready(spectral_floor=0.0, floor_r_min=0.05, floor_r_max=0.4,
               floor_survival="blur12")
    with torch.no_grad():
        m.theta.zero_(); m.gain.zero_(); m.floor.zero_()
        out = m.pixel_residual(x)
    assert torch.allclose(out, x, atol=1e-12), float((out - x).abs().max())


def test_三個參數全零時加法項開著也仍是恆等():
    x = _image()
    m = _ready(floor_r_max=0.4, floor_survival="blur12")
    with torch.no_grad():
        m.theta.zero_(); m.gain.zero_(); m.floor.zero_()
        out = m.pixel_residual(x)
    assert float((out - x).abs().max()) < 1e-12


# ---------------------------------------------------------------------------
# 二、只動加法項
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kw", [
    {"floor_r_max": 0.4},
    {"floor_r_min": 0.5},
    {"floor_survival": "blur12"},
    {"floor_survival": "blur1"},
    {"floor_r_min": 0.05, "floor_r_max": 0.4, "floor_survival": "blur12"},
])
def test_新旋鈕不影響相位與增益的閘(kw):
    """相位／增益的閘由 `r_min`／`r_max`／`survival_weight` 決定，
    三個新旋鈕一位都不得改到它——否則「兩半各自有自己的頻帶」這件事就沒有
    做到，量到的會是兩半一起被搬走。"""
    ref = _ready()
    m = _ready(**kw)
    assert torch.equal(m.freq_gate, ref.freq_gate)
    assert torch.equal(m.gate(), ref.gate())
    assert torch.equal(m.gain_gate(), ref.gain_gate())
    assert torch.equal(m.tex_gate, ref.tex_gate)


@pytest.mark.parametrize("kw", [
    {"floor_r_max": 0.4},
    {"floor_survival": "blur12"},
])
def test_新旋鈕確實改動了價目表(kw):
    """上一條若因為兩者都沒作用而成立，就沒有鑑別力。"""
    assert not torch.equal(_ready(**kw).floor_price(), _ready().floor_price())


def test_相位那一半的survival_weight不動加法項的價目表():
    """`survival_weight` 只乘在相位／增益的閘上。加法項要走 `floor_survival`
    ——這正是這一組旋鈕存在的理由：兩半可以押不同的頻帶。"""
    ref = _ready()
    m = _ready(survival_weight="blur12")
    assert torch.equal(m.floor_price(), ref.floor_price())
    assert not torch.equal(m.freq_gate, ref.freq_gate)


def test_加法項的帶壓窄之後乘性那一半仍在高頻():
    """把加法項關進 r <= 0.4，而相位的閘在 r > 0.4 上仍然是開的。"""
    m = _ready(floor_r_max=0.4)
    fy = torch.fft.fftfreq(BLOCK, dtype=DT) * 2.0
    fx = torch.fft.rfftfreq(BLOCK, dtype=DT) * 2.0
    r = torch.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)
    high = r > 0.4
    assert float(m.floor_price()[high].abs().max()) == 0.0
    # 相位的閘在同一批頻格上仍有開度（扣掉 fx=0／fx=N/2 那兩行）
    inner = high.clone()
    inner[:, 0] = False
    inner[:, -1] = False
    assert float(m.freq_gate[inner].max()) > 0.0


# ---------------------------------------------------------------------------
# 三、共軛對稱依賴的那兩行仍然是零
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kw", [
    {"floor_r_min": 0.05},
    {"floor_r_max": 0.4},
    {"floor_r_min": 0.0, "floor_r_max": 2.0},
    {"floor_survival": "blur12"},
    {"floor_r_min": 0.0, "floor_r_max": 2.0, "floor_survival": "blur12"},
])
@pytest.mark.parametrize("gate", ["uniform", "complement", "watson"])
def test_fx為零與fx為block一半那兩行在新的帶下仍然是零(kw, gate):
    price = _ready(floor_gate=gate, **kw).floor_price()
    assert float(price[..., :, 0].abs().max()) == 0.0
    assert float(price[..., :, -1].abs().max()) == 0.0


# ---------------------------------------------------------------------------
# 四、總預算不隨三個旋鈕改變
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kw", [
    {"floor_r_max": 0.4},
    {"floor_r_min": 0.5},
    {"floor_r_min": 0.05},
    {"floor_survival": "blur12"},
    {"floor_survival": "blur1"},
    {"floor_r_min": 0.05, "floor_r_max": 0.4, "floor_survival": "blur12"},
])
@pytest.mark.parametrize("gate", ["uniform", "complement", "complement_rank",
                                  "watson"])
def test_總預算與預設相同(kw, gate):
    """參考點固定是「相位那一半的帶通 × jpeg_luma」的平均值，四個旋鈕
    （含 floor_gate）都動不了它。"""
    ref = _ready(floor_gate="uniform")
    m = _ready(floor_gate=gate, **kw)
    assert float(m.floor_price().mean()) == pytest.approx(
        float(ref.floor_price().mean()), rel=1e-9)


def test_帶壓窄之後每一格的定價被抬高():
    """L1 總量守恆的直接後果，也是這個選擇的代價：同樣多的預算擠進更少的
    頻格。不抬高的話 `floor_r_max` 在中等值上就等於一個關閉開關。"""
    ref = _ready().floor_price()
    m = _ready(floor_r_max=0.4).floor_price()
    band = m > 0
    assert float(m[band].mean()) > float(ref[band].mean())


# ---------------------------------------------------------------------------
# 五、錯誤直接拋出，不靜默退回預設
# ---------------------------------------------------------------------------

def test_未知的floor_survival直接拋錯():
    with pytest.raises(ValueError, match="floor_survival"):
        PhaseResidual(size=64, block=16, spectral_floor=0.04,
                      floor_survival="not_a_weight")


def test_未知的floor_survival在建構時就拋錯而不是等到prepare_gates():
    """錯誤若拖到防禦迴圈裡逐張呼叫的 `prepare_gates` 才拋，會被淹在批次
    輸出裡。"""
    with pytest.raises(ValueError, match="floor_survival"):
        PhaseResidual(size=64, block=16, spectral_floor=0.0,
                      floor_survival="blur3")


def test_加法項的上界不大於下界時拋錯():
    m = _module(floor_r_min=0.5, floor_r_max=0.3)
    with pytest.raises(ValueError, match="通帶是空的"):
        m.prepare_gates(_image())


def test_參考帶為空時拋錯而不是回傳全零的價目表():
    """全零的價目表會讓加法項悄悄失效，而 CSV 上 `spectral_floor` 那一欄
    照樣寫著非零值。格點上的最大半徑是 sqrt(1 + (1 - 2/block)^2) < 1.38。"""
    m = _module(r_min=1.4, floor_r_min=0.12)
    with pytest.raises(ValueError, match="總預算參考點為零"):
        m.prepare_gates(_image())


def test_格點上的最大半徑確實小於上一條用的下界():
    """上一條的 1.4 若其實落在通帶內，那條測試就沒有在測它宣稱的東西。"""
    fy = torch.fft.fftfreq(BLOCK, dtype=DT) * 2.0
    fx = torch.fft.rfftfreq(BLOCK, dtype=DT) * 2.0
    r = torch.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)
    r = r[:, 1:-1]                      # fx=0 與 fx=N/2 兩行一律為零
    assert float(r.max()) < 1.4
    assert float(r.max()) == pytest.approx(
        math.sqrt(1.0 + (1.0 - 2.0 / BLOCK) ** 2))


# ---------------------------------------------------------------------------
# 六、存活加權的方向：把預算往低頻搬
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["blur1", "blur12"])
def test_存活加權把加法項的預算往低頻搬(name):
    """`w = (1 + Σ_σ exp(-2π²σ²f²)) / (1 + |S|)` 隨頻率單調遞減，故加權後
    低頻那一半分到的比例必須上升。總量守恆使這是一個純粹的重分配。"""
    fy = torch.fft.fftfreq(BLOCK, dtype=DT) * 2.0
    fx = torch.fft.rfftfreq(BLOCK, dtype=DT) * 2.0
    r = torch.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)
    low = r <= 0.5

    ref = _ready().floor_price()
    m = _ready(floor_survival=name).floor_price()
    share_ref = float(ref[low].sum() / ref.sum())
    share_new = float(m[low].sum() / m.sum())
    assert share_new > share_ref
    # 權重本身確實不是全 1，否則上面那條沒有鑑別力
    w = survival_weight(name, BLOCK, DEV, DT)
    assert float(w.min()) < 0.99

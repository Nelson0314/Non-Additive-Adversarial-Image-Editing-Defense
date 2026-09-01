"""可學的空間包絡：`floor_envelope`／`floor_envelope_k`／`floor_envelope_scope`。

存在理由
────────────────────────────────────────────────────────────────────
本模組原本沒有任何**可學的空間定位**。紋理閘會定位，但它由原圖算出、固定
不動；`--floor-survival` 已經在**頻率**上做軟性挑選，空間上仍然均勻；
`src/residual/` 底下沒有任何 patch 或 mask 的參數化。

這一層加的就是那個自由度：由 K 組中心 `(cy, cx)`、K 個空間尺度 `s`、一個
徑向低通截止 `f_c`、一個強度 `beta` 參數化的平滑窗，五個純量與 `theta`／
`gain`／`floor` 走同一條 PGD。總預算被縮放回原本的平均值，所以「集中」的
意思是「同樣的總量堆到少數位置，那些位置的定價因此被抬高」。

本檔釘住的事
────────────────────────────────────────────────────────────────────
    1. **`floor_envelope = "none"` 時逐位元等於加這個旋鈕之前**，而且前向
       連 `envelope()` 都不呼叫（用會爆的樁證明，與 `test_coarsen.py` 的
       `F.interpolate` 樁同型）。
    2. **`beta = 0` 時包絡逐位元是 1，輸出逐位元等於關閉**。這一條比第 1 條
       強：它是在**開著**的程式路徑上量的，證明的是構造本身退得回去，
       不是靠一個 if 繞過去。
    3. **梯度真的傳得到那五個純量**，而且是五個都傳得到——少一個不會拋錯，
       只會讓那一維永遠停在初值。
    4. **總預算不隨包絡改變**：價目表被縮放回同一個平均值。
    5. `project()` 把五個純量投影回由網格導出的可行集。
    6. 三個旗標從 CLI 一路轉交到 `PhaseResidual`，且寫進 CSV。
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

from src.defense.param_pgd import PhaseParam
from src.residual.texture_rephase import (
    FLOOR_ENVELOPES,
    FLOOR_ENVELOPE_SCOPES,
    PhaseResidual,
    radial_gate,
    radial_radius,
    window_centres,
)

ROOT = Path(__file__).resolve().parents[1]
SIZE, BLOCK, HOP = 128, 32, 8
DT = torch.float64


def _image(seed: int = 3) -> torch.Tensor:
    """左半平坦、右半有紋理，讓紋理閘既不是全零也不是全一。"""
    g = torch.Generator().manual_seed(seed)
    x = torch.rand(1, 3, SIZE, SIZE, generator=g, dtype=DT)
    x[:, :, :, :SIZE // 2] = 0.45
    return x


def _module(**kw) -> PhaseResidual:
    base = dict(size=SIZE, block=BLOCK, hop=HOP, r_min=0.12, theta_max=1.0,
                gain_max=1.0, energy_quantile=0.0, freq_weight="jpeg_luma",
                freq_weight_power=0.25, spectral_floor=0.04)
    base.update(kw)
    return PhaseResidual(**base).to(DT)


def _filled(**kw) -> PhaseResidual:
    """三個場都填非零值——全零的話任何比較都是套套邏輯。"""
    m = _module(**kw)
    m.prepare_gates(_image())
    with torch.no_grad():
        m.theta.normal_(generator=torch.Generator().manual_seed(11))
        m.gain.normal_(generator=torch.Generator().manual_seed(12))
        m.floor.uniform_(-1, 1, generator=torch.Generator().manual_seed(13))
    return m


# ---- 1. 關閉時逐位元等於現況 ----

def test_預設是關閉的():
    m = _module()
    assert m.floor_envelope == "none"
    assert m.floor_envelope_k == 1
    assert m.floor_envelope_scope == "floor"


def test_關閉時輸出逐位元等於加這個旋鈕之前():
    x = _image()
    assert torch.equal(_filled().pixel_residual(x),
                       _filled(floor_envelope="none").pixel_residual(x))


def test_關閉時前向完全不呼叫包絡(monkeypatch):
    """「逐位元等於加這個旋鈕之前」的直接證據。

    拿兩個關閉的模組互比是套套邏輯——兩邊走的是同一條新程式。真正要證的是
    **新程式在關閉時沒有多做任何事**，所以把 `envelope` 換成會爆的樁：
    三個場全開仍跑得完，就表示它沒被碰到。
    """
    m = _filled()

    def boom(self):
        raise AssertionError("floor_envelope='none' 不該呼叫 envelope()")

    monkeypatch.setattr(PhaseResidual, "envelope", boom)
    assert torch.isfinite(m.pixel_residual(_image())).all()


def test_開啟時前向確實會呼叫包絡(monkeypatch):
    """上一項的對照：換成會爆的樁時，開著就必須爆。否則樁根本沒裝上。"""
    m = _filled(floor_envelope="gauss")

    def boom(self):
        raise AssertionError("樁有裝上")

    monkeypatch.setattr(PhaseResidual, "envelope", boom)
    with pytest.raises(AssertionError, match="樁有裝上"):
        m.pixel_residual(_image())


# ---- 2. beta = 0 時包絡恆為 1，且輸出逐位元等於關閉 ----

@pytest.mark.parametrize("scope", list(FLOOR_ENVELOPE_SCOPES))
@pytest.mark.parametrize("k", [1, 2, 3])
def test_beta為零時包絡逐位元是1(scope, k):
    """`(1 - 0.0) + 0.0 * S == 1.0` 在 IEEE754 上是精確的。

    近似相等不夠：包絡是**乘**上去的，`x * (1 + 1e-16)` 與 `x` 不同位，
    而那個差異在防禦圖上看不出來、在 CSV 上也沒有欄位。
    """
    m = _module(floor_envelope="gauss", floor_envelope_k=k,
                floor_envelope_scope=scope)
    m.prepare_gates(_image())
    env = m.envelope()
    assert env.shape == (1, m.n_blocks, BLOCK, BLOCK // 2 + 1)
    assert bool((env == 1.0).all())


@pytest.mark.parametrize("scope", list(FLOOR_ENVELOPE_SCOPES))
@pytest.mark.parametrize("k", [1, 2, 3])
def test_beta為零時輸出逐位元等於關閉(scope, k):
    """比第一組更強：這是在**開著**的程式路徑上量的。

    第一組證的是「關掉時不執行」，這一組證的是「執行了但構造上是恆等」
    ——包絡恆為 1 時退回現況這件事因此不依賴任何 if。
    """
    x = _image()
    base = _filled()
    on = _filled(floor_envelope="gauss", floor_envelope_k=k,
                 floor_envelope_scope=scope)
    assert torch.equal(base.pixel_residual(x), on.pixel_residual(x))


def test_beta為零時價目表也逐位元不變():
    """`envelope_price` 的縮放比在 env 全為 1 時必須精確是 1.0。

    參考量若取 `price.mean()`（形狀 (n, nb)）而分母取 `(price*env).mean()`
    （形狀 (1, L, n, nb)），兩者的元素個數差 L 倍、浮點求和順序不同，比值
    會落在 1 附近而不是 1.0，逐位元恆等就破了——而症狀是零。
    """
    m = _filled(floor_envelope="gauss")
    price = m.floor_price()
    out = m.envelope_price(price, m.envelope())
    assert torch.equal(out, price.expand_as(out))


def test_beta不為零時輸出確實改變():
    """恆等測試的對照組：不改變的話上面幾條全部是空的。"""
    x = _image()
    base = _filled()
    on = _filled(floor_envelope="gauss")
    with torch.no_grad():
        on.env_beta.fill_(1.0)
        on.env_sigma.fill_(0.25)
    assert not torch.equal(base.pixel_residual(x), on.pixel_residual(x))


# ---- 3. 梯度傳得到五個純量 ----

ENV_PARAMS = ("env_center", "env_sigma", "env_fc", "env_beta")


@pytest.mark.parametrize("scope", list(FLOOR_ENVELOPE_SCOPES))
def test_梯度傳得到包絡的每一個參數(scope):
    """五個都要傳得到。少一個不會拋錯，只會讓那一維永遠停在初值。

    **beta 必須先離開 0**：beta = 0 時 `d env / d c = beta * (...) = 0`，
    中心與尺度的梯度在構造上就是零。那不是斷了，是那一點的真實梯度；要量
    「傳不傳得到」就必須在 beta > 0 的地方量。
    """
    m = _filled(floor_envelope="gauss", floor_envelope_scope=scope)
    with torch.no_grad():
        m.env_beta.fill_(0.5)
        m.env_sigma.fill_(0.4)
        m.env_center[0, 0] = 0.3
    m.pixel_residual(_image()).pow(2).sum().backward()
    for name in ENV_PARAMS:
        g = getattr(m, name).grad
        assert g is not None, name
        assert g.shape == getattr(m, name).shape, name
        assert torch.isfinite(g).all(), name
        assert float(g.abs().sum()) > 0, name


def test_beta的梯度在beta等於零時仍然非零():
    """零初始化不是死點：beta 必須能自己離開 0，否則整層永遠不會啟動。

    `d env / d beta = S - 1` 與 `F - 1`，兩者在 beta = 0 時都不是零。
    """
    m = _filled(floor_envelope="gauss")
    m.pixel_residual(_image()).pow(2).sum().backward()
    assert float(m.env_beta.grad.abs().sum()) > 0


def test_關閉時包絡的參數不進最佳化():
    par = PhaseParam(size=SIZE, block=BLOCK, hop=HOP, r_min=0.12,
                     gain_ratio=1.0, spectral_floor=0.04)
    par.reset(_image(), seed=0)
    assert len(par.params()) == 3


def test_開啟時包絡的五個純量接在既有三者之後():
    """位置不能插在前面：`_load_weights` 是按位置 zip 的。"""
    par = PhaseParam(size=SIZE, block=BLOCK, hop=HOP, r_min=0.12,
                     gain_ratio=1.0, spectral_floor=0.04,
                     floor_envelope="gauss")
    par.reset(_image(), seed=0)
    ps = par.params()
    assert len(ps) == 7
    m = par.module
    for got, want in zip(ps, [m.theta, m.gain, m.floor, m.env_center,
                              m.env_sigma, m.env_fc, m.env_beta]):
        assert got is want


# ---- 4. 總預算不隨包絡改變 ----

def test_總預算不隨包絡改變():
    """加法項的四個既有旋鈕都被縮放回同一個參考平均值，包絡沿用同一條規則。

    理由與 `_build_floor_price` 相同：等失真對齊是對 `radius` 二分搜尋，而
    `radius` 只驅動 theta 與 gain、碰不到加法項。不補回總量的話加法項的
    預算直接掉下去，二分搜尋抬高 `radius` 來補，同一列就同時混進「加性
    變少乘性變多」與「加性換了位置」兩件事。
    """
    m = _filled(floor_envelope="gauss")
    with torch.no_grad():
        m.env_beta.fill_(1.0)
        m.env_sigma.fill_(0.3)
        m.env_fc.fill_(0.4)
    price = m.floor_price()
    out = m.envelope_price(price, m.envelope())
    ref = (price * torch.ones_like(out)).mean().detach()
    assert float(out.mean().detach()) == pytest.approx(float(ref), rel=1e-10)


def test_集中之後尖峰的定價被抬高():
    """「擾動超級大但只在一個位置」在本構造裡就是這一條。

    總量固定、包絡把價目從別處收回來堆到凸包底下，於是那一塊的每一格定價
    被抬高。不會發生的話，這一層就只是把加法項整體調小。
    """
    m = _filled(floor_envelope="gauss")
    price = m.floor_price()
    with torch.no_grad():
        m.env_beta.fill_(1.0)
        m.env_sigma.fill_(0.25)
        m.env_fc.fill_(m.fc_max)
    out = m.envelope_price(price, m.envelope())
    assert float(out.max()) > 2.0 * float(price.max())


def test_包絡不讓通帶外的頻格復活():
    """`fx = 0` 與 `fx = block//2` 兩行必須維持為零。

    rfft2 的共軛對稱依賴它們（模組 docstring 第 3 點）。破壞了輸出仍然是
    實數、只是那兩行的幅度不再保留，**而且沒有任何症狀**。
    """
    m = _filled(floor_envelope="gauss")
    with torch.no_grad():
        m.env_beta.fill_(1.0)
        m.env_sigma.fill_(0.3)
    out = m.envelope_price(m.floor_price(), m.envelope())
    assert float(out[..., 0].abs().max()) == 0.0
    assert float(out[..., -1].abs().max()) == 0.0


def test_scope為floor時不動相位與增益的閘():
    """預設 scope 只碰加法項。碰到了會讓「加性換位置」與「乘性換位置」
    混在同一列，而兩者的總量處理不同。"""
    base = _filled()
    on = _filled(floor_envelope="gauss")
    with torch.no_grad():
        on.env_beta.fill_(1.0)
        on.env_sigma.fill_(0.25)
    assert torch.equal(base.gate(), on.gate())
    assert torch.equal(base.gain_gate(), on.gain_gate())


def test_scope為all時連乘性那一半也被定位():
    """兩個 scope 必須真的不同，否則旗標是裝飾。"""
    x = _image()
    a = _filled(floor_envelope="gauss", floor_envelope_scope="floor")
    b = _filled(floor_envelope="gauss", floor_envelope_scope="all")
    for m in (a, b):
        with torch.no_grad():
            m.env_beta.fill_(1.0)
            m.env_sigma.fill_(0.25)
            m.env_fc.fill_(0.5)
    assert not torch.equal(a.pixel_residual(x), b.pixel_residual(x))


def test_集中確實把殘差搬到凸包底下():
    """構造的核心主張：中心搬到哪裡，殘差的質心就跟著搬。

    量的是殘差能量在 y 方向的質心——包絡只在 y 上偏移，x 上不動。
    """
    x = _image()

    def centroid_y(m):
        r = (m.pixel_residual(x) - x).abs().mean(dim=(0, 1)).sum(dim=1)
        idx = torch.arange(SIZE, dtype=DT)
        return float((r * idx).sum() / r.sum())

    up = _filled(floor_envelope="gauss", floor_envelope_scope="all")
    down = _filled(floor_envelope="gauss", floor_envelope_scope="all")
    for m, cy in ((up, -0.5), (down, 0.5)):
        with torch.no_grad():
            m.env_beta.fill_(1.0)
            m.env_sigma.fill_(0.3)
            m.env_center[0, 0] = cy
    assert centroid_y(up) < centroid_y(down)


# ---- 5. 可行集 ----

def test_project把五個純量投影回可行集():
    """**投影到可行集，不是只在前向夾。**

    只在前向夾的話，被夾住的座標梯度是零，sign-PGD 會把參數一路推到界外
    再也回不來，而報表上看不出來（與 `theta_cap` 同一個理由）。
    """
    par = PhaseParam(size=SIZE, block=BLOCK, hop=HOP, r_min=0.12,
                     gain_ratio=1.0, spectral_floor=0.04,
                     floor_envelope="gauss")
    par.reset(_image(), seed=0)
    m = par.module
    with torch.no_grad():
        m.env_center.fill_(9.0)
        m.env_sigma.fill_(-9.0)
        m.env_fc.fill_(9.0)
        m.env_beta.fill_(9.0)
    par.project()
    assert float(m.env_center.abs().max()) <= 1.0
    assert float(m.env_sigma.min()) == pytest.approx(m.sigma_min)
    assert float(m.env_fc.max()) == pytest.approx(m.fc_max)
    assert float(m.env_beta.max()) == 1.0


def test_四個邊界全部由網格導出():
    """沒有一個是憑感覺挑的常數——換 block／hop／size 它們就跟著換。"""
    m = _module()
    assert m.sigma_min == pytest.approx(2.0 * HOP / SIZE)
    assert m.sigma_max == pytest.approx(2.0 * math.sqrt(2.0))
    assert m.fc_min == pytest.approx(2.0 / BLOCK)
    assert m.fc_max == pytest.approx(math.sqrt(2.0))
    # `fc_max` 必須真的是格點上存在的最大半徑，不是一個湊出來的數。
    assert float(radial_radius(BLOCK, torch.device("cpu"), DT).max()) \
        <= m.fc_max + 1e-12


def test_視窗中心的座標與unfold的視窗序一致():
    """順序錯掉的話包絡會被轉置，而輸出仍然是一張合法的影像。"""
    m = _module()
    xy = window_centres(m.side, HOP, SIZE, torch.device("cpu"), DT)
    assert xy.shape == (m.n_blocks, 2)
    # row-major：前 side 個共用同一個 y、x 遞增。
    assert bool((xy[:m.side, 0] == xy[0, 0]).all())
    assert bool((xy[:m.side, 1].diff() > 0).all())
    assert xy[0].tolist() == [-1.0, -1.0]
    assert xy[-1].tolist() == [1.0, 1.0]


def test_低通與帶通用的是同一套座標():
    """兩邊若用不同的座標約定，低通的截止就會對到別的地方。"""
    r = radial_radius(BLOCK, torch.device("cpu"), DT)
    gate = radial_gate(BLOCK, 0.5, torch.device("cpu"), DT, 0.9)
    inside = gate > 0
    assert float(r[inside].min()) >= 0.5
    assert float(r[inside].max()) <= 0.9


# ---- 6. 拒絕靜默失效的設定 ----

def test_只有加法項卻沒有加法項可乘時拒絕啟動():
    """包絡的梯度會恆為零，而 CSV 上那一欄照樣寫著 gauss。"""
    with pytest.raises(ValueError, match="spectral_floor"):
        _module(floor_envelope="gauss", spectral_floor=0.0)


def test_scope為all時不需要加法項():
    """`all` 乘的是相位與增益的閘，與 `spectral_floor` 無關。"""
    m = _module(floor_envelope="gauss", floor_envelope_scope="all",
                spectral_floor=0.0)
    m.prepare_gates(_image())
    assert torch.isfinite(m.pixel_residual(_image())).all()


@pytest.mark.parametrize("bad", ["Gauss", "gaussian", "", "uniform"])
def test_未知的包絡名在建構時就拋錯(bad):
    """靜默回退會讓一整批掃描跑成基準的重複。"""
    with pytest.raises(ValueError, match="floor_envelope"):
        _module(floor_envelope=bad)


@pytest.mark.parametrize("bad", ["Floor", "both", ""])
def test_未知的scope在建構時就拋錯(bad):
    with pytest.raises(ValueError, match="floor_envelope_scope"):
        _module(floor_envelope="gauss", floor_envelope_scope=bad)


@pytest.mark.parametrize("bad", [0, -1, 2.5])
def test_不合法的凸包個數要拋錯(bad):
    with pytest.raises(ValueError, match="floor_envelope_k"):
        _module(floor_envelope="gauss", floor_envelope_k=bad)


def test_關閉時呼叫包絡直接拋錯():
    """回傳一張全 1 的張量會讓「有沒有開」變得看不出來。"""
    m = _filled()
    with pytest.raises(RuntimeError, match="floor_envelope"):
        m.envelope()


def test_單一凸包不走軟聯集的浮點路徑():
    """`1 - (1 - b)` 在浮點上不等於 `b`（0.3 會變成 0.30000000000000004）。

    K = 1 特判是為了讓「只有一個凸包」與單一高斯逐位元一致。
    """
    one = _module(floor_envelope="gauss", floor_envelope_k=1)
    one.prepare_gates(_image())
    with torch.no_grad():
        one.env_beta.fill_(1.0)
        one.env_sigma.fill_(0.4)
        one.env_fc.fill_(one.fc_max)
    d2 = (one.win_xy ** 2).sum(-1)
    want = torch.exp(-d2 / (2.0 * torch.tensor(0.4, dtype=DT) ** 2))
    got = one.envelope()[0, :, 1, 1] / torch.exp(
        -(one.freq_r[1, 1] / one.fc_max) ** 2)
    assert torch.allclose(got, want, atol=0, rtol=1e-12)


def test_多個凸包的軟聯集落在零與一之間():
    m = _module(floor_envelope="gauss", floor_envelope_k=3)
    m.prepare_gates(_image())
    with torch.no_grad():
        m.env_beta.fill_(1.0)
        m.env_sigma.fill_(0.6)
        m.env_center.copy_(torch.tensor([[-0.6, -0.6], [0.0, 0.0], [0.6, 0.6]],
                                        dtype=DT))
    env = m.envelope()
    assert float(env.min()) >= 0.0
    assert float(env.max()) <= 1.0 + 1e-12


# ---- 7. 旗標一路轉交，並寫進 CSV ----

def test_PhaseParam把三個旗標轉交下去():
    par = PhaseParam(size=SIZE, block=BLOCK, hop=HOP, r_min=0.12,
                     spectral_floor=0.04, floor_envelope="gauss",
                     floor_envelope_k=2, floor_envelope_scope="all")
    par.reset(_image(), seed=0)
    assert par.module.floor_envelope == "gauss"
    assert par.module.floor_envelope_k == 2
    assert par.module.floor_envelope_scope == "all"
    assert par.module.env_center.shape == (2, 2)


def test_phase_ablation的build把三個旗標轉交下去():
    sys.path.insert(0, str(ROOT / "scripts"))
    from phase_ablation import build

    par, _, _ = build("phase_gain", seed=0, block=BLOCK, hop=HOP,
                      gain_ratio=1.0, spectral_floor=0.04,
                      floor_envelope="gauss", floor_envelope_k=2,
                      floor_envelope_scope="all")
    assert par.floor_envelope == "gauss"
    assert par.floor_envelope_k == 2
    assert par.floor_envelope_scope == "all"


def test_三個旗標的CLI預設值是關閉的():
    sys.path.insert(0, str(ROOT / "scripts"))
    import ip2p_run

    ns = ip2p_run.build_parser().parse_args(["--out", "x"])
    assert ns.floor_envelope == "none"
    assert ns.floor_envelope_k == 1
    assert ns.floor_envelope_scope == "floor"


def test_三個旗標的CLI選項與模組的常數一致():
    """選項清單若各寫一份，加了新的包絡形式就會有一邊漏掉。"""
    sys.path.insert(0, str(ROOT / "scripts"))
    import ip2p_run

    ap = ip2p_run.build_parser()
    opts = {a.dest: a.choices for a in ap._actions}
    assert list(opts["floor_envelope"]) == list(FLOOR_ENVELOPES)
    assert list(opts["floor_envelope_scope"]) == list(FLOOR_ENVELOPE_SCOPES)


@pytest.mark.parametrize("col", ["floor_envelope", "floor_envelope_k",
                                 "floor_envelope_scope"])
def test_三個旗標都是CSV的欄位(col):
    """CLAUDE.md：每個未載的參數都要成為 CSV 的**欄位**，不是註解。

    比對字面字串而非執行整條管線：這裡不載入 IP2P（需要 GPU 與權重），而
    漏掉欄位的失效方式正是「那一行不存在」。
    """
    src = (ROOT / "scripts" / "ip2p_run.py").read_text(encoding="utf-8")
    assert f'"{col}":' in src


def test_學出來的包絡參數也進得了CSV():
    """旗標記的是設定，這幾欄記的是 PGD 把凸包放到了哪裡。

    沒有它就只能從防禦圖用眼睛猜，而「包絡有沒有真的動」與「動了有沒有用」
    是兩件事，前者必須是數字。
    """
    par = PhaseParam(size=SIZE, block=BLOCK, hop=HOP, r_min=0.12,
                     spectral_floor=0.04, floor_envelope="gauss")
    assert par.envelope_state() == {}          # reset 之前
    par.reset(_image(), seed=0)
    with torch.no_grad():
        par.module.env_beta.fill_(0.5)
        par.module.env_center[0, 1] = 0.5
    st = par.envelope_state()
    assert st["env_beta"] == 0.5
    assert st["env_cx0"] == 0.5
    # 中心也換算成像素座標：正規化座標讀不出「離主體多遠」。
    assert st["env_px0"] == pytest.approx(0.75 * SIZE)
    assert st["env_py0"] == pytest.approx(0.5 * SIZE)


def test_關閉時不寫學出來的那幾欄():
    par = PhaseParam(size=SIZE, block=BLOCK, hop=HOP, r_min=0.12,
                     spectral_floor=0.04)
    par.reset(_image(), seed=0)
    assert par.envelope_state() == {}


def test_續跑時參數個數不同會拋錯而不是靜默對錯位置():
    """`_load_weights` 按位置 zip。開了包絡之後參數由 3 個變成 7 個，拿舊的
    `__w.pt` 續跑會把 `theta` 的值寫到別的張量上——那不會拋錯、不會有症狀。
    """
    off = PhaseParam(size=SIZE, block=BLOCK, hop=HOP, r_min=0.12,
                     gain_ratio=1.0, spectral_floor=0.04)
    on = PhaseParam(size=SIZE, block=BLOCK, hop=HOP, r_min=0.12,
                    gain_ratio=1.0, spectral_floor=0.04,
                    floor_envelope="gauss")
    x = _image()
    off.reset(x, seed=0)
    on.reset(x, seed=0)
    assert len(off.params()) != len(on.params())
    assert "不可續跑" in (ROOT / "scripts" / "ip2p_run.py").read_text(
        encoding="utf-8")

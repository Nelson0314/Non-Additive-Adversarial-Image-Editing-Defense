"""身分損失接在 VAE 一次往返上（`--loss facelock`）的六個不變量。

移植自 arXiv:2411.16832 的官方 `methods.py`。六件事各自失敗時都沒有症狀：

1. **符號。** `run_param_pgd` 最小化，原文梯度上升。寫反時防禦圖被推向
   「更像本人」，而 `trace.csv` 的曲線看起來仍在下降。
2. **只走 VAE，不碰 UNet。** 這是這個損失存在的唯一理由；不小心引到取樣器
   的話成本會跳兩個數量級而讀數看起來一樣。
3. **兩個啟動排程。** 官方程式碼有、論文正文沒有。沒接上 `step_hook` 時
   會變成三項全程開啟——**那是另一個目標而旗標名稱不變**。
4. **參照端是原圖的臉。** 不是重建圖的臉；用後者會把「VAE 重建得像不像」
   也算進去。
5. **值域。** `decode_latent` 回 `[-1,1]`、`encode_image` 吃 `[0,1]`，兩邊
   約定相反；接錯時嵌入落在訓練分布外而餘弦全部貼近零。
6. **臉不在支撐內。** 本專案的威脅模型凍結受保護主體，故重建圖的臉只能經由
   VAE 的感受野被支撐內的內容影響。損失必須對支撐內的改動有梯度，否則這一批
   問的問題根本問不出來。

用假的 VAE（可微、形狀正確）跑，不載擴散權重。
"""

import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.defense.facelock_loss import (FACELOCK_START_FR,  # noqa: E402
                                       FACELOCK_START_LPIPS,
                                       FACELOCK_W_LATENT, FACELOCK_W_LPIPS,
                                       make_facelock_loss)

SIDE = 128


class FakeIP2P:
    """**無損**往返的可微假 VAE：`D(E(x)) = x` 逐位元成立。

    公式那幾條要用它：往返有損時 `x' = x` 上的三項都不是它們的極值，
    量到的就不是「式子對不對」而是「這個假 VAE 有多爛」。
    **記下被呼叫過哪些方法**，供第 2 條檢查。
    """

    def __init__(self):
        self.calls = []

    def encode_image(self, x01):
        self.calls.append("encode_image")
        return x01 * 2.0 - 1.0

    def decode_latent(self, z, use_ckpt=False):
        self.calls.append("decode_latent")
        return z

    def __getattr__(self, name):
        raise AssertionError(
            f"facelock 損失呼叫了 {name!r}——它只能用 VAE 的編碼與解碼。")


class FakeIP2PLossy(FakeIP2P):
    """8 倍降取樣 ＋ **一條全域耦合路徑**的假 VAE。

    第 6 條要用它。**只有降取樣是不夠的**：`avg_pool` 是局部的，升回去之後
    右下角的改動仍然到不了左上角的臉框，那一條會假紅。真實的 SD VAE 解碼器
    在最低解析度上有 self-attention，是**全域耦合**的——這裡用一個全域平均項
    當它的最小模型。係數取小，讓局部結構仍然主導。

    無損的假 VAE 在那一條上則會給出假的**通過**（LPIPS 項本來就是全域的），
    故那一條把另外兩項的權重關成 0，只留身分項。
    """

    #: 全域路徑的權重。0 的話這個類別退化成純局部，第 6 條就測不到東西。
    GLOBAL_MIX = 0.05

    def encode_image(self, x01):
        self.calls.append("encode_image")
        return F.avg_pool2d(x01 * 2.0 - 1.0, 8)

    def decode_latent(self, z, use_ckpt=False):
        self.calls.append("decode_latent")
        up = F.interpolate(z, scale_factor=8, mode="bilinear",
                           align_corners=False)
        return up + self.GLOBAL_MIX * z.mean()


def _img(seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(1, 3, SIDE, SIDE, generator=g)


def _face_mask(top=16, left=16, side=40):
    m = torch.zeros(1, 1, SIDE, SIDE)
    m[..., top:top + side, left:left + side] = 1.0
    return m


def _make(steps=100, cls=FakeIP2P, **kw):
    ip2p = cls()
    x = _img()
    fn = make_facelock_loss(ip2p, x_clean=x, face_mask=_face_mask(),
                            steps=steps, **kw)
    return ip2p, x, fn


def test_只用了vae的編碼與解碼():
    ip2p, x, fn = _make()
    fn(x.clone())
    assert set(ip2p.calls) == {"encode_image", "decode_latent"}, ip2p.calls


def test_原圖處三項各自的值都合理():
    """在 `x' = x` 上：latent 差為 0、LPIPS 為 0、身分餘弦為 1。
    故損失恰等於身分那一項的 1.0（兩個排程全開時）。"""
    _ip2p, x, fn = _make()
    fn._advance(100)
    v = float(fn(x.clone()))
    assert v == pytest.approx(1.0, abs=2e-3), v


def test_符號_把臉改壞會讓損失下降():
    """最小化的方向必須是「更不像本人」。寫反時這一條會紅。"""
    _ip2p, x, fn = _make()
    fn._advance(100)
    base = float(fn(x.clone()))
    broken = x.clone()
    broken[..., 16:56, 16:56] = 1.0 - broken[..., 16:56, 16:56]
    assert float(fn(broken)) < base


def test_兩個啟動排程真的會關掉那兩項():
    """沒到啟動點時身分項與 LPIPS 項都不計入，只剩 latent 那一項。
    在 `x' = x` 上 latent 差為 0，故損失恰為 0。"""
    _ip2p, x, fn = _make(steps=100)
    fn._advance(0)
    assert float(fn(x.clone())) == pytest.approx(0.0, abs=1e-6)
    # 身分項在 0.35N 之後才開；0.30N 時仍然關著。
    fn._advance(30)
    assert float(fn(x.clone())) == pytest.approx(0.0, abs=1e-6)
    fn._advance(35)
    assert float(fn(x.clone())) == pytest.approx(1.0, abs=2e-3)


def test_沒接step_hook時視為全開():
    """`step_hook` 沒接上是會靜默改變目標的失效；此時必須是「全開」
    而不是「全關」——全關的話損失只剩 latent 項，而旗標名稱仍寫 facelock。"""
    _ip2p, x, fn = _make()
    assert float(fn(x.clone())) == pytest.approx(1.0, abs=2e-3)


def test_身分項對臉框外的改動有梯度():
    """第 6 條，也是這一批要問的問題本身：**臉在像素上是凍結的**，
    衣物上的圖樣只能經由 VAE 的感受野去改變重建出來的臉。

    另外兩項的權重關成 0，只留身分項——否則 LPIPS 是全域的，這一條會拿到
    一個與感受野無關的假通過。"""
    _ip2p, x, fn = _make(cls=FakeIP2PLossy, w_latent=0.0, w_lpips=0.0)
    fn._advance(100)
    d = torch.zeros_like(x, requires_grad=True)
    fn((x + d).clamp(0, 1)).backward()
    assert d.grad is not None
    far = d.grad[..., 80:, 80:]          # 右下角，完全在臉框（16..56）之外
    assert float(far.abs().sum()) > 0, "臉框外的改動對身分項拿不到梯度"


def test_無損往返時身分項對臉框外的改動沒有梯度():
    """對照：往返無損時重建圖的臉逐位元等於原圖的臉，故臉框外的改動
    **不可能**動到身分項。這一條把上一條的通過歸因到感受野，而不是別的東西。"""
    _ip2p, x, fn = _make(cls=FakeIP2P, w_latent=0.0, w_lpips=0.0)
    fn._advance(100)
    d = torch.zeros_like(x, requires_grad=True)
    fn((x + d).clamp(0, 1)).backward()
    far = d.grad[..., 80:, 80:]
    assert float(far.abs().sum()) == 0.0


def test_權重與排程的值域檢查():
    with pytest.raises(ValueError, match="steps"):
        _make(steps=0)
    with pytest.raises(ValueError, match="w_latent"):
        _make(w_latent=-1.0)
    with pytest.raises(ValueError, match="start_fr"):
        _make(start_fr=1.5)


def test_移植的預設值就是官方程式碼的值():
    """論文正文只給 λ 一個符號、未給數值；兩個排程正文完全沒寫。
    改這四個常數等於改成另一個方法，故釘住。"""
    assert FACELOCK_W_LATENT == 0.2
    assert FACELOCK_W_LPIPS == 1.0
    assert FACELOCK_START_FR == 0.35
    assert FACELOCK_START_LPIPS == 0.25

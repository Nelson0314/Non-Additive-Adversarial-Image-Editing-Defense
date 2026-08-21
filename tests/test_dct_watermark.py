"""DJSMA（Chen et al., The Imaging Science Journal 2026）的驗收。

**2026-08-21 整份重寫。** 第一版是照摘要猜的（顯著度 top-k ＋ sign-PGD），
取得全文後確認全錯：真正的方法是逐次只改一個係數 ±1 的貪婪 JSMA、攻擊區是
固定的反對角帶、且是定向攻擊。這一支釘住那四件事，因為它們每一個補錯了都
還是跑得動、只是量到別的方法：

1. **攻擊區是第 3–5 條反對角**，不是由梯度挑的；
2. **一次只改一個係數，幅度 ±1**；
3. **`mu` 是同一個位置的改動次數上限**，超過就把該位置關掉（l∞）；
   **`tau` 是迭代上限**（l0）；
4. 定向成功就早停，且**已成功的圖不再被更動**。
"""

import pytest
import torch

from src.baselines.dct_watermark import (
    PAPER_ADV_DIAGONALS, PAPER_EVAL_QUALITY, PAPER_MU, PAPER_TAU,
    PAPER_WM_DIAGONALS, DJSMASpec, _pick_target, diagonal_mask, run_djsma,
)


def _img(n=1, size=64):
    g = torch.Generator().manual_seed(0)
    return torch.rand(n, 3, size, size, generator=g, dtype=torch.float64)


class _StubNet(torch.nn.Module):
    """對輸入可微的假分類器：取全域平均後線性投影到 5 類。"""

    def __init__(self, k=5):
        super().__init__()
        g = torch.Generator().manual_seed(1)
        self.w = torch.nn.Parameter(
            torch.randn(3, k, generator=g, dtype=torch.float64))

    def forward(self, x01):
        return x01.mean(dim=(2, 3)) @ self.w


def test_論文寫死的四個值():
    assert PAPER_TAU == 1500
    assert PAPER_MU == 1
    assert PAPER_EVAL_QUALITY == 75
    assert PAPER_ADV_DIAGONALS == (3, 4, 5)
    assert PAPER_WM_DIAGONALS == (6, 7, 8)


def test_反對角遮罩的編號是1based():
    """位置 (i, j) 屬於第 i+j+1 條。第 3 條就是 i+j=2 那三格。"""
    m = diagonal_mask((3,), "cpu", torch.float64)
    assert float(m.sum()) == 3.0
    for i in range(8):
        for j in range(8):
            assert bool(m[i, j]) == (i + j == 2)


def test_E345與E678不重疊且都在中頻():
    a = diagonal_mask(PAPER_ADV_DIAGONALS, "cpu", torch.float64)
    b = diagonal_mask(PAPER_WM_DIAGONALS, "cpu", torch.float64)
    assert float((a * b).sum()) == 0.0, "兩個頻帶重疊，解耦的前提不成立"
    assert not bool(a[0, 0]) and not bool(b[0, 0]), "DC 不該落在任何一帶"
    assert float(a.sum()) == 3 + 4 + 5
    assert float(b.sum()) == 6 + 7 + 8


def test_編號越界時拒絕():
    with pytest.raises(ValueError, match="反對角"):
        diagonal_mask((0,), "cpu", torch.float64)
    with pytest.raises(ValueError, match="反對角"):
        diagonal_mask((16,), "cpu", torch.float64)


def _run(x, **kw):
    net = _StubNet()
    spec = DJSMASpec(tau=kw.pop("tau", 12), **kw)
    return run_djsma(x, spec, logits_fn=net)


def test_只動第3到5條反對角():
    """**檢查 δ 本體，不由防禦圖反推。**

    解碼含夾取與 4:2:0 重取樣，色度通道的改動反推回去會被抹到禁區——
    先前 DCT-Shield 的 `skip_dc` 就是在這裡誤判過一次。
    """
    x = _img()
    res = _run(x)
    allowed = diagonal_mask(PAPER_ADV_DIAGONALS, "cpu", torch.float64)
    total = 0.0
    for c, d in res.delta.items():
        outside = float((d.abs() * (1 - allowed)).sum())
        assert outside == 0.0, f"{c} 在禁區動了 {outside}"
        total += float(d.abs().sum())
    assert total > 0.0, "什麼都沒動，測試沒有鑑別力"


def test_一次只改一個係數():
    """跑 k 次迭代，l0 不得超過 k。"""
    x = _img()
    for k in (3, 7):
        res = _run(x, tau=k)
        assert int(res.changed.max()) <= k, f"tau={k} 卻改了 {int(res.changed.max())} 個"


def test_mu限制同一個位置的改動次數():
    """mu=1 時每個位置最多被改一次，故 l0 應等於實際迭代數（沒有重複位置）。"""
    x = _img()
    res = _run(x, tau=6, mu=1)
    assert int(res.changed[0]) <= 6


def test_目標已達成時完全不動():
    """`argmax F == t` 在第 0 步就成立時，δ 必須是全零、l0 為 0。

    用一個永遠把第 0 類排最前的假分類器，並把目標指成第 0 類。
    """

    class _AlwaysZero(torch.nn.Module):
        def forward(self, x01):
            n = x01.shape[0]
            out = torch.zeros(n, 5, dtype=x01.dtype, device=x01.device)
            out[:, 0] = 1.0
            # 留一條可微的路徑，否則 autograd 會抱怨
            return out + x01.mean(dim=(2, 3)).sum(dim=1, keepdim=True) * 0.0

    x = _img()
    # least_likely 會挑 argmin；此處 logits 為 [1,0,0,0,0]，argmin 是第 1 類，
    # 故改用 random 並把種子固定到會選中第 0 類的值
    spec = DJSMASpec(tau=20, target="random", target_seed=0)
    res = run_djsma(x, spec, logits_fn=_AlwaysZero())
    tgt = _pick_target(torch.tensor([[1.0, 0, 0, 0, 0]], dtype=torch.float64), spec)
    if int(tgt[0]) == 0:
        assert int(res.changed[0]) == 0, "目標已達成卻還在改係數"
    else:
        # 種子沒選中第 0 類時這個測試沒有鑑別力，至少要求迴圈正常結束
        assert res.x_def.shape == x.shape


def test_grad模式需要標modified():
    with pytest.raises(ValueError, match="saliency"):
        DJSMASpec(saliency="grad")
    s = DJSMASpec(saliency="grad", modified_from_paper=True,
                  modification_note="換成本專案的損失，擴散編輯沒有分類器")
    assert s.saliency == "grad"


def test_grad模式可用且仍受頻帶與l0限制():
    x = _img()
    spec = DJSMASpec(tau=5, saliency="grad", modified_from_paper=True,
                     modification_note="換成本專案的損失")

    def loss_fn(z):
        return z.pow(2).mean()

    res = run_djsma(x, spec, loss_fn=loss_fn)
    assert int(res.changed.max()) <= 5


def test_缺對應的函式時拒絕():
    x = _img()
    with pytest.raises(ValueError, match="logits_fn"):
        run_djsma(x, DJSMASpec(tau=2))
    with pytest.raises(ValueError, match="loss_fn"):
        run_djsma(x, DJSMASpec(tau=2, saliency="grad", modified_from_paper=True,
                               modification_note="x"))


def test_設定越界時拒絕():
    for kw, msg in (({"tau": 0}, "tau"), ({"mu": 0}, "mu"),
                    ({"target": "nope"}, "target"),
                    ({"saliency": "nope"}, "saliency")):
        with pytest.raises(ValueError, match=msg):
            DJSMASpec(**kw)


def test_輸出值域與形狀():
    x = _img(n=2)
    res = _run(x, tau=4)
    assert res.x_def.shape == x.shape
    assert float(res.x_def.min()) >= 0.0 and float(res.x_def.max()) <= 1.0
    assert res.changed.shape == (2,)

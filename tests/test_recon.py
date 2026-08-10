"""A 段（DEC-016）重建下限壓縮的三段邏輯：挑參數、還原、停止規則。

抽成不依賴 SD 的純函式就是為了在這裡驗。`descend` 決定 A2 跑多久，而 A2 的
硬停止條件是整個 A 段的安全機制——停太晚，解碼器把原圖背起來，防禦在 latent
上的擾動就傳不到輸出（見 `src/defense/recon.py` 的模組 docstring）。
"""

import pytest
import torch
import torch.nn as nn

from src.defense.recon import (acutance_band, acutance_feasible,
                               blunting_penalty, decoder_tunable, descend,
                               reconstruction_loss, restored)
from src.metrics.acutance import acutance


class Toy(nn.Module):
    """含 GroupNorm、有 bias 的 conv、無 bias 的 conv 與 Linear 各一。"""

    def __init__(self):
        super().__init__()
        self.norm = nn.GroupNorm(2, 4)
        self.conv = nn.Conv2d(4, 4, 3, padding=1)
        self.conv_nb = nn.Conv2d(4, 4, 3, padding=1, bias=False)
        self.proj = nn.Linear(4, 4)


def test_只挑_GroupNorm_affine_與_conv_bias():
    names = [n for n, _ in decoder_tunable(Toy())]
    assert names == ["norm.weight", "norm.bias", "conv.bias"]


def test_無可調參數時拋出():
    """靜默回傳空清單的話，優化器會什麼都不更新而下限「剛好沒降」，
    症狀與「這張圖壓不動」完全一樣。"""
    with pytest.raises(ValueError, match="逐圖微調"):
        decoder_tunable(nn.Sequential(nn.Linear(4, 4)))


def test_restored_把數值與_requires_grad_還原():
    """逐圖微調是針對單張影像的，下一張必須從 stock 權重重新開始。"""
    m = Toy()
    m.requires_grad_(False)
    params = [p for _, p in decoder_tunable(m)]
    before = [p.detach().clone() for p in params]
    with restored(params):
        assert all(p.requires_grad for p in params)
        with torch.no_grad():
            for p in params:
                p.add_(1.0)
    for p, b in zip(params, before):
        assert torch.equal(p, b)
        assert p.requires_grad is False
        assert p.grad is None


def _blur(x):
    return torch.nn.functional.avg_pool2d(x, 3, 1, 1)


def test_銳利度帶保證起點可行():
    """帶的半寬取「容差」與「舊下限自身偏差」的大者，故舊下限一定落在帶內。
    少了這一條，本來就鈍的影像會在第 0 步就違反約束而沒有任何可行落點。"""
    assert acutance_band(0.60, band=0.05) == pytest.approx(0.40)
    assert acutance_band(0.9935, band=0.05) == pytest.approx(0.05)


def test_鈍化_hinge_在帶內不施力而更鈍時施力():
    torch.manual_seed(0)
    x = torch.rand(1, 3, 64, 64)
    floor = _blur(x)                       # 假想的舊下限
    band = acutance_band(acutance(x, floor)["acutance_ratio"])

    assert float(blunting_penalty(floor, x, band)) == pytest.approx(0, abs=1e-6)
    assert float(blunting_penalty(_blur(floor), x, band)) > 0.05


def test_過銳也要罰():
    """單邊版本把最佳化推到反方向：實測衝到銳利度比 2.94，那是加雜訊。"""
    torch.manual_seed(0)
    x = torch.rand(1, 3, 64, 64)
    band = 0.05
    sharper = (x + 0.3 * torch.randn_like(x)).clamp(0, 1)
    assert acutance(x, sharper)["acutance_ratio"] > 1.0 + band
    assert float(blunting_penalty(sharper, x, band)) > 0.0


def test_可行性判準與_hinge_用同一個帶():
    """兩者分岔正是本專案反覆抓到的缺陷型態，故要釘住。"""
    band = 0.05
    ok = acutance_feasible(band)
    assert ok({"acutance_ratio": 1.0 - band}) is True
    assert ok({"acutance_ratio": 1.0 + band}) is True
    assert ok({"acutance_ratio": 1.0 + 2 * band}) is False


def test_鈍化_hinge_可關閉且不影響其餘兩項():
    x = torch.rand(1, 3, 32, 32)
    y = _blur(x)
    lp = lambda a, b: (a - b).pow(2).mean()  # noqa: E731
    off = reconstruction_loss(y, x, lp, 1.0, 0.5, gamma_acut=0.0)
    assert float(off) == pytest.approx(
        float(lp(y, x)) + 0.5 * float((y - x).abs().mean()))


def _toy_descend(target, steps=50, lr=0.5, log_every=1):
    """把 |w| 當成「下限」：一個能被壓到零、逐步單調下降的最小例子。"""
    w = torch.nn.Parameter(torch.tensor([1.0]))
    return descend([w], forward=lambda: w * 1.0,
                   loss_fn=lambda y: (y ** 2).mean(),
                   measure=lambda y: {"lpips": float(y.abs())},
                   steps=steps, lr=lr, key="lpips", target=target,
                   log_every=log_every, tag="toy")


def test_達到目標即停而不跑到收斂():
    """這是 A2 的安全機制本身：目標一達到就停，剩下的步數不跑。"""
    hist, s = _toy_descend(target=0.5)
    assert s["reached"] is True
    assert s["stop_step"] < 50, "達標後仍跑滿步數"
    assert hist[-1]["lpips"] <= 0.5


def test_未達目標如實記錄而不拋出():
    """一張圖的容量上限是要報的結果，不能讓它毀掉整批量測。"""
    hist, s = _toy_descend(target=-1.0, steps=5)
    assert s["reached"] is False
    assert s["stop_step"] == 5
    assert s["best"] == pytest.approx(min(h["lpips"] for h in hist))


def test_回到最佳步而不是末步():
    """指標在逐步之間會震盪，取末步等於把震盪的相位當成結果。"""
    w = torch.nn.Parameter(torch.tensor([1.0]))
    seen = []

    def measure(y):
        # 第 0、1、2 步分別是 1.0、0.1、9.0：最佳在中間，末步最差。
        v = [1.0, 0.1, 9.0][len(seen)]
        seen.append(v)
        return {"lpips": v}

    _, s = descend([w], forward=lambda: w * 1.0,
                   loss_fn=lambda y: (y ** 2).mean(),
                   measure=measure, steps=2, lr=0.1, key="lpips",
                   target=None, log_every=1, tag="toy")
    assert s["best_step"] == 1 and s["best"] == pytest.approx(0.1)


def test_target_為_None_時跑滿步數():
    hist, s = _toy_descend(target=None, steps=8, log_every=4)
    assert s["reached"] is False and s["stop_step"] == 8
    assert [h["step"] for h in hist] == [0, 4, 8]

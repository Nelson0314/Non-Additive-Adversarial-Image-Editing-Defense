"""WaNet 式三元對照的位移場參數化（`src/defense/param_pgd.py`）。

五件事會靜默出錯，逐條釘住：零位移的恆等性（壞了就是每一張圖都先付一筆
無償失真，而且小到不會被任何報表抓到）、位移的單位真的是像素、投影夾的是
粗網格本身（夾錯地方預算就沒有定義）、梯度回得來（斷了 PGD 會安靜地什麼
都不學）、以及 `warp_rand` 與 `warp_roundtrip` 抽到**同一個**隨機場
（不同的話，②③ 之差就混進了抽樣變異，整個比較失效）。

全部在 CPU 上執行。
"""

import pytest
import torch

from src.defense.param_pgd import (
    WarpParam, WarpRandomParam, WarpRoundTripParam,
)


def _img(seed=0, size=64):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(1, 3, size, size, generator=g)


def test_零位移逐位等於原圖():
    """像素中心基準網格＋align_corners=False，邊長為 2 的冪時是精確恆等。

    `align_corners=True` 配 `linspace` 做不到（512 上誤差 5.8e-5），
    那是一筆無償的失真地板。與相位 θ=0、明暗場 m=0 同性質。
    """
    for size in (64, 512):
        x = _img(size=size)
        p = WarpParam(radius=8.0)
        p.reset(x, 0)
        assert torch.equal(p.render(x), x), f"size={size} 不是逐位相等"


def test_零位移的往返也逐位等於原圖():
    x = _img(size=64)
    p = WarpRoundTripParam(radius=8.0)
    p.reset(x, 0)
    with torch.no_grad():
        p.c.zero_()
    assert torch.equal(p.render(x), x)


def test_位移的單位是像素():
    """粗網格填 1.0 → 上採樣後恆為 1 px → 影像整體平移一格。"""
    x = _img(size=64)
    p = WarpParam(radius=4.0, grid=8)
    p.reset(x, 0)
    with torch.no_grad():
        p.c[:, 0].fill_(1.0)      # 只動 x 方向
        p.c[:, 1].zero_()
    y = p.render(x)
    assert torch.allclose(y[:, :, :, :-1], x[:, :, :, 1:], atol=1e-6)


def test_參數量是粗網格大小乘二():
    x = _img()
    p = WarpParam(radius=4.0, grid=16)
    p.reset(x, 0)
    assert p.params()[0].shape == (1, 2, 16, 16)


def test_梯度回得來():
    x = _img()
    p = WarpParam(radius=4.0)
    p.reset(x, 0)
    # 零位移處雙線性的梯度不為零（權重對座標的導數是相鄰像素差），
    # 但為了不依賴那件事，先給一個非零的場。
    with torch.no_grad():
        p.c.normal_(generator=torch.Generator().manual_seed(7))
    p.render(x).pow(2).mean().backward()
    assert float(p.params()[0].grad.abs().sum()) > 0


def test_投影夾的是粗網格本身():
    """雙三次上採樣會過衝，夾在上採樣之後預算就沒有定義。"""
    x = _img()
    p = WarpParam(radius=4.0)
    p.reset(x, 0)
    with torch.no_grad():
        p.c.fill_(100.0)
    p.project()
    assert float(p.c.detach().max()) == pytest.approx(4.0, abs=1e-6)


def test_隨機版不最佳化且真的動了():
    x = _img()
    r = WarpRandomParam(radius=8.0)
    r.reset(x, 3)
    assert r.params() == []
    assert not torch.allclose(r.render(x), x, atol=1e-3)
    assert float(r.c.detach().abs().max()) <= 8.0 + 1e-6


def test_往返版與隨機版用同一個場():
    """②③ 之差要是「幾何」而不是「抽樣運氣」，兩者的 c 必須逐位相同。"""
    x = _img()
    a, b = WarpRandomParam(radius=8.0), WarpRoundTripParam(radius=8.0)
    a.reset(x, 11)
    b.reset(x, 11)
    assert torch.equal(a.c, b.c)


def test_殘餘幾何遠小於單次_warp_但不是零():
    """`−f` 只是 `f` 的一階逆。殘餘量的量級是 `radius² / 粗網格間距`。

    用 `effective_displacement` 直接量搬移量，不用像素差——像素差把幾何與
    內插 artifact 混在一起，而這一格要釘住的正是幾何那一半。
    """
    x = _img(size=512)
    single, trip = WarpRandomParam(radius=4.0), WarpRoundTripParam(radius=4.0)
    single.reset(x, 4)
    trip.reset(x, 4)
    r_single = float(single.effective_displacement(x).pow(2).sum(1).sqrt().mean())
    r_trip = float(trip.effective_displacement(x).pow(2).sum(1).sqrt().mean())
    assert r_trip < 0.2 * r_single
    assert r_trip > 0.0          # 不是精確逆，殘餘幾何確實存在


def test_單次_warp_的實際搬移就是上採樣後的場():
    x = _img(size=128)
    p = WarpRandomParam(radius=3.0)
    p.reset(x, 9)
    d = p._displacement(x)
    eff = p.effective_displacement(x)
    # 邊界一圈被 padding_mode="border" 夾住，故只比內部。
    assert torch.allclose(eff[:, :, 8:-8, 8:-8], d[:, :, 8:-8, 8:-8], atol=1e-3)


def test_往返後內插_artifact_還在():
    """幾何幾乎回到原點，但影像不會回到原圖——那個差就是這一格要量的東西。"""
    x = _img(size=128)
    trip = WarpRoundTripParam(radius=4.0)
    trip.reset(x, 4)
    assert not torch.allclose(trip.render(x), x, atol=1e-3)


def test_隨機版同種子可重現_不同種子不同():
    x = _img()
    a, b, c = (WarpRandomParam(radius=8.0) for _ in range(3))
    a.reset(x, 5)
    b.reset(x, 5)
    c.reset(x, 6)
    assert torch.equal(a.c, b.c)
    assert not torch.equal(a.c, c.c)


def test_粗網格太小時拋錯():
    with pytest.raises(ValueError, match="至少為 2"):
        WarpParam(radius=4.0, grid=1)


def test_邊界用_border_不造出黑框():
    """`zeros` 會在邊緣造出一圈黑，那是位移場以外的東西。"""
    x = torch.full((1, 3, 64, 64), 0.7)
    p = WarpParam(radius=8.0, grid=4)
    p.reset(x, 0)
    with torch.no_grad():
        p.c.fill_(8.0)
    with torch.no_grad():
        y = p.render(x)
    assert float(y.min()) == pytest.approx(0.7, abs=1e-5)


def test_三個條件的名字():
    assert WarpParam(radius=1.0).name == "warp"
    assert WarpRandomParam(radius=1.0).name == "warp_rand"
    assert WarpRoundTripParam(radius=1.0).name == "warp_roundtrip"


def test_三個條件都接上了驅動與_build():
    """接不上就是跑批次時才發現，而那時卡已經佔住了。"""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    from ip2p_run import PHASE_CONDS
    from phase_ablation import WARP_RADIUS_HI, WARP_RADIUS_LO, build

    for name, cls in (("warp", WarpParam), ("warp_rand", WarpRandomParam),
                      ("warp_roundtrip", WarpRoundTripParam)):
        assert name in PHASE_CONDS
        param, lo, hi = build(name, seed=0)
        assert isinstance(param, cls)
        assert param.name == name
        assert (lo, hi) == (WARP_RADIUS_LO, WARP_RADIUS_HI)
        assert param.grid == 16


def test_粗網格邊長是_CSV_的欄位():
    """本專案指定、論文未載的參數必須是欄位而不是註解（CLAUDE.md）。"""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    from ip2p_run import build_parser

    args = build_parser().parse_args(
        ["--out", "x", "--conditions", "warp", "--radius", "8",
         "--warp-grid", "24"])
    assert args.warp_grid == 24
    src = (Path(__file__).resolve().parent.parent
           / "scripts" / "ip2p_run.py").read_text(encoding="utf-8")
    assert '"warp_grid": args.warp_grid,' in src


def test_最佳化那一格不得用_latent_norm():
    """`latent_norm`（‖E(x′)‖₂）對位移場在**零位移處是帶折點的局部極小**。

    實測（`runs/ip2p_warp/step_probe_latent_norm.csv`）：梯度完全正常
    （absmean 3.4e−2、零元素比例 0.0000），但每走一步損失都上升
    （105.95 → 110.07），sign PGD 於是在 0 與 ±α 之間形成週期 2 的振盪、
    `|c|` 恆等於 α。第一輪就是這樣跑出 `dists=0.0001`——**看起來像「最佳化
    買不到東西」，其實是最佳化沒有動過**。這一格若被改回去，整個 ① vs ②
    的判定會靜默變成假的否證，故釘住。
    """
    from pathlib import Path
    sh = (Path(__file__).resolve().parent.parent
          / "scripts" / "warp_triad.sh").read_text(encoding="utf-8")
    assert 'LOSS_OPT="--loss encoder_target"' in sh
    assert "--conditions warp --radius" in sh


def test_步長診斷探針的_CLI_可用():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    import warp_step_probe  # noqa: F401
    assert warp_step_probe.RESOLUTION == 512

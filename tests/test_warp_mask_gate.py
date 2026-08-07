"""位移場的遮罩閘 — `site_warp.WarpResidual` 的 `gate`（2026-08-08 處置 B）。

inpainting 威脅模型下攻擊方會把遮罩內整片重畫，我方在該區域的位移**沒有
防禦價值卻照樣被 `L_fid` 全額收費**。閘要保證的是三件事，本檔逐條釘住：

1. 不給閘時與加閘之前**逐位元相同**（img2img 的既有結果不受影響）；
2. 閘住的區域位移恆為零，且該處 `flow` 的梯度為零（預算真的沒花進去）；
3. 邊界不是硬跳——閘乘在粗網格上，過渡帶展開為一個網格間距。

第 3 條是這個實作與「乘在全解析度位移場上」的唯一差別，而後者在
`attention_box` 的硬邊矩形上會產生撕裂狀瑕疵。少了這條斷言，兩種寫法在
其餘全部斷言上都通過。
"""

import pytest
import torch

from src.residual.site_warp import WarpResidual


def half_mask(size: int) -> torch.Tensor:
    """右半邊為 1（= 攻擊方要重畫的區域），與 `data/masks.py` 同一約定。"""
    m = torch.zeros(1, 1, size, size)
    m[..., size // 2:] = 1.0
    return m


# ---------------------------------------------------------------------------
# coarse_gate
# ---------------------------------------------------------------------------


def test_閘為未被遮罩覆蓋的面積比例():
    g = WarpResidual.coarse_gate(half_mask(64), 8)
    assert g.shape == (1, 1, 8, 8)
    assert torch.allclose(g[..., :4], torch.ones(1, 1, 8, 4))
    assert torch.allclose(g[..., 4:], torch.zeros(1, 1, 8, 4))


def test_比一個網格小的遮罩不會整個消失():
    """降採樣取面積平均而非最近鄰。取最近鄰時只看格中心，一個比格小的遮罩
    會整個消失，而症狀是「閘看起來全開、預算照樣花在會被重畫的區域」。"""
    m = torch.zeros(1, 1, 64, 64)
    m[..., 0:2, 0:2] = 1.0          # 2×2，落在 8×8 格的一角
    g = WarpResidual.coarse_gate(m, 8)
    assert g[0, 0, 0, 0] == pytest.approx(1.0 - 4 / 64)
    assert g[0, 0, 0, 1] == pytest.approx(1.0)


def test_遮罩形狀不符即拋出():
    with pytest.raises(ValueError, match=r"\(1, 1, H, W\)"):
        WarpResidual.coarse_gate(torch.zeros(1, 64, 64), 8)


def test_閘與粗網格形狀不符即拋出():
    """廣播會靜默給出一個閘錯位置的位移場，故在建構時就擋下。"""
    with pytest.raises(ValueError, match="形狀"):
        WarpResidual(size=64, grid_size=8, gate=torch.ones(1, 1, 4, 4))


# ---------------------------------------------------------------------------
# 不給閘 → 逐位元不變
# ---------------------------------------------------------------------------


def test_不給閘時與加閘之前逐位元相同():
    x = torch.rand(1, 3, 64, 64)
    a = WarpResidual(size=64, grid_size=8, max_disp=2.0, init_std=0.4, seed=3)
    b = WarpResidual(size=64, grid_size=8, max_disp=2.0, init_std=0.4, seed=3,
                     gate=torch.ones(1, 1, 8, 8))
    assert torch.equal(a.pixel_residual(x), b.pixel_residual(x))
    assert a.gate is None


def test_閘不進state_dict():
    """閘由遮罩決定、隨 `phi.pt` 的 `build` 欄還原，不是可訓練參數。放進
    `state_dict` 會讓舊 φ 因為缺鍵而載入失敗，而那與「參數形狀不符」的
    真正錯誤混在同一個例外裡。"""
    m = WarpResidual(size=64, grid_size=8, gate=torch.ones(1, 1, 8, 8))
    assert list(m.state_dict()) == ["flow"]


# ---------------------------------------------------------------------------
# 閘住的區域：位移為零、梯度為零
# ---------------------------------------------------------------------------


def test_遮罩內位移為零而遮罩外不為零():
    g = WarpResidual.coarse_gate(half_mask(64), 8)
    m = WarpResidual(size=64, grid_size=8, max_disp=2.0, init_std=0.5, seed=1,
                     gate=g)
    d = m.displacement(64, 64)
    # 遮罩內部（避開過渡帶：閘乘在粗網格上，最後一個開著的格會滲入一格）
    assert d[..., 40:].abs().max() == 0.0
    assert d[..., :24].abs().max() > 0.0


def test_遮罩內的flow梯度為零():
    """預算有沒有真的省下來，看的是梯度而不是位移：位移為零但梯度不為零時
    最佳化仍會往那裡推，只是被閘擋住，`L_fid` 照樣收費。"""
    g = WarpResidual.coarse_gate(half_mask(64), 8)
    m = WarpResidual(size=64, grid_size=8, max_disp=2.0, init_std=0.3, seed=2,
                     gate=g)
    x = torch.rand(1, 3, 64, 64)
    m.pixel_residual(x).pow(2).sum().backward()
    assert m.flow.grad[..., 4:].abs().max() == 0.0
    assert m.flow.grad[..., :4].abs().max() > 0.0


def test_閘住整張圖時輸出即原圖():
    m = WarpResidual(size=64, grid_size=8, max_disp=2.0, init_std=0.5, seed=4,
                     gate=torch.zeros(1, 1, 8, 8))
    x = torch.rand(1, 3, 64, 64)
    with torch.no_grad():
        assert torch.equal(m.pixel_residual(x), x)


# ---------------------------------------------------------------------------
# 邊界的平滑性
# ---------------------------------------------------------------------------


def test_邊界的過渡帶展開為一個網格間距():
    """閘乘在粗網格上，故位移量沿橫向的逐像素落差以 (一格的位移 / 網格間距)
    為量級。乘在全解析度位移場上時，同一條線上會出現一次等於全幅的跳變。
    """
    size, grid = 64, 8
    g = WarpResidual.coarse_gate(half_mask(size), grid)
    m = WarpResidual(size=size, grid_size=grid, max_disp=4.0, gate=g)
    with torch.no_grad():
        m.flow.fill_(1.0)                     # 全域等幅，落差只可能來自閘
        d = m.displacement(size, size)[0, 0]  # x 分量，(H, W)
    step = (d[:, 1:] - d[:, :-1]).abs().max()
    span = size // grid                        # 一個網格間距 = 8 px
    assert step < 1.0 / span * 1.5, "過渡帶比一個網格間距窄，等於硬邊"
    assert step > 0.0


def test_disp_stats記錄有沒有加閘():
    """兩種形態的位移統計看起來一樣，事後只憑 disp_max 分不出這一格是不是
    只動了脈絡。"""
    plain = WarpResidual(size=64, grid_size=8).disp_stats()
    assert plain["mask_gated"] is False and plain["gate_open_frac"] == 1.0

    g = WarpResidual.coarse_gate(half_mask(64), 8)
    gated = WarpResidual(size=64, grid_size=8, gate=g).disp_stats()
    assert gated["mask_gated"] is True
    assert gated["gate_open_frac"] == pytest.approx(0.5)

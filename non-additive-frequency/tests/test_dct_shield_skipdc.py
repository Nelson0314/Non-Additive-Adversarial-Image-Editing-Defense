"""`DCTShieldSpec.skip_dc` —— 重現落差的檢定旗標，**不是論文的東西**。

2026-08-20 的診斷：本專案實作出的 δ 在每個係數上都吃滿 ±1，而 DC 是最飽和的
位置（Y 1.02／Cb 1.13／Cr 0.96 個量化階）。DC 是區塊的平均亮度與色度，整階
平移即每個 8×8 區塊整體變亮或變色，平坦區的可見方格由此而來。論文報的失真是
LPIPS 0.267、本專案量到 0.469，而 PSNR 反而高 1.83 dB——能量更小卻更醜。

要釘三件事：

1. 預設**必須是 False**，否則會靜默改掉被重現的方法；
2. 打開時 DC 的 δ 恆為零，其餘係數不受影響；
3. 打開時 `modified_from_paper` 由呼叫端負責標註——這裡只驗旗標本身。
"""

import pytest
import torch

from src.baselines.dct_shield import (
    PAPER_EPS, PAPER_GAMMA, DCTShieldSpec, run_dct_shield,
)
from src.baselines.jpeg_codec import jpeg_encode


class _StubSD:
    """只需要 `encode_image`：`run_dct_shield` 的預設損失是 `‖E(x')‖₂`。

    取每 8×8 區塊的平均當 latent，形狀對得上真的 VAE，且對輸入可微。
    """

    def encode_image(self, x01, use_ckpt: bool = False):
        return torch.nn.functional.avg_pool2d(x01, 8)


def _img(size=64):
    g = torch.Generator().manual_seed(0)
    return torch.rand(1, 3, size, size, generator=g, dtype=torch.float64)


def test_預設不跳過DC():
    assert DCTShieldSpec(name="x").skip_dc is False


def test_跳過DC卻沒標modified時拒絕():
    """檢定用的變體混進 baseline 的表而看不出來，是最容易犯的錯。"""
    with pytest.raises(ValueError, match="skip_dc"):
        DCTShieldSpec(name="x", skip_dc=True)


def _delta(spec, x):
    """取最佳化出來的 δ 本體。

    **不能由防禦圖反推**：解碼含夾取與 4:2:0 重取樣，AC 的擾動會滲進反推出
    的 DC，於是「DC 是否為零」在像素端根本驗不出來。`run_dct_shield` 自
    2026-08-20 起直接回傳 δ。
    """
    return run_dct_shield(_StubSD(), x, spec).delta


def test_打開時DC完全不動():
    x = _img()
    spec = DCTShieldSpec(name="nodc", eps=PAPER_EPS, gamma=PAPER_GAMMA, steps=8,
                         skip_dc=True, modified_from_paper=True,
                         modification_note="排除 DC，檢定重現落差用")
    d = _delta(spec, x)
    for c, t in d.items():
        assert torch.allclose(t[..., 0, 0], torch.zeros_like(t[..., 0, 0]), atol=1e-6), \
            f"{c} 的 DC 被動到了"


def test_關閉時DC會被動到():
    # 對照組：不跳過時 DC 應該確實有非零的 δ，否則上面那個測試沒有鑑別力。
    x = _img()
    spec = DCTShieldSpec(name="base", eps=PAPER_EPS, gamma=PAPER_GAMMA, steps=8)
    d = _delta(spec, x)
    assert any(t[..., 0, 0].abs().sum() > 0 for t in d.values()), \
        "沒有任何通道的 DC 被動到，對照組不成立"


def test_跳過DC不影響AC():
    """AC 仍然照常最佳化——旗標只該關掉一個係數，不是整體縮小。"""
    x = _img()
    kw = dict(eps=PAPER_EPS, gamma=PAPER_GAMMA, steps=8)
    d = _delta(DCTShieldSpec(name="nodc", skip_dc=True, modified_from_paper=True,
                             modification_note="排除 DC，檢定重現落差用", **kw), x)
    ac = d["Y"][..., 1:, 1:].abs()
    assert float(ac.mean()) > 0.0, "AC 完全沒動，旗標把整體關掉了"

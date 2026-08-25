"""區塊置換探針的守門測試（`scripts/residual_permute_probe.py`）。

這支探針判定 `SURVEY_ARCHITECTURE` 第三節的 H1（DCT-Shield 統計驅動、本方法
實現驅動），而它的整個論證建立在「置換**只**動區塊的位置、不動區塊的內容」
上。若置換順手改了內容（能量、色彩關係、邊界），量到的就不是 H1 而是別的
東西，且不會有任何症狀——輸出仍是一張看起來合理的殘差圖。故逐條釘住。

全部在 CPU 上執行，不需要擴散模型。
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from residual_permute_probe import permute_blocks  # noqa: E402


def _residual(seed=0, c=3, h=64, w=64):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(1, c, h, w, generator=g) * 0.05


def _tiles(t, block):
    """把 (1,C,H,W) 拆成一組可雜湊的方格內容。"""
    _, c, h, w = t.shape
    return sorted(
        tuple(t[0, :, i:i + block, j:j + block].flatten().tolist())
        for i in range(0, h, block)
        for j in range(0, w, block)
    )


def test_置換保留每個方格的內容逐位元不變():
    r = _residual()
    p = permute_blocks(r, 32, seed=0)
    assert _tiles(r, 32) == _tiles(p, 32)


def test_置換不改變總能量():
    r = _residual()
    p = permute_blocks(r, 32, seed=0)
    assert torch.allclose(r.pow(2).sum(), p.pow(2).sum())


def test_置換確實打亂位置():
    """不打亂的話整支探針量的是 0，而那看起來像「H1 成立」。"""
    r = _residual()
    assert not torch.equal(r, permute_blocks(r, 16, seed=0))


def test_三個通道共用同一個排列():
    """逐通道各自排列會連色彩結構一起破壞，歸因就混掉了。

    構造一個各通道為常數的殘差：共用排列時每個方格內三通道的比例不變。
    """
    r = torch.zeros(1, 3, 64, 64)
    for i in range(2):
        for j in range(2):
            r[0, 0, i * 32:(i + 1) * 32, j * 32:(j + 1) * 32] = 0.1 * (2 * i + j + 1)
            r[0, 1] = r[0, 0] * 2.0
            r[0, 2] = r[0, 0] * 3.0
    p = permute_blocks(r, 32, seed=1)
    assert torch.allclose(p[0, 1], p[0, 0] * 2.0)
    assert torch.allclose(p[0, 2], p[0, 0] * 3.0)


def test_同種子可重現_不同種子給不同排列():
    r = _residual()
    assert torch.equal(permute_blocks(r, 16, seed=7), permute_blocks(r, 16, seed=7))
    assert not torch.equal(permute_blocks(r, 16, seed=7),
                           permute_blocks(r, 16, seed=8))


def test_只有一個方格時是恆等():
    r = _residual(h=32, w=32)
    assert torch.equal(permute_blocks(r, 32, seed=3), r)


def test_邊長不整除時拋錯而不是默默切齊():
    """切不齊時若靜默捨去，殘差的邊緣會被丟掉而 RMS 悄悄變小。"""
    r = _residual(h=60, w=64)
    with pytest.raises(ValueError, match="整數倍"):
        permute_blocks(r, 32, seed=0)


def test_形狀不合時拋錯():
    with pytest.raises(ValueError, match=r"\(1,C,H,W\)"):
        permute_blocks(torch.zeros(2, 3, 64, 64), 32, seed=0)

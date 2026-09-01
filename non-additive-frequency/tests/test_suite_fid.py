"""`MetricSuite.fid`：使用者 2026-08-19 定案的指標清單含 FID。

FID 是**分布指標**，吃兩組影像而不是一對，故不在 `pairwise` 裡。三件事要釘：
同一組對自己是 0、不同分布為正、少於兩張時拒絕而不是靜默回傳 NaN。

最後一條是刻意的：協方差在單張上無定義，回傳 NaN 會讓小樣本數字悄悄進表。
可信度下限（`FID_MIN_TRUSTED = 150`）另由報表端負責，不在此攔阻——煙霧測試
要跑得動。
"""

import pytest
import torch

from src.metrics.suite import MetricSuite


@pytest.fixture(scope="module")
def suite():
    return MetricSuite()


def test_同一組影像的FID為零(suite):
    g = torch.Generator().manual_seed(0)
    a = torch.rand(4, 3, 64, 64, generator=g)
    assert suite.fid(a, a.clone()) == pytest.approx(0.0, abs=1e-3)


def test_不同分布的FID為正且大於同分布(suite):
    g = torch.Generator().manual_seed(1)
    a = torch.rand(6, 3, 64, 64, generator=g)
    b = (a + 0.3 * torch.rand(6, 3, 64, 64, generator=g)).clamp(0, 1)
    assert suite.fid(a, b) > suite.fid(a, a.clone())


def test_兩組張數可以不同(suite):
    # FID 比的是兩個高斯，兩側樣本數不必相等。
    g = torch.Generator().manual_seed(2)
    a = torch.rand(6, 3, 64, 64, generator=g)
    b = torch.rand(4, 3, 64, 64, generator=g)
    assert suite.fid(a, b) >= 0.0


def test_單張時拒絕(suite):
    a = torch.rand(3, 3, 64, 64)
    with pytest.raises(ValueError, match="至少需要 2 張"):
        suite.fid(a[:1], a)


def test_可信度下限是150(suite):
    # 與 DCT-Shield（arXiv:2504.17894）Table 1 的 150 張一致；報表端據此留空。
    assert MetricSuite.FID_MIN_TRUSTED == 150

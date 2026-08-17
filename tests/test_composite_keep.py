"""`apa_baseline.composite_keep`：把保留區換成同一份參照。

存在的理由是量測上的缺陷而不是美觀：`sdedit(keep01=...)` 的保留區換回的是
輸入圖的內容，於是 `edit_orig` 的保留區是原圖、`edit_def` 的保留區是防禦圖，
兩者的 LPIPS 會把防禦擾動本身算進「編輯被推開多少」。遮罩從只有頭擴大到
整個人之後這一塊足以主導讀數，故必須有這一步，且必須測到「保留區歸零」。
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from apa_baseline import composite_keep  # noqa: E402


def _mask(h=16, w=16):
    m = torch.zeros(1, 1, h, w)
    m[..., 2:8, 3:9] = 1.0
    return m


def test_沒有遮罩時原樣回傳():
    x = torch.rand(1, 3, 16, 16)
    out = composite_keep(torch.zeros_like(x), x, None)
    assert out is x


def test_遮罩內換成參照遮罩外保留原值():
    ref = torch.zeros(1, 3, 16, 16)
    x = torch.ones(1, 3, 16, 16)
    m = _mask()
    out = composite_keep(ref, x, m)
    assert torch.equal(out[..., 2:8, 3:9], torch.zeros(1, 3, 6, 6))
    assert out[..., 0, 0].allclose(torch.ones(3))


def test_兩條分支合成後在遮罩內逐位相同():
    """這是整個函式的目的：保留區對 LPIPS 的貢獻必須歸零。"""
    ref = torch.rand(1, 3, 16, 16)
    a = torch.rand(1, 3, 16, 16)
    b = torch.rand(1, 3, 16, 16)
    m = _mask()
    ca, cb = composite_keep(ref, a, m), composite_keep(ref, b, m)
    inside = m.expand_as(ca) > 0
    assert torch.equal(ca[inside], cb[inside])
    assert not torch.equal(ca[~inside], cb[~inside])


def test_羽化的邊界照比例混合而不是二值化():
    ref = torch.zeros(1, 3, 4, 4)
    x = torch.ones(1, 3, 4, 4)
    m = torch.full((1, 1, 4, 4), 0.25)
    out = composite_keep(ref, x, m)
    assert torch.allclose(out, torch.full((1, 3, 4, 4), 0.75))

"""對照張的格子尺寸。**原生 512 不縮放這件事要能被檢查到。**

判「單純劣化還是原圖不可辨」看的是接縫與鱗片重影，縮放會把它們平均掉；
`runs/obedience_audit` 的判定就是在原生解析度上做的。
"""

import sys
from pathlib import Path

import pytest
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from edit_contact_sheet import RESOLUTION, TILE, build_sheet, tile_of  # noqa: E402


@pytest.fixture()
def png(tmp_path):
    def make(name, value):
        arr = (torch.full((RESOLUTION, RESOLUTION, 3), value) * 255).to(
            torch.uint8).numpy()
        p = tmp_path / name
        Image.fromarray(arr).save(p)
        return p
    return make


def test_原生尺寸時不經過任何內插(png):
    """棋盤格在縮放之後會被平均掉，逐位元相等即證明沒有縮放。"""
    arr = torch.zeros(RESOLUTION, RESOLUTION, 3)
    arr[::2, ::2] = 1.0
    p = Path(png("checker.png", 0.0))
    Image.fromarray((arr * 255).to(torch.uint8).numpy()).save(p)
    got = tile_of(p, RESOLUTION)
    assert got.shape == (3, RESOLUTION, RESOLUTION)
    assert float(got.max()) == pytest.approx(1.0)
    assert float(got.min()) == pytest.approx(0.0)


def test_較小的格子確實縮放(png):
    p = png("flat.png", 0.4)
    assert tile_of(p, 128).shape == (3, 128, 128)


def test_版面尺寸跟著格子走(png):
    a, b = png("a.png", 0.2), png("b.png", 0.8)
    small = build_sheet([(1, a, b)], tile=TILE)
    full = build_sheet([(1, a, b)], tile=RESOLUTION)
    assert full.shape[-1] > small.shape[-1]
    assert full.shape[-1] == 2 * RESOLUTION + 3 * 10

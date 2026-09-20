"""直方圖匹配這個「空模型」本身要成立，匹配後的位移才讀得懂。

最重要的一條是 `test_global_monotone_map_is_undone`：載體是純顏色全域映射，
所以匹配必須有能力把一個全域單調映射幾乎還原。還不回去的話，匹配後的位移
就不是「色彩校正還不回去的那一部分」，而是匹配自己的殘差。
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.metrics.colour_normalised import BINS, match_histogram  # noqa: E402


def smooth_image(seed: int = 0, size: int = 128) -> torch.Tensor:
    """低頻的彩色影像：直方圖有寬度，又不是逐點白雜訊。"""
    generator = torch.Generator().manual_seed(seed)
    coarse = torch.rand((1, 3, 8, 8), generator=generator)
    return torch.nn.functional.interpolate(
        coarse, size=(size, size), mode="bicubic", align_corners=False).clamp(0, 1)


def advcf_curve(pieces: int = 64, radius: float = 5.0):
    from src.defense.color_param import ColorCurveParam
    carrier = ColorCurveParam(radius=radius, pieces=pieces,
                              bound_mode="advcf", init_jitter=1.0)
    return carrier


def test_matching_an_image_to_itself_is_near_identity():
    x = smooth_image(seed=3)
    y = match_histogram(x, x)
    # 256 格量化的上限是半格。
    assert float((y - x).abs().mean()) < 1.0 / BINS


def test_lut_is_monotone():
    x = smooth_image(seed=4)
    ref = smooth_image(seed=5)
    matched = match_histogram(x, ref)
    for c in range(3):
        src = x[0, c].reshape(-1)
        dst = matched[0, c].reshape(-1)
        order = torch.argsort(src)
        run = dst[order]
        assert bool((run[1:] - run[:-1] >= -1e-6).all()), f"通道 {c} 的查表不單調"


def portrait() -> torch.Tensor:
    """真的人像：肖像只走過 RGB 立方體的一小塊，合成圖的直方圖比它寬。"""
    from PIL import Image
    import numpy as np
    path = Path(__file__).resolve().parent.parent / "data/portraits/man/man_00.png"
    array = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4, 5])
def test_global_monotone_map_is_undone(seed):
    """AdvCF 的曲線套上去之後，匹配回原圖的殘差要落到量化底線。

    底線是 `2/BINS`：256 格的格寬本身。殘差掉到這個量級，表示匹配把整個
    全域單調映射都還回去了，剩下的只有量化。
    """
    for x in (smooth_image(seed=6, size=256), portrait()):
        carrier = advcf_curve()
        carrier.reset(x, seed=seed)
        with torch.no_grad():
            y = carrier.render(x).clamp(0, 1)
        shifted = float((y - x).abs().mean())
        assert shifted > 0.005, f"這條曲線幾乎沒動到影像（平均位移 {shifted:.4f}）"
        residual = float((match_histogram(y, x) - x).abs().mean())
        assert residual < 2.0 / BINS, (
            f"匹配沒有還原到量化底線：原位移 {shifted:.4f}、殘差 {residual:.4f}")


def test_shape_is_checked():
    x = smooth_image(seed=8)
    with pytest.raises(ValueError):
        match_histogram(x, smooth_image(seed=8, size=64))
    with pytest.raises(ValueError):
        match_histogram(x[0], x[0])

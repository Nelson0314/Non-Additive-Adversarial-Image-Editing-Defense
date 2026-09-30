"""影像載入器明給尺寸時須同時滿足寬高契約。"""
import sys
from pathlib import Path

from PIL import Image
import pytest

from immunization_core.io import load_image_tensor


@pytest.mark.parametrize("width,height,size,shape", [
    (16, 8, 16, (16, 16)), (8, 16, 16, (16, 16)),
    (16, 16, 16, (16, 16)), (16, 8, None, (8, 16)),
])
def test_explicit_size_checks_both_dimensions(tmp_path, width, height, size, shape):
    path = tmp_path / "input.png"
    Image.new("RGB", (width, height), (31, 127, 254)).save(path)
    tensor = load_image_tensor(path, "cpu", size=size)
    assert tuple(tensor.shape) == (1, 3, *shape)
    assert tensor.min() >= 0 and tensor.max() <= 1

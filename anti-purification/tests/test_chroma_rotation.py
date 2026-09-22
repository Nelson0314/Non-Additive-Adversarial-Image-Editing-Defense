import math

import torch

from src.defense.lowfreq_color import ChromaAffineParam
from src.defense.ncf_param import rgb_to_lab


def _image(seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(1, 3, 64, 64, generator=g, dtype=torch.float32)


def _param(rotation_deg, amplitude=1.0):
    return ChromaAffineParam(
        target_mean=[60.0, 25.0, -20.0],
        target_cov=[[80.0, 0.0, 0.0], [0.0, 40.0, 0.0], [0.0, 0.0, 40.0]],
        support=torch.ones(1, 1, 64, 64), radius=0.2, max_gain=1.0,
        amplitude=amplitude, isometric=True, rotation_deg=rotation_deg)


def test_rotation_matrix_is_the_requested_angle():
    x = _image()
    p = _param(90.0)
    p.reset(x, seed=0)
    m = p.chroma_matrix()
    want = torch.tensor([[0.0, -1.0], [1.0, 0.0]], dtype=m.dtype)
    assert torch.allclose(m, want, atol=1e-9)


def test_rotation_keeps_unit_singular_values():
    x = _image()
    for deg in (30.0, 90.0, 180.0):
        p = _param(deg)
        p.reset(x, seed=0)
        s = torch.linalg.svdvals(p.chroma_matrix())
        assert torch.allclose(s, torch.ones(2, dtype=s.dtype), atol=1e-9)


def test_a_bigger_angle_moves_more_colour():
    x = _image()
    moved = []
    for deg in (30.0, 90.0, 180.0):
        p = _param(deg)
        p.reset(x, seed=0)
        moved.append(float((rgb_to_lab(p.render(x))[:, 1:]
                            - rgb_to_lab(x)[:, 1:]).pow(2).mean()))
    assert moved[0] < moved[1] < moved[2]


def test_lab_isometry_does_not_bound_the_rgb_high_frequency():
    """色度平面上的等距**不保證** RGB 的高通殘差不上升。

    奇異值恰為 1 是在 Lab 的 (a,b) 平面上成立的；`highfreq_report` 的操作性
    讀數在 RGB 上，而 Lab→RGB 是非線性的（gamma 加 3×3），所以同一個色度梯度
    轉到不同色相之後，映進 RGB 的梯度可以變大。

    **而且上界隨影像內容變，不是角度的函數。** 兩組量到的形狀完全不同：
    真實照片（`task_env_weather_70149`，512²）0–135 度是 0.966–0.980、
    180 度 1.062；這裡的均勻隨機圖 30 度就 1.079、90 度峰值 1.338、180 度
    1.100。所以不加高頻這條約束在等距臂上**既不是構造保證、也不是一個通用的
    角度上界**，只能逐影像逐角度照量。這一筆把該事實釘住，避免把 Lab 的保證
    誤讀成 RGB 的保證。
    """
    x = _image()
    ratios = []
    for deg in (0.0, 30.0, 90.0, 135.0, 180.0):
        p = _param(deg)
        p.reset(x, seed=0)
        ratios.append(p.diagnostics(x)['hf_ratio_rgb_total'])
    assert max(ratios) > 1.0, '等距不保證 RGB 高頻不上升，這一筆就是那個反例'


def test_rotation_requires_the_isometric_arm():
    try:
        ChromaAffineParam(
            target_mean=[60.0, 25.0, -20.0],
            target_cov=[[80.0, 0.0, 0.0], [0.0, 40.0, 0.0], [0.0, 0.0, 40.0]],
            support=torch.ones(1, 1, 64, 64), isometric=False,
            rotation_deg=90.0)
    except ValueError as e:
        assert 'isometric' in str(e)
    else:
        raise AssertionError('非等距臂給 rotation_deg 必須拋錯，不得靜默忽略')


def test_rotation_deg_appears_in_diagnostics():
    x = _image()
    p = _param(180.0)
    p.reset(x, seed=0)
    assert p.diagnostics(x)['rotation_deg'] == 180.0
    assert math.isclose(p.rotation_deg, 180.0)

import torch

from scripts.color_ceiling import COLUMNS, build_param


def _kwargs(**extra):
    base = dict(support=torch.ones(1, 1, 64, 64),
                target_mean=[60.0, 25.0, -20.0],
                target_cov=[[80.0, 0.0, 0.0], [0.0, 40.0, 0.0], [0.0, 0.0, 40.0]],
                blur_sigma=24.0)
    base.update(extra)
    return base


def test_columns_carry_the_four_readouts_and_the_appearance_group():
    for name in ('arm', 'image', 'class', 'seed', 'eval_seed',
                 'delta_e_target', 'amplitude', 'delta_e_reached',
                 'edit_lpips',
                 'subject_id_def', 'subject_id_orig', 'subject_id_drop',
                 'n_faces_edit_def', 'n_faces_edit_orig',
                 'input_psnr', 'input_dists', 'final_deltaE00',
                 'final_hf_rgb_total', 'final_outside_support_max_abs'):
        assert name in COLUMNS


def test_build_param_rejects_an_unknown_arm():
    x = torch.rand(1, 3, 64, 64)
    try:
        build_param('nope', x, **_kwargs())
    except ValueError as e:
        assert 'nope' in str(e)
    else:
        raise AssertionError('未知的臂必須拋錯，不得靜默走預設')


def test_isometric_arm_is_wired_to_the_isometric_flag():
    x = torch.rand(1, 3, 64, 64)
    p = build_param('chroma_isometric', x, **_kwargs(rotation_deg=90.0))
    assert p.isometric is True
    assert p.rotation_deg == 90.0


def test_bounded_arm_is_not_isometric():
    x = torch.rand(1, 3, 64, 64)
    p = build_param('chroma_bounded', x, **_kwargs())
    assert p.isometric is False
    assert p.max_gain == 1.0


def test_region_palette_without_regions_raises():
    x = torch.rand(1, 3, 64, 64)
    try:
        build_param('region_palette', x, **_kwargs())
    except ValueError as e:
        assert 'regions' in str(e)
    else:
        raise AssertionError('region_palette 沒有 regions 必須拋錯')

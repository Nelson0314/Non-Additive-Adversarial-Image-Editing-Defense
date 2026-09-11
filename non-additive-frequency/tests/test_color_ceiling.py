import torch

from scripts.color_ceiling import COLUMNS, build_param


def _kwargs(**extra):
    base = dict(support=torch.ones(1, 1, 64, 64),
                target_mean=[60.0, 25.0, -20.0],
                target_cov=[[80.0, 0.0, 0.0], [0.0, 40.0, 0.0], [0.0, 0.0, 40.0]],
                blur_sigma=24.0)
    base.update(extra)
    return base


def test_columns_carry_the_readouts_and_the_appearance_group():
    for name in ('arm', 'image', 'class', 'seed', 'eval_seed',
                 'delta_e_target', 'amplitude', 'delta_e_reached',
                 'edit_lpips',
                 'subject_id_def', 'subject_id_orig', 'subject_id_drop',
                 'subject_box_iou_edit_def',
                 # 語意取代了臉數：臉數只回答「畫面上有幾張臉」，會把旁人算
                 # 進來，而主體位置上有沒有臉由 box_iou 與空的 id 表達。
                 'edit_clip_orig', 'edit_clip_def', 'edit_clip_drop',
                 'edit_siglip_orig', 'edit_siglip_def', 'edit_siglip_drop',
                 'input_psnr', 'input_dists', 'final_deltaE00',
                 'final_hf_rgb_total', 'final_outside_support_max_abs',
                 'attack_batch', 'attack_precision'):
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


def test_collision_arm_records_its_objective_trace():
    for name in ('collision_first', 'collision_last', 'collision_at_anchor',
                 'collision_steps', 'collision_region_area',
                 'collision_ring_area'):
        assert name in COLUMNS


def test_collision_arm_is_not_isometric():
    """碰撞臂要容許收縮：奇異值恰為 1 時兩塊區域的均值差長度被保住，推不動。"""
    x = torch.rand(1, 3, 64, 64)
    p = build_param('collision', x, **_kwargs())
    assert p.isometric is False
    assert p.max_gain == 1.0


def test_collision_region_takes_a_region_name_not_a_class_name():
    """區域由設定逐類指定，不由類名推。

    指令類可以有很多個（帽子、太陽眼鏡、皇冠……）卻共用同一塊區域，靠類名
    硬判的話每加一個指令就要改程式，而且加錯了會靜默走到別的區域。
    """
    from scripts.color_ceiling import collision_region
    x = torch.rand(1, 3, 64, 64)
    clothes = torch.zeros(1, 1, 64, 64)
    clothes[:, :, 20:40, 10:30] = 1.0
    assert torch.equal(collision_region(x, 'clothes', clothes), clothes)


def test_collision_region_rejects_an_unknown_region():
    from scripts.color_ceiling import collision_region
    x = torch.rand(1, 3, 64, 64)
    try:
        collision_region(x, 'nope', torch.ones(1, 1, 64, 64))
    except ValueError as e:
        assert 'nope' in str(e)
    else:
        raise AssertionError('未知的區域必須拋錯，不得靜默走預設')


def test_max_reach_target_is_a_recognised_value():
    """`max_reach` 不是等失真錨點，是各臂自己的幅度 1.0。"""
    import json
    from pathlib import Path
    spec = json.loads(Path('configs/color_scaleup.json').read_text(encoding='utf-8'))
    assert 'max_reach' in spec['delta_e_targets']
    assert 6.3 in spec['delta_e_targets']
    assert len(spec['images']) == 20
    assert len(spec['classes']) == 6
    assert {c['region'] for c in spec['classes']} == {
        'clothes', 'head_ring', 'outside_subject'}


def test_columns_record_the_effective_gain_after_amplitude_interpolation():
    """`isometric` 記的是插值前的設定，插值後的有效增益要另外記。

    (1-a)·I + a·R(90度) 在 a = 0.5 上的奇異值是 0.707——那是收縮不是等距，
    只看 `isometric=1` 會誤述機制。
    """
    for name in ('effective_gain_max', 'effective_gain_min', 'palette_id_second'):
        assert name in COLUMNS


def test_amplitude_interpolation_contracts_a_rotation():
    from src.defense.lowfreq_color import ChromaAffineParam
    x = torch.rand(1, 3, 64, 64)
    p = ChromaAffineParam(
        target_mean=[60.0, 25.0, -20.0],
        target_cov=[[80.0, 0.0, 0.0], [0.0, 40.0, 0.0], [0.0, 0.0, 40.0]],
        support=torch.ones(1, 1, 64, 64), radius=0.2, max_gain=1.0,
        isometric=True, rotation_deg=90.0, amplitude=0.5)
    p.reset(x, seed=0)
    assert abs(float(torch.linalg.svdvals(p.chroma_matrix()).max()) - 1.0) < 1e-9
    eff = float(torch.linalg.svdvals(p.effective_chroma_matrix()).max())
    assert abs(eff - 0.7071) < 1e-3


def test_face_counts_are_not_reported():
    """臉數不進報表。

    它只回答「畫面上有幾張臉」，多人畫面上會把旁人算進來；主體位置上還有
    沒有臉已經由 `subject_box_iou_edit_def` 與空的 `subject_id_def` 表達。
    """
    for name in ('n_faces_orig', 'n_faces_edit_orig', 'n_faces_edit_def'):
        assert name not in COLUMNS

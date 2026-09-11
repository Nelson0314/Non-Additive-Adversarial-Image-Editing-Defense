import pytest

from src.defense.criterion import criterion_score, normalise_terms, softmin


BASE = dict(id_def=0.80, id_control=0.85, clip_def=0.06, clip_control=0.08,
            quality_def=5.0, quality_control=5.0, subject_iou=0.6)


def test_三項都與未防禦相同時每項都是一():
    t = normalise_terms(id_def=0.85, id_control=0.85, clip_def=0.08,
                        clip_control=0.08, quality_def=5.0, quality_control=5.0,
                        subject_iou=0.7)
    assert t['id_norm'] == 1.0 and t['dir_norm'] == 1.0 and t['use_norm'] == 1.0
    assert criterion_score(t, tau=0.0) == 1.0


def test_任一條垮掉分數就低_因為判準是連言():
    only_id = normalise_terms(**{**BASE, 'id_def': 0.05})
    only_use = normalise_terms(**{**BASE, 'quality_def': 500.0})
    assert criterion_score(only_id, tau=0.0) < 0.1
    assert criterion_score(only_use, tau=0.0) < 0.1


def test_指令那一項照算但不進目標函數():
    flat = normalise_terms(**{**BASE, 'clip_def': 0.0})
    assert flat['dir_norm'] == 0.0
    assert criterion_score(flat, tau=0.0) == min(flat['id_norm'],
                                                 flat['use_norm'])
    assert criterion_score(flat, tau=0.0) > 0.9
    assert criterion_score(flat, tau=0.0,
                           keys=('id_norm', 'dir_norm', 'use_norm')) == 0.0


def test_主體框上沒有臉時身分項記零且旗標分開():
    t = normalise_terms(**{**BASE, 'subject_iou': 0.0, 'id_def': ''})
    assert t['id_norm'] == 0.0
    assert t['subject_lost'] == 1
    assert t['control_subject_lost'] == 0


def test_未防禦本身就找不到臉時防禦拿不到分():
    t = normalise_terms(**{**BASE, 'id_control': '', 'id_def': '',
                           'subject_iou': 0.0})
    assert t['id_norm'] == 1.0
    assert t['control_subject_lost'] == 1


def test_未防禦的指令沒往前走時方向項不給分():
    t = normalise_terms(**{**BASE, 'clip_control': 0.0, 'clip_def': 0.0})
    assert t['dir_norm'] == 1.0
    assert t['control_direction_flat'] == 1


def test_品質越差使用性越低():
    worse = normalise_terms(**{**BASE, 'quality_def': 20.0})
    same = normalise_terms(**BASE)
    assert worse['use_norm'] < same['use_norm'] == 1.0


def test_每項都夾在零到一之間():
    t = normalise_terms(**{**BASE, 'id_def': 2.0, 'clip_def': -0.5,
                           'quality_def': 0.5})
    for k in ('id_norm', 'dir_norm', 'use_norm'):
        assert 0.0 <= t[k] <= 1.0


def test_軟極小介於極小與平均之間且隨_tau_單調():
    v = [0.2, 0.7, 0.9]
    assert softmin(v, tau=0.0) == pytest.approx(0.2)
    a, b = softmin(v, tau=0.05), softmin(v, tau=0.5)
    assert 0.2 - 1e-9 <= a <= b <= sum(v) / 3


def test_缺項直接拋錯():
    with pytest.raises(KeyError):
        criterion_score({'id_norm': 0.5, 'dir_norm': 0.5})

import math

import torch

from src.defense.carrier_search import (KNOBS, START, build_carrier, describe,
                                        grid_of, spearman)
from src.defense.color_search import evolution_search_batched


MEAN = [71.3, -5.9, 1.8]
COV = [[210.0, 4.0, -3.0], [4.0, 36.0, 2.0], [-3.0, 2.0, 30.0]]
SECOND = [60.6, 8.9, 33.9]


def _image(seed=0, size=48):
    g = torch.Generator().manual_seed(seed)
    return (0.3 + 0.4 * torch.rand(1, 3, size, size, generator=g)).clamp(0, 1)


def _supports(x):
    frame = torch.ones_like(x[:, :1])
    clothes = torch.zeros_like(x[:, :1])
    clothes[..., 24:, 8:40] = 1.0
    return frame, clothes


def _build(x, values):
    frame, clothes = _supports(x)
    return build_carrier(values, x, frame_support=frame, clothes_support=clothes,
                         frame_palette=(MEAN, COV), clothes_palette=(SECOND, COV))


def test_網格值對應到二的冪():
    assert [grid_of(v) for v in range(6)] == [1, 2, 4, 8, 16, 32]
    assert grid_of(-5) == 1 and grid_of(99) == 32


def test_起點是現行方法_全域仿射且場為零():
    x = _image()
    c = _build(x, START)
    assert c.stages[0].grid == 1 and c.stages[1].grid == 1
    assert float(c.stages[0].delta.abs().max()) == 0.0
    assert float(c.stages[1].delta.abs().max()) == 0.0


def test_同一組旋鈕值建出逐位元相同的載體():
    x = _image(1)
    v = dict(START, frame_grid=3.0, frame_scale=0.6, field_seed=42.0)
    a, b = _build(x, v), _build(x, v)
    assert torch.equal(a.render(x), b.render(x))


def test_換一個場種子就換一張防禦圖():
    x = _image(2)
    v = dict(START, frame_grid=3.0, frame_scale=0.6, field_seed=1.0)
    w = dict(v, field_seed=2.0)
    assert not torch.equal(_build(x, v).render(x), _build(x, w).render(x))


def test_振幅為零時場不起作用():
    x = _image(3)
    v = dict(START, frame_grid=3.0, frame_scale=0.0, field_seed=5.0)
    c = _build(x, v)
    assert float(c.stages[0].delta.abs().max()) == 0.0


def test_旋鈕描述與實際建出來的載體一致():
    x = _image(4)
    v = dict(START, frame_grid=2.0, clothes_grid=4.0, lock_luminance=0.0)
    c = _build(x, v)
    d = describe(v)
    assert d['knob_frame_grid'] == c.stages[0].grid == 4
    assert d['knob_clothes_grid'] == c.stages[1].grid == 16
    assert d['knob_lock_luminance'] == 0
    assert c.stages[0].lock_luminance is False


def test_批次搜尋在玩具目標上會下降():
    target = {k.name: 0.5 * (k.lo + k.hi) for k in KNOBS}

    def evaluate_many(batch):
        return [sum((v[k.name] - target[k.name]) ** 2 / (k.span ** 2 + 1e-9)
                    for k in KNOBS) for v in batch]

    out = evolution_search_batched(list(KNOBS), evaluate_many, budget=90,
                                   children=6, seed=0, start=START)
    assert out.best_score < evaluate_many([START])[0]
    assert out.evaluations == 90


def test_秩相關的邊界情形():
    assert spearman([1, 2, 3], [1, 2, 3]) == 1.0
    assert spearman([1, 2, 3], [3, 2, 1]) == -1.0
    assert math.isnan(spearman([1, 2], [1, 2]))
    assert math.isnan(spearman([1, 1, 1], [1, 2, 3]))


def test_有臉遮罩時串成三段且臉那一段只動臉():
    x = _image(7)
    frame, clothes = _supports(x)
    face = torch.zeros_like(x[:, :1])
    face[..., 4:20, 4:20] = 1.0
    v = dict(START, face_grid=2.0, face_scale=0.8, face_amplitude=1.0,
             frame_amplitude=0.05, clothes_amplitude=0.05, field_seed=11.0)
    c = build_carrier(v, x, frame_support=frame, clothes_support=clothes,
                      face_support=face, frame_palette=(MEAN, COV),
                      clothes_palette=(SECOND, COV))
    assert [s for s in c.tags] == ['frame', 'clothes', 'face']
    assert c.stages[2].grid == 4
    assert float(c.stages[2].delta.abs().max()) > 0


def test_沒有臉遮罩時退化成兩段():
    x = _image(8)
    frame, clothes = _supports(x)
    c = build_carrier(dict(START), x, frame_support=frame,
                      clothes_support=clothes, frame_palette=(MEAN, COV),
                      clothes_palette=(SECOND, COV))
    assert c.tags == ['frame', 'clothes']


def test_配色由旋鈕在庫裡挑且夾在範圍內():
    from src.defense.carrier_search import pick
    bank = [(f'mean{k}', f'cov{k}') for k in range(20)]
    assert pick(bank, 0.0) == bank[0]
    assert pick(bank, 3.4) == bank[3]
    assert pick(bank, 19.6) == bank[19]
    assert pick(bank, -5) == bank[0]
    assert pick(bank, 99) == bank[19]


def test_臉那一段吃自己的配色():
    x = _image(9)
    frame, clothes = _supports(x)
    face = torch.zeros_like(x[:, :1])
    face[..., 4:20, 4:20] = 1.0
    v = dict(START, face_scale=0.0, face_amplitude=1.0)
    skin = ([55.0, 12.0, 18.0], COV)
    c = build_carrier(v, x, frame_support=frame, clothes_support=clothes,
                      face_support=face, frame_palette=(MEAN, COV),
                      clothes_palette=(SECOND, COV), face_palette=skin)
    assert c.stages[2].target_mean.tolist() == skin[0]

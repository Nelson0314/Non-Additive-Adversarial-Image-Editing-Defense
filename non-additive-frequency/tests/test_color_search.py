import torch

from src.defense.color_search import Knob, evolution_search


def test_finds_the_minimum_of_a_smooth_function():
    knobs = [Knob('a', -5.0, 5.0), Knob('b', -5.0, 5.0)]
    calls = {'n': 0}

    def f(v):
        calls['n'] += 1
        return (v['a'] - 1.5) ** 2 + (v['b'] + 2.0) ** 2

    out = evolution_search(knobs, f, budget=120, children=4, seed=1)
    assert out.evaluations == 120
    assert calls['n'] == 120
    assert out.best_score < 0.25
    assert abs(out.best['a'] - 1.5) < 0.5
    assert abs(out.best['b'] + 2.0) < 0.5


def test_respects_the_bounds():
    knobs = [Knob('a', 0.0, 1.0)]
    seen = []

    def f(v):
        seen.append(v['a'])
        return -v['a']

    out = evolution_search(knobs, f, budget=40, children=3, seed=2)
    assert all(0.0 <= s <= 1.0 for s in seen)
    assert 0.0 <= out.best['a'] <= 1.0


def test_handles_a_noisy_objective_without_diverging():
    knobs = [Knob('a', -3.0, 3.0)]
    g = torch.Generator().manual_seed(7)

    def f(v):
        return (v['a'] - 1.0) ** 2 + float(torch.randn(1, generator=g)) * 0.05

    out = evolution_search(knobs, f, budget=80, children=4, seed=3)
    assert abs(out.best['a'] - 1.0) < 1.0


def test_history_records_every_evaluation():
    knobs = [Knob('a', 0.0, 1.0)]
    out = evolution_search(knobs, lambda v: v['a'], budget=17, children=4, seed=4)
    assert len(out.history) == 17
    assert out.history[0][0] == 0


def test_rejects_degenerate_settings():
    for kw in ({'budget': 0}, {'children': 0}):
        try:
            evolution_search([Knob('a', 0.0, 1.0)], lambda v: 0.0, **kw)
        except ValueError:
            pass
        else:
            raise AssertionError(f'{kw} 必須拋錯')
    try:
        evolution_search([], lambda v: 0.0, budget=5)
    except ValueError:
        pass
    else:
        raise AssertionError('沒有旋鈕必須拋錯')


def test_starting_point_is_used_and_clipped():
    knobs = [Knob('a', 0.0, 1.0)]
    first = []
    evolution_search(knobs, lambda v: (first.append(v['a']), 0.0)[1],
                     budget=1, children=1, seed=5, start={'a': 9.0})
    assert first[0] == 1.0

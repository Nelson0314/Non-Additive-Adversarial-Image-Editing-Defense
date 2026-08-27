"""分階段訓練（階段二 ＋ 信賴域）。

三件事各自釘住：算子的給法（排程）、信賴域的行為、以及**關閉時逐位元等於
加這一組旗標之前**。最後一件是本專案的慣例——新旗鈕的預設值必須是「什麼都
沒發生」，否則既有批次的可比性會被靜默改掉。
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch

from src.defense.param_pgd import AdditiveParam, Stage2Result, run_stage2_pgd
from src.defense.purify_aware import (
    STAGE2_OPS, make_sequenced_ops_transform, resolve_stage2_ops,
    stage2_schedule,
)

ROOT = Path(__file__).resolve().parents[1]


def _load_runner():
    spec = importlib.util.spec_from_file_location(
        "ip2p_run_stage2", ROOT / "scripts" / "ip2p_run.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# 排程
# ---------------------------------------------------------------------------

def test_cycle_is_exact_round_robin():
    names = ["identity", "blur1", "crop10"]
    assert stage2_schedule(names, 7, order="cycle") == [
        "identity", "blur1", "crop10", "identity", "blur1", "crop10", "identity"]


def test_shuffle_covers_each_operator_once_per_round():
    names = ["identity", "blur1", "blur2", "crop10"]
    plan = stage2_schedule(names, 12, order="shuffle", seed=3)
    for start in (0, 4, 8):
        assert sorted(plan[start:start + 4]) == sorted(names)


def test_shuffle_is_not_a_fixed_order():
    """洗牌的重點就是拿掉固定的相位關係；若每一輪都一樣就退化成 cycle。"""
    names = ["identity", "blur1", "blur2", "crop10"]
    plan = stage2_schedule(names, 40, order="shuffle", seed=0)
    rounds = {tuple(plan[i:i + 4]) for i in range(0, 40, 4)}
    assert len(rounds) > 1


def test_random_can_be_unbalanced_within_a_round():
    """`random` 的已知缺點：短期覆蓋不均。這一條是它與 shuffle 的差別本身。"""
    names = ["identity", "blur1", "blur2", "crop10"]
    plan = stage2_schedule(names, 40, order="random", seed=0)
    windows = [sorted(plan[i:i + 4]) for i in range(0, 40, 4)]
    assert any(w != sorted(names) for w in windows)


def test_schedule_is_reproducible_under_seed():
    a = stage2_schedule(["identity", "blur1", "crop10"], 15, seed=7)
    b = stage2_schedule(["identity", "blur1", "crop10"], 15, seed=7)
    assert a == b


def test_ramp_uses_only_weak_operators_in_the_first_half():
    names = ["identity", "blur1", "blur2", "crop15"]
    plan = stage2_schedule(names, 20, order="cycle", ramp=True)
    assert all(STAGE2_OPS[n][1] <= 1 for n in plan[:10])
    assert any(STAGE2_OPS[n][1] > 1 for n in plan[10:])


def test_ramp_off_is_the_default_and_uses_the_whole_pool():
    names = ["identity", "blur2"]
    assert stage2_schedule(names, 4, order="cycle") == [
        "identity", "blur2", "identity", "blur2"]


def test_unknown_operator_raises():
    with pytest.raises(ValueError, match="未知的淨化算子"):
        resolve_stage2_ops(["identity", "sharpen"])


def test_unknown_order_raises():
    with pytest.raises(ValueError, match="未知的順序"):
        stage2_schedule(["identity"], 3, order="alternating")


def test_transform_rejects_step_outside_the_plan():
    tf = make_sequenced_ops_transform(["identity"], 3)
    x = torch.rand(1, 3, 16, 16)
    with pytest.raises(ValueError, match="超出階段二"):
        tf(x, 3)


def test_transform_applies_the_planned_operator():
    tf = make_sequenced_ops_transform(["identity", "blur2"], 2, order="cycle")
    x = torch.rand(1, 3, 32, 32)
    assert torch.equal(tf(x, 0), x)
    assert not torch.equal(tf(x, 1), x)


# ---------------------------------------------------------------------------
# 信賴域
# ---------------------------------------------------------------------------

def _toy(seed: int = 0):
    """一個把損失降到最低就等於「δ 全部貼上界」的玩具問題。"""
    torch.manual_seed(seed)
    # 值域留白，讓 clamp 不會把梯度歸零——否則測到的是夾取不是信賴域。
    x01 = torch.rand(1, 3, 16, 16) * 0.5 + 0.25
    param = AdditiveParam(radius=0.1)
    param.reset(x01, seed)
    return x01, param


def test_stage2_keeps_the_gain_it_promised_to_keep():
    x01, param = _toy()

    def loss_fn(x):
        return -(x.mean())            # 越亮越好；階段一把 δ 推到上界

    with torch.no_grad():
        param.delta.fill_(0.1)        # 假裝階段一已經收斂

    def bad(x, step):
        return 1.0 - x            # 「淨化」把亮暗反過來，梯度因此指向
                                  # 「把 δ 拉回去」，正是信賴域要擋的方向

    res = run_stage2_pgd(x01, param, loss_fn, steps=8, alpha=0.02,
                         transform=bad, trust_frac=0.95, check_every=2)
    with torch.no_grad():
        gain_now = float(loss_fn(x01)) - float(loss_fn(param.render(x01)))
    assert gain_now >= 0.95 * res.gain_stage1
    assert res.reverts > 0
    assert res.alpha_final < res.alpha_init


def test_stage2_returns_the_last_accepted_snapshot():
    x01, param = _toy(1)

    def loss_fn(x):
        return -(x.mean())

    with torch.no_grad():
        param.delta.fill_(0.1)
    res = run_stage2_pgd(x01, param, loss_fn, steps=4, alpha=0.05,
                         transform=lambda x, s: 1.0 - x, trust_frac=1.0,
                         check_every=1)
    assert torch.equal(res.x_def, param.render(x01).detach())
    assert isinstance(res, Stage2Result)


def test_stage2_stops_when_the_step_shrinks_past_the_floor():
    x01, param = _toy(2)

    def loss_fn(x):
        return -(x.mean())

    with torch.no_grad():
        param.delta.fill_(0.1)
    res = run_stage2_pgd(x01, param, loss_fn, steps=200, alpha=0.05,
                         transform=lambda x, s: 1.0 - x, trust_frac=1.0,
                         check_every=1, alpha_min_ratio=1 / 8)
    assert res.stopped_early
    assert res.steps_run < 200


def test_stage2_refuses_when_stage_one_bought_nothing():
    """增益不為正時比例約束沒有意義。**不可以靜默改用別的判準跑下去。**"""
    x01, param = _toy(3)

    def loss_fn(x):
        return x.mean()               # 越暗越好，而 δ=0.1 讓它變亮

    with torch.no_grad():
        param.delta.fill_(0.1)
    with pytest.raises(ValueError, match="階段一的增益不為正"):
        run_stage2_pgd(x01, param, loss_fn, steps=4, alpha=0.01,
                       transform=lambda x, s: x, trust_frac=0.95)


@pytest.mark.parametrize("kw,msg", [
    ({"steps": 0}, "steps 必須為正"),
    ({"trust_frac": 1.5}, "trust_frac 必須落在"),
    ({"check_every": 0}, "check_every 必須為正"),
])
def test_stage2_argument_validation(kw, msg):
    x01, param = _toy(4)
    base = dict(steps=4, alpha=0.01, transform=lambda x, s: x)
    base.update(kw)
    with pytest.raises(ValueError, match=msg):
        run_stage2_pgd(x01, param, lambda x: -(x.mean()), **base)


# ---------------------------------------------------------------------------
# CLI：預設值、守門、以及「關著時不呼叫階段二」
# ---------------------------------------------------------------------------

def test_defaults_are_off():
    mod = _load_runner()
    args = mod.build_parser().parse_args(["--out", "o", "--data", "d"])
    assert args.stage2_steps == 0
    assert args.stage2_order == "shuffle"
    assert args.stage2_trust == 0.95
    assert "identity" in args.stage2_ops


def test_stage2_is_not_called_when_disabled():
    """**關著時逐位元等於加這組旗標之前**：階段二那一支根本沒被呼叫。"""
    mod = _load_runner()

    def boom(*a, **k):
        raise AssertionError("stage2_steps=0 時不應該呼叫 run_stage2_pgd")

    mod.run_stage2_pgd = boom
    called = {}

    class FakeRes:
        x_def = torch.rand(1, 3, 8, 8)

    def fake_pgd(*a, **k):
        called["yes"] = True
        return FakeRes()

    mod.run_param_pgd = fake_pgd
    args = mod.build_parser().parse_args(
        ["--out", "o", "--data", "d", "--radius", "1.0"])
    x01 = torch.rand(1, 3, 64, 64)
    out = mod.defend(None, None, "phase", x01, args, lambda x: x.mean())
    assert called and out[0].shape == FakeRes.x_def.shape


def test_stage2_is_called_when_enabled():
    mod = _load_runner()
    seen = {}

    def fake_stage2(x01, param, loss_fn, args):
        seen["steps"] = args.stage2_steps
        return torch.zeros(1, 3, 8, 8), {"stage2_reverts": 0}

    mod._run_stage2 = fake_stage2

    class FakeRes:
        x_def = torch.rand(1, 3, 8, 8)

    mod.run_param_pgd = lambda *a, **k: FakeRes()
    args = mod.build_parser().parse_args(
        ["--out", "o", "--data", "d", "--radius", "1.0", "--stage2-steps", "40"])
    x01 = torch.rand(1, 3, 64, 64)
    out = mod.defend(None, None, "phase", x01, args, lambda x: x.mean())
    assert seen["steps"] == 40
    assert out[4]["stage2_reverts"] == 0


@pytest.mark.parametrize("extra,msg", [
    (["--stage2-steps", "40"], "不可與預算模式"),
    (["--stage2-steps", "40", "--radius", "1.0", "--update", "adam"],
     "只支援 sign 更新"),
])
def test_stage2_guards(extra, msg):
    mod = _load_runner()
    mod.run_param_pgd = lambda *a, **k: type("R", (), {
        "x_def": torch.rand(1, 3, 8, 8)})()
    args = mod.build_parser().parse_args(["--out", "o", "--data", "d"] + extra)
    with pytest.raises(SystemExit, match=msg):
        mod.defend(None, None, "phase", torch.rand(1, 3, 64, 64), args,
                   lambda x: x.mean())


def test_stage2_rejects_conditions_without_an_optimiser():
    mod = _load_runner()
    args = mod.build_parser().parse_args(
        ["--out", "o", "--data", "d", "--radius", "1.0", "--stage2-steps", "8"])
    with pytest.raises(SystemExit, match="沒有階段一的解"):
        mod.defend(None, None, "phase_rand", torch.rand(1, 3, 64, 64), args,
                   lambda x: x.mean())

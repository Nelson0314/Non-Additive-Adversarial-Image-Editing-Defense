"""收斂評估與 early stop。

這一支釘的是**收斂怎麼判**。專案已經犯過一次錯：拿隨機目標的逐步損失當收斂
訊號，而那個抖動是取樣變異（實測 0.16–0.61）不是參數在漂。所以評估必須是
決定性的，而 early stop 必須與**歷史最佳**比而不是與上一次比。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from src.defense.param_pgd import AdditiveParam, run_param_pgd  # noqa: E402

import ip2p_run  # noqa: E402


def _toy():
    g = torch.Generator().manual_seed(0)
    return torch.rand(1, 3, 16, 16, generator=g)


def test_defaults_leave_behaviour_unchanged():
    """不給旗標時逐位元等於加這組機制之前：不評估、不早停。"""
    a = ip2p_run.build_parser().parse_args(["--out", "o"])
    assert a.eval_every == 0
    assert a.patience == 0


def test_no_eval_means_no_early_stop():
    x = _toy()
    r = run_param_pgd(x, AdditiveParam(radius=0.05), lambda y: y.pow(2).mean(),
                      steps=30, patience=2)
    assert r.stop_reason == "max_steps"
    assert r.stopped_at == 30
    assert r.best_eval is None


def test_early_stop_fires_once_the_objective_stops_improving():
    """半徑用滿之後損失就不動了，早停要在那之後觸發。"""
    x = _toy()
    f = lambda y: y.pow(2).mean()
    r = run_param_pgd(x, AdditiveParam(radius=0.05), f, steps=400,
                      eval_fn=f, eval_every=10, patience=3, min_delta=0.01)
    assert r.stop_reason == "early_stop"
    assert r.stopped_at < 400
    assert r.best_eval is not None


def test_early_stop_compares_against_the_best_not_the_previous():
    """與**歷史最佳**比。逐次比較會被單次雜訊帶著走——一個偶然的小改善就把
    耐心歸零，早停因此永遠不觸發。"""
    x = _toy()
    seq = iter([1.0, 0.5, 0.9, 0.51, 0.9, 0.52, 0.9, 0.53] + [0.9] * 40)
    calls = {"n": 0}

    def fake_eval(_):
        calls["n"] += 1
        return torch.tensor(next(seq))

    r = run_param_pgd(x, AdditiveParam(radius=0.05), lambda y: y.pow(2).mean(),
                      steps=200, eval_fn=fake_eval, eval_every=1,
                      patience=3, min_delta=0.01)
    # 0.5 之後再也沒有低於 0.5×(1−0.01)，故第 3 次未改善時就要停。
    assert r.stop_reason == "early_stop"
    assert r.best_eval == pytest.approx(0.5)


def test_eval_values_are_recorded_in_history():
    x = _toy()
    f = lambda y: y.pow(2).mean()
    r = run_param_pgd(x, AdditiveParam(radius=0.05), f, steps=40,
                      eval_fn=f, eval_every=10)
    evals = [h for h in r.history if "eval" in h]
    assert len(evals) >= 4
    assert all(h["eval"] > 0 for h in evals)


def test_trace_columns_are_written():
    src = (ROOT / "scripts" / "ip2p_run.py").read_text(encoding="utf-8")
    assert 'write_csv(args.out / "trace.csv"' in src
    for col in ("stop_reason", "stopped_at", "best_eval"):
        assert f'"{col}"' in src, col


def test_deterministic_eval_is_deterministic():
    """固定評估連續呼叫必須給同一個值，否則曲線讀不出趨勢。"""
    from src.defense.image_guidance_loss import make_image_guidance_loss

    class FakeUNet:
        def __call__(self, z, t, encoder_hidden_states=None):
            class O:
                # **輸出必須依賴影像條件那四個通道**（4:8），否則 cond 與 base
                # 恆等、差恆為零，這個測試就什麼都沒驗到。
                sample = z[:, :4] + z[:, 4:8] * float(t.item()) * 0.001
            return O()

    class FakeSched:
        alphas_cumprod = torch.linspace(0.999, 0.001, 1000)

    class FakeIP2P:
        unet = FakeUNet()
        device = "cpu"
        scheduler = FakeSched()
        _null_emb_cache = torch.zeros(1, 77, 768)

        def image_latents(self, x):
            return x[:, :4] if x.shape[1] >= 4 else x.repeat(1, 2, 1, 1)[:, :4]

    fn = make_image_guidance_loss(FakeIP2P(), zt_mode="noise", t_min=1,
                                  t_max=1000, seed=0)
    fixed = fn.make_fixed(4, 12345)
    x = torch.rand(1, 3, 8, 8)
    a, b = float(fixed(x)), float(fixed(x))
    assert a == b, "固定評估不可隨呼叫改變"
    assert float(fn(x)) != float(fn(x)), "訓練損失本來就該每次重抽"

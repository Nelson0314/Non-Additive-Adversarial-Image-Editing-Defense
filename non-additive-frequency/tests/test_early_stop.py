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


# ---- 權重保存與續跑 ----

def test_weight_flags_default_off():
    a = ip2p_run.build_parser().parse_args(["--out", "o"])
    assert a.save_weights is False
    assert a.resume_weights is None
    assert a.skip_existing is False


def test_weights_path_shape():
    from pathlib import Path as P
    assert ip2p_run.weights_path(P("r"), "img", "phase_gain").name == \
        "img__phase_gain__w.pt"


def test_resume_refuses_a_different_construction(tmp_path):
    """形狀不合要拋錯。**不可靜默略過**——那會讓「續跑」載進去的是別的
    構造的參數，而報表上看不出來。"""
    import torch as T

    T.save([T.zeros(3, 4)], tmp_path / "img__phase_gain__w.pt")

    class P:
        def params(self):
            return [T.zeros(5, 6)]

        def project(self):
            pass

    class A:
        resume_weights = tmp_path
        _cur_image = "img"

    with pytest.raises(SystemExit, match="形狀"):
        ip2p_run._load_weights(P(), None, A(), "phase_gain")


def test_resume_reports_zero_when_no_file(tmp_path):
    """沒有對應檔案就從零起步並記 0，不假裝續跑過。"""
    class A:
        resume_weights = tmp_path
        _cur_image = "nothing"

    assert ip2p_run._load_weights(None, None, A(), "phase_gain") == 0


def test_resume_loads_and_projects(tmp_path):
    import torch as T

    T.save([T.full((2, 3), 0.5)], tmp_path / "img__phase_gain__w.pt")
    tgt = T.zeros(2, 3)

    class P:
        projected = False

        def params(self):
            return [tgt]

        def project(self):
            P.projected = True

    class A:
        resume_weights = tmp_path
        _cur_image = "img"

    assert ip2p_run._load_weights(P(), None, A(), "phase_gain") == 1
    assert float(tgt.abs().mean()) == pytest.approx(0.5)
    assert P.projected, "載入之後必須投影回可行集"


# ---- 續跑必須發生在參數張量建好之後，且不可被 reset 清掉 ----
#
# 這一組釘的是一次真實跑次的當機與它底下的靜默失效：
#
#   AttributeError: 'NoneType' object has no attribute 'theta'
#
# 成因是 `defend()` 曾在 `run_param_pgd` **之前**呼叫 `_load_weights`，而
# 參數張量要到 `run_param_pgd` 內部的 `param.reset()` 才存在；就算閃過當機，
# 那一次 `reset()` 也會把載進去的值換掉——不拋錯、不留症狀。


def _phase_gain_param(**kw):
    """與 `build("phase_gain", ...)` 同一個構造：`phase_on` ＋ `gain_ratio > 0`
    ＋ `spectral_floor > 0`，`params()` 因此是 theta／gain／floor 三個張量。
    只把 `size` 由 512 調小，讓測試不必花整張圖的時間。"""
    from src.defense.param_pgd import PhaseParam
    return PhaseParam(size=64, block=16, radius=1.0, gain_ratio=0.3,
                      spectral_floor=0.05, phase_on=True, **kw)


def _phase_x01():
    g = torch.Generator().manual_seed(3)
    return torch.rand(1, 3, 64, 64, generator=g)


def test_phase_family_has_no_params_before_reset():
    """根因之一：相位族在 `reset()` 之前沒有參數張量。載入排在
    `run_param_pgd` 之前就會撞到這個 `AttributeError`。"""
    param = _phase_gain_param()
    assert param.module is None
    with pytest.raises(AttributeError, match="theta"):
        param.params()


def test_reset_replaces_the_tensors_rather_than_zeroing_them():
    """根因之二，也是本修正真正要擋的靜默失效：`reset()` **換掉張量本身**。
    先載再讓 `run_param_pgd` reset 一次，載進去的值會無聲消失。"""
    param, x01 = _phase_gain_param(), _phase_x01()
    param.reset(x01, 0)
    first = param.params()
    assert len(first) == 3, "phase_gain ＋ spectral_floor 應有三個張量"
    with torch.no_grad():
        for t in first:
            t.fill_(0.25)
    param.reset(x01, 0)
    second = param.params()
    assert all(a is not b for a, b in zip(first, second)), \
        "reset 若是就地歸零，這個修正的理由就不成立"
    assert all(float(t.detach().abs().max()) == 0.0 for t in second)


def _saved_phase_weights(tmp_path, value=0.05, name="img__phase_gain__w.pt"):
    probe, x01 = _phase_gain_param(), _phase_x01()
    probe.reset(x01, 0)
    saved = [torch.full_like(t.detach(), value) for t in probe.params()]
    torch.save(saved, tmp_path / name)
    return saved


def _resume_args(tmp_path, image="img"):
    class A:
        resume_weights = tmp_path
        _cur_image = image
    return A()


def test_resume_weights_survive_into_the_optimisation(tmp_path):
    """載進去的值必須**在第一步的前向就已經在參數裡**，而且跑完還在。

    這是本修正的重點：只驗「沒有當機」是不夠的，被後續的 `reset()` 清成零
    也不會當機。"""
    _saved_phase_weights(tmp_path, value=0.05)
    param, x01 = _phase_gain_param(), _phase_x01()
    args = _resume_args(tmp_path)

    resumed = []

    def post_reset(p):
        resumed.append(ip2p_run._load_weights(p, x01, args, "phase_gain"))

    at_first_forward = []

    def loss_fn(x_def):
        if not at_first_forward:
            at_first_forward.extend(
                float(t.detach().abs().mean()) for t in param.params())
        return x_def.sum()

    # `step_size=0` 讓參數不被更新，留下的就只有「載進去的那組值」。
    res = run_param_pgd(x01, param, loss_fn, steps=2, step_size=0.0,
                        post_reset=post_reset)

    assert resumed == [1], "載到了要記 1，而且只呼叫一次"
    assert at_first_forward == pytest.approx([0.05, 0.05, 0.05]), \
        "第一步的前向就必須看到載進來的值，不是零"
    assert [float(t.detach().abs().mean()) for t in param.params()] == \
        pytest.approx([0.05, 0.05, 0.05]), "跑完之後載進來的值仍須在參數裡"
    assert res.x_def.shape == x01.shape


def test_resume_actually_changes_the_output(tmp_path):
    """載入若是靜默失效（被 reset 清掉），輸出會與從零起步逐位元相同。
    這一條把「有沒有真的載到」釘在**交出去的圖**上。"""
    _saved_phase_weights(tmp_path, value=0.05)
    args = _resume_args(tmp_path)
    x01 = _phase_x01()

    def run(post_reset):
        param = _phase_gain_param()
        return run_param_pgd(x01, param, lambda x: x.sum(), steps=1,
                             step_size=0.0, post_reset=post_reset).x_def

    cold = run(None)
    warm = run(lambda p: ip2p_run._load_weights(p, x01, args, "phase_gain"))
    assert not torch.equal(cold, warm)


def test_resume_reports_zero_when_no_file_and_still_runs(tmp_path):
    """沒有對應檔就從零起步、記 0，不當機。"""
    param, x01 = _phase_gain_param(), _phase_x01()
    args = _resume_args(tmp_path, image="nothing")
    resumed = []
    res = run_param_pgd(
        x01, param, lambda x: x.sum(), steps=1, step_size=0.0,
        post_reset=lambda p: resumed.append(
            ip2p_run._load_weights(p, x01, args, "phase_gain")))
    assert resumed == [0]
    assert all(float(t.detach().abs().max()) == 0.0 for t in param.params()), \
        "沒載到就該是從零起步"
    assert res.x_def.shape == x01.shape


def test_resume_refuses_a_different_shape_through_the_hook(tmp_path):
    """形狀不合仍然拋錯，掛到 `post_reset` 上之後也一樣。"""
    probe, x01 = _phase_gain_param(), _phase_x01()
    probe.reset(x01, 0)
    torch.save([torch.zeros(t.shape[0], t.shape[1] + 1, *t.shape[2:])
                for t in probe.params()],
               tmp_path / "img__phase_gain__w.pt")
    args = _resume_args(tmp_path)
    param = _phase_gain_param()
    with pytest.raises(SystemExit, match="形狀"):
        run_param_pgd(x01, param, lambda x: x.sum(), steps=1, step_size=0.0,
                      post_reset=lambda p: ip2p_run._load_weights(
                          p, x01, args, "phase_gain"))


def test_resume_refuses_a_different_tensor_count(tmp_path):
    """張量個數不合（例如存檔時沒開 `--spectral-floor`）也要拋錯。"""
    torch.save([torch.zeros(2, 3)], tmp_path / "img__phase_gain__w.pt")
    args = _resume_args(tmp_path)
    param, x01 = _phase_gain_param(), _phase_x01()
    with pytest.raises(SystemExit, match="構造不同"):
        run_param_pgd(x01, param, lambda x: x.sum(), steps=1, step_size=0.0,
                      post_reset=lambda p: ip2p_run._load_weights(
                          p, x01, args, "phase_gain"))


def test_post_reset_default_leaves_the_loop_bit_identical():
    """不給 `post_reset` 與顯式給 `None`／給一個什麼都不做的函式，
    三者的輸出逐位元相同——這個鉤子本身不得動到任何狀態。"""
    x01 = _toy()

    def run(**kw):
        return run_param_pgd(x01, AdditiveParam(radius=0.05),
                             lambda x: (x * x).sum(), steps=3, seed=7,
                             **kw).x_def

    a, b, c = run(), run(post_reset=None), run(post_reset=lambda p: None)
    assert torch.equal(a, b) and torch.equal(a, c)


# ---- defend() 這一層：旗標沒給時不得掛上鉤子 ----


def _defend_args(tmp_path, **over):
    argv = ["--out", str(tmp_path), "--radius", "1.0",
            "--conditions", "phase_gain", "--gain-ratio", "0.3",
            "--spectral-floor", "0.05"]
    args = ip2p_run.build_parser().parse_args(argv)
    args._cur_image = "img"
    for k, v in over.items():
        setattr(args, k, v)
    return args


class _StubResult:
    x_def = torch.zeros(1, 3, 8, 8)
    stop_reason = "max_steps"
    stopped_at = 0
    best_eval = None
    history: list = []


def _defend_capturing(monkeypatch, args):
    """跑 `defend()` 的相位那一支，把傳給 `run_param_pgd` 的 kwargs 攔下來。
    不載入任何擴散模型權重。"""
    seen = {}

    def fake_pgd(x01, param, loss_fn, **kw):
        seen.update(kw)
        # 照真品的順序：先 reset，鉤子才跑。順序寫反的話這個測試就驗不到
        # 「載入發生在張量存在之後」。
        param.reset(x01, kw.get("seed", 0))
        if kw.get("post_reset") is not None:
            kw["post_reset"](param)
        return _StubResult()

    monkeypatch.setattr(ip2p_run, "run_param_pgd", fake_pgd)
    # `build()` 用的是 512，這裡換成同構造的小尺寸，省掉整張圖的時間。
    real_build = ip2p_run.build
    monkeypatch.setattr(ip2p_run, "build",
                        lambda *a, **k: (_phase_gain_param(),
                                         *real_build(*a, **k)[1:]))
    out = ip2p_run.defend(None, None, "phase_gain", _phase_x01(), args,
                          lambda x: x.sum())
    return seen, out


def test_defend_passes_no_hook_when_resume_is_off(monkeypatch, tmp_path):
    """不給 `--resume-weights` 時 `post_reset` 必須是 `None`，
    呼叫路徑與加入這個參數之前相同。"""
    seen, out = _defend_capturing(monkeypatch, _defend_args(tmp_path))
    assert seen["post_reset"] is None
    assert out[4]["resumed"] == 0


def test_defend_resumes_through_the_hook(monkeypatch, tmp_path):
    """給了 `--resume-weights` 且有對應檔時，`resumed` 記 1，
    而載入是經由 `post_reset` 發生的（即在 `reset()` 之後）。"""
    store = tmp_path / "w"
    store.mkdir()
    _saved_phase_weights(store, value=0.05)
    args = _defend_args(tmp_path, resume_weights=store)
    seen, out = _defend_capturing(monkeypatch, args)
    assert callable(seen["post_reset"])
    assert out[4]["resumed"] == 1


def test_defend_reports_zero_when_the_image_has_no_file(monkeypatch, tmp_path):
    """給了旗標但這張圖沒有對應檔：`resumed` 記 0，不當機。"""
    store = tmp_path / "w"
    store.mkdir()
    args = _defend_args(tmp_path, resume_weights=store)
    seen, out = _defend_capturing(monkeypatch, args)
    assert callable(seen["post_reset"])
    assert out[4]["resumed"] == 0


def test_defend_refuses_resume_in_budget_mode(tmp_path):
    """預算模式（不給 `--radius`）不呼叫 `_load_weights`、也不寫 `resumed` 欄，
    給了旗標會整批從零開始練而報表上看不出來。**寧可拒絕**。"""
    args = _defend_args(tmp_path, radius=None, resume_weights=tmp_path)
    with pytest.raises(SystemExit, match="--resume-weights 不可與預算模式"):
        ip2p_run.defend(None, None, "phase_gain", _phase_x01(), args,
                        lambda x: x.sum())

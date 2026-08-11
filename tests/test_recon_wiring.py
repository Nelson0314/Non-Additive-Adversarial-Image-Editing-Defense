"""A 段（DEC-016）接進五段流程的三個接點：起點、解碼器、雜湊。

這三件事各自都有「壞掉但沒有症狀」的形態，故各釘一條：

- 起點沒換成 `z*` → 重建仍是舊下限，看起來只是「A 段效果不好」
- 解碼器的微調值沒還原 → 上一張圖的過擬合污染下一張，看起來是「某些圖較好」
- 新旋鈕無條件進 `config_hash` → 既有批次每一格判為未完成而整批重跑
"""

import pytest
import torch
import torch.nn as nn

from src.defense.generator import DefenseGenerator
from src.defense.recon import ReconAdapter, decoder_tunable
from src.experiment.executors import RunConfig
from src.utils.cellid import config_hash


class ToyDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.GroupNorm(2, 4)
        self.conv = nn.Conv2d(4, 4, 3, padding=1)


def _adapter(dec, fill=3.0):
    named = dict(dec.named_parameters())
    return ReconAdapter(
        torch.zeros(1, 4, 8, 8),
        {k: torch.full_like(named[k], fill)
         for k, _ in decoder_tunable(dec)})


def test_applied_在區塊內生效離開即還原():
    dec = ToyDecoder()
    before = {k: v.detach().clone() for k, v in dec.named_parameters()}
    with _adapter(dec).applied(dec):
        assert torch.allclose(dec.norm.weight, torch.full_like(dec.norm.weight, 3.0))
        assert torch.allclose(dec.conv.bias, torch.full_like(dec.conv.bias, 3.0))
    for k, v in dec.named_parameters():
        assert torch.equal(v, before[k]), f"{k} 沒有還原"


def test_未被微調的參數不受影響():
    """只寫進 `decoder` 裡列到的名稱。多寫等於改動 A 段沒有量過的東西。"""
    dec = ToyDecoder()
    w = dec.conv.weight.detach().clone()
    with _adapter(dec).applied(dec):
        assert torch.equal(dec.conv.weight, w)


def test_名稱對不上直接拋出():
    """依名稱而非位置對齊：位置對齊會在結構改動時靜默寫到別的張量上。"""
    dec = ToyDecoder()
    bad = ReconAdapter(torch.zeros(1, 4, 8, 8),
                       {"不存在的層.bias": torch.zeros(4)})
    with pytest.raises(KeyError, match="解碼器沒有這些參數"):
        with bad.applied(dec):
            pass


def test_payload_來回不變():
    dec = ToyDecoder()
    a = _adapter(dec)
    b = ReconAdapter.from_payload(a.to_payload())
    assert torch.equal(a.z_star, b.z_star)
    assert set(a.decoder) == set(b.decoder)
    for k in a.decoder:
        assert torch.equal(a.decoder[k], b.decoder[k])


# ---------------------------------------------------------------------------
# 生成路徑的起點
# ---------------------------------------------------------------------------


class _StubModule:
    site = "apa"
    enabled = False

    def pixel_residual(self, x):
        return None

    def disable(self):
        pass

    def enable(self):
        pass


class _StubSD:
    """記下 `prepare()` 實際拿了什麼當起點。"""

    def __init__(self):
        self.encode_calls = 0
        self.seen = None

    def encode_text(self, p):
        return torch.zeros(1, 4, 4)

    def timesteps(self, k, t_max=None):
        return torch.arange(k)

    def conditioning_for(self, x, vae_ckpt=False):
        from contextlib import nullcontext
        return nullcontext()

    def encode_image(self, x, use_ckpt=False):
        self.encode_calls += 1
        return torch.zeros(1, 4, 8, 8)

    def ddim_inversion(self, z0, emb, ts, k):
        self.seen = z0.detach().clone()
        return z0


def test_有_adapter_時起點取_z_star_而不呼叫編碼器():
    sd = _StubSD()
    z = torch.full((1, 4, 8, 8), 0.7)
    gen = DefenseGenerator(sd, _StubModule(), k_inv=2,
                           recon=ReconAdapter(z, {}))
    gen.prepare(torch.rand(1, 3, 64, 64))
    assert sd.encode_calls == 0, "仍呼叫了 encode_image，A1 等於沒有接上"
    assert torch.equal(sd.seen, z)


def test_無_adapter_時走原本的編碼器():
    sd = _StubSD()
    gen = DefenseGenerator(sd, _StubModule(), k_inv=2)
    gen.prepare(torch.rand(1, 3, 64, 64))
    assert sd.encode_calls == 1


# ---------------------------------------------------------------------------
# config_hash
# ---------------------------------------------------------------------------


def _cell_cfg(cfg: RunConfig):
    """`config_hash` 吃的必填鍵，其餘固定，只讓 `module_params` 變動。"""
    return {
        "spec_version": 1, "model": "sd", "resolution": 512, "guidance": 7.5,
        "steps": 50, "strength": 0.6, "gpu": "RTX-3090", "precision": "fp32",
        "condition": "apa", "loss_params": {}, "optim_params": {},
        "lr": None, "tau": None, "purify": None, "seed": 0,
        "image_id": "horse_00",
        "module_params": cfg.module_params(),
    }


def test_recon_關閉時雜湊與新增欄位之前逐位相同():
    """既有批次（`runs/s3t20_merged` 等）續跑時每一格必須仍判為已完成。
    這裡直接釘住 `module_params` 的鍵集合——新增鍵無條件出現就是那個缺陷。"""
    cfg = RunConfig()
    assert cfg.recon is False
    assert set(cfg.module_params()) == {
        "warp_grid_size", "warp_max_disp", "warp_resample",
        "apa_lora_rank", "apa_latent_max_rank", "apa_latent_const_rank",
        "random_init_std",
    }


def test_recon_開啟時雜湊必須改變():
    """換了重建起點就是換一組實驗，沿用舊產物會把兩條路徑的結果混在一起。"""
    off, on = RunConfig(), RunConfig(recon=True)
    assert config_hash(_cell_cfg(off)) != config_hash(_cell_cfg(on))


@pytest.mark.parametrize("field,value", [
    ("recon_objective", "dists"),
    ("recon_a1_steps", 400),
    ("recon_a1_lr", 0.04),
    ("recon_a2_steps", 100),
    ("recon_a2_lr", 1e-3),
    ("recon_floor_ratio", 0.7),
    ("recon_gamma_acut", 0.0),
    ("recon_acut_band", 0.1),
    ("recon_w_pixel", 0.15),
])
def test_每個_A_段旋鈕都改變雜湊(field, value):
    """漏掉任一個，改了它的批次會沿用前一批的產物而沒有症狀。"""
    base = RunConfig(recon=True)
    other = RunConfig(recon=True, **{field: value})
    assert config_hash(_cell_cfg(base)) != config_hash(_cell_cfg(other))


# ---------------------------------------------------------------------------
# solve() 的端到端形狀
# ---------------------------------------------------------------------------


class _ToyVAE:
    def __init__(self, dec):
        self.decoder = dec


class _ToySD:
    """夠 `solve()` 跑完的最小替身：編碼是降採樣、解碼是升採樣過玩具解碼器。

    存在的理由是 `solve()` 只在 GPU 上跑過，而它把四個 callable 串起來——
    其中兩個的參數個數不同（`measure` 一個、`pairwise` 兩個）。實測上機時
    正是在這裡以 `TypeError: measure() takes 1 positional argument but 2
    were given` 中止，而那一步在 A1 的 200 步**之後**才執行，等於整批白跑。
    """

    def __init__(self):
        self.vae = _ToyVAE(ToyDecoder())

    def encode_image(self, x01, use_ckpt=False):
        return torch.nn.functional.avg_pool2d(x01.mean(1, keepdim=True)
                                              .repeat(1, 4, 1, 1), 4)

    def decode_latent(self, z, use_ckpt=False):
        y = self.vae.decoder.conv(self.vae.decoder.norm(z))
        y = torch.nn.functional.interpolate(y, scale_factor=4)
        return y[:, :3].clamp(0, 1)


def test_solve_跑得完並回傳可套用的_adapter():
    from src.defense import recon as reconmod

    torch.manual_seed(0)
    sd, x = _ToySD(), torch.rand(1, 3, 32, 32)

    def perceptual(a, b):
        return (a - b).pow(2).mean()

    def pairwise(a, b):
        from src.metrics.acutance import acutance
        return {"lpips": float((a - b).abs().mean()), **acutance(a, b)}

    def measure(y):
        return pairwise(x, y)

    adapter, history, summary = reconmod.solve(
        sd, x, perceptual, measure, pairwise, key="lpips",
        a1_steps=3, a1_lr=0.01, a2_steps=3, a2_lr=0.01, floor_ratio=0.5,
        w_pixel=0.5, gamma_acut=1.0, acut_band=0.05, resp_seed=0,
        resp_scale=0.05, log_every=1)

    assert adapter.z_star.shape == sd.encode_image(x).shape
    assert set(adapter.decoder) == {"norm.weight", "norm.bias", "conv.bias"}
    assert {h["phase"] for h in history} == {"A1", "A2"}
    # 探針必須真的量到東西：兩個都是 0 代表 `pairwise` 被餵了同一張圖。
    assert summary["resp_a1"] > 0 and summary["resp_a2"] > 0
    with adapter.applied(sd.vae.decoder):
        pass


def test_solve_把微調值還原給下一張圖():
    """`solve` 內部用 `restored` 包住 A2。沒有還原的話第二張圖是從第一張的
    過擬合權重出發，而那看起來只是「後面的圖比較好壓」。"""
    from src.defense import recon as reconmod

    sd, x = _ToySD(), torch.rand(1, 3, 32, 32)
    before = {k: v.detach().clone() for k, v in sd.vae.decoder.named_parameters()}

    def pairwise(a, b):
        from src.metrics.acutance import acutance
        return {"lpips": float((a - b).abs().mean()), **acutance(a, b)}

    reconmod.solve(sd, x, lambda a, b: (a - b).pow(2).mean(),
                   lambda y: pairwise(x, y), pairwise, key="lpips",
                   a1_steps=2, a1_lr=0.01, a2_steps=2, a2_lr=0.01,
                   floor_ratio=0.5, w_pixel=0.5, gamma_acut=0.0,
                   acut_band=0.05, resp_seed=0, resp_scale=0.05, log_every=1)

    for k, v in sd.vae.decoder.named_parameters():
        assert torch.equal(v, before[k]), f"{k} 沒有還原"


# ---------------------------------------------------------------------------
# apa_lora_rank = 0：拿掉不起作用的階段一
# ---------------------------------------------------------------------------


def test_lora_rank_0_只留_latent_一個成員():
    """實測 LoRA 的 B 因子恆為零、`--recon` 後又不再訓練，那 1594368 個參數
    對輸出沒有作用。留著的代價是論文會描述一個沒有在跑的東西。"""
    from src.residual.site_apa import build_apa

    m = build_apa(None, steps=2, latent_size=8, lora_rank=0,
                  latent_max_rank=4, latent_const_rank=2)
    assert list(m.param_groups()) == ["stage2"], "組名變了，StageSpec 會抓不到"
    assert len(m.members) == 1


def test_lora_rank_0_的方向參數仍然抓得到():
    """射線縮放依能力挑成員。寫死 `members[1]` 的話，縮到錯的參數上不會有
    任何症狀——失真照樣解到 τ，只是解的不是攻擊的那一半。"""
    from src.experiment.executors import direction_param
    from src.residual.site_apa import build_apa

    m = build_apa(None, steps=2, latent_size=8, lora_rank=0,
                  latent_max_rank=4, latent_const_rank=2)
    p = direction_param(m)
    assert p is m.members[0].tensor.V


def test_φ_初始仍為零殘差():
    """關掉 LoRA 不可以破壞「φ=0 逐位元等同未注入」，x_base 的定義靠它。"""
    from src.residual.site_apa import build_apa

    m = build_apa(None, steps=2, latent_size=8, lora_rank=0,
                  latent_max_rank=4, latent_const_rank=2)
    hook = m.eps_hook(torch.arange(2), 2)
    eps = torch.randn(1, 4, 8, 8)
    assert torch.equal(hook(eps.clone(), 0, torch.tensor(0)), eps)


# ---------------------------------------------------------------------------
# 投影式約束（改良 1–3）
# ---------------------------------------------------------------------------


class _Scalable:
    """`build(k)` 的最小替身：失真隨 k 單調上升，且帶一個 φ=0 的下限。

    下限這一項是本專案生成路徑的實際形狀（VAE 來回誤差），而投影解的是
    **增量**——把它寫進替身，才驗得到「floor 有沒有被減掉」。
    """

    def __init__(self, floor=0.08, slope=0.02):
        self.p = torch.nn.Parameter(torch.ones(1))
        self.floor, self.slope = floor, slope

    def metric(self):
        return self.floor + self.slope * float(self.p.data.abs().sum())


def _projector(obj, delta, tol=1e-4, max_iter=40):
    """把 `budget_projector` 的數值核心搬到替身上重跑一次。

    直接呼叫 `budget_projector` 需要 SD、指標套件與 ImageEntry；此處驗的是
    「投影後 φ 是否恰好落在球面上」這件事本身，那是純數值的。
    """
    state = {"k": 1.0, "base": None}

    def render(scale):
        saved = obj.p.data.clone()
        obj.p.data = saved * float(scale)
        out = obj.metric()
        obj.p.data = saved
        return out

    def project(step=0):
        if state["base"] is None:
            state["base"] = render(0.0)
        target = state["base"] + delta
        lo, hi = 0.0, max(1.0, state["k"])
        while render(hi) < target:
            hi *= 2.0
        k = 0.5 * (lo + hi)
        for _ in range(max_iter):
            got = render(k)
            if abs(got - target) < tol:
                break
            lo, hi = (k, hi) if got < target else (lo, k)
            k = 0.5 * (lo + hi)
        obj.p.data = obj.p.data * float(k)
        return {"proj_k": k, "proj_metric": got, "proj_floor": state["base"]}

    return project


def test_投影後恰好落在預算球面上():
    o = _Scalable()
    log = _projector(o, delta=0.04)()
    assert o.metric() == pytest.approx(0.08 + 0.04, abs=1e-3)
    assert log["proj_floor"] == pytest.approx(0.08)


def test_投影解的是增量而不是絕對值():
    """生成路徑的 φ=0 下限不是零。投影若解絕對值，Δ 小於下限時無解，而那
    正是 τ=0.05／0.10 對 site apa 「結構上不可達」的成因。"""
    o = _Scalable(floor=0.08)
    _projector(o, delta=0.04)()
    assert o.metric() - 0.08 == pytest.approx(0.04, abs=1e-3)


def test_連續投影後仍留在球面上():
    """梯度步會把 φ 推離球面，投影要把它拉回來。連續做才看得出狀態有沒有
    在多次呼叫之間累積錯誤（例如把上一次的 k 重複乘進去）。"""
    o = _Scalable()
    proj = _projector(o, delta=0.04)
    for _ in range(5):
        with torch.no_grad():          # 模擬一次梯度步
            o.p.data = o.p.data * 1.7
        proj()
        assert o.metric() == pytest.approx(0.12, abs=1e-3)


def test_投影旗標未給預算時拋出():
    """預算就是這個機制本身，沒有預設值可以沿用；靜默取一個值會讓訓練綁在
    一個與段 2 不同的球面上，而報表上兩者都叫 Δ。"""
    from src.experiment.executors import CONDITION_SPECS
    assert CONDITION_SPECS["apa_pj"].project is True
    assert CONDITION_SPECS["apa"].project is False


def test_project_budget_未給時不進雜湊():
    from src.experiment.executors import RunConfig
    assert "project" not in RunConfig().module_params()
    assert "project" in RunConfig(project_budget=0.04).module_params()


def test_投影式條件的隨機起點在_build_module_而非訓練路徑():
    """模塊有多條建構路徑：段 1 訓練、段 0 的 `_probe_lr`、量測腳本。起點只加
    在其中一條的話，其餘每一條都會各自以「放大到 2048 倍仍達不到 Δ」中止一次
    ——那正是本專案 2026-08-11 實測到的失效順序。"""
    import inspect

    from src.experiment import executors

    src = inspect.getsource(executors.build_module)
    assert "condition_spec(condition).project" in src, (
        "隨機起點不在 build_module 裡，段 0 的學習率探測會拿到零方向")
    train = inspect.getsource(executors._train_nonadditive)
    assert "randn" not in train, "訓練路徑上仍有一份重複的起點初始化"


def test_A段接上時不探測階段一的學習率():
    """`optim_config` 已把 `align_steps` 歸零，那個學習率不會被任何一格用到。
    實測每條件約 20 分鐘（5 個候選 × 60 步），純浪費。"""
    import inspect

    from src.experiment import executors

    src = inspect.getsource(executors.calibrate_lr)
    assert "skip_align" in src and "uses_stage_a" in src
    # 三處必須共用同一個謂詞：只改其中兩處不會報錯，只會多燒 910 秒或在段 1
    # 才以缺鍵中止（見 `uses_stage_a` 的說明）。
    # 謂詞的本體與它自己的說明各算一次，其餘任何一處都是就地展開。
    whole = inspect.getsource(executors)
    assert whole.count('cfg.recon and spec.site == "apa"') == 2, (
        "政策謂詞又被就地展開了；它必須只出現在 uses_stage_a 裡")
    for fn in (executors.optim_config, executors._train_nonadditive,
               executors._train_random):
        assert 'spec.site == "apa"' not in inspect.getsource(fn).replace(
            "uses_stage_a", ""), f"{fn.__name__} 仍在就地比對 site 名稱"


def test_投影式條件關掉_LPIPS_hinge_但保留銳利度與色偏(tmp_path):
    """同一件事不可由兩個機制同時管：投影已保證失真落在預算球面上，hinge
    要嘛恆為零、要嘛與投影拉扯。銳利度與色偏保留——它們不是縮放能保證的
    性質，投影管不到。"""
    from dataclasses import replace as dc_replace

    from src.experiment import executors
    from tests.test_executors import make_res

    res = make_res(tmp_path)
    res.cfg = dc_replace(res.cfg, project_budget=0.04, attn_mask_tau=0.5)
    entry = res.image("dog_00")
    pj = executors.loss_config(res, executors.condition_spec("apa_pj"), entry)
    ap = executors.loss_config(res, executors.condition_spec("apa"), entry)
    assert pj.gamma_lpips == 0.0
    assert ap.gamma_lpips > 0.0
    assert pj.gamma_acut == ap.gamma_acut and pj.gamma_chroma == ap.gamma_chroma


# ---------------------------------------------------------------------------
# 投影模式下的停止準則
# ---------------------------------------------------------------------------


def _hist(n, monitor, pen=0.0):
    return [{"step": i, "attn_suppressed": monitor(i), "fid_pen_lpips": pen,
             "fid_pen_acut": pen} for i in range(n)]


def test_投影模式下約束恆滿足時仍judge得到平台():
    """hinge 恆為零在投影模式下代表「一直被綁得剛剛好」，不是「還沒被綁住」。
    預設的 `require_constraint` 分不出這兩者，實測三格全部跑滿 250 步，而跑滿
    上限的格子依本專案的協議不可用於跨條件比較。"""
    from src.defense.optimize import plateau_stop

    h = _hist(60, lambda i: 1.0 - 0.5 ** i, pen=0.0)   # 監看量早已走平
    kw = dict(patience=20, tol=1e-4, min_steps=25,
              monitor_key="attn_suppressed")
    stop_default, _ = plateau_stop(h, **kw)
    stop_proj, reason = plateau_stop(h, require_constraint=False, **kw)
    assert stop_default is False, "預設行為變了：罰項模式不該在約束未啟動時停"
    assert stop_proj is True and reason


def test_投影式條件把該旗標關掉而其餘條件不變(tmp_path):
    from dataclasses import replace as dc_replace

    from src.experiment import executors
    from tests.test_executors import make_res

    res = make_res(tmp_path)
    res.cfg = dc_replace(res.cfg, project_budget=0.04, attn_mask_tau=0.5)
    entry = res.image("dog_00")
    pj = executors.optim_config(res, executors.condition_spec("apa_pj"), entry)
    ap = executors.optim_config(res, executors.condition_spec("apa"), entry)
    assert pj.stop_require_constraint is False
    assert ap.stop_require_constraint is True


def test_參數組由_ConditionSpec_明寫而非由_site_名稱推導():
    """先前是 `"stage2" if site == "apa" else "default"`，與本專案「依能力分派、
    不比對名稱」的慣例相反。新增一個同樣有兩階段的 site 時，漏改的後果可能是
    落回 `default` 而把兩個階段的參數一起更新——那與設定的兩階段結構是兩件
    不同的事，且沒有症狀。"""
    import inspect

    from src.experiment.executors import CONDITION_SPECS, optim_config

    src = inspect.getsource(optim_config)
    assert 'spec.site == "apa"' not in src.replace("uses_stage_a", "")
    for name, spec in CONDITION_SPECS.items():
        want = ("stage2", "stage1") if spec.site == "apa" else ("default",
                                                                "default")
        assert (spec.train_group, spec.align_group)[:len(want)] == want, name

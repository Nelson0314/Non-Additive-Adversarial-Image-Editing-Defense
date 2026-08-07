"""計算層的**契約**測試 — `src/experiment/executors.py`。

## 這裡驗什麼、不驗什麼

驗的是「executor 有沒有照契約辦事」：

- 回傳 `(artifacts, extra_meta)`，且 `artifacts` 裡的每一條路徑執行後真的存在
  （續跑判定 `ProgressWriter.is_done` 只檢查存在性，列了卻沒產生的路徑會讓
  該格永遠重跑，而症狀是「這段怎麼跑不完」）；
- 產物落盤符合 `docs/CODE_2026-08-05.md` §4；
- 學習率確實只從 `Calibration.get()` 取得，未校準時拋出；
- 各段呼叫 `optimize()` / `run_pgd()` / `solve_k()` 時傳的參數正確。

**不驗**收斂、不驗數值——那需要真實 SDXL 權重與 GPU。凡是只有在真實模型上
才成立的性質，本檔一律不下斷言，改由主 session 在 RTX 5090 上驗（清單見
本檔末尾的註解）。

## 為什麼不載入任何擴散模型

`R`（同失真隨機對照）走位移場，整條路徑是 `grid_sample`，完全不碰 SD。
故 train → rayscale → control → eval → report 這一整串可以用 `FakeSD` 跑通，
而被測的仍是同一份 executor 程式。三個非加性條件與五篇 baseline 的
SD 呼叫則以 `unittest.mock.patch` 攔在 `optimize` / `run_pgd` 這一層。
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from src.defense.optimize import OptimConfig, OptimResult
from src.experiment import executors, grid
from src.experiment.runner import cell_config, run_stage
from src.utils.calibration import Calibration, CalibrationMismatch
from src.utils.cellid import config_hash
from src.utils.progress import ProgressWriter

DATA = Path(__file__).resolve().parent.parent / "data" / "lo_aligned"
SIZE = 32


# ---------------------------------------------------------------------------
# 假模型
# ---------------------------------------------------------------------------


class FakeSD:
    """只提供 executor 實際會呼叫的介面。多的不給——多給一個方法就等於
    讓測試涵蓋一條沒有人會走的路徑。"""

    def __init__(self):
        self.device = torch.device("cpu")
        self.sdedit_calls = []

    def encode_text(self, prompt):
        return torch.zeros(1, 4, 8)

    def uncond_prompt(self, batch: int = 1):
        return torch.zeros(1, 4, 8)

    def latent_shape(self, h, w):
        return (1, 4, h // 8, w // 8)

    def sample_edit_noise(self, z_like, seed):
        g = torch.Generator().manual_seed(int(seed))
        return torch.randn(z_like.shape, generator=g)

    def sdedit(self, x01, emb, noise, num_steps, strength=0.5,
               guidance_scale=1.0, emb_uncond=None, **kw):
        self.sdedit_calls.append({
            "num_steps": num_steps, "strength": strength,
            "guidance_scale": guidance_scale,
            "noise_sum": float(noise.sum()),
        })
        # 輸出必須依賴 x 與 noise，否則兩側的比較恆為零、測試沒有鑑別力
        return (x01 * 0.7 + 0.15 + float(noise.mean()) * 0.01).clamp(0, 1)

    def decode_latent(self, z, use_ckpt=False):
        return z[:, :3].clamp(0, 1)


class FakeSuite:
    """`MetricSuite` 的替身。`lpips` 取平均絕對差乘一個常數：射線縮放的
    二分搜尋要的是**單調且可達**，不是真的 LPIPS。"""

    LPIPS_GAIN = 20.0

    def pairwise(self, a, b):
        d = (a - b).abs()
        return {
            "psnr": 42.0, "linf": float(d.max()), "ssim": 0.9,
            "vif_p": 0.8, "fsim": 0.95,
            "lpips": float(d.mean()) * self.LPIPS_GAIN,
            "dists": 0.1, "acutance_ratio": 1.0,
            "rms": float((d ** 2).mean().sqrt()),
            "frac_gt_16_255": float((d > 16 / 255).float().mean()),
        }

    def niqe(self, x):
        return float("nan")

    def semantic(self, x, prompt):
        return {"clip": float(x.mean()), "siglip": float(x.std())}

    def release_vlm(self):
        """真身會把 CLIP 與 SigLIP 移出顯存（1,352 MB）。替身記錄被呼叫的
        次數，供 `test_訓練前釋放語意權重` 檢查呼叫點沒有被拿掉。"""
        self.released = getattr(self, "released", 0) + 1

    def full(self, a, b, prompt=None):
        out = self.pairwise(a, b)
        out["niqe_a"], out["niqe_b"] = self.niqe(a), self.niqe(b)
        if prompt is not None:
            for k, v in self.semantic(a, prompt).items():
                out[f"{k}_a"] = v
            for k, v in self.semantic(b, prompt).items():
                out[f"{k}_b"] = v
        return out


BASE = {
    "spec_version": 1, "model": "fake-sd", "resolution": SIZE,
    "guidance": 7.5, "steps": 3, "strength": 0.6,
    "gpu": "test", "precision": "fp32", "lr": None,
}


CALIB_KEYS = ("lr.N1", "lr.N2", "lr.N3_stage1", "lr.N3_stage2",
              "stop_tol.shared_mass", "stop_tol.edit_shift")


def make_res(tmp_path, images=("dog_00", "cat_00"), with_calib=True,
             calib_keys=CALIB_KEYS, **cfg_kw) -> executors.Resources:
    cfg = executors.RunConfig(
        resolution=SIZE, guidance=7.5, strength=0.6, steps=3, seed=7,
        train_n_edit=2, k_inv=2, max_steps=3, align_steps=2, probe_steps=2,
        warp_grid_size=4, warp_max_disp=6.0, random_init_std=0.5,
        # attention 擷取要掃真實 UNet 的 attn2 層，`FakeSD` 沒有 UNet。
        # 這是替身的明示宣告，不是效能選項——預設必須是開，
        # 由 `test_attention擷取預設為開` 釘住。
        capture_attn=False,
        target_image="", **cfg_kw,
    )
    base = dict(BASE, loss_params=cfg.loss_params(),
                module_params=cfg.module_params(),
                optim_params=cfg.optim_params())
    entries = executors.load_lo_aligned(DATA, SIZE, torch.device("cpu"),
                                        ids=list(images))
    res = executors.Resources(
        sd=FakeSD(), suite=FakeSuite(), batch_dir=tmp_path / "b1",
        base_config=base, cfg=cfg,
        images={e.image_id: e for e in entries},
        y_target=torch.full((1, 3, SIZE, SIZE), 0.5),
    )
    res.batch_dir.mkdir(parents=True, exist_ok=True)
    if with_calib:
        table = Calibration()
        fixed = {"stop_tol.shared_mass": 3e-4, "stop_tol.edit_shift": 5e-4}
        for key in calib_keys:
            table.put(key, fixed.get(key, 0.01), res.calib_context,
                      note="測試固定值")
        res.calib = table
    return res


def fake_optim_result(res, entry) -> OptimResult:
    hist = [{"step": i, "stage": "default", "stage_step": i, "lr": 0.01,
             "loss": 1.0 - 0.1 * i, "L_def": 0.5, "L_fid": 0.2,
             "edit_shift": 0.01 * i, "grad_norm": 0.3,
             "fid_psnr": 30.0, "fid_linf": 0.05, "fid_lpips": 0.1}
            for i in range(3)]
    return OptimResult(
        history=hist,
        x_def=(entry.x01 * 0.9 + 0.05),
        x_base=entry.x01, x_base0=entry.x01,
        seconds=1.0, steps_done=len(hist),
        stage_reports=[{"group": "default", "lr_key": "lr.N2", "lr": 0.01,
                        "max_steps": 3, "steps": 3, "stop_reason": "ok"}],
        stop_reason="ok",
    )


def assert_artifacts_exist(res, artifacts):
    """續跑判定只檢查存在性；列了卻沒產生的路徑會讓該格永遠重跑。"""
    assert artifacts, "executor 沒有回報任何產物，續跑判定會永遠判為未完成"
    for a in artifacts:
        assert not Path(a).is_absolute(), f"{a} 是絕對路徑，必須相對批次目錄"
        assert "\\" not in a, f"{a} 含反斜線；跨平台一律用 posix 路徑"
        assert (res.batch_dir / a).exists(), f"回報了不存在的產物：{a}"


# ---------------------------------------------------------------------------
# 條件表
# ---------------------------------------------------------------------------


def test_條件表與格點的條件逐鍵對應():
    """格點列了某個條件、計算層卻不認識它，症狀是整段跑到該條件才失敗。"""
    assert set(executors.CONDITION_SPECS) == set(grid.CONDITIONS)


def test_未知條件立刻拋出而非猜一個預設():
    with pytest.raises(KeyError, match="計算層沒有定義"):
        executors.condition_spec("N9")


def test_calib與report沒有格點式的計算層():
    """硬塞進格點框架只會得到一個「零格、永遠成功」的段。"""
    for stage in ("calib", "report"):
        with pytest.raises(KeyError, match="run_calibration"):
            executors.make_executor(stage)


# ---------------------------------------------------------------------------
# 學習率的唯一入口
# ---------------------------------------------------------------------------


def test_學習率鍵逐條件不同且來自校準表(tmp_path):
    res = make_res(tmp_path)
    keys = {c: executors.optim_config(res, executors.condition_spec(c))
            .stages[0].lr_key for c in grid.NONADDITIVE}
    assert keys == {"N1": "lr.N1", "N2": "lr.N2", "N3": "lr.N3_stage2"}
    assert len(set(keys.values())) == 3, "學習率不可跨條件共用"


def test_N3的兩階段分別取用不同的組與鍵(tmp_path):
    res = make_res(tmp_path)
    cfg = executors.optim_config(res, executors.condition_spec("N3"))
    assert cfg.stages[0].group == "stage2", "階段二只更新 latent"
    assert cfg.align_group == "stage1", "階段一只更新 LoRA"
    assert cfg.align_lr_key == "lr.N3_stage1"
    assert cfg.align_steps > 0
    assert cfg.stages[0].lr_key != cfg.align_lr_key, (
        "兩階段的參數量綱不同，Adam 每步位移約等於 lr，共用一個值代表兩種步長"
    )


def test_N1向校準表索取stop_tol(tmp_path):
    """停止門檻沒有回退路徑，未校準時 `resolve_stop_tol` 會拋出。"""
    res = make_res(tmp_path)
    cfg = executors.optim_config(res, executors.condition_spec("N1"))
    assert cfg.stop_tol == pytest.approx(3e-4)
    assert cfg.unet_ckpt is False, (
        "cross-attention 擷取與 UNet checkpoint 不相容，實測 backward 中止"
    )


def test_N2的edit_shift門檻同樣由校準表取得(tmp_path):
    """2026-08-05 收緊。before：`edit_shift` 走 `MONITOR_TOL` 的 1e-4——
    那是 SD v1.4／512² 的實測值，靜默沿用到 SDXL／1024²。"""
    res = make_res(tmp_path)
    assert executors.optim_config(
        res, executors.condition_spec("N2")).stop_tol == pytest.approx(5e-4)


def test_未校準edit_shift時N2直接拋出(tmp_path):
    from src.utils.calibration import CalibrationMismatch

    res = make_res(tmp_path, calib_keys=("lr.N1", "lr.N2", "lr.N3_stage1",
                                         "lr.N3_stage2", "stop_tol.shared_mass"))
    with pytest.raises(CalibrationMismatch):
        executors.optim_config(res, executors.condition_spec("N2"))


def test_沒有校準表時取學習率直接拋出(tmp_path):
    """不回退到預設值——未校準的值沿用正是本專案重複十次的缺陷。"""
    res = make_res(tmp_path, with_calib=False)
    with pytest.raises(CalibrationMismatch, match="段 0"):
        res.require_calib()


def test_校準context恰好是必填欄位不多不少(tmp_path):
    """`Calibration.get` 比對的是完全相等而非子集。"""
    from src.utils.calibration import REQUIRED_CONTEXT

    res = make_res(tmp_path)
    assert set(res.calib_context) == set(REQUIRED_CONTEXT)


# ---------------------------------------------------------------------------
# 段 1：訓練
# ---------------------------------------------------------------------------


def test_train_非加性條件呼叫optimize並帶入校準表(tmp_path):
    res = make_res(tmp_path)
    entry = res.image("dog_00")
    cell = grid.Cell("train", "N2", "dog_00")
    seen = {}

    def fake_optimize(sd, module, x01, cfg, loss_cfg, purifiers, **kw):
        seen.update(cfg=cfg, loss_cfg=loss_cfg, kw=kw,
                    purifiers=[p.kind for p in purifiers])
        return fake_optim_result(res, entry)

    with patch.object(executors, "optimize", fake_optimize):
        arts, extra = executors.train_executor(cell, {"res": res})

    assert seen["kw"]["calib"] is res.calib, "校準表必須傳給 optimize"
    assert seen["kw"]["calib_context"] == res.calib_context
    assert seen["kw"]["y_target"] is res.y_target, "targeted_output 需要目標影像"
    assert seen["cfg"].stages[0].lr_key == "lr.N2"
    assert seen["loss_cfg"].defense_mode == "targeted_output"
    assert seen["loss_cfg"].tau_lpips == pytest.approx(grid.TRAIN_TAU)
    assert "identity" in seen["purifiers"], "訓練期的 𝒫 必須含恆等算子"
    assert_artifacts_exist(res, arts)
    assert extra["steps_used"] == 3 and extra["stop_reason"] == "ok"
    assert extra["lr"] == pytest.approx(0.01), "實際採用的 lr 必須留在 meta"


def test_train_targeted_attn不傳目標影像(tmp_path):
    """N1 的著力點是注意力質量，沒有目標影像這回事。"""
    res = make_res(tmp_path)
    entry = res.image("dog_00")
    seen = {}

    def fake_optimize(sd, module, x01, cfg, loss_cfg, purifiers, **kw):
        seen.update(kw=kw, loss_cfg=loss_cfg)
        return fake_optim_result(res, entry)

    with patch.object(executors, "optimize", fake_optimize):
        executors.train_executor(grid.Cell("train", "N1", "dog_00"),
                                 {"res": res})
    assert seen["kw"]["y_target"] is None
    assert seen["loss_cfg"].defense_mode == "targeted_attn"


def test_train_產物落盤符合CODE第四節(tmp_path):
    """`meta.json` + `phi.pt` + `train.csv` + 影像，缺一不可。"""
    res = make_res(tmp_path)
    entry = res.image("dog_00")
    with patch.object(executors, "optimize",
                      lambda *a, **k: fake_optim_result(res, entry)):
        arts, _ = executors.train_executor(
            grid.Cell("train", "N2", "dog_00"), {"res": res})
    names = {Path(a).name for a in arts}
    assert {"meta.json", "phi.pt", "train.csv", "orig.png", "x_def.png",
            "residual.png"} <= names
    d = res.batch_dir / "N2" / "dog_00"
    assert (d / "history.png").exists(), "優化曲線是診斷的最低要求"


def test_train_的meta雜湊與骨架算的一致(tmp_path):
    """兩者不一致時，續跑判定會判為未完成而永遠重跑，且沒有其他症狀。"""
    import json

    res = make_res(tmp_path)
    entry = res.image("dog_00")
    cell = grid.Cell("train", "N2", "dog_00")
    with patch.object(executors, "optimize",
                      lambda *a, **k: fake_optim_result(res, entry)):
        executors.train_executor(cell, {"res": res})
    meta = json.loads((res.batch_dir / "N2" / "dog_00" / "meta.json")
                      .read_text(encoding="utf-8"))
    assert meta["config_hash"] == config_hash(cell_config(cell, res.base_config))
    assert meta["cell_id"] == cell.cell_id()


def test_train_baseline走run_pgd且帶入該篇的spec(tmp_path):
    from src.baselines.pgd import PGDResult

    res = make_res(tmp_path)
    entry = res.image("dog_00")
    seen = {}

    def fake_run_pgd(sd, x01, spec, **kw):
        seen.update(spec=spec, kw=kw)
        return PGDResult(delta01=torch.zeros_like(x01) + 0.01,
                         x_adv01=(x01 + 0.01).clamp(0, 1),
                         history=[{"step": 0, "loss": 1.0,
                                   "delta_linf01": 0.01, "delta_l2": 1.0}],
                         seconds=0.5)

    with patch.object(executors, "run_pgd", fake_run_pgd):
        arts, extra = executors.train_executor(
            grid.Cell("train", "photoguard_c", "dog_00"), {"res": res})

    assert seen["spec"].name == "photoguard_c"
    assert seen["kw"]["strength"] == pytest.approx(res.cfg.strength), (
        "PhotoGuard-c 的 img2img 版 strength 原始碼沒有，必須由呼叫端明給"
    )
    assert seen["kw"]["seed"] == res.cfg.seed
    assert extra["steps_used"] == seen["spec"].steps, "步數照原論文"
    assert extra["lr"] is None, "PGD 的步長來自 spec，不走校準表"
    assert extra["modified_from_paper"] == seen["spec"].modified_from_paper
    assert_artifacts_exist(res, arts)


def test_train_baseline的phi是加性delta(tmp_path):
    from src.baselines.pgd import PGDResult

    res = make_res(tmp_path)
    with patch.object(executors, "run_pgd",
                      lambda sd, x01, spec, **kw: PGDResult(
                          delta01=torch.zeros_like(x01) + 0.02,
                          x_adv01=(x01 + 0.02).clamp(0, 1), history=[])):
        executors.train_executor(grid.Cell("train", "dia_r", "dog_00"),
                                 {"res": res})
    payload = executors.load_phi(res.batch_dir / "dia_r" / "dog_00" / "phi.pt")
    assert payload["parameterization"] == "additive"
    assert payload["delta01"].abs().max() == pytest.approx(0.02)


def test_train_隨機對照不做任何最佳化(tmp_path):
    """R 走與非加性條件相同的參數化，但參數取高斯隨機。"""
    res = make_res(tmp_path)
    with patch.object(executors, "optimize",
                      lambda *a, **k: pytest.fail("R 不得呼叫 optimize")):
        arts, extra = executors.train_executor(
            grid.Cell("train", "R", "dog_00"), {"res": res})
    assert extra["steps_used"] == 0
    assert "random" in extra["stop_reason"]
    assert_artifacts_exist(res, arts)
    payload = executors.load_phi(res.batch_dir / "R" / "dog_00" / "phi.pt")
    assert payload["parameterization"] == "warp"
    assert payload["state_dict"]["flow"].abs().max() > 0, "隨機方向不得為零"


def test_train_隨機對照的方向逐影像獨立且可重現(tmp_path):
    res = make_res(tmp_path)
    for img in ("dog_00", "cat_00"):
        executors.train_executor(grid.Cell("train", "R", img), {"res": res})
    a = executors.load_phi(res.batch_dir / "R" / "dog_00" / "phi.pt")
    b = executors.load_phi(res.batch_dir / "R" / "cat_00" / "phi.pt")
    assert not torch.equal(a["state_dict"]["flow"], b["state_dict"]["flow"])

    res2 = make_res(tmp_path / "again")
    executors.train_executor(grid.Cell("train", "R", "dog_00"), {"res": res2})
    c = executors.load_phi(res2.batch_dir / "R" / "dog_00" / "phi.pt")
    assert torch.equal(a["state_dict"]["flow"], c["state_dict"]["flow"]), (
        "同一張影像在不同執行中必須得到同一個隨機方向"
    )


# ---------------------------------------------------------------------------
# φ 的落盤與射線縮放
# ---------------------------------------------------------------------------


def test_射線縮放只乘攻擊那一半的參數():
    """N3 的階段一是保真對齊的結果，一起縮放等於在改變重建本身。"""
    from src.residual.site_warp import WarpResidual

    mod = WarpResidual(size=SIZE, grid_size=4, init_std=0.3, seed=1)
    assert executors.direction_param(mod) is mod.flow

    class _Unknown:
        pass

    with pytest.raises(TypeError, match="射線縮放的方向參數"):
        executors.direction_param(_Unknown())


def test_rayscale_落在目標失真上並留下逐τ產物(tmp_path):
    res = make_res(tmp_path)
    executors.train_executor(grid.Cell("train", "R", "dog_00"), {"res": res})

    cell = grid.Cell("rayscale", "R", "dog_00", tau=0.05)
    arts, extra = executors.rayscale_executor(cell, {"res": res})

    assert extra["tau_target"] == pytest.approx(0.05)
    assert extra["tau_achieved"] == pytest.approx(0.05, abs=0.005), (
        "solve_k 的容差是 0.005；超出表示縮放沒有真的落在預算上"
    )
    assert_artifacts_exist(res, arts)
    names = {Path(a).name for a in arts}
    assert {"phi_tau0.05.pt", "x_def_tau0.05.png", "residual_tau0.05.png",
            "fidelity_tau0.05.csv", "meta_tau0.05.json"} == names


def test_rayscale_縮放後的phi可獨立重建出同一張圖(tmp_path):
    """段 3 是由 `phi_tau{τ}.pt` 重建 x_def 的，兩者不一致等於評測的不是
    段 2 宣告的那個失真。"""
    res = make_res(tmp_path)
    entry = res.image("dog_00")
    executors.train_executor(grid.Cell("train", "R", "dog_00"), {"res": res})
    executors.rayscale_executor(
        grid.Cell("rayscale", "R", "dog_00", tau=0.05), {"res": res})

    payload = executors.load_phi(
        res.batch_dir / "R" / "dog_00" / "phi_tau0.05.pt")
    x = executors.materialize(payload, res, entry)
    got = res.suite.pairwise(entry.x01, x)["lpips"]
    assert got == pytest.approx(0.05, abs=0.005)


def test_rayscale_缺段一產物時明確指出缺哪個檔(tmp_path):
    res = make_res(tmp_path)
    with pytest.raises(FileNotFoundError, match="phi.pt"):
        executors.rayscale_executor(
            grid.Cell("rayscale", "R", "dog_00", tau=0.05), {"res": res})


def test_rayscale_達不到目標時拋出而非取最接近值(tmp_path):
    """「達不到 τ」與「達到了」在下游完全不同。"""
    res = make_res(tmp_path)
    executors.train_executor(grid.Cell("train", "R", "dog_00"), {"res": res})
    with pytest.raises(ValueError, match="達不到"):
        executors.rayscale_executor(
            grid.Cell("rayscale", "R", "dog_00", tau=50.0), {"res": res})


def test_phi格式不符時拒絕讀取(tmp_path):
    res = make_res(tmp_path)
    p = res.batch_dir / "bad.pt"
    torch.save({"format": 999}, p)
    with pytest.raises(ValueError, match="φ 格式"):
        executors.load_phi(p)


# ---------------------------------------------------------------------------
# 段 3：φ=0 對照與評測
# ---------------------------------------------------------------------------


def _chain(res, condition="R", image="dog_00", tau=0.05,
           purify=("identity", 0.0), seed=0):
    """跑完 train → rayscale → control → eval 的一條鏈，回傳 eval 的結果。"""
    executors.train_executor(grid.Cell("train", condition, image), {"res": res})
    executors.rayscale_executor(
        grid.Cell("rayscale", condition, image, tau=tau), {"res": res})
    ctrl = executors.control_executor(
        grid.Cell("control", "phi0", image, purify=purify, seed=seed),
        {"res": res})
    ev = executors.eval_executor(
        grid.Cell("eval", condition, image, tau=tau, purify=purify, seed=seed),
        {"res": res})
    return ctrl, ev


def test_control_產物落在共用目錄且與條件無關(tmp_path):
    res = make_res(tmp_path)
    (arts, extra) = executors.control_executor(
        grid.Cell("control", "phi0", "dog_00", purify=("blur", 1.0), seed=2),
        {"res": res})
    assert_artifacts_exist(res, arts)
    assert all(a.startswith("control/dog_00/purify/blur_1/") for a in arts), (
        "對照跨 9 個條件共用，故不得寫進任何一個條件的目錄下"
    )
    assert extra["purify_kind"] == "blur" and extra["seed"] == 2


def test_eval_兩側共用同一組評測噪聲(tmp_path):
    """兩條分支的噪聲不同時，量到的偏移主要來自噪聲差異而非防禦。"""
    res = make_res(tmp_path)
    _chain(res, seed=3)
    noises = [c["noise_sum"] for c in res.sd.sdedit_calls]
    assert len(noises) == 2 and noises[0] == pytest.approx(noises[1])


def test_eval_噪聲種子與訓練錯開(tmp_path):
    res = make_res(tmp_path)
    assert executors.eval_noise_seed(res, 0) == res.cfg.seed + 10_000
    assert (executors.eval_noise_seed(res, 1)
            != executors.eval_noise_seed(res, 0))


def test_eval_效果量的是對照減防禦(tmp_path):
    res = make_res(tmp_path)
    (_, ctrl_extra), (arts, row) = _chain(res)
    assert row["effect_siglip"] == pytest.approx(
        row["edit_siglip_a"] - row["edit_siglip_b"])
    assert row["effect_abs"] == pytest.approx(row["effect_siglip"]), (
        "判定用 SigLIP：CLIP 分不出編輯是否發生（標準差大於均值）"
    )
    assert "effect_clip" in row, "對齊文獻仍需 CLIP 欄位"
    assert_artifacts_exist(res, arts)
    assert {Path(a).name for a in arts} == {
        "x_purified.png", "edit_seed0.png", "metrics_seed0.json"}


def test_eval_缺對照時明確指出必須先跑control(tmp_path):
    res = make_res(tmp_path)
    executors.train_executor(grid.Cell("train", "R", "dog_00"), {"res": res})
    executors.rayscale_executor(
        grid.Cell("rayscale", "R", "dog_00", tau=0.05), {"res": res})
    with pytest.raises(FileNotFoundError, match="對照"):
        executors.eval_executor(
            grid.Cell("eval", "R", "dog_00", tau=0.05,
                      purify=("identity", 0.0), seed=0), {"res": res})


def test_eval_缺段二產物時明確指出缺哪個檔(tmp_path):
    res = make_res(tmp_path)
    executors.control_executor(
        grid.Cell("control", "phi0", "dog_00", purify=("identity", 0.0),
                  seed=0), {"res": res})
    with pytest.raises(FileNotFoundError, match="phi_tau0.05.pt"):
        executors.eval_executor(
            grid.Cell("eval", "R", "dog_00", tau=0.05,
                      purify=("identity", 0.0), seed=0), {"res": res})


def test_eval_的x_def快取不會跨格串味(tmp_path):
    """快取鍵少一個軸，另一個 τ 的格就會拿到上一格的防禦圖而毫無症狀。"""
    res = make_res(tmp_path)
    executors.train_executor(grid.Cell("train", "R", "dog_00"), {"res": res})
    for tau in (0.05, 0.10):
        executors.rayscale_executor(
            grid.Cell("rayscale", "R", "dog_00", tau=tau), {"res": res})
    a = executors._x_def_for(res, "R", "dog_00", 0.05)
    b = executors._x_def_for(res, "R", "dog_00", 0.10)
    c = executors._x_def_for(res, "R", "dog_00", 0.05)
    assert not torch.equal(a, b), "不同 τ 必須得到不同的防禦圖"
    assert torch.equal(a, c)


# ---------------------------------------------------------------------------
# 不可用的淨化算子
# ---------------------------------------------------------------------------


def test_相依不齊的淨化算子標成skipped而非逐格失敗(tmp_path):
    """`CONSECUTIVE_FAILURE_LIMIT` 是 10，同一算子連續 5 個種子，
    兩個不可用的算子相鄰就會誤觸「系統性失敗」而把整段停掉。"""
    res = make_res(tmp_path)
    cells = [
        grid.Cell("eval", "N1", "dog_00", tau=0.20,
                  purify=("cnn_denoise_substitute", 0.0), seed=0),
        grid.Cell("eval", "N1", "dog_00", tau=0.20,
                  purify=("jpeg", 75), seed=0),
    ]
    out = executors.annotate_unavailable(cells, res)
    assert out[0].skipped and "相依不齊" in out[0].skip_reason
    assert not out[1].skipped, "相依齊備的算子不得被標成 skipped"


def test_已有skip_reason的格不被覆寫(tmp_path):
    res = make_res(tmp_path)
    cell = grid.Cell("eval", "N3", "dog_00", tau=0.05,
                     purify=("cnn_denoise_substitute", 0.0), seed=0,
                     skip_reason="低於 VAE 重建下限")
    assert executors.annotate_unavailable([cell], res)[0].skip_reason == \
        "低於 VAE 重建下限"


# ---------------------------------------------------------------------------
# 事前檢查
# ---------------------------------------------------------------------------


def test_事前檢查點名缺MIST圖與PromptFlare的解析度限制(tmp_path):
    """條件明給而非取 `grid.CONDITIONS` 的預設：promptflare 自 2026-08-06
    起因機時裁決不在格點內（`grid.EXCLUDED`），但那個檢查必須留著——
    把它加回 BASELINES 時，解析度限制仍然成立。"""
    res = make_res(tmp_path)
    res.cfg.resolution = 1024
    warns = " ".join(executors.preflight(res, conditions=("mist", "promptflare")))
    assert "MIST.png" in warns
    assert "promptflare" in warns and "512" in warns


def test_訓練前釋放語意權重(tmp_path):
    """CLIP 與 SigLIP 合計 1,352 MB，訓練迴圈一次也不碰，而 N1 的訓練圖
    在 1024² 下差約 600 MB 就 OOM（RTX 3090 實測）。呼叫點被拿掉的症狀
    只有 OOM，且只在顯存較小的卡上出現，故以測試釘住。"""
    res = make_res(tmp_path)
    executors.train_executor(grid.Cell("train", "R", "dog_00"), {"res": res})
    assert getattr(res.suite, "released", 0) >= 1, "段 1 沒有釋放語意權重"


def test_事前檢查提醒兩個門檻仍是舊預算的值(tmp_path):
    res = make_res(tmp_path)
    warns = " ".join(executors.preflight(res, conditions=()))
    assert "tau_acut" in warns and "tau_chroma" in warns


# ---------------------------------------------------------------------------
# 與骨架接起來
# ---------------------------------------------------------------------------


def test_接上run_stage後續跑判定生效(tmp_path):
    """`artifacts` 的存在性判定是靠 executor 回報的路徑，故必須端到端驗一次。"""
    res = make_res(tmp_path)
    w = ProgressWriter(res.batch_dir, env={"gpu": "test"})
    try:
        cells = [grid.Cell("train", "R", i) for i in ("dog_00", "cat_00")]
        first = run_stage("train", cells, executors.make_executor("train"), w,
                          res.base_config, ctx={"res": res}, verbose=False)
        assert first.done == 2 and first.failed == 0
        second = run_stage("train", cells, executors.make_executor("train"), w,
                           res.base_config, ctx={"res": res}, verbose=False)
        assert second.resumed == 2 and second.done == 0

        (res.batch_dir / "R" / "dog_00" / "phi.pt").unlink()
        third = run_stage("train", cells, executors.make_executor("train"), w,
                          res.base_config, ctx={"res": res}, verbose=False)
        assert third.done == 1 and third.resumed == 1, (
            "產物被清掉的那一格必須重跑"
        )
    finally:
        w.release()


def test_報表由逐格紀錄彙整出grid_csv(tmp_path):
    import csv as _csv

    res = make_res(tmp_path)
    w = ProgressWriter(res.batch_dir, env={"gpu": "test"})
    try:
        img = "dog_00"
        run_stage("train", [grid.Cell("train", "R", img)],
                  executors.make_executor("train"), w, res.base_config,
                  ctx={"res": res}, verbose=False)
        run_stage("rayscale", [grid.Cell("rayscale", "R", img, tau=0.05)],
                  executors.make_executor("rayscale"), w, res.base_config,
                  ctx={"res": res}, verbose=False)
        ctrl = [grid.Cell("control", "phi0", img, purify=("identity", 0.0),
                          seed=s) for s in range(grid.MIN_SEEDS)]
        run_stage("control", ctrl, executors.make_executor("control"), w,
                  res.base_config, ctx={"res": res}, verbose=False)
        ev = [grid.Cell("eval", "R", img, tau=0.05, purify=("identity", 0.0),
                        seed=s) for s in range(grid.MIN_SEEDS)]
        run_stage("eval", ev, executors.make_executor("eval"), w,
                  res.base_config, ctx={"res": res}, verbose=False)
    finally:
        w.release()

    out = executors.run_report(res)
    assert out["n_rows"] == grid.MIN_SEEDS
    rows = list(_csv.DictReader(
        (res.batch_dir / "grid.csv").open(encoding="utf-8")))
    assert list(rows[0])[:4] == ["cell_id", "config_hash", "condition",
                                 "image_id"], "欄位順序須依 CODE §4"
    assert rows[0]["subtask"] == "dog"
    assert rows[0]["stop_reason"], "train 的診斷欄位必須併進來"
    assert rows[0]["tau_achieved"], "rayscale 的失真實測值必須併進來"
    assert rows[0]["retention"] not in ("", None)
    assert rows[0]["retention_usable"] in ("True", "False")


def test_報表把不可用的retention標出來(tmp_path):
    """先驗實驗那種 −43、−98 的數字必須在資料層就被標出。"""
    rows = [
        {"condition": "N1", "image_id": "a", "tau": 0.2,
         "purify_kind": "identity", "effect_abs": 0.001},
        {"condition": "N1", "image_id": "a", "tau": 0.2,
         "purify_kind": "identity", "effect_abs": -0.001},
        {"condition": "N1", "image_id": "a", "tau": 0.2,
         "purify_kind": "jpeg", "effect_abs": 0.5},
    ]
    executors._fill_retention(rows)
    assert rows[-1]["retention_usable"] is False, (
        "identity 的效果落在量測噪聲內時，retention 不可用"
    )


# ---------------------------------------------------------------------------
# 小工具
# ---------------------------------------------------------------------------


def test_csv欄位取全部列的聯集(tmp_path):
    """只認第一列的鍵會靜默丟掉後面的欄位，而 CSV 看起來仍然完整。"""
    import csv as _csv

    p = executors.write_csv(tmp_path / "x.csv",
                            [{"a": 1}, {"a": 2, "b": 3}])
    rows = list(_csv.DictReader(p.open(encoding="utf-8")))
    assert rows[1]["b"] == "3"


def test_x0軌跡的取樣固定含頭尾():
    trace = list(range(12))
    got = executors._sample_trace(trace, every=5)
    assert got[0] == 0 and got[-1] == 11
    assert got == [0, 5, 10, 11]


def test_資料集只收prompts_yaml宣告過的內容():
    entries = executors.load_lo_aligned(DATA, 8, torch.device("cpu"))
    assert len(entries) == 24, "六類各 4 張"
    assert all(e.prompts and e.content for e in entries)
    assert "overview" not in {e.image_id for e in entries}, (
        "資料集總覽圖不得被當成一張待防禦影像"
    )


def test_資料集抽樣讓前k張落在k個不同類別上():
    picked = executors.load_lo_aligned(DATA, 8, torch.device("cpu"), n=3,
                                       seed=1)
    assert len({e.group for e in picked}) == 3


def test_資料集的樣本數只有一個入口():
    """N 由 3 擴到 150 只能改一個設定值。"""
    for n in (2, 5, 9):
        assert len(executors.load_lo_aligned(
            DATA, 8, torch.device("cpu"), n=n, seed=1)) == n


def test_未知影像id立刻拋出(tmp_path):
    res = make_res(tmp_path)
    with pytest.raises(KeyError, match="不在本批"):
        res.image("nope_99")


# ---------------------------------------------------------------------------
# 只有在真實 SDXL 上才驗得到的性質（本檔刻意不涵蓋）
# ---------------------------------------------------------------------------
#
# 下列各項在 `FakeSD` 上恆為真或恆無意義，必須在 RTX 5090 + 真實 SDXL 上
# 逐項確認。清單同步於交接報告：
#
# 1. `optimize()` 在 N1／N2／N3 上真的會收斂（`stop_reason` 非空），而不是
#    跑滿 `max_steps`。跑滿的那一格量到的是「走到哪裡」不是能力。
# 2. 段 0 挑出的學習率在 250 步的正式訓練下仍然可行——本檔只驗它從校準表
#    取得，不驗那個值好不好。
# 3. `stop_tol.shared_mass` 的量級。`MONITOR_TOL` 對它刻意留空。
# 4. 位移場在 `--warp-max-disp` 下能否達到 τ=0.35（段 0 的 `warp_reach.csv`）。
# 5. N3 的 VAE 重建下限在 SDXL 上是否仍為 LPIPS 0.1434——`grid.py` 的
#    `GENERATIVE_LPIPS_FLOOR` 是 SD v1.4 的實測值。
# 6. 五篇 baseline 的 `prepare` 在 SDXL 上跑不跑得起來（AdvPaint 的 QKV
#    recorder、PromptFlare 的 token 白名單、Mist 的 VAE 取樣）。
# 7. 各淨化算子的 `available`：IMPRESS、DiffPure、Adverse Cleaner、
#    CNN 去噪替代品在遠端環境是否齊備。
# 8. 記憶體：`materialize` 對 N3 每次都重跑一條 k_inv 步的生成鏈，
#    段 2 的二分搜尋最多 28 次；1024² 下的峰值必須實測。


def test_兩個lr探測都用臨時校準表取設定():
    """段 0 **正在產生**校準表，`res.calib` 此時必為 None，而 `optim_config`
    會向校準表索取 `stop_tol`。兩個探測函式都必須把自己那張臨時表換進去。

    以原始碼檢查而非實跑：`_probe_align_lr` 只有帶 `align_lr_key` 的條件
    （N3／apa）走得到，而建 APA 模組需要真實 UNet，`FakeSD` 刻意不提供
    （見該類別的 docstring）。

    2026-08-06 修正。before：`_probe_align_lr` 寫的是 `optim_config(res, spec)`，
    在 GPU 上以 `CalibrationMismatch: calibration.json 尚未產生` 中止段 0。
    """
    import inspect

    for fn in (executors._probe_lr, executors._probe_align_lr):
        # 排除 docstring：它記著修正前的原貌（`修改論文方法要記 before/after`），
        # 那段文字本身就含要被禁止的呼叫形式。
        src = inspect.getsource(fn).replace(fn.__doc__ or "\0", "")
        assert "optim_config(res, spec)" not in src, (
            f"{fn.__name__} 直接把未校準的 res 交給 optim_config")
        assert "calib=tmp" in src, (
            f"{fn.__name__} 沒有把臨時校準表換進去")


def test_align探測把迴圈中途的發散帶進probe列(tmp_path):
    """`align` 的發散有兩條路，兩條都必須在 `lr_probe.csv` 留下 `diverged`。

    第 0 步發散時 `raise_if_diverged` 拋 `Diverged`，由 except 記下；但迴圈
    中途出現非有限值時 `align` 是以 `break` 記一列 history 後**正常返回**
    ——還原最佳步的 φ 之後 `x_align` 仍可用，不該拋例外。

    2026-08-07 修正。before：正常返回路徑不看 history 的 `diverged`，於是
    第二條路的候選只留下 `final_loss=inf` 與 `finite=False`。而 `write_csv`
    取全部列的鍵聯集，該欄因此整張表都不存在——`optimize.py` 明寫「該事實
    要進 lr_probe.csv 供報告引用」卻沒有做到。v14 段 0 實測暴露：
    `lr.N3_stage1=0.1` 在第 46 步發散，CSV 沒有 `diverged` 欄。
    """
    res = make_res(tmp_path)
    entry = res.image("dog_00")
    x = entry.x01

    module = MagicMock()
    module.enabled = True
    gen = MagicMock()
    gen.prepare.return_value = None
    gen.generate.return_value = x.clone()

    def fake_align(*a, **kw):
        # `align` 中途發散時留下的那一列：φ 已還原故第一個回傳值是有限的
        return x.clone(), [{"step": 0, "align_loss": 0.0, "fid_lpips": 0.1,
                            "fid_psnr_total": 30.0},
                           {"step": 46, "align_loss": float("inf"),
                            "fid_lpips": float("nan"),
                            "fid_psnr_total": float("nan"),
                            "diverged": True}]

    with patch.object(executors, "build_module", return_value=module), \
         patch.object(executors, "DefenseGenerator", return_value=gen), \
         patch("src.defense.optimize.align", fake_align):
        row = executors._probe_align_lr(res, "N3", entry, 0.1, steps=60)

    assert row["diverged"] is True, "中途發散沒有被帶進 probe 列"
    assert row["finite"] is False
    assert row["steps"] == 2, "步數應取實際跑過的 history 長度"


def test_未發散的probe列也明寫diverged欄(tmp_path):
    """`write_csv` 取全部列的鍵**聯集**：省略 `diverged` 時，該欄只在恰好有
    候選發散的批次裡出現。讀 CSV 的人因此無法分辨「沒有候選發散」與
    「這一欄根本沒被寫出來」——b3 段 0 的 `lr_probe.csv` 正是後者的樣子。
    """
    res = make_res(tmp_path)
    entry = res.image("dog_00")
    with patch.object(executors, "optimize",
                      return_value=fake_optim_result(res, entry)), \
         patch.object(executors, "build_module", return_value=MagicMock()):
        row = executors._probe_lr(res, "N2", entry, "lr.N2", 0.01, steps=3)
    assert row["diverged"] is False


def test_shared_tokens進入config_hash(tmp_path):
    """`shared_tokens` 決定 N1 的 L_def 對哪一格 token 施力——改了它就是換一個
    目標函數。不進雜湊的話，改了值之後舊格會被續跑判定視為完成而靜默沿用，
    而報表上看不出任何差別。這正是 A7 點名的缺陷型態。

    2026-08-06 補入。before：本欄只是 `LossConfig` 的預設值，既不在
    `RunConfig` 也不在 `loss_params()`，且沒有 CLI 入口。
    """
    a = make_res(tmp_path / "a")
    b = make_res(tmp_path / "b", shared_tokens=(76,))
    assert a.cfg.loss_params()["shared_tokens"] == [0]
    assert b.cfg.loss_params()["shared_tokens"] == [76]

    cell = grid.Cell("train", "N1", "dog_00")
    base_a = dict(BASE, loss_params=a.cfg.loss_params(),
                  module_params=a.cfg.module_params(),
                  optim_params=a.cfg.optim_params())
    base_b = dict(BASE, loss_params=b.cfg.loss_params(),
                  module_params=b.cfg.module_params(),
                  optim_params=b.cfg.optim_params())
    assert (config_hash(cell_config(cell, base_a))
            != config_hash(cell_config(cell, base_b))), \
        "換了 shared_tokens 卻算出同一個雜湊，舊格會被沿用"


def test_shared_tokens傳得到損失設定(tmp_path):
    res = make_res(tmp_path, shared_tokens=(76,))
    cfg = executors.loss_config(res, executors.condition_spec("N1"))
    assert cfg.shared_tokens == (76,)


def test_外部素材依本批解析度縮放(tmp_path):
    """`load_image_tensor(size=...)` 必須把素材縮到本批解析度。

    MIST.png 是 1440×1440 的固定素材，本批在 1024²。不縮放時錯誤發生在
    `mist.loss_fn` 的 `mse_sum(zx, ctx.z_target)`：latent 邊長 180 對 128，
    訊息裡只有兩個數字，看不出來源是一張沒有縮放的素材。
    """
    from PIL import Image

    src = tmp_path / "big.png"
    Image.new("RGB", (45, 45), (12, 34, 56)).save(src)
    dev = torch.device("cpu")

    raw = executors.load_image_tensor(src, dev)
    assert raw.shape == (1, 3, 45, 45), "不給 size 時不該改動尺寸"

    scaled = executors.load_image_tensor(src, dev, size=SIZE)
    assert scaled.shape == (1, 3, SIZE, SIZE)
    assert float(scaled.min()) >= 0.0 and float(scaled.max()) <= 1.0


def test_mist的target縮到本批解析度(tmp_path):
    """`--mist-target` 與 `--target-image` 是兩個入口，縮放不可只做其中一個。

    2026-08-06 修正。before：`baseline_kwargs` 直接
    `load_image_tensor(path, res.device)`，兩張圖的 mist 格全部以
    `RuntimeError: The size of tensor a (128) must match the size of
    tensor b (180)` 失敗。
    """
    from PIL import Image

    tgt = tmp_path / "MIST.png"
    Image.new("RGB", (45, 45), (0, 0, 0)).save(tgt)
    res = make_res(tmp_path, mist_target=str(tgt))
    kw = executors.baseline_kwargs("mist", res, res.image("dog_00"))
    assert kw["target01"].shape[-1] == res.cfg.resolution, (
        "mist 的 target 沒有縮到本批解析度")


def test_dia用本批的checkpoint設定(tmp_path):
    """DIA 的損失把整條反演留在同一張圖上，1024² 下不 checkpoint 就 OOM。

    數值中性，故沿用 `unet_ckpt` 而不新增旗標；此處釘住它確實傳得到。
    """
    res = make_res(tmp_path)
    entry = res.image("dog_00")
    for name in ("dia_pt", "dia_r"):
        assert executors.baseline_kwargs(name, res, entry)["use_ckpt"] is True

    off = make_res(tmp_path / "off", unet_ckpt=False)
    assert executors.baseline_kwargs(
        "dia_r", off, off.image("dog_00"))["use_ckpt"] is False


def test_dia的context把checkpoint交給eps():
    """旗標要真的到達 `sd._eps`，否則 `baseline_kwargs` 傳了也沒有效果。"""
    import inspect

    from src.baselines import dia

    src = inspect.getsource(dia._step)
    assert "use_ckpt=ctx.use_ckpt" in src, "_step 沒有把 checkpoint 設定交給 _eps"
    assert "use_ckpt" in inspect.signature(dia.prepare).parameters

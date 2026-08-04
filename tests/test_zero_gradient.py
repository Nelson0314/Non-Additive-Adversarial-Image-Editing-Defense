"""釘住「目標函數在起點的梯度是不是零」這一類的缺陷。

φ = 0 是每一次最佳化的起點。若目標函數在該點的梯度精確為零，最佳化永遠
離不開起點，而症狀是「跑完了但什麼都沒動」——與「跑了但沒效果」在外部
分不出來。本專案踩過兩次：

    divergence  LEDGER 5.3／7.4   KL 在最小值處，實測 grad_norm = 0.000e+00
    untargeted  LEDGER 5.10       LPIPS 在 y_def = y_orig 處是最小值

第二個嚴重得多：`untargeted` 是 59 個有記錄的 run 的 100%（5.2）。它至今
能跑起來，靠的是淨化 EOT——`eot_pairs("rotate", ...)` 第 1 步會換成 blur，
兩條分支因此不再逐元素相同（5.11）。

本檔用小張量與真實的 piq.LPIPS 直接驗證，不需要 SD 權重。
"""

import pytest
import torch


def _obj(mode="untargeted", margin=1.0):
    from src.defense.objective import DefenseObjective, LossConfig

    return DefenseObjective(
        LossConfig(margin=margin, defense_mode=mode), torch.device("cpu"))


def _pair(delta=0.0, seed=0):
    """回傳 (y_def 需要梯度, y_orig 常數)。delta = 0 時兩者逐元素相同。"""
    g = torch.Generator().manual_seed(seed)
    y = torch.rand(1, 3, 64, 64, generator=g)
    yd = y if delta == 0.0 else (
        y + delta * torch.randn(y.shape, generator=g)).clamp(0, 1)
    return yd.clone().requires_grad_(True), y.detach()


def _grad(term, x):
    term.backward()
    return float(x.grad.norm())


# ---------------------------------------------------------------------------
# untargeted：起點梯度精確為零（LEDGER 5.10）
# ---------------------------------------------------------------------------


def test_untargeted_在兩分支相同時梯度精確為零():
    yd, yo = _pair(delta=0.0)
    obj = _obj()
    term = obj.defense_term([yd], [yo])
    # hinge 仍是啟動的：margin 1.0 − 距離 0 = 1.0，看起來「有損失可以下降」
    assert float(term) == pytest.approx(1.0, abs=1e-6)
    # 但梯度是零。這兩件事同時成立，正是它難以察覺的原因
    assert _grad(term, yd) == 0.0


def test_untargeted_在兩分支略有差異時梯度非零():
    yd, yo = _pair(delta=0.01)
    obj = _obj()
    assert _grad(obj.defense_term([yd], [yo]), yd) > 0.0


def test_淨化算子的輪替第0步是identity():
    """5.11 的機制：第 0 步取 identity 才會落在零梯度點上。"""
    from src.defense.optimize import eot_pairs
    from src.purify.ops import default_train_set

    ps = default_train_set()
    kinds = [ps[eot_pairs("rotate", s, 1, len(ps))[0][0]].kind for s in range(4)]
    assert kinds == ["identity", "blur", "jpeg", "identity"]


# ---------------------------------------------------------------------------
# 沒有這個缺陷的兩個目標函數（LEDGER 5.13）
# ---------------------------------------------------------------------------


def test_targeted_在起點梯度非零():
    """有目標模式最小化 d(y_def, y_target)，起點不在最小值處。"""
    yd, _ = _pair(delta=0.0)
    g = torch.Generator().manual_seed(1)
    y_target = torch.rand(1, 3, 64, 64, generator=g)
    obj = _obj(mode="targeted")
    assert _grad(obj.defense_term([yd], [yd.detach()], y_target=y_target), yd) > 0.0


def test_encoder_項在目標為零張量時梯度非零():
    """`min ‖E_vae(x_def) − 0‖²`：只要 E_vae(x) ≠ 0 起點就有梯度。"""
    z = torch.rand(1, 4, 8, 8).requires_grad_(True)
    obj = _obj()
    assert _grad(obj.encoder_term(z, torch.zeros_like(z)), z) > 0.0


# ---------------------------------------------------------------------------
# 這個性質必須是「距離在最小值處」而不是「hinge 飽和」
# ---------------------------------------------------------------------------


def test_零梯度的來源是距離的最小值而非hinge飽和():
    """把 margin 設成 0 讓 hinge 完全不啟動，梯度同樣是零。

    這排除掉「是 hinge 在 clamp 處梯度為零」這個競爭解釋：margin = 0 時
    `max(0, 0 − 0)` 落在 clamp 的邊界上，而 margin = 1.0 時 hinge 明明是
    啟動的（損失 1.0），兩種情形梯度都是零。真正的原因是 LPIPS 本身在
    y_def = y_orig 處取到最小值。
    """
    yd, yo = _pair(delta=0.0)
    term = _obj(margin=1.0).defense_term([yd], [yo])
    assert float(term) == pytest.approx(1.0, abs=1e-6)   # hinge 啟動中
    assert _grad(term, yd) == 0.0                         # 梯度仍是零


# ---------------------------------------------------------------------------
# 停止門檻與監看量綁在一起（LEDGER 6.21）
# ---------------------------------------------------------------------------


def test_每個監看量各有自己的停止門檻():
    """`plateau_stop` 比的是絕對改善量，而兩個監看量的動態範圍差 39 倍。"""
    from src.defense.optimize import MONITOR_TOL, resolve_stop_tol

    assert resolve_stop_tol(None, "edit_shift") == MONITOR_TOL["edit_shift"]
    assert resolve_stop_tol(None, "attn_div") == MONITOR_TOL["attn_div"]
    # attn_div 的範圍小一個量級，門檻也必須小一個量級
    assert MONITOR_TOL["edit_shift"] / MONITOR_TOL["attn_div"] == pytest.approx(10.0)


def test_呼叫端明寫的門檻優先():
    from src.defense.optimize import resolve_stop_tol

    assert resolve_stop_tol(3e-4, "attn_div") == 3e-4


def test_未校準的監看量必須拋出而不是沿用別人的門檻():
    """這正是這道缺陷的形狀：為某個量校準的門檻被沿用而沒有症狀。"""
    from src.defense.optimize import resolve_stop_tol

    with pytest.raises(KeyError, match="沒有校準過的 stop_tol"):
        resolve_stop_tol(None, "some_new_metric")


def test_用edit_shift的門檻會讓crossattn在半途停下():
    """以真實的 attn_div 軌跡驗證，不是推論。

    `runs/gate_suppress/horse_00__PF__history.json` 是 60 步的 `suppress`
    最佳化。用 1e-4（edit_shift 的門檻）會在第 30 步判定收斂，砍掉後半段的
    53% 改善；用 1e-5（attn_div 的門檻）不會。
    """
    import json
    from pathlib import Path

    from src.defense.optimize import plateau_stop

    root = Path(__file__).resolve().parents[1]
    h = json.load(
        open(root / "runs/gate_suppress/horse_00__PF__history.json"))
    assert len(h) == 60

    def first_stop(tol):
        for i in range(25, len(h)):
            stop, _ = plateau_stop(h[:i + 1], 20, tol, 25,
                                   require_constraint=False,
                                   monitor_key="attn_div")
            if stop:
                return i
        return None

    assert first_stop(1e-4) == 30          # edit_shift 的門檻：半途就停
    assert first_stop(1e-5) is None        # attn_div 的門檻：跑滿 60 步

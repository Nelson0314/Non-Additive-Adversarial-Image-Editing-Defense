"""釘住「目標函數在起點的梯度是不是零」這一件事。

φ = 0 是每一次最佳化的起點。若目標函數在該點的梯度精確為零，最佳化永遠
離不開起點，而症狀是「跑完了但什麼都沒動」——與「跑了但沒效果」在外部
分不出來。本專案踩過兩次：

    divergence  LEDGER 5.3／7.4   KL 在最小值處，實測 grad_norm = 0.000e+00
    untargeted  LEDGER 5.10       LPIPS 在 y_def = y_orig 處是最小值

第二個嚴重得多：`untargeted` 涵蓋 59 個有紀錄批次的 **100%**（LEDGER 5.2）。

2026-08-05 的改動：`untargeted` 與 `divergence` 兩個模式都已從程式中移除，
三個非加性條件（N1／N2／N3）一律 targeted（`DESIGN` §2.1）。本檔因此分成
兩半：

* 前半驗證**現行**的兩個 targeted 形式在 φ = 0 處梯度非零。
* 後半以手工重建的算式**保存被淘汰的行為**，並釘住「舊模式的名字必須被
  明確拒絕」。這一段不是歷史註腳——它是本專案最大的一次無症狀失效，
  程式若靜默接受舊字串，同一批無效資料會再產生一次。

本檔用小張量與真實的 piq.LPIPS 直接驗證，不需要 SD 權重。
"""

import pytest
import torch

from src.defense.objective import DefenseObjective, LossConfig


def _obj(**kw):
    return DefenseObjective(LossConfig(**kw), torch.device("cpu"))


def _pair(delta=0.0, seed=0):
    """回傳 (y_def 需要梯度, y_orig 常數)。delta = 0 時兩者逐元素相同。

    delta = 0 即「φ = 0 且淨化為 identity」那個起點：防禦分支與原圖分支跑的
    是同一張圖、同一組噪聲，兩條輸出逐元素相同。
    """
    g = torch.Generator().manual_seed(seed)
    y = torch.rand(1, 3, 64, 64, generator=g)
    yd = y if delta == 0.0 else (
        y + delta * torch.randn(y.shape, generator=g)).clamp(0, 1)
    return yd.clone().requires_grad_(True), y.detach()


def _grad(term, x):
    term.backward()
    return float(x.grad.norm())


def _maps(n_layers=2, n_tokens=77, q=16, seed=0, requires_grad=True):
    """假的 cross-attention 分佈：(B, Q, T)，已在 token 維度正規化。

    不需要 SD——本檔要驗的是損失對分佈的梯度，不是 UNet 怎麼算出那個分佈。
    """
    g = torch.Generator().manual_seed(seed)
    out = []
    for i in range(n_layers):
        a = torch.rand(1, q, n_tokens, generator=g)
        a = a / a.sum(dim=-1, keepdim=True)
        out.append(a.clone().requires_grad_(requires_grad))
    return out


# ---------------------------------------------------------------------------
# 現行行為：兩個 targeted 形式在起點梯度非零
# ---------------------------------------------------------------------------


def test_targeted_output_在起點梯度非零():
    """N2／N3 階段二：min d(y_def, y_target)。

    起點是 y_def = y_orig（φ=0、淨化為 identity），而目標 y_target 是另一張
    影像，故起點不在最小值處。這正是 untargeted 沒有的性質。
    """
    yd, yo = _pair(delta=0.0)
    g = torch.Generator().manual_seed(1)
    y_target = torch.rand(1, 3, 64, 64, generator=g)
    obj = _obj(defense_mode="targeted_output")
    assert _grad(obj.defense_term([yd], [yo], y_target), yd) > 0.0


def test_targeted_output_取MSE時起點梯度亦非零():
    """N3 階段二的 R_a 是 ‖·‖²（`DESIGN` §4），不是 LPIPS。兩種度量都要驗。"""
    yd, yo = _pair(delta=0.0, seed=2)
    g = torch.Generator().manual_seed(3)
    y_target = torch.rand(1, 3, 64, 64, generator=g)
    obj = _obj(defense_mode="targeted_output", target_metric="mse")
    assert _grad(obj.defense_term([yd], [yo], y_target), yd) > 0.0


def test_targeted_output_不需要淨化輪替就能起步():
    """這是本次改動的核心：起步不再依賴「抗淨化」這個附帶目標。

    untargeted 之所以能跑起來，是靠訓練期淨化算子輪替到第 1 步換成 blur，
    兩條分支因此不再逐元素相同（LEDGER 5.11）。也就是說主損失的啟動器是
    另一個目標。此處以 identity 算子（即兩條分支逐元素相同）驗證 targeted
    在同一情境下仍有梯度。
    """
    from src.purify.ops import Purifier

    yd, yo = _pair(delta=0.0, seed=4)
    identity = Purifier("identity")
    assert torch.equal(identity.forward(yo), yo), "測試前提：identity 逐元素不變"

    g = torch.Generator().manual_seed(5)
    y_target = torch.rand(1, 3, 64, 64, generator=g)
    obj = _obj(defense_mode="targeted_output")
    assert _grad(obj.defense_term([yd], [yo], y_target), yd) > 0.0


def test_targeted_attn_在起點梯度非零():
    """N1：min (1 − shared token 的注意力質量)。

    φ = 0 時該質量是一個由影像決定的中間值，不是極值，故梯度一般非零。
    這與「與未防禦參照的散度」相反——後者在 φ=0 時兩個分佈逐元素相同，
    散度取到最小值，梯度精確為零。
    """
    maps = _maps(seed=10)
    obj = _obj(defense_mode="targeted_attn")
    term = obj.attention_term([maps])
    assert 0.0 < float(term) < 1.0, "質量必須是中間值，不是 0 或 1"
    term.backward()
    assert all(m.grad is not None for m in maps)
    assert sum(float(m.grad.abs().sum()) for m in maps) > 0.0


def test_targeted_attn_的梯度指向shared_token():
    """方向也要對：梯度必須把質量往 shared token 推，不是隨便有值就好。

    L = 1 − Σ_{τ∈shared} A[·,τ]，故 ∂L/∂A[·,τ] 對 shared 位置為負
    （沿負梯度走會把該格的質量調高），對其餘位置為零。
    """
    maps = _maps(n_layers=1, seed=11)
    obj = _obj(defense_mode="targeted_attn", shared_tokens=(0,))
    obj.attention_term([maps]).backward()
    g = maps[0].grad
    assert float(g[..., 0].max()) < 0.0, "shared token 的梯度必須為負"
    assert float(g[..., 1:].abs().max()) == 0.0, "其餘 token 不該直接收到力"


def test_encoder_項在目標為零張量時梯度非零():
    """`min ‖E_vae(x_def) − 0‖²`：只要 E_vae(x) ≠ 0 起點就有梯度。"""
    z = torch.rand(1, 4, 8, 8).requires_grad_(True)
    assert _grad(_obj().encoder_term(z, torch.zeros_like(z)), z) > 0.0


# ---------------------------------------------------------------------------
# 被淘汰的行為：保存下來，並確認它已無法被選用
# ---------------------------------------------------------------------------


def test_已淘汰的untargeted_在起點梯度精確為零():
    """本專案最大的一次無症狀失效，以手工重建的算式保存。

    `L = max(0, m − LPIPS(y_def, y_orig))`。在 y_def = y_orig 處：

    * hinge **是啟動的**——margin 1.0 減距離 0 等於 1.0，損失看起來有東西
      可以下降；
    * 但梯度精確為零，因為 LPIPS 在該點取到最小值。

    這兩件事同時成立，正是它難以察覺的原因：逐步日誌上 L_def 是一個大於零
    的數，只有 grad_norm 那一欄是 0.000e+00。實測涵蓋 59 個有紀錄批次的
    100%。改為 targeted 之後這個形式不再存在於程式中，本測試以手工算式
    重現它，使該教訓不依賴那段程式碼還在不在。
    """
    yd, yo = _pair(delta=0.0)
    obj = _obj()
    margin = 1.0
    term = torch.clamp(margin - obj.distance(yd, yo), min=0.0)
    assert float(term) == pytest.approx(margin, abs=1e-6), "hinge 啟動中"
    assert _grad(term, yd) == 0.0, "而梯度精確為零"


def test_已淘汰的untargeted_的零梯度來自距離而非hinge飽和():
    """排除「是 hinge 在 clamp 邊界上梯度為零」這個競爭解釋。

    margin = 0 時 `max(0, 0 − 0)` 落在 clamp 的邊界上，梯度為零；margin = 1.0
    時 hinge 明明是啟動的（損失 1.0），梯度同樣是零。真正的原因是 LPIPS 本身
    在 y_def = y_orig 處取到最小值——換一個 margin、換一種距離都不能解決，
    只有換成 targeted 才能。
    """
    obj = _obj()
    for margin in (0.0, 1.0):
        yd, yo = _pair(delta=0.0)
        term = torch.clamp(margin - obj.distance(yd, yo), min=0.0)
        assert _grad(term, yd) == 0.0


def test_已淘汰的untargeted_靠淨化輪替才脫離起點():
    """它至今能跑起來的機制：輪替第 0 步是 identity，第 1 步換成 blur。

    兩條分支因此不再逐元素相同，梯度才不是零。也就是說「抗淨化」這個附帶
    目標同時是主損失的啟動器——一個附帶目標關掉就會讓主目標完全失效的設計。
    """
    from src.defense.optimize import eot_pairs
    from src.purify.ops import default_train_set

    ps = default_train_set()
    kinds = [ps[eot_pairs("rotate", s, 1, len(ps))[0][0]].kind for s in range(4)]
    assert kinds[0] == "identity", "第 0 步落在零梯度點上"
    assert kinds[1] != "identity", "第 1 步換算子才打破對稱"

    # 兩分支略有差異時梯度確實非零——即「換了算子之後才動得了」
    yd, yo = _pair(delta=0.01)
    obj = _obj()
    term = torch.clamp(1.0 - obj.distance(yd, yo), min=0.0)
    assert _grad(term, yd) > 0.0


def test_舊模式名稱必須被明確拒絕():
    """靜默接受 `defense_mode="untargeted"` 會讓同一批無效資料再產生一次。

    訊息必須說明淘汰理由，否則使用者看到例外的第一反應是把它改回去。
    """
    with pytest.raises(ValueError, match="已移除"):
        LossConfig(defense_mode="untargeted")
    with pytest.raises(ValueError, match="grad_norm"):
        LossConfig(defense_mode="untargeted")


def test_未知的defense_mode必須報錯():
    with pytest.raises(ValueError, match="未知的 defense_mode"):
        LossConfig(defense_mode="minimax")


# ---------------------------------------------------------------------------
# 停止門檻與監看量綁在一起（LEDGER 6.21）
# ---------------------------------------------------------------------------


def test_停止門檻沒有任何回退路徑():
    """`plateau_stop` 比的是絕對改善量，而不同監看量的動態範圍差 39 倍。

    2026-08-05 收緊：先前 `edit_shift` 有回退值（1e-4，量在 SD v1.4／512²／
    site PF 上），只有 `shared_mass` 會拋出，即同一道規則對兩個監看量鬆緊
    不一。門檻與學習率是同一種量，故比照 `resolve_lr` 改為完全無回退。
    """
    from src.defense.optimize import resolve_stop_tol

    for key in ("edit_shift", "attn_div", "shared_mass", "some_new_metric"):
        with pytest.raises(KeyError, match="stop_tol"):
            resolve_stop_tol(None, key)


def test_呼叫端明寫的門檻優先():
    from src.defense.optimize import resolve_stop_tol

    assert resolve_stop_tol(3e-4, "attn_div") == 3e-4


def test_先驗門檻只作為量級紀錄留存():
    """兩個舊值仍要查得到——它們是「該用什麼量級」的唯一參考——
    但不得成為任何一條解析路徑的回傳值。"""
    from src.defense import optimize as O

    assert not hasattr(O, "MONITOR_TOL"), (
        "舊名保留會讓呼叫端以為它仍是生效的表"
    )
    assert O.LEGACY_MONITOR_TOL["edit_shift"] / \
        O.LEGACY_MONITOR_TOL["attn_div"] == pytest.approx(10.0)


def test_N1的監看量必須由校準表取得():
    """`shared_mass` 是 targeted 之後的新監看量，`attn_div` 的 1e-5 不適用。

    前者是「導向 shared token 的質量」、後者是「1 − 內容 token 的質量」，
    兩者不是同一個量，動態範圍未實測。沿用會重演 LEDGER 6.21：門檻只比平均
    改善率低 1.6 倍而不是一個量級，實測在半途就判定收斂且回報「已收斂」。
    """
    from src.defense.optimize import DEFENSE_MONITOR, resolve_stop_tol

    assert DEFENSE_MONITOR["targeted_attn"] == "shared_mass"
    with pytest.raises(KeyError, match="stop_tol"):
        resolve_stop_tol(None, "shared_mass")


def test_用edit_shift的門檻會讓注意力路徑在半途停下():
    """以真實的軌跡驗證，不是推論。

    `runs/gate_suppress/horse_00__PF__history.json` 是 60 步的注意力目標
    最佳化。用 1e-4（edit_shift 的門檻）會在第 30 步判定收斂，砍掉後半段的
    53% 改善；用 1e-5（該量自己的門檻）不會。
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
    assert first_stop(1e-5) is None        # 該量自己的門檻：跑滿 60 步


# ---------------------------------------------------------------------------
# N1 的 decoy 位置（2026-08-06，SDXL 實測後補）
# ---------------------------------------------------------------------------

def test_起點質量過低時拋出而非空跑():
    """`1 − mass` 在 mass=0 處是**最大值**，梯度為零，最佳化不會有任何更新。

    這與已移除的 `untargeted` 完全同型（A1），而該缺陷涵蓋先驗實驗 59 個
    有紀錄批次的 100%——因為「損失不動」在 log 上看起來只是「還沒開始降」。
    故必須在第 0 步就拋出，不能靠人盯。

    SDXL 實測：token 0（BOS）的質量 7.2e-06、token 76（末位 PAD）1.59e-02，
    相差三個數量級，門檻 1e-3 落在中間。
    """
    from src.defense.optimize import MIN_SHARED_MASS

    assert 7.2e-06 < MIN_SHARED_MASS < 4.84e-03, (
        "門檻必須把 BOS（7.2e-06）判為不可用、把末位 PAD（4.84e-03，"
        "有內容 prompt 下的值）判為可用")


def test_防護確實接在N1的第0步上():
    """釘住呼叫點：常數存在但沒被用上時，症狀仍然是整段空跑。"""
    import inspect

    from src.defense import optimize as O

    src = inspect.getsource(O._build_attn_step)
    assert "MIN_SHARED_MASS" in src, "N1 的 step_fn 沒有檢查起點質量"
    assert "global_step == 0" in src, "檢查必須只在第 0 步做，不是每步"

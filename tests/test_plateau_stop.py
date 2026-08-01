"""停止準則 `plateau_stop` 的邏輯。

抽成純函數就是為了能在沒有 SD 的情況下驗證這段——它決定每一格跑多久，
是本協議的核心。E21–E23 §5.4 的問題（固定步數讓不同格子被不同的東西綁住）
若在這裡出錯，會再產生一批不可跨 site 比較的資料。
"""

import pytest

from src.defense.optimize import plateau_stop

PAT, TOL, MIN = 20, 1e-4, 25


def _hist(shifts, pen_lpips=0.0, pen_acut=0.0):
    return [{"edit_shift": s, "fid_pen_lpips": pen_lpips, "fid_pen_acut": pen_acut}
            for s in shifts]


def test_步數不足時不停():
    """下限存在的理由：太早停會把「約束還沒啟動」誤判成收斂。"""
    stop, _ = plateau_stop(_hist([0.1] * 10, pen_lpips=0.5), PAT, TOL, MIN)
    assert stop is False


def test_約束未啟動時不停即使進展已停():
    """這是本準則的關鍵條件。進展停了但沒碰到失真預算，代表步數不足或
    學習率太小，不是「該方法在此預算下的能力」已經量到。E21 的 τ=0.10 就是
    這種情形：沒有任何一格碰到預算，末端 6/6 仍在上升。
    """
    flat = _hist([0.1] * 40, pen_lpips=0.0, pen_acut=0.0)
    stop, _ = plateau_stop(flat, PAT, TOL, MIN)
    assert stop is False, "約束沒啟動過就不算收斂"


def test_約束啟動且進展停止時停():
    hist = _hist([0.1] * 40, pen_lpips=0.3)
    stop, reason = plateau_stop(hist, PAT, TOL, MIN)
    assert stop is True
    assert "1.0e-04" in reason and "20" in reason


def test_鈍化懲罰也算約束啟動():
    """LPIPS ∩ 鈍化是交集式雙 hinge，任一道綁住都算被綁住。"""
    hist = _hist([0.1] * 40, pen_lpips=0.0, pen_acut=0.2)
    stop, _ = plateau_stop(hist, PAT, TOL, MIN)
    assert stop is True


def test_仍在上升時不停():
    """E23 的教訓：25 步時 site S 領先，100 步時反轉。末端仍在上升的格子
    量到的是「走到哪裡」，不是能力。

    上升速率取 E23 的實測值 5.4e-4／步（site P 由 25 步的 net 0.0784 走到
    100 步的 0.1186），即這個測試釘的是「真實資料裡那一格必須被判為未收斂」。
    """
    rising = _hist([0.05 + 5.4e-4 * i for i in range(40)], pen_lpips=0.3)
    stop, _ = plateau_stop(rising, PAT, TOL, MIN)
    assert stop is False


def test_改善量遠低於門檻時停():
    """`edit_shift` 在 1e-4 量級來回抖動時必須停。初版用相對改善率，該情形
    下分母趨近零、判定被噪聲主導，40 步跑滿都不會停（tiny-SD 端到端實測）。
    """
    noisy = _hist([1e-4 + (1e-5 if i % 2 else -1e-5) for i in range(40)],
                  pen_lpips=0.3)
    stop, _ = plateau_stop(noisy, PAT, TOL, MIN)
    assert stop is True


def test_下降也視為進展已停():
    """改善量為負代表已越過最佳點，繼續跑只是浪費機時。"""
    falling = _hist([0.2 - 0.001 * i for i in range(40)], pen_lpips=0.3)
    stop, _ = plateau_stop(falling, PAT, TOL, MIN)
    assert stop is True


def test_只看觀察窗而非全部歷史():
    """前段大幅上升、後段持平，應該停。若誤用全部歷史算斜率會漏掉。"""
    hist = _hist([0.01 * i for i in range(30)] + [0.30] * 20, pen_lpips=0.3)
    stop, _ = plateau_stop(hist, PAT, TOL, MIN)
    assert stop is True


def test_可關閉約束條件():
    """診斷用：想單獨看「進展停了沒」時可關掉第一個條件。"""
    flat = _hist([0.1] * 40, pen_lpips=0.0)
    assert plateau_stop(flat, PAT, TOL, MIN)[0] is False
    assert plateau_stop(flat, PAT, TOL, MIN, require_constraint=False)[0] is True


def test_patience小於二必須報錯():
    """切不成前後兩半就無法比較改善量；靜默接受會讓準則變成比較單點。"""
    with pytest.raises(ValueError, match="patience"):
        plateau_stop(_hist([0.1] * 40, pen_lpips=0.3), 1, TOL, MIN)


def test_歷史短於觀察窗時不停():
    """min_steps 小於 patience 時仍不得在窗填滿前判定。"""
    stop, _ = plateau_stop(_hist([0.1] * 5, pen_lpips=0.3), PAT, TOL, min_steps=2)
    assert stop is False


def test_只認係數非零的hinge():
    """這條是端到端測試逼出來的。`fidelity_term` 一律計算四道 hinge 的
    懲罰值，但係數為零的那幾道不進梯度、不構成約束。反過來，實際綁住一格的
    可能是 LPIPS 與鈍化以外的那道：tiny-SD 的端到端測試中 site C 出現
    pen_lpips = pen_acut = 0 而 pen_linf = 0.098（×100 = 9.8，佔 L_fid 全部），
    與 E13 量到的「L∞ hinge 把 site S 完全節流」同一回事。

    若判準只認寫死的兩道，這種格子會永遠判不到「約束已啟動」而跑滿上限。
    """
    from src.defense.optimize import active_constraint_keys
    from src.defense.objective import LossConfig

    hist = [{"edit_shift": 0.1, "fid_pen_lpips": 0.0, "fid_pen_acut": 0.0,
             "fid_pen_linf": 0.098} for _ in range(40)]
    assert plateau_stop(hist, PAT, TOL, MIN)[0] is False, "預設鍵看不到 L∞"
    assert plateau_stop(hist, PAT, TOL, MIN,
                        constraint_keys=("fid_pen_linf",))[0] is True

    # 預設的 LossConfig：LPIPS、鈍化、色度、L∞ 四道係數非零，PSNR 為 0
    keys = active_constraint_keys(LossConfig())
    assert keys == ("fid_pen_lpips", "fid_pen_acut", "fid_pen_chroma",
                    "fid_pen_linf")
    # E15 之後主網格用 beta_linf=0，此時 L∞ 不再是約束
    assert active_constraint_keys(LossConfig(beta_linf=0.0)) == (
        "fid_pen_lpips", "fid_pen_acut", "fid_pen_chroma")


def test_全部係數為零時拒絕():
    """沒有任何約束時「約束啟動並穩定」沒有定義，不可靜默當成無約束收斂。"""
    from src.defense.optimize import active_constraint_keys
    from src.defense.objective import LossConfig

    with pytest.raises(ValueError, match="係數都是零"):
        active_constraint_keys(LossConfig(
            gamma_lpips=0.0, gamma_acut=0.0, gamma_chroma=0.0,
            beta_linf=0.0, gamma_psnr=0.0))

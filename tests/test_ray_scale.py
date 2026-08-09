"""`src/metrics/ray_scale.py` 的契約 —— 段 2 的全部工作都在這裡。

`code_health.py` 指出這個模組先前沒有任何測試**直接** import 它：它只透過
`executors` 被間接帶到，所以「二分到不了目標時會拋出」「容差是多少」這些
性質沒有被任何斷言釘住，而段 2 每一格都依賴它們。

用純函數的假 `lpips_fn` 驗，不需要模型：這一層的邏輯與影像無關。
"""
import pytest

from src.metrics.ray_scale import solve_k


def _linear(scale=1.0):
    """一個單調遞增、k=0 時為 0 的假距離函數。"""
    calls = []

    def make(k):
        calls.append(k)
        return k                      # 「影像」就是 k 本身

    def dist(x):
        return scale * float(x)

    return dist, make, calls


def test_解得到目標且落在容差內():
    dist, make, _ = _linear()
    x, got, k = solve_k(dist, make, 0.20)
    assert got == pytest.approx(0.20, abs=0.005), (
        "段 2 宣告該格的失真是 τ，整張匹配失真的比較表都建立在該宣告上")
    assert k == pytest.approx(0.20, abs=0.01)


def test_雙向可行_目標低於或高於起點都解得到():
    """訓練點不必是最大的 τ：`TRAIN_TAU` 由 0.35 降到 0.20 之後，
    段 2 得同時往上與往下縮放（DEC-001）。"""
    dist, make, _ = _linear()
    for tau in (0.05, 0.10, 0.35, 0.50):
        _, got, _ = solve_k(dist, make, tau)
        assert got == pytest.approx(tau, abs=0.005), f"τ={tau} 解不到"


def test_達不到目標時拋出而不是取最接近的值():
    """**這是本模組最重要的一條。** 靜默取最接近的值會讓該格的失真宣告
    與實際不符，而報表不會顯示任何異常——那正是本專案反覆出現的失敗型態。
    """
    # 上界被夾住：無論 k 多大，距離都不超過 0.1
    def capped(x):
        return min(0.1, float(x))

    def make(k):
        return k

    with pytest.raises(ValueError):
        solve_k(capped, make, 0.35)


def test_起點恆等時不會誤判為已達成():
    """φ=0 的模塊在任何 k 上都回傳同一張圖，距離恆為 0。
    那時目標不可達，必須拋出而不是回報成功。"""
    def zero(x):
        return 0.0

    def make(k):
        return k

    with pytest.raises(ValueError):
        solve_k(zero, make, 0.20)

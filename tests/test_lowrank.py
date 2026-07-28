"""低秩殘差張量與秩排程的正確性測試 — spec §4.1、§4.2、E1-T2。

本檔不依賴 SD 模型，純數學層，本機 CPU 即可執行。
"""

import math

import pytest
import torch

from src.residual.lowrank import SCHEDULES, LowRankResidual, envelope, rank_at


# ---------------------------------------------------------------- envelope


@pytest.mark.parametrize("p", [1, 2, 3, 4, 5])
def test_envelope_端點(p):
    """LRDM 式 (17) 在 d=0 為 1、d=1 為 0。p=3 時係數為 a=-10, b=15, c=-6。"""
    assert envelope(0.0, p) == pytest.approx(1.0)
    assert envelope(1.0, p) == pytest.approx(0.0, abs=1e-12)


def test_envelope_p3_係數展開():
    """明確驗算 p=3：1 - 10 + 15 - 6 = 0。"""
    assert envelope(1.0, 3) == pytest.approx(1 - 10 + 15 - 6, abs=1e-12)


@pytest.mark.parametrize("p", [1, 2, 3, 4, 5])
def test_envelope_區間內單調遞減(p):
    vals = [envelope(i / 50.0, p) for i in range(51)]
    assert all(a >= b - 1e-12 for a, b in zip(vals, vals[1:]))
    assert all(-1e-12 <= v <= 1.0 + 1e-12 for v in vals)


# ---------------------------------------------------------------- rank_at


def test_rank_at_const():
    assert rank_at("const", t=0, t_max=100, max_rank=32, const_rank=8) == 8
    assert rank_at("const", t=100, t_max=100, max_rank=32, const_rank=8) == 8
    # 超過 max_rank 時夾住
    assert rank_at("const", t=0, t_max=100, max_rank=32, const_rank=99) == 32


def test_rank_at_線性排程端點():
    """LI 在乾淨端為 0、高噪端為 R_m；LD 相反。"""
    assert rank_at("LI", t=0, t_max=100, max_rank=32) == 0
    assert rank_at("LI", t=100, t_max=100, max_rank=32) == 32
    assert rank_at("LD", t=0, t_max=100, max_rank=32) == 32
    assert rank_at("LD", t=100, t_max=100, max_rank=32) == 0


def test_rank_at_多項式排程端點():
    """PD = ceil(envelope·R_m)，由 R_m 降至 0；PI 相反。"""
    assert rank_at("PD", t=0, t_max=100, max_rank=32) == 32
    assert rank_at("PD", t=100, t_max=100, max_rank=32) == 0
    assert rank_at("PI", t=0, t_max=100, max_rank=32) == 0
    assert rank_at("PI", t=100, t_max=100, max_rank=32) == 32


def test_rank_at_對照公式():
    """逐點比對 LRDM 式 (15)、(16)、(18)、(19) 的 ceil 結果。"""
    R_m, T, p = 32, 100, 3
    for t in range(0, T + 1, 7):
        d = t / T
        assert rank_at("LI", t, T, R_m) == math.ceil(d * R_m)
        assert rank_at("LD", t, T, R_m) == math.ceil((1 - d) * R_m)
        assert rank_at("PD", t, T, R_m, p=p) == min(
            R_m, math.ceil(max(0.0, envelope(d, p)) * R_m)
        )
        assert rank_at("PI", t, T, R_m, p=p) == min(
            R_m, math.ceil(max(0.0, 1 - envelope(d, p)) * R_m)
        )


@pytest.mark.parametrize("sched", SCHEDULES)
def test_rank_at_永遠落在合法區間(sched):
    for t in range(0, 101):
        r = rank_at(sched, t, 100, max_rank=32, const_rank=8)
        assert 0 <= r <= 32


def test_rank_at_未知排程報錯():
    with pytest.raises(ValueError, match="未知的秩排程"):
        rank_at("bogus", 0, 100, 32)


def test_rank_at_const_缺參數報錯():
    with pytest.raises(ValueError, match="const_rank"):
        rank_at("const", 0, 100, 32)


# ---------------------------------------------------- LowRankResidual：秩


@pytest.mark.parametrize("r", [1, 2, 4, 8])
def test_殘差的秩精確等於設定值(r):
    """E1-T2：對產生的 Δ 做 SVD，數值秩必須等於 r。"""
    torch.manual_seed(20260728)
    mod = LowRankResidual(steps=1, channels=3, height=32, width=32, max_rank=8)
    # V 初始為零，需先填入非零值才能測秩
    with torch.no_grad():
        mod.V.normal_(0, 0.02)

    delta = mod(step=0, rank=r)
    assert delta.shape == (3, 32, 32)
    for c in range(3):
        sv = torch.linalg.svdvals(delta[c].double())
        # 判準必須是「相對」的：einsum 於 float32 計算，捨入誤差正比於
        # 奇異值本身的量級（sv[0] 約 1e-3 時，殘留奇異值約 1e-10）。
        # 用絕對閾值會隨 init_std 改變而誤判。
        assert sv[r - 1] / sv[0] > 1e-6, f"channel {c}: 第 {r} 個奇異值相對過小"
        if r < 32:
            ratio = (sv[r] / sv[0]).item()
            assert ratio < 1e-6, (
                f"channel {c}: 第 {r+1} 個奇異值相對值應為機器誤差量級，實得 {ratio:.2e}"
            )
        assert torch.linalg.matrix_rank(delta[c].double(), rtol=1e-6).item() == r


def test_秩零回傳零張量():
    mod = LowRankResidual(steps=1, channels=4, height=16, width=16, max_rank=4)
    with torch.no_grad():
        mod.V.normal_(0, 1.0)
    delta = mod(step=0, rank=0)
    assert delta.shape == (4, 16, 16)
    assert torch.all(delta == 0)


def test_只使用前r個分量():
    """rank=r 的輸出必須與「把第 r 個之後的分量歸零」的結果相同。"""
    torch.manual_seed(1)
    mod = LowRankResidual(steps=1, channels=2, height=12, width=12, max_rank=6)
    with torch.no_grad():
        mod.V.normal_(0, 0.1)
    partial = mod(step=0, rank=3)

    with torch.no_grad():
        mod.U[:, :, 3:, :] = 0
        mod.V[:, :, 3:, :] = 0
    full = mod(step=0, rank=6)
    assert torch.allclose(partial, full, atol=1e-6)


# ------------------------------------------------ LowRankResidual：初始化


def test_初始化殘差為零():
    """V=0 使初始 Δ=0，即 x_def = x。"""
    torch.manual_seed(0)
    mod = LowRankResidual(steps=4, channels=4, height=64, width=64, max_rank=32)
    for s in range(4):
        assert torch.all(mod(step=s) == 0)


def test_初始化時梯度可流向V():
    """A 高斯／B 零慣例：初始 ∂L/∂V ≠ 0，第一步更新 V。"""
    torch.manual_seed(0)
    mod = LowRankResidual(steps=1, channels=2, height=8, width=8, max_rank=4)
    target = torch.randn(2, 8, 8)
    loss = ((mod(step=0) - target) ** 2).mean()
    loss.backward()

    assert mod.V.grad is not None and mod.V.grad.abs().sum() > 0, "V 應取得非零梯度"
    # U 的梯度在 V=0 時為零，這是預期行為，非錯誤
    assert mod.U.grad is not None
    assert mod.U.grad.abs().sum() == pytest.approx(0.0, abs=1e-12)


def test_一步更新後兩者皆有梯度():
    torch.manual_seed(0)
    mod = LowRankResidual(steps=1, channels=2, height=8, width=8, max_rank=4)
    opt = torch.optim.SGD(mod.parameters(), lr=0.1)
    target = torch.randn(2, 8, 8)

    for _ in range(2):
        opt.zero_grad()
        loss = ((mod(step=0) - target) ** 2).mean()
        loss.backward()
        opt.step()

    assert mod.U.grad.abs().sum() > 0
    assert mod.V.grad.abs().sum() > 0


# ------------------------------------------------ LowRankResidual：其他


def test_參數量對上spec的表():
    """spec §4.3 參數量表：site P (R_m=32) = 98,304；site L (k_inv=10, R_m=32) = 163,840。"""
    site_p = LowRankResidual(steps=1, channels=3, height=512, width=512, max_rank=32)
    assert site_p.num_parameters() == 98_304
    assert site_p.num_parameters(rank=8) == 24_576

    site_l = LowRankResidual(steps=10, channels=4, height=64, width=64, max_rank=32)
    assert site_l.num_parameters() == 163_840
    assert site_l.num_parameters(rank=8) == 40_960


def test_實際參數張量大小與宣告一致():
    mod = LowRankResidual(steps=10, channels=4, height=64, width=64, max_rank=32)
    total = sum(p.numel() for p in mod.parameters())
    assert total == mod.num_parameters()


def test_max_rank_超過較小維度時報錯():
    with pytest.raises(ValueError, match="秩不可能高於"):
        LowRankResidual(steps=1, channels=3, height=8, width=16, max_rank=12)


def test_rank_超出範圍報錯():
    mod = LowRankResidual(steps=1, channels=3, height=16, width=16, max_rank=4)
    with pytest.raises(ValueError, match="超出"):
        mod(step=0, rank=5)

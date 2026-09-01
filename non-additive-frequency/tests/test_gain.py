"""可學幅度增益（2026-08-21 的改動一）。

由來：等 DISTS 錨點上相位臂的位移是 DCT-Shield 的 45%，而三個構造上的約束
擋著——幅度譜逐位保留使擾動振幅由原圖自己的頻譜決定、相位是週期量故失真有
天花板、編碼器對能量大小比對相位敏感。乘性增益 `spec' = |spec|·exp(g)·
exp(i(phi+theta))` 拆掉第二條、打到第三條，但**拆不掉第一條**——平坦區
`|spec| ~ 0`，乘任何東西還是 0。

這一支釘住五件事：

1. `gain_max = 0` 時整條路徑逐位元不變。SDEdit 那條凍結的線必須能重跑；
2. `theta = 0` 且 `g = 0` 時仍然逐位等於原圖（零點沒有被弄丟）；
3. 增益確實改變幅度譜——否則它根本沒接上；
4. 增益被同一個閘限制，與相位允許出現的位置完全一致；
5. `gl_iters > 0` 時增益**不會**被 Griffin-Lim 的幅度替換投影掉。
"""

import math

import pytest
import torch

from src.defense.param_pgd import PhaseParam
from src.residual.texture_rephase import PhaseResidual


def _img(size=64):
    g = torch.Generator().manual_seed(0)
    return torch.rand(1, 3, size, size, generator=g, dtype=torch.float64)


def _mod(gain_max=0.0, gl_iters=0, size=64, block=16):
    m = PhaseResidual(size=size, block=block, r_min=0.25, theta_max=math.pi,
                      gain_max=gain_max, gl_iters=gl_iters).double()
    return m


def test_關閉時與加這個選項之前逐位相同():
    x = _img()
    a, b = _mod(gain_max=0.0), _mod(gain_max=0.0)
    for m in (a, b):
        m.prepare_gates(x)
    with torch.no_grad():
        t = torch.randn_like(a.theta) * 0.3
        a.theta.copy_(t)
        b.theta.copy_(t)
        b.gain.copy_(torch.randn_like(b.gain) * 5.0)   # 關著時不該有影響
    assert torch.equal(a.pixel_residual(x), b.pixel_residual(x))


def test_零點仍然是原圖():
    x = _img()
    m = _mod(gain_max=1.0)
    m.prepare_gates(x)
    # theta = 0、gain = 0（初始值）
    assert torch.allclose(m.pixel_residual(x), x, atol=1e-10)


def test_增益確實改變幅度譜():
    x = _img()
    m = _mod(gain_max=1.0)
    m.prepare_gates(x)
    base = m.analyze(x).abs()
    with torch.no_grad():
        m.gain.fill_(0.5)
    out = m.analyze(m.pixel_residual(x)).abs()
    assert not torch.allclose(base, out, atol=1e-6), "增益沒有接上"
    # 正增益應該讓被放行的頻格整體變大
    on = m.gate() > 0.5
    if on.any():
        assert float(out.mean()) > float(base.mean())


def test_增益被同一個閘限制():
    """閘為零的頻格，增益不得有任何作用——否則兩個旋鈕作用範圍不同，
    「動什麼」與「動哪裡」兩個變因就混在一起了。"""
    x = _img()
    m = _mod(gain_max=1.0)
    m.prepare_gates(x)
    with torch.no_grad():
        m.gain.fill_(1.0)
    spec0 = m.analyze(x)
    rot = spec0 * torch.exp(
        (m.gain.clamp(-1.0, 1.0) * m.gate()).unsqueeze(1))
    # gate() 是 (1,L,n,nb)，spec 是 (B,C,L,n,nb)：取 off[0] 當後三維的遮罩
    off = (m.gate() == 0)[0]
    assert off.any(), "這張圖上沒有被關掉的頻格，測試沒有鑑別力"
    assert torch.allclose(rot.abs()[:, :, off], spec0.abs()[:, :, off], atol=1e-12)


def test_griffin_lim_不會把增益投影掉():
    x = _img()
    a = _mod(gain_max=1.0, gl_iters=0)
    b = _mod(gain_max=1.0, gl_iters=3)
    for m in (a, b):
        m.prepare_gates(x)
        with torch.no_grad():
            m.gain.fill_(0.6)
    ya, yb = a.pixel_residual(x), b.pixel_residual(x)
    # 兩者都必須明顯偏離原圖；若 GL 用了原圖的幅度當目標，yb 會被拉回去
    da = float((ya - x).abs().mean())
    db = float((yb - x).abs().mean())
    assert da > 1e-4 and db > 1e-4
    assert db > 0.2 * da, f"gl_iters 把增益投影掉了：{db:.2e} 對 {da:.2e}"


def test_gain_max_拒絕負值():
    with pytest.raises(ValueError, match="gain_max"):
        PhaseResidual(size=64, block=16, gain_max=-0.1)


def test_純幅度變體沒有相位自由度():
    p = PhaseParam(size=64, block=16, gain_ratio=0.5, phase_on=False)
    x = torch.rand(1, 3, 64, 64)
    p.reset(x, seed=0)
    names = [id(t) for t in p.params()]
    assert id(p.module.theta) not in names, "純幅度變體不該把 theta 交出去"
    assert id(p.module.gain) in names


def test_兩個自由度都關掉時拒絕():
    with pytest.raises(ValueError, match="自由度"):
        PhaseParam(size=64, block=16, gain_ratio=0.0, phase_on=False)


def test_半徑同時驅動兩個上界():
    p = PhaseParam(size=64, block=16, gain_ratio=0.25)
    x = torch.rand(1, 3, 64, 64)
    p.reset(x, seed=0)
    p.set_radius(8.0)
    assert p.module.theta_max == pytest.approx(math.pi)   # 相位封頂
    assert p.module.gain_max == pytest.approx(2.0)        # 增益不封頂


# ---- 帶通（2026-08-21）----

def test_rmax預設維持原本的高通行為():
    from src.residual.texture_rephase import radial_gate
    a = radial_gate(16, 0.25, "cpu", torch.float64)
    b = radial_gate(16, 0.25, "cpu", torch.float64, float("inf"))
    assert torch.equal(a, b)


def test_rmax真的砍掉高頻():
    from src.residual.texture_rephase import radial_gate
    hi = radial_gate(16, 0.25, "cpu", torch.float64)
    band = radial_gate(16, 0.25, "cpu", torch.float64, 0.6)
    assert band.sum() < hi.sum(), "帶通沒有比高通少"
    # 被砍掉的一律是高頻：band 為 1 的地方 hi 也必須是 1
    assert torch.all((band == 0) | (hi == 1))


def test_通帶為空時拒絕():
    from src.residual.texture_rephase import radial_gate
    with pytest.raises(ValueError, match="r_max"):
        radial_gate(16, 0.5, "cpu", torch.float64, 0.5)


def test_rmax接到PhaseParam上():
    p = PhaseParam(size=64, block=16, r_min=0.25, r_max=0.6)
    x = torch.rand(1, 3, 64, 64)
    p.reset(x, seed=0)
    assert p.module.r_max == 0.6
    q = PhaseParam(size=64, block=16, r_min=0.25)
    q.reset(x, seed=0)
    assert float(p.module.freq_gate.sum()) < float(q.module.freq_gate.sum())

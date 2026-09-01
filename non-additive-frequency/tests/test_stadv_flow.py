"""stAdv（arXiv:1801.02612）三個零件接到位移場臂上（`src/defense/stadv_flow.py`）。

釘的是五件會**靜默**錯掉的事：

1. `L_flow` 的根號在鄰居和的**裡面**。寫成根號在外不會拋錯，只會變成另一個
   正則項——本檔用一個兩者數值不同的場（14 vs 12）把它分開。
2. `--warp-grid` 等於影像邊長時上採樣退化成**恆等**，粗網格係數本身就是原文
   的稠密場。退化不成立的話，「稠密」這兩個字就是假的，而報表上看不出來。
3. `f ≡ 0` 時 `L_flow` 的梯度**有限**。原文的式子在那裡是 0/0，autograd 給
   NaN，而位移場的預設起點正是全零。
4. L-BFGS 那條路徑會因為收斂而提早停，且 `stopped_at`／`stop_reason` 記對。
5. 不加旗標時行為與加這一組東西之前相同（三個旗標預設關閉、關閉時損失函數
   物件原樣傳出）。

全部在 CPU 上執行。
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from src.defense.param_pgd import (  # noqa: E402
    AdditiveParam, WarpParam, run_param_pgd,
)
from src.defense.stadv_flow import (  # noqa: E402
    NEIGHBOURHOODS, flow_pair_count, flow_tv_loss,
)

import ip2p_run  # noqa: E402


TINY = 1e-12          # sqrt(TINY) = 1e-6，四項也只有 4e-6，比 1e-4 的容差小


def _field(du, dv):
    """(1, 2, H, W)：通道 0 = Δu、通道 1 = Δv。"""
    return torch.stack([torch.tensor(du, dtype=torch.float64),
                        torch.tensor(dv, dtype=torch.float64)]).unsqueeze(0)


# ─────────────────────────────── 一、根號的位置 ───────────────────────────────

def test_根號在鄰居和的裡面而不是外面():
    """原文：`Σ_p Σ_q sqrt(‖Δu^p−Δu^q‖² + ‖Δv^p−Δv^q‖²)`。

    這個場刻意讓像素 (0,0) 的右鄰與下鄰**同時**有位移差：根號在內時兩項分別
    是 3 與 4（和 7），根號在外時併成 sqrt(9+16) = 5。整張的總和因此是
    14（在內）對 12（在外）。兩者都是「平滑正則」，寫錯不會有任何症狀。
    """
    f = _field(du=[[0.0, 3.0], [4.0, 0.0]], dv=[[0.0, 0.0], [0.0, 0.0]])
    inside = float(flow_tv_loss(f, eps=TINY, neighbourhood="right_down"))

    # 根號在外的那一版（**這不是原文的式子**，只在這裡造出來當對照）：
    # 逐 p 先把它所有鄰居的差平方加起來，再開一次根號。
    right = (f[..., :, :-1] - f[..., :, 1:]).pow(2).sum(dim=1)   # (1, H, W-1)
    down = (f[..., :-1, :] - f[..., 1:, :]).pow(2).sum(dim=1)    # (1, H-1, W)
    acc = torch.zeros(1, 2, 2, dtype=torch.float64)
    acc[..., :, :-1] += right
    acc[..., :-1, :] += down
    outside = float(torch.sqrt(acc + TINY).sum())

    assert inside == pytest.approx(14.0, abs=1e-4)
    assert outside == pytest.approx(12.0, abs=1e-4)
    assert abs(inside - outside) > 1.0


def test_兩個通道的平方和在同一個根號底下():
    """`sqrt(‖Δu 差‖² + ‖Δv 差‖²)`：3–4–5 直角三角形，一項就是 5。"""
    f = _field(du=[[0.0, 3.0]], dv=[[0.0, 4.0]])
    assert float(flow_tv_loss(f, eps=TINY, neighbourhood="right_down")) == \
        pytest.approx(5.0, abs=1e-4)


# ─────────────────────────── 二、鄰域是我方指定的旋鈕 ───────────────────────────

def test_四鄰域恰為右下兩鄰的兩倍():
    """同一對相鄰像素在四鄰域裡被算兩次（p→q 與 q→p），故恰為兩倍。

    原文沒有寫 N(p) 是哪一種，這個倍率就是「選錯不會拋錯、只會差一個 τ」的
    具體大小。
    """
    g = torch.Generator().manual_seed(0)
    f = torch.randn(1, 2, 6, 5, generator=g, dtype=torch.float64)
    a = float(flow_tv_loss(f, eps=1e-6, neighbourhood="right_down"))
    b = float(flow_tv_loss(f, eps=1e-6, neighbourhood="four"))
    assert b == pytest.approx(2.0 * a, rel=1e-9)


def test_八鄰域比四鄰域多算對角線():
    g = torch.Generator().manual_seed(1)
    f = torch.randn(1, 2, 6, 5, generator=g, dtype=torch.float64)
    four = float(flow_tv_loss(f, eps=1e-6, neighbourhood="four"))
    eight = float(flow_tv_loss(f, eps=1e-6, neighbourhood="eight"))
    assert eight > four


def test_項數與邊界不做_padding():
    """全零場的 `L_flow` 恰為「鄰居對數 × sqrt(eps)」——影像外的鄰居不計入。

    做 padding 的話邊界會多出一圈項，那些項在加了 eps 之後仍然貢獻
    sqrt(eps)，等於把邊界長度混進正則項的常數裡。
    """
    eps = 4e-4                       # sqrt = 0.02，讀數上分得開
    for name in NEIGHBOURHOODS:
        f = torch.zeros(1, 2, 7, 5, dtype=torch.float64)
        n = flow_pair_count(7, 5, name)
        assert float(flow_tv_loss(f, eps=eps, neighbourhood=name)) == \
            pytest.approx(n * math.sqrt(eps), rel=1e-9)
    assert flow_pair_count(7, 5, "right_down") == 7 * 4 + 6 * 5
    assert flow_pair_count(7, 5, "four") == 2 * (7 * 4 + 6 * 5)


def test_兩個參數都是必填的關鍵字():
    """原文的式子沒有 eps、也沒有寫 N(p) 是哪一種，故不可以有預設值。"""
    f = torch.zeros(1, 2, 4, 4)
    with pytest.raises(TypeError):
        flow_tv_loss(f)                                   # 兩個都沒給
    with pytest.raises(TypeError):
        flow_tv_loss(f, eps=1e-6)                         # 缺鄰域
    with pytest.raises(TypeError):
        flow_tv_loss(f, neighbourhood="four")             # 缺 eps


def test_不合法的輸入直接拋錯():
    with pytest.raises(ValueError, match="鄰域"):
        flow_tv_loss(torch.zeros(1, 2, 4, 4), eps=1e-6, neighbourhood="six")
    with pytest.raises(ValueError, match="eps 必須為正"):
        flow_tv_loss(torch.zeros(1, 2, 4, 4), eps=0.0, neighbourhood="four")
    with pytest.raises(ValueError, match="位移場"):
        flow_tv_loss(torch.zeros(1, 3, 4, 4), eps=1e-6, neighbourhood="four")


# ─────────────────────────── 三、f ≡ 0 處的梯度 ───────────────────────────

def test_原文的式子在全零場上真的會給_NaN():
    """這是 eps 存在的**根本原因**，不是預防性的保險。

    `sqrt` 在 0 的導數是 +∞，鏈式規則再乘上「平方和對 f 的導數為 0」，
    得到 inf × 0 = NaN。加 eps 是與原文的差異（modified_from_paper）。
    """
    f = torch.zeros(1, 2, 3, 3, requires_grad=True)
    d = f[..., :, :-1] - f[..., :, 1:]
    torch.sqrt(d.pow(2).sum(dim=1)).sum().backward()      # 沒有 eps 的原式
    assert torch.isnan(f.grad).any()


def test_加了_eps_之後全零場的梯度有限():
    f = torch.zeros(1, 2, 3, 3, requires_grad=True)
    flow_tv_loss(f, eps=1e-8, neighbourhood="four").backward()
    assert torch.isfinite(f.grad).all()
    # 全零是這一項的最小值，梯度恰為零。
    assert float(f.grad.abs().max()) == 0.0


def test_非零場的梯度回得到粗網格係數():
    x = torch.rand(1, 3, 32, 32, generator=torch.Generator().manual_seed(0))
    p = WarpParam(radius=4.0, grid=8)
    p.reset(x, 0)
    with torch.no_grad():
        p.c.normal_(generator=torch.Generator().manual_seed(3))
    flow_tv_loss(p.flow_field(x), eps=1e-8, neighbourhood="four").backward()
    assert torch.isfinite(p.c.grad).all()
    assert float(p.c.grad.abs().sum()) > 0


# ─────────────────── 四、稠密場：warp-grid 等於邊長時是恆等 ───────────────────

@pytest.mark.parametrize("size", [64, 512])
def test_粗網格邊長等於影像邊長時上採樣是恆等(size):
    """退化不成立的話，「逐像素稠密場」這句話就是假的，而報表上看不出來。"""
    x = torch.zeros(1, 3, size, size)
    p = WarpParam(radius=4.0, grid=size)
    p.reset(x, 0)
    with torch.no_grad():
        p.c.normal_(generator=torch.Generator().manual_seed(1))
    field = p.flow_field(x)
    assert field.shape == (1, 2, size, size)
    assert torch.equal(field, p.c)              # 逐位元，不是近似


def test_稠密場的參數量是兩倍像素數():
    """原文的自由度是 `2·H·W`。"""
    x = torch.zeros(1, 3, 64, 64)
    p = WarpParam(radius=4.0, grid=64)
    p.reset(x, 0)
    assert p.params()[0].numel() == 2 * 64 * 64


def test_粗網格較小時正則項算的是上採樣後的場():
    x = torch.zeros(1, 3, 64, 64)
    p = WarpParam(radius=4.0, grid=16)
    p.reset(x, 0)
    assert p.flow_field(x).shape == (1, 2, 64, 64)


# ─────────────────────────── 五、L-BFGS 那條路徑 ───────────────────────────

def _toy():
    return torch.rand(1, 3, 16, 16, generator=torch.Generator().manual_seed(0))


def test_lbfgs_是合法的更新規則且會降低損失():
    x = _toy()
    f = lambda y: y.pow(2).mean()
    r = run_param_pgd(x, AdditiveParam(radius=0.05), f, steps=5,
                      update="lbfgs", step_size=0.01, log_every=1)
    assert r.history[0]["loss"] > r.history[-1]["loss"]


def test_未知的更新規則仍然拋錯():
    with pytest.raises(ValueError, match="sign／adam／lbfgs"):
        run_param_pgd(_toy(), AdditiveParam(radius=0.05),
                      lambda y: y.pow(2).mean(), steps=1, update="newton")


def test_lbfgs_會因收斂而提早停並記下停在第幾步():
    """`--steps` 只是上限：實際停止由 `eval_fn` ＋ `patience` 決定。

    原文沒有載明 L-BFGS 的迭代次數，所以步數不可以寫死。停止的位置與原因
    必須落在既有的 `stopped_at`／`stop_reason` 兩欄裡，否則報表上分不出
    「跑滿上限」與「收斂了」。
    """
    x = _toy()
    f = lambda y: y.pow(2).mean()
    r = run_param_pgd(x, AdditiveParam(radius=0.05), f, steps=200,
                      update="lbfgs", step_size=0.01,
                      eval_fn=f, eval_every=1, patience=3, min_delta=0.01)
    assert r.stop_reason == "early_stop"
    assert 0 < r.stopped_at < 200
    assert r.best_eval is not None
    # 停在哪一步就只有幾筆紀錄，不是跑滿之後才回頭截斷。
    assert max(h["step"] for h in r.history) == r.stopped_at - 1


def test_lbfgs_沒觸發早停時兩欄是跑滿的值():
    x = _toy()
    r = run_param_pgd(x, AdditiveParam(radius=0.05),
                      lambda y: y.pow(2).mean(), steps=4, update="lbfgs",
                      step_size=0.01)
    assert (r.stop_reason, r.stopped_at) == ("max_steps", 4)


def test_lbfgs_逐步記下_line_search_的評估次數():
    """L-BFGS 的一步不等價於一次前向後向，不記下來就無法與 sign 並排比代價。"""
    x = _toy()
    f = lambda y: y.pow(2).mean()
    r = run_param_pgd(x, AdditiveParam(radius=0.05), f, steps=3,
                      update="lbfgs", step_size=0.01, log_every=1)
    assert all(h["closures"] >= 1 for h in r.history)


def test_lbfgs_之後參數仍然落在可行集裡():
    """line search 會走出半徑，每個 outer iteration 之後投影一次收回來。"""
    x = _toy()
    p = AdditiveParam(radius=0.01)
    r = run_param_pgd(x, p, lambda y: (y - 1.0).pow(2).mean(), steps=6,
                      update="lbfgs", step_size=1.0)
    assert float(p.delta.detach().abs().max()) <= 0.01 + 1e-9
    assert r.stopped_at == 6


def test_sign_路徑不記_closures():
    """新欄位只出現在 L-BFGS 的列上，sign 的列逐位元與加它之前相同。"""
    x = _toy()
    r = run_param_pgd(x, AdditiveParam(radius=0.05),
                      lambda y: y.pow(2).mean(), steps=3, log_every=1)
    assert all("closures" not in h for h in r.history)


def test_sign_路徑的更新式沒有被_lbfgs_那條路改到():
    """手算一條 sign-PGD 軌跡逐位元對照。

    為了讓 L-BFGS 的 closure 進得來，迴圈的內層被拆成了兩條路徑；拆錯的話
    sign 那一條會安靜地換一個更新式，而所有既有結果都會平移。
    """
    x = _toy()
    f = lambda y: (y - 0.3).pow(2).mean()
    p = AdditiveParam(radius=1.0)
    run_param_pgd(x, p, f, steps=3, step_size=0.01)

    d = torch.zeros_like(x, requires_grad=True)
    for _ in range(3):
        g, = torch.autograd.grad(f((x + d).clamp(0.0, 1.0)), d)
        with torch.no_grad():
            d.sub_(0.01 * torch.sign(g))
            d.clamp_(-1.0, 1.0)
    assert torch.equal(p.delta.detach(), d.detach())


def test_位移場配_lbfgs_與流場正則能一起跑到收斂():
    """三個零件接在一起：稠密場、L-BFGS、`L_adv + τ·L_flow`。

    起點是全零場（`WarpParam` 的預設），也就是 `L_flow` 的 0/0 那一點；
    整條跑得完就代表 eps 真的把那個 NaN 擋掉了。
    """
    x = torch.rand(1, 3, 32, 32, generator=torch.Generator().manual_seed(2))
    target = torch.roll(x, shifts=2, dims=-1)
    p = WarpParam(radius=6.0, grid=32)

    def loss(y):
        return ((y - target).pow(2).mean()
                + 1e-4 * flow_tv_loss(p.flow_field(x), eps=1e-8,
                                      neighbourhood="right_down"))

    r = run_param_pgd(x, p, loss, steps=60, update="lbfgs", step_size=0.5,
                      eval_fn=loss, eval_every=1, patience=5, min_delta=1e-4)
    assert torch.isfinite(p.c.detach()).all()
    assert float(p.c.detach().abs().max()) > 0.0        # 真的走了
    assert r.history[0]["loss"] > r.history[-1]["loss"]
    assert r.stop_reason in ("early_stop", "max_steps")
    assert r.best_eval is not None


# ─────────────────────── 六、預設關閉：行為逐位元不變 ───────────────────────

def test_三個旗標預設關閉():
    a = ip2p_run.build_parser().parse_args(["--out", "o"])
    assert a.flow_tau == 0.0
    assert a.flow_eps is None
    assert a.flow_neighbourhood is None
    assert a.update == "sign"


def test_關閉時損失函數原樣傳出():
    """`is` 而不是等值：換了物件就代表呼叫路徑不再逐位元相同。"""
    a = ip2p_run.build_parser().parse_args(["--out", "o"])
    base = lambda y: y.pow(2).mean()
    x = torch.zeros(1, 3, 8, 8)
    assert ip2p_run._with_flow_tv(WarpParam(radius=1.0), x, a, base) is base


def test_打開時損失是主損失加上_tau_乘_L_flow():
    a = ip2p_run.build_parser().parse_args(
        ["--out", "o", "--conditions", "warp", "--flow-tau", "0.05",
         "--flow-eps", "1e-8", "--flow-neighbourhood", "four"])
    x = torch.rand(1, 3, 16, 16, generator=torch.Generator().manual_seed(4))
    p = WarpParam(radius=4.0, grid=16)
    p.reset(x, 0)
    with torch.no_grad():
        p.c.normal_(generator=torch.Generator().manual_seed(5))
    base = lambda y: y.pow(2).mean()
    base._fixed_eval = base
    combined = ip2p_run._with_flow_tv(p, x, a, base)
    expect = float(base(p.render(x)).detach()) + 0.05 * float(flow_tv_loss(
        p.flow_field(x), eps=1e-8, neighbourhood="four").detach())
    assert float(combined(p.render(x)).detach()) == pytest.approx(
        expect, rel=1e-6)
    # **評估函數也要包**：不包的話訓練最小化 A、收斂判定看 B。
    assert float(combined._fixed_eval(p.render(x)).detach()) == pytest.approx(
        expect, rel=1e-6)


def test_接到不是位移場的參數化上直接拒絕():
    a = ip2p_run.build_parser().parse_args(
        ["--out", "o", "--conditions", "add", "--flow-tau", "0.05",
         "--flow-eps", "1e-8", "--flow-neighbourhood", "four"])
    with pytest.raises(SystemExit, match="位移場"):
        ip2p_run._with_flow_tv(AdditiveParam(radius=0.05),
                               torch.zeros(1, 3, 8, 8), a,
                               lambda y: y.pow(2).mean())


# ─────────────────────────── 七、守門與 CSV 欄位 ───────────────────────────

def _args(*extra):
    return ip2p_run.build_parser().parse_args(
        ["--out", "o", "--conditions", "warp", *extra])


def test_開了_tau_就必須明給_eps():
    with pytest.raises(SystemExit, match="--flow-eps"):
        ip2p_run.validate_flow_args(
            _args("--flow-tau", "0.05", "--flow-neighbourhood", "four"))


def test_開了_tau_就必須明給鄰域():
    with pytest.raises(SystemExit, match="--flow-neighbourhood"):
        ip2p_run.validate_flow_args(
            _args("--flow-tau", "0.05", "--flow-eps", "1e-8"))


def test_給了設定卻沒開_tau_也拒絕():
    with pytest.raises(SystemExit, match="沒有被用到"):
        ip2p_run.validate_flow_args(_args("--flow-eps", "1e-8"))


def test_只接在_warp_上():
    with pytest.raises(SystemExit, match="只接在 warp"):
        ip2p_run.validate_flow_args(ip2p_run.build_parser().parse_args(
            ["--out", "o", "--conditions", "warp_rand", "--flow-tau", "0.05",
             "--flow-eps", "1e-8", "--flow-neighbourhood", "four"]))


def test_預設不觸發任何守門():
    ip2p_run.validate_flow_args(ip2p_run.build_parser().parse_args(
        ["--out", "o"]))
    ip2p_run.validate_convergence_args(ip2p_run.build_parser().parse_args(
        ["--out", "o"]))


def test_lbfgs_沒配收斂判定就拒絕():
    """步數不可以寫死，故 `--update lbfgs` 必須配 --eval-every 與 --patience。"""
    with pytest.raises(SystemExit, match="不寫死步數"):
        ip2p_run.validate_convergence_args(_args("--update", "lbfgs"))
    with pytest.raises(SystemExit, match="不寫死步數"):
        ip2p_run.validate_convergence_args(
            _args("--update", "lbfgs", "--eval-every", "10"))
    ip2p_run.validate_convergence_args(
        _args("--update", "lbfgs", "--eval-every", "10", "--patience", "5"))


def test_三個設定都是_CSV_的欄位():
    """原文沒有的 eps、原文未載明而由我方指定的鄰域，都必須是欄位不是註解。"""
    src = (ROOT / "scripts" / "ip2p_run.py").read_text(encoding="utf-8")
    for col in ("flow_tau", "flow_eps", "flow_neighbourhood", "flow_loss"):
        assert f'"{col}"' in src, col

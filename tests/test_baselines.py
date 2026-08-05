"""baseline 攻擊層：規格必須逐項等於原始碼查證的結果。

本檔的定位與其他測試不同。這裡驗的**不是我們的設計是否自洽**，而是
**我們有沒有忠實重現別人的方法**。任何一個值被「合理化」成一個看起來
比較整齊的數字，整批比較就失效，而且事後補不回來。

出處一律是 `docs/SOURCE_AUDIT_2026-08-05.md`（含逐字原始碼佐證），
不是各篇論文的正文——七篇裡沒有一篇的正文與其原始碼完全一致。

全部測試在 CPU 上跑，不需要 SD 權重。
"""

import dataclasses

import pytest
import torch

from src.baselines import REGISTRY
from src.baselines.pgd import (BaselineSpec, ValueRange, initial_point,
                               project, step_size_at, update_direction)

# ---------------------------------------------------------------------------
# 1. 規格對照表 —— 直接抄自 SOURCE_AUDIT §10，改動這裡等於宣稱查證結果有誤
# ---------------------------------------------------------------------------

E = 1.0 / 255.0

#           name              eps            eps_pixel01     norm    steps  update
AUDIT = {
    "photoguard_c":  (16.0,          8.0,           "l2",   200, "normalized_grad"),
    "mist":          (32 * E,        16 * E,        "linf", 100, "sign"),
    "dia_pt":        (0.05,          0.025,         "linf",  20, "sign"),
    "dia_r":         (0.05,          0.025,         "linf",  20, "sign"),
    "advpaint":      (0.06,          0.03,          "linf", 100, "sign"),
    "promptflare":   (12 * E,        6 * E,         "linf", 400, "sign"),
}


@pytest.mark.parametrize("name", sorted(AUDIT))
def test_規格逐項等於原始碼查證(name):
    eps, eps01, norm, steps, update = AUDIT[name]
    s = REGISTRY[name]
    assert s.eps == pytest.approx(eps), f"{name} 的 eps 與查證不符"
    assert s.eps_pixel01 == pytest.approx(eps01), f"{name} 的像素域 eps 與查證不符"
    assert s.norm == norm
    assert s.steps == steps
    assert s.update_rule == update


def test_五篇的值域全部是負一到一():
    """三篇 baseline 在 [-1,1] 上最佳化而論文報的是像素域數字，兩者差兩倍，
    且沒有任何症狀——輸出仍是一張合理的防禦圖，只是強度錯一倍。

    本專案的張量介面是 [0,1]，故每篇都必須各自記錄其最佳化值域。
    """
    for name, s in REGISTRY.items():
        assert (s.value_range.lo, s.value_range.hi) == (-1.0, 1.0), name
        assert s.value_range.note, f"{name} 未記錄值域的出處"


def test_換算關係與值域一致():
    """`eps_pixel01` 不是獨立輸入的數字，必須等於 eps 除以值域跨距。

    Mist 的原始碼有乘 2（`epsilon/255.0 * (1-(-1))`）故像素域為 16/255；
    PromptFlare 沒有乘（`eps /= 255` 直接施加在 [-1,1] 上）故只有 6/255。
    兩篇看起來都寫 `eps/255`，實際差兩倍——這正是必須逐篇查原始碼的理由。
    """
    for name, s in REGISTRY.items():
        assert s.eps_pixel01 == pytest.approx(s.eps / s.value_range.scale), name


def test_promptflare的像素域預算是論文宣稱的一半():
    """獨立釘住這一項，因為它最容易在「照論文數字」時被改錯。

    照論文的 12/255 實作等於給它兩倍預算，baseline 被不當增強，
    而我們的方法會顯得比實際更差。
    """
    s = REGISTRY["promptflare"]
    assert s.eps == pytest.approx(12 * E), "程式碼中的名目值是 12/255"
    assert s.eps_pixel01 == pytest.approx(6 * E), "像素域只有 6/255"


def test_advpaint取原始碼的一百步而非論文的兩百五十步():
    assert REGISTRY["advpaint"].steps == 100


def test_photoguard是L2不是Linf():
    """論文 Table 9 寫 ℓ∞ 16/255，但 complex notebook 實跑
    `torch.renorm(d_x, p=2, dim=0, maxnorm=16)`，ℓ∞ 呼叫整段被註解。

    兩者是不同的失真幾何：L2 球允許少數像素遠超過均攤值，不可互相換算。
    """
    s = REGISTRY["photoguard_c"]
    assert s.norm == "l2"
    assert s.update_rule == "normalized_grad", "原始碼是歸一化梯度，不是 sign"
    assert s.grad_reps == 10


def test_每篇都記錄了與論文正文的落差():
    """七篇裡沒有一篇的正文與原始碼完全一致。落差本身要寫進論文，
    故它必須存在於程式中而非只在文件裡。"""
    for name, s in REGISTRY.items():
        assert s.discrepancy_note, f"{name} 未記錄與論文正文的落差"
        assert s.source, f"{name} 未記錄出處"


# ---------------------------------------------------------------------------
# 2. 建構期的驗證
# ---------------------------------------------------------------------------

def _minimal(**kw):
    base = dict(
        name="t", source="test", value_range=ValueRange(-1.0, 1.0, note="test"),
        eps=0.06, eps_pixel01=0.03, norm="linf", steps=10, step_size=0.01,
        step_schedule="constant", update_rule="sign", objective="minimize",
        init_rule="none", grad_reps=1,
        needs_target_image=False, needs_mask=False,
        modified_from_paper=False, modification_note="",
        prepare=lambda *a, **k: {}, loss_fn=lambda *a, **k: torch.zeros(()),
        discrepancy_note="test",
    )
    base.update(kw)
    # eps_pixel01 必須等於 eps / 值域跨距，spec 建構時會驗證。
    # 測試改 eps 時自動跟著推導，避免寫出一組自相矛盾的規格。
    if "eps_pixel01" not in kw:
        vr = base["value_range"]
        base["eps_pixel01"] = base["eps"] / vr.scale
    return BaselineSpec(**base)


def test_宣稱改寫過就必須寫明改了什麼():
    """`modified_from_paper=True` 而說明為空時，報表會印出一列看起來像
    原論文設定的數字。那比不標註更糟。"""
    with pytest.raises(ValueError, match="modification_note"):
        _minimal(modified_from_paper=True, modification_note="")
    _minimal(modified_from_paper=True, modification_note="mask 設為全圖")


@pytest.mark.parametrize("field,bad", [
    ("norm", "l1"), ("update_rule", "adam"), ("objective", "both"),
    ("steps", 0), ("eps", -1.0),
])
def test_未知或不合法的設定即拋出(field, bad):
    with pytest.raises(ValueError):
        _minimal(**{field: bad})


def test_三篇改寫過的都標註了():
    """AdvPaint 與 PromptFlare 的 mask 設為全圖、PhotoGuard-c 的 img2img 版
    由本專案移植——三者都不是原論文設定。"""
    for name in ("advpaint", "promptflare", "photoguard_c"):
        s = REGISTRY[name]
        assert s.modified_from_paper is True, name
        assert len(s.modification_note) > 20, f"{name} 的說明過於簡略"


def test_registry的鍵等於spec的名稱():
    for k, s in REGISTRY.items():
        assert k == s.name


# ---------------------------------------------------------------------------
# 3. PGD 的三個組件
# ---------------------------------------------------------------------------

def test_值域換算來回一致():
    vr = ValueRange(-1.0, 1.0, note="test")
    x01 = torch.rand(1, 3, 8, 8)
    assert torch.allclose(vr.to01(vr.from01(x01)), x01, atol=1e-6)
    assert vr.scale == 2.0


def test_linf投影把擾動限制在eps內():
    spec = _minimal(norm="linf", eps=0.06)
    x = torch.zeros(1, 3, 16, 16)
    adv = x + torch.randn_like(x)          # 遠超過 eps
    out = project(adv, x, spec)
    assert float((out - x).abs().max()) <= spec.eps + 1e-6


def test_l2投影把擾動限制在半徑內():
    """PhotoGuard-c 用的是 `torch.renorm(p=2, dim=0, maxnorm=eps)`，
    約束作用在整張圖展平後的單一向量，不是逐通道。"""
    spec = _minimal(norm="l2", eps=16.0)
    x = torch.zeros(1, 3, 64, 64)
    adv = x + torch.randn_like(x) * 10
    out = project(adv, x, spec)
    assert float((out - x).flatten().norm()) <= spec.eps + 1e-4


def test_l2投影不逐通道():
    """逐通道約束會讓實際的整體半徑變成 √3 倍。"""
    spec = _minimal(norm="l2", eps=16.0)
    x = torch.zeros(1, 3, 64, 64)
    out = project(x + torch.randn_like(x) * 10, x, spec)
    d = out - x
    per_ch = torch.stack([d[0, c].norm() for c in range(3)])
    assert float(per_ch.max()) < spec.eps - 1e-3, "單一通道不應獨自達到上限"


def test_sign與歸一化梯度是不同的方向():
    """PhotoGuard-c 是歸一化梯度、其餘四篇是 sign。硬套同一種會改變
    每一步走的方向，且不會有任何症狀。"""
    g = torch.tensor([[[[3.0, -0.001]]]])
    s = update_direction(g, _minimal(update_rule="sign"))
    n = update_direction(g, _minimal(update_rule="normalized_grad"))
    assert torch.equal(s, torch.tensor([[[[1.0, -1.0]]]]))
    assert float(n.flatten().norm()) == pytest.approx(1.0, abs=1e-5)
    assert abs(float(n[0, 0, 0, 1])) < 1e-3, "小分量在歸一化下仍然小"


def test_advpaint的步長線性衰減到百分之一():
    """原始碼：`step_size - (step_size - step_size/100)/iters * iter`。"""
    s = REGISTRY["advpaint"]
    assert s.step_schedule == "linear_decay_1pct"
    first = step_size_at(s, 0)
    last = step_size_at(s, s.steps - 1)
    assert first == pytest.approx(s.step_size)
    # 公式 `ss - (ss - ss/100)/iters * iter` 在 iter = iters 才恰好降到 1%，
    # 故最後一步（iter = iters-1）略高於該值。不可為了讓數字整齊而改公式。
    assert last < first / 40, "衰減幅度不足"
    assert step_size_at(s, s.steps) == pytest.approx(s.step_size / 100)


def test_固定步長的四篇不衰減():
    for name in ("photoguard_c", "mist", "dia_pt", "promptflare"):
        s = REGISTRY[name]
        assert step_size_at(s, 0) == pytest.approx(step_size_at(s, s.steps - 1)), name


def test_起點規則逐篇不同且都落在預算內():
    """Mist 由 ℓ∞ 球內均勻取樣起步、DIA 由 ℓ1 球、其餘由原圖起步。
    起點會影響最終解落在球面的哪一側，逐位元不同，不可統一。

    `initial_point` 產生的是各篇公式的原始輸出，**值域夾限是 `run_pgd` 的
    職責**（它在取得起點後立刻 `clamp(vr.lo, vr.hi)`，與 advertorch 的
    `LinfPGDAttack` 在 rand_init 後夾限的作法一致）。兩者分工：
    這裡驗的是「起點在預算內」，夾限由下一個測試驗。
    """
    # x 必須由固定的 generator 產生，不可用全域 RNG：DIA-PT 的 `l1_ball`
    # 起點依賴 x_ref，用全域 RNG 會讓本測試的結果取決於前面跑過哪些測試。
    g = torch.Generator().manual_seed(1234)
    x = torch.rand(1, 3, 16, 16, generator=g) * 2 - 1
    for name, s in REGISTRY.items():
        adv = initial_point(x, s, generator=torch.Generator().manual_seed(0))
        assert adv.shape == x.shape, name
        d = adv - x
        # 直接量距離，不用「投影後不變」當判準：`project` 的職責是
        # 「投影回球內**再夾回值域**」，而隨機起點會把貼近值域邊界的像素
        # 推到界外，夾限因此一定會動到它們。那是正確行為，不是超出預算。
        if s.init_rule == "l1_ball":
            continue        # 見 test_DIA的L1起點在某些輸入下超出預算
        if s.norm == "linf":
            assert float(d.abs().max()) <= s.eps + 1e-6, name
        else:
            assert float(d.flatten().norm()) <= s.eps + 1e-4, name


@pytest.mark.xfail(reason="DIA 原始碼自身的缺陷，處置待裁決；見 SOURCE_AUDIT §11",
                   strict=True)
def test_DIA的L1起點在某些輸入下超出預算():
    """釘住一個**忠實轉寫自 DIA 原始碼的缺陷**，不是釘住正確行為。

    `_l1_projection` 出自 Croce & Hein 的 AutoAttack，其箱型約束假設
    `x ∈ [0,1]`；DIA 把它套用在 `[-1,1]` 的影像張量上（`DIA_PT.py:241-248`）。
    當 `s1 + c ≥ 0` 時投影分支整個被跳過，回傳值停在中間變數 `u`
    （其值域可達 ±1），起點因而遠離 eps 球。

    實測（16² 影像、eps=0.05）：多數 x 得到 ‖d‖₁ = 0.05004 正常，
    但 x 的 generator seed 為 7 時得到 **‖d‖₁ = 14.01、‖d‖∞ = 1.499**，
    是預算的 30 倍。

    後果：`run_pgd` 在起點只做值域夾限、不做投影（與 DIA 自身的迴圈一致），
    故**第一次梯度是在一個遠超預算的點上算的**。這會讓 DIA 的比較基礎與
    其他四篇不對等。

    處置有三個選項，見 `SOURCE_AUDIT §11`，需使用者裁決：
    (a) 照抄不改，在論文中如實陳述；(b) 在 `run_pgd` 的起點加投影並標為改寫；
    (c) 改用 DIA 的另一個變體 DIA-R（`init_rule="none"`，不受影響）。

    本測試以 `xfail(strict=True)` 標記：缺陷還在時它「如預期失敗」，
    一旦有人修掉它會變成非預期通過而立刻顯現，不會被靜默吸收。
    """
    s = REGISTRY["dia_pt"]
    g = torch.Generator().manual_seed(7)
    x = torch.rand(1, 3, 16, 16, generator=g) * 2 - 1
    adv = initial_point(x, s, generator=torch.Generator().manual_seed(0))
    assert float((adv - x).abs().max()) <= s.eps + 1e-6


def test_隨機起點在夾限後仍在值域內():
    """隨機初始化會把靠近值域邊界的像素推到界外，故必須夾限。
    不夾的話送進 VAE 的是一張不存在的影像。"""
    for name in ("mist", "advpaint"):
        s = REGISTRY[name]
        assert s.init_rule == "uniform_linf"
        x = torch.full((1, 3, 16, 16), s.value_range.hi)   # 全部貼在上界
        adv = initial_point(x, s, generator=torch.Generator().manual_seed(0))
        assert float(adv.max()) > s.value_range.hi, "前提：未夾限時會超出"
        clamped = adv.clamp(s.value_range.lo, s.value_range.hi)
        assert float(clamped.max()) <= s.value_range.hi + 1e-6, name


def test_原圖起步的三篇初值等於原圖():
    x = torch.rand(1, 3, 16, 16) * 2 - 1
    for name in ("photoguard_c", "advpaint", "promptflare"):
        s = REGISTRY[name]
        if s.init_rule != "none":
            continue
        assert torch.equal(initial_point(x, s, generator=None), x), name


def test_目標方向逐篇不同():
    """`minimize` 推向目標（PhotoGuard-c、Mist 的 textural）、
    `maximize` 推離參照（DIA）。方向弄反會讓攻擊變成保護。"""
    assert REGISTRY["photoguard_c"].objective == "minimize"
    assert REGISTRY["dia_pt"].objective == "maximize"


def test_需要目標影像的篇目與查證一致():
    """PhotoGuard-c 的 target 是零張量（中性灰）、Mist 是高對比的 MIST.png，
    兩者不可混用同一份 target。"""
    assert REGISTRY["photoguard_c"].needs_target_image is True
    assert REGISTRY["mist"].needs_target_image is True
    assert REGISTRY["dia_pt"].needs_target_image is False


def test_needs_mask記錄的是方法本身的需求():
    """`needs_mask` 描述的是**該方法結構上是否需要遮罩**，不是我們怎麼設定它。

    AdvPaint 與 PromptFlare 原生是 inpainting 防禦，其損失定義在遮罩上，
    這個事實不會因為我們把遮罩設成全圖而消失。把旗標改成 False 會抹掉
    「這兩篇經過改寫」這個必須寫進論文的資訊；改寫方式記在
    `modification_note`，兩者分工不同。
    """
    inpainting_native = {"advpaint", "promptflare"}
    for name, s in REGISTRY.items():
        assert s.needs_mask is (name in inpainting_native), name
        if s.needs_mask:
            assert "mask" in s.modification_note, f"{name} 未說明遮罩如何處理"


def test_spec不可變():
    """設定在跑到一半被改掉會讓 config_hash 與實際執行的內容脫節。"""
    with pytest.raises(dataclasses.FrozenInstanceError):
        REGISTRY["mist"].steps = 1

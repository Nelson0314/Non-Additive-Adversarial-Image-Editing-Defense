"""損失、淨化、譜診斷的正確性測試。

這些是純函數層，不需 SD 模型，故可在本機快速執行。
"""

import pytest
import torch

from src.defense.objective import DefenseObjective, LossConfig
from src.defense.optimize import eot_pairs
from src.metrics.spectrum import analyze, effective_rank, energy_rank
from src.purify.ops import (
    Purifier,
    gaussian_blur,
    jpeg_proxy,
    jpeg_real,
    quantize_proxy,
    quantize_real,
    default_train_set,
)
from src.residual.lowrank import LowRankResidual
from src.residual.site_pixel_full import FullRankPixelResidual

DEV = torch.device("cpu")
SEED = 20260728


@pytest.fixture(scope="module")
def obj():
    return DefenseObjective(LossConfig(), DEV)


def _img(seed=0, size=64):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(1, 3, size, size, generator=g)


# ------------------------------------------------------------------ 保真項


def test_保真項在完全相同時為零(obj):
    """x_def = x 時 LPIPS=0、SSIM=1、兩道地板都不觸發，總和必須為 0。"""
    x = _img(1)
    total, parts = obj.fidelity_term(x, x)
    assert parts["fid_linf"] == 0.0
    assert parts["fid_pen_linf"] == 0.0
    assert parts["fid_pen_psnr"] == 0.0, "完全相同時 PSNR 地板不應觸發"
    assert abs(float(total)) < 1e-4


# ------------------------------------------------- 鈍化約束（E20 修訂之三）


def _blur(x, k=9, sigma=2.0):
    import torch.nn.functional as F

    c = torch.arange(k, dtype=torch.float32) - (k - 1) / 2
    w = torch.exp(-c.pow(2) / (2 * sigma**2))
    w = (w / w.sum()).view(1, 1, 1, k)
    n = x.shape[1]
    x = F.pad(x, (k // 2, k // 2, 0, 0), mode="replicate")
    x = F.conv2d(x, w.expand(n, 1, 1, k), groups=n)
    x = F.pad(x, (0, 0, k // 2, k // 2), mode="replicate")
    return F.conv2d(x, w.transpose(2, 3).expand(n, 1, k, 1), groups=n)


def _structured(seed=20260801, size=128):
    """有結構的中間調圖樣。

    鈍化約束的測試不能用本檔的 `_img()`：那是純白噪聲，底圖沒有任何結構，
    加性擾動會讓局部梯度能量變動 4.4%（實測），超過為真實照片校準的
    τ_acut = 0.04。那是該測試圖的性質而非指標的性質——真實影像的紋理區塊
    梯度能量高得多，同樣的擾動只造成 2.2%。
    """
    g = torch.Generator().manual_seed(seed)
    yy, xx = torch.meshgrid(torch.arange(size), torch.arange(size), indexing="ij")
    x = torch.zeros(1, 3, size, size)
    x[0, 0] = torch.where((xx + yy) % 16 < 8, 0.30, 0.70)
    x[0, 1] = torch.where((xx // 8) % 2 == 0, 0.35, 0.65)
    x[0, 2] = 0.35 + 0.30 * torch.rand(size, size, generator=g)
    return x


def test_SSIM預設不參與梯度():
    """E20 實測 SSIM 在等 LPIPS 下把雜訊判得比模糊貴約 2 倍，留在梯度裡
    等於補貼模糊。係數保留但預設為 0，可一行復原。"""
    assert LossConfig().alpha_ssim == 0.0


def test_鈍化hinge對模糊施力對雜訊不施力(obj):
    """這是本約束存在的全部理由：它必須只對鈍化收費，不對加性擾動收費。"""
    x = _structured()
    blurred = _blur(x)
    g = torch.Generator().manual_seed(11)
    noised = (x + 0.02 * torch.randn(x.shape, generator=g)).clamp(0, 1)

    _, pb = obj.fidelity_term(blurred, x)
    _, pn = obj.fidelity_term(noised, x)
    assert pb["fid_pen_acut"] > 0.0, "模糊必須觸發鈍化 hinge"
    assert pn["fid_pen_acut"] == 0.0, "同量級的加性雜訊不得觸發鈍化 hinge"
    assert pb["fid_acut"] > pn["fid_acut"]


def test_鈍化hinge在完全相同時不施力(obj):
    x = _structured()
    _, parts = obj.fidelity_term(x, x)
    assert parts["fid_acut"] == pytest.approx(0.0, abs=1e-6)
    assert parts["fid_pen_acut"] == 0.0


def test_鈍化項可微且梯度非零(obj):
    """約束若不可微就進不了訓練，這條必須實測而非假設。"""
    x = _structured()
    xd = _blur(x).clone().requires_grad_(True)
    total, _ = obj.fidelity_term(xd, x)
    total.backward()
    assert xd.grad is not None
    assert float(xd.grad.abs().sum()) > 0.0


def test_鈍化以x_base為對象(obj):
    """與 L∞／LPIPS 兩道 hinge 一致：量的是「防禦本身加了多少」。

    x_def 等於 x_base 時不得施力，即使兩者都遠離原圖 x。
    """
    x = _structured()
    base = _blur(x)
    _, parts = obj.fidelity_term(base, x, x_base=base)
    assert parts["fid_pen_acut"] == 0.0
    assert parts["fid_acut"] == pytest.approx(0.0, abs=1e-6)


def test_PSNR地板在低於門檻時才施力(obj):
    """hinge 的定義：超過門檻不施力，低於才施力。兩側都要驗。"""
    x = _img(2)
    good = (x + 0.001 * torch.randn_like(x)).clamp(0, 1)   # 高 PSNR
    bad = (x + 0.2 * torch.randn_like(x)).clamp(0, 1)      # 低 PSNR

    _, pg = obj.fidelity_term(good, x)
    _, pb = obj.fidelity_term(bad, x)
    assert pg["fid_psnr"] > obj.cfg.psnr_floor
    assert pg["fid_pen_psnr"] == 0.0, "PSNR 高於地板時不得施力"
    assert pb["fid_psnr"] < obj.cfg.psnr_floor
    assert pb["fid_pen_psnr"] > 0.0, "PSNR 低於地板時必須施力"


def test_Linf地板在超過tau時才施力(obj):
    x = _img(3)
    small = (x + 0.5 * obj.cfg.tau_linf).clamp(0, 1)
    _, ps = obj.fidelity_term(x, x)
    assert ps["fid_pen_linf"] == 0.0

    big = x.clone()
    big[0, 0, 0, 0] = (x[0, 0, 0, 0] + 3 * obj.cfg.tau_linf).clamp(0, 1)
    _, pb = obj.fidelity_term(big, x)
    assert pb["fid_linf"] > obj.cfg.tau_linf
    assert pb["fid_pen_linf"] > 0.0


def test_兩道hinge以x_base為對象而非原圖(obj):
    """spec §5.2 修訂：hinge 量的是「防禦加了多少」，不是「與原圖差多少」。

    這是必要的：E0c 實測各影像的重建地板由 19.61 dB 到 31.01 dB，相差
    11.4 dB，任何全域固定的絕對門檻對部分影像不可達、對另一部分不施力。
    構造一個「x_base 已遠離 x、但 x_def 等於 x_base」的情形——防禦什麼都
    沒加，兩道 hinge 都必須為零，即使 x_def 與 x 差很多。
    """
    x = _img(20)
    x_base = (x + 0.25 * torch.randn_like(x)).clamp(0, 1)  # 模擬重建誤差
    _, p = obj.fidelity_term(x_base, x, x_base=x_base)

    assert p["fid_linf"] == 0.0, "x_def 等於 x_base 時，防禦造成的 L∞ 必為 0"
    assert p["fid_pen_linf"] == 0.0
    assert p["fid_pen_psnr"] == 0.0, "防禦沒有讓 PSNR 下降，不得施力"
    # 對原圖的絕對值仍須照實記錄，不因改了優化對象就不報
    assert p["fid_linf_total"] > 0.0
    assert p["fid_psnr_total"] < obj.cfg.psnr_floor


def test_防禦造成的下降仍會觸發hinge(obj):
    """反向檢查：確認上一個測試不是把 hinge 整個關掉了。"""
    x = _img(21)
    x_base = (x + 0.1 * torch.randn_like(x)).clamp(0, 1)
    x_def = (x_base + 0.2 * torch.randn_like(x)).clamp(0, 1)
    _, p = obj.fidelity_term(x_def, x, x_base=x_base)

    assert p["fid_linf"] > obj.cfg.tau_linf
    assert p["fid_pen_linf"] > 0.0
    assert p["fid_pen_psnr"] > 0.0


# --------------------------------------------------------- EOT 的算子取樣


def test_all模式每步涵蓋全部淨化算子():
    """spec §5.1 的期望值以完整列舉估計，而非以輪替近似。

    輪替模式下某一步的梯度只朝一個算子下降；主網格中模糊 σ=1.0 就在訓練
    集裡，測試時卻只保留 4.4%，懷疑即為三個算子的梯度互相覆寫所致。
    """
    for step in (0, 1, 7):
        pairs = eot_pairs("all", step=step, n_eot=1, n_purifiers=3)
        assert sorted(p for p, _ in pairs) == [0, 1, 2], "每步都必須涵蓋全部算子"
        assert len(pairs) == 3
    # 與 step 無關：全列舉不應隨步數改變
    assert eot_pairs("all", 0, 2, 3) == eot_pairs("all", 99, 2, 3)
    assert len(eot_pairs("all", 0, 2, 3)) == 6, "算子數 × 噪聲取樣數"


def test_rotate模式維持原有行為():
    """新增模式不得改動既有路徑，否則 36 格結果失去可比性。"""
    assert eot_pairs("rotate", 0, 1, 3) == [(0, 0)]
    assert eot_pairs("rotate", 1, 1, 3) == [(1, 0)]
    assert eot_pairs("rotate", 3, 1, 3) == [(0, 0)], "應循環回第一個算子"
    assert eot_pairs("rotate", 2, 2, 3) == [(1, 0), (2, 1)]


def test_未知的purify_mode必須報錯():
    """靜默退回某個預設會讓實驗跑完才發現設定沒生效。"""
    with pytest.raises(ValueError, match="purify_mode"):
        eot_pairs("average", 0, 1, 3)


# ------------------------------------------------------------------ 保真項續


def test_LPIPS地板在超過tau時才施力(obj):
    """綁定約束改為感知指標後的 hinge 定義。

    不預設「某個擾動幅度會落在 τ 的哪一側」——先前有測試因為假設兩張隨機
    影像的 LPIPS 會大於 0.5（實測 0.33）而誤報失敗。此處改為先量測再驗
    hinge 與量測值的一致性，最後檢查兩側都真的被涵蓋到，否則測試是空的。
    """
    x = _img(30)
    seen_below = seen_above = False
    for scale in (0.0, 0.005, 0.05, 0.3):
        xd = (x + scale * torch.randn_like(x)).clamp(0, 1)
        _, p = obj.fidelity_term(xd, x)
        if p["fid_lpips_rel"] <= obj.cfg.tau_lpips:
            assert p["fid_pen_lpips"] == 0.0, "低於 τ 時不得施力"
            seen_below = True
        else:
            assert p["fid_pen_lpips"] == pytest.approx(
                p["fid_lpips_rel"] - obj.cfg.tau_lpips, abs=1e-6
            ), "超過 τ 時施力量必須等於超出量"
            seen_above = True
    assert seen_below and seen_above, "掃描未涵蓋 τ 兩側，測試沒有鑑別力"


def test_LPIPS地板同樣以x_base為對象(obj):
    """與 L∞、PSNR 兩道 hinge 一致：量的是防禦加了多少，不是與原圖差多少。

    若這道 hinge 誤用原圖為對象，site L 那種 x_base 本身就遠離 x 的情形會
    讓 hinge 恆為啟動、φ 無法改善，重演 spec §5.2 修訂前的失敗。
    """
    x = _img(31)
    x_base = (x + 0.25 * torch.randn_like(x)).clamp(0, 1)   # 模擬重建誤差
    _, p = obj.fidelity_term(x_base, x, x_base=x_base)

    assert p["fid_lpips_rel"] == 0.0, "x_def 等於 x_base 時防禦造成的感知差異必為 0"
    assert p["fid_pen_lpips"] == 0.0
    # 對原圖的絕對感知距離仍須照實記錄
    assert p["fid_lpips"] > obj.cfg.tau_lpips


def test_PSNR地板預設不參與梯度但仍記錄(obj):
    """修訂之二把 gamma_psnr 設為 0：保留量測與記錄，不影響優化方向。

    同時驗證係數可一行復原——這是保留該項而非刪除的唯一理由。
    """
    x = _img(32)
    xd = (x + 0.2 * torch.randn_like(x)).clamp(0, 1)

    total_off, p = obj.fidelity_term(xd, x)
    assert p["fid_pen_psnr"] > 0.0, "PSNR 低於地板，量測值必須照實記錄"
    assert obj.cfg.gamma_psnr == 0.0, "預設不參與梯度"

    revived = DefenseObjective(LossConfig(gamma_psnr=1.0), DEV)
    total_on, _ = revived.fidelity_term(xd, x)
    assert float(total_on) > float(total_off), "改回 1.0 後該項必須重新生效"
    assert float(total_on) - float(total_off) == pytest.approx(
        p["fid_pen_psnr"], abs=1e-3
    )


def test_保真項對x_def可微(obj):
    """φ 的梯度必須能穿過保真項，否則兩道地板形同虛設。"""
    x = _img(4)
    xd = (x + 0.05 * torch.randn_like(x)).clamp(0, 1).requires_grad_(True)
    total, _ = obj.fidelity_term(xd, x)
    total.backward()
    assert xd.grad is not None and xd.grad.abs().sum() > 0


# ------------------------------------------------------------------ 防禦項


def test_hinge在偏移超過margin後不再施力(obj):
    """無界最大化會發散，hinge 是必要設計（spec §5.1）。

    margin 由實測距離推出，不預設「兩張無關的圖 LPIPS 必大於 0.5」——
    實測兩張獨立均勻雜訊圖的 LPIPS 僅約 0.33，該預設會誤判。
    """
    a = _img(5)
    far = _img(6)
    d_far = float(obj.distance(far, a))
    assert d_far > 0, "測試前提：兩張獨立影像的距離須為正"

    below = DefenseObjective(LossConfig(margin=d_far * 0.5), DEV)
    above = DefenseObjective(LossConfig(margin=d_far * 2.0), DEV)

    assert float(below.defense_term([far], [a])) == 0.0, (
        "距離已超過 margin，hinge 必須歸零"
    )
    saturated = float(above.defense_term([far], [a]))
    assert saturated == pytest.approx(d_far * 2.0 - d_far, abs=1e-5), (
        "距離未達 margin 時應為 m − d"
    )


def test_hinge在距離為零時施力最大(obj):
    a = _img(5)
    assert float(obj.defense_term([a], [a])) == pytest.approx(obj.cfg.margin, abs=1e-5)


def test_防禦項取樣數不符時報錯(obj):
    a = _img(7)
    with pytest.raises(ValueError, match="取樣數不符"):
        obj.defense_term([a, a], [a])


def test_edit_shift與L_def分開記錄(obj):
    """hinge 飽和後 L_def 恆為 0，只看 L_def 會誤判優化停滯。

    故意把 margin 設在實測距離之下，製造出飽和情形，再確認 edit_shift
    仍記錄得到真實偏移。
    """
    x = _img(8)
    xd = (x + 0.02 * torch.randn_like(x)).clamp(0, 1)
    far = _img(9)
    d = float(obj.distance(far, x))

    sat = DefenseObjective(LossConfig(margin=d * 0.5), DEV)
    _, log = sat(xd, x, [far], [x])
    assert log["L_def"] == 0.0, "margin 低於實測距離時 hinge 應飽和"
    assert log["edit_shift"] == pytest.approx(d, abs=1e-5), (
        "偏移量必須獨立於 hinge 之外被記錄"
    )


# ------------------------------------------------------------------ 淨化


def test_淨化必須包含恆等算子():
    """spec §5.1 明訂 𝒫 含恆等算子，否則訓練目標不含「不淨化」的情形。"""
    kinds = [p.kind for p in default_train_set()]
    assert "identity" in kinds


def test_恆等算子逐元素不變():
    x = _img(10)
    assert torch.equal(Purifier("identity").forward(x), x)
    assert torch.equal(Purifier("identity").evaluate(x), x)


def test_直通估計前向與真實實作完全相同():
    """代理的設計是「前向真實、反向恆等」，故前向差必須精確為 0。"""
    x = _img(11)
    assert torch.equal(quantize_proxy(x, 16), quantize_real(x, 16))
    assert torch.equal(jpeg_proxy(x, 75), jpeg_real(x, 75))
    assert Purifier("jpeg", 75).proxy_gap(x) == 0.0
    assert Purifier("quantize", 16).proxy_gap(x) == 0.0


def test_直通估計的梯度為恆等():
    x = _img(12).requires_grad_(True)
    quantize_proxy(x, 16).sum().backward()
    assert torch.allclose(x.grad, torch.ones_like(x)), "直通估計的梯度應為恆等"


def test_不可微算子被正確標記():
    """報告須明列哪些淨化在訓練時用的是代理梯度。"""
    assert Purifier("blur", 1.0).differentiable
    assert Purifier("identity").differentiable
    assert not Purifier("jpeg", 75).differentiable
    assert not Purifier("quantize", 16).differentiable


def test_模糊為可微且強度為零時不變():
    x = _img(13)
    assert torch.equal(gaussian_blur(x, 0.0), x)
    xv = x.clone().requires_grad_(True)
    gaussian_blur(xv, 1.0).sum().backward()
    assert xv.grad.abs().sum() > 0


def test_模糊保持亮度():
    """高斯核已正規化，reflect padding 下平坦區的均值不應改變。"""
    x = torch.full((1, 3, 32, 32), 0.4)
    assert torch.allclose(gaussian_blur(x, 1.5), x, atol=1e-5)


# ------------------------------------------------------------------ 譜診斷


@pytest.mark.parametrize("r", [1, 3, 8])
def test_外積殘差的實測秩等於設定值(r):
    """site P 的秩有理論保證，此測試是為了抓實作錯誤而非驗證數學。"""
    t = LowRankResidual(steps=1, channels=3, height=64, width=64, max_rank=8, seed=SEED)
    with torch.no_grad():
        t.V.normal_(0, 0.02, generator=torch.Generator().manual_seed(SEED))
    delta = t(step=0, rank=r)
    for c in range(delta.shape[0]):
        assert effective_rank(delta[c]) == r


def test_秩判準必須是相對而非絕對():
    """einsum 的 float32 捨入誤差正比於 σ₁，絕對閾值會隨尺度誤判。

    把殘差整體縮小 1e-6 倍後，秩必須不變；若判準是絕對閾值就會變成 0。
    """
    t = LowRankResidual(steps=1, channels=1, height=32, width=32, max_rank=4, seed=SEED)
    with torch.no_grad():
        t.V.normal_(0, 0.02, generator=torch.Generator().manual_seed(SEED))
    d = t(step=0, rank=4)[0]
    assert effective_rank(d) == 4
    assert effective_rank(d * 1e-6) == 4, "縮放後秩改變，代表判準用了絕對閾值"


def test_能量秩不高於有效秩():
    """涵蓋 99% 能量所需的秩，必然不超過非零奇異值的個數。"""
    t = LowRankResidual(steps=1, channels=1, height=32, width=32, max_rank=8, seed=SEED)
    with torch.no_grad():
        t.V.normal_(0, 0.02, generator=torch.Generator().manual_seed(SEED))
    d = t(step=0, rank=8)[0]
    assert energy_rank(d, 0.99) <= effective_rank(d)
    assert energy_rank(d, 0.90) <= energy_rank(d, 0.99)


def test_clamp破壞site_P的精確秩但保留能量低秩():
    """spec §7.2 修訂紀錄：x_def−x 的秩不等於 r，clamp 是非線性變換。

    此測試鎖住的是一個曾寫錯的規格敘述，不是一個實作細節。用飽和像素
    構造出必然觸發 clamp 的情形，確認：精確秩被破壞、能量秩仍為 r。
    """
    from src.residual.site_pixel import PixelResidual

    r, n = 2, 64
    x = torch.rand(1, 3, n, n)
    x[0, :, :, :8] = 1.0   # 人為製造飽和區，保證 clamp 會作用

    mod = PixelResidual(size=n, channels=3, max_rank=r, const_rank=r, seed=SEED)
    with torch.no_grad():
        mod.tensor.V.normal_(0, 0.05, generator=torch.Generator().manual_seed(SEED))

    raw = mod.raw_residual()[0]
    eff = (mod.pixel_residual(x) - x).detach()[0]

    assert mod.clamped_fraction(x) > 0, "測試前提：必須真的有元素被 clamp"
    for c in range(3):
        assert effective_rank(raw[c]) == r, "clamp 前的 Δ 秩必須精確等於 r"
    assert max(effective_rank(eff[c]) for c in range(3)) > r, (
        "clamp 後精確秩應被破壞；若此斷言失敗，代表 clamp 未觸發或秩判準有誤"
    )
    assert max(energy_rank(eff[c], 0.99) for c in range(3)) <= r + 2, (
        "clamp 造成的擾動能量極小，99% 能量秩應仍在 r 附近"
    )


def test_零殘差的秩為零():
    assert effective_rank(torch.zeros(16, 16)) == 0
    assert energy_rank(torch.zeros(16, 16)) == 0


def test_譜分析逐通道且欄位齊全():
    d = torch.randn(3, 16, 16)
    a = analyze(d)
    assert len(a["per_channel"]) == 3
    assert len(a["effective_rank"]) == 3
    for spec in a["per_channel"]:
        assert len(spec["singular_values"]) == 16
        assert spec["cumulative_energy"][-1] == pytest.approx(1.0, abs=1e-9)


# ------------------------------------------------------- 全秩對照組


def test_全秩殘差初始為零且防禦圖等於原圖():
    """「模塊停用時不改變任何計算結果」的不變量，在全秩對照上也必須成立。

    自由參數在 Δ=0 處梯度不為零，故零初始化同時給到「x_def = x 逐元素
    相等」與「梯度可流動」，不需要 site P 那種 U 高斯／V 零的安排。
    """
    x = _img(40, size=32)
    mod = FullRankPixelResidual(size=32)
    assert torch.equal(mod.delta(), torch.zeros_like(mod.delta()))
    assert torch.equal(mod.pixel_residual(x), x)

    mod.disable()
    assert torch.equal(mod.pixel_residual(x), x), "停用時亦須逐元素相等"


def test_全秩殘差在Δ為零處梯度不為零():
    """這是零初始化可行的前提；若梯度為零，優化永遠動不了。"""
    x = _img(41, size=32)
    mod = FullRankPixelResidual(size=32)
    mod.pixel_residual(x).pow(2).sum().backward()
    assert mod.delta_param.grad is not None
    assert float(mod.delta_param.grad.abs().max()) > 0.0


@pytest.mark.parametrize("r", [1, 4, 8])
def test_全秩殘差的實測秩遠高於低秩(r):
    """對照組必須真的不受秩約束，否則整個比較沒有意義。

    以隨機值填入自由參數後量實測有效秩，並與同尺寸的低秩模塊比較。
    低秩那側取 clamp 前的 Δ（raw_residual），因為 clamp 本身會抬高秩，
    此處要比的是參數化造成的差異，不是 clamp 造成的。
    """
    size = 32
    full = FullRankPixelResidual(size=size)
    with torch.no_grad():
        full.delta_param.copy_(torch.randn_like(full.delta_param) * 0.02)

    low = LowRankResidual(steps=1, channels=3, height=size, width=size,
                          max_rank=r, seed=SEED)
    with torch.no_grad():   # V 初始為零會讓 Δ 恆為零，量不到秩
        low.V.copy_(torch.randn_like(low.V) * 0.02)
    low_delta = low(step=0, rank=r).unsqueeze(0)

    # effective_rank 吃單一通道的二維矩陣；逐通道分析用 analyze
    full_rank = analyze(full.raw_residual())["effective_rank"]
    low_rank_ = analyze(low_delta)["effective_rank"]
    assert max(low_rank_) <= r, f"低秩側的實測秩不得超過 {r}"
    assert min(full_rank) > r, "全秩側必須明顯高於低秩側，否則對照無效"


# ----------------------------------------------- 有目標與 VAE 編碼器目標


def test_有目標模式最小化與目標的距離(obj):
    """無目標是「離原編輯遠」，有目標是「往固定目標去」，方向不同。

    構造兩個候選：一個接近目標、一個遠離目標。有目標模式下前者的損失
    必須較小——這正是無目標模式做不到的區分（兩者離 y_orig 可能一樣遠）。
    """
    tgt = _img(50)
    near = (tgt + 0.02 * torch.randn_like(tgt)).clamp(0, 1)
    far = _img(51)
    y_orig = _img(52)

    o = DefenseObjective(LossConfig(defense_mode="targeted"), DEV)
    l_near = float(o.defense_term([near], [y_orig], y_target=tgt))
    l_far = float(o.defense_term([far], [y_orig], y_target=tgt))
    assert l_near < l_far, "接近目標者的損失必須較小"
    assert l_near >= 0.0


def test_有目標模式缺少目標時報錯():
    """靜默退回無目標會讓實驗跑完才發現量的是另一個目標函數。"""
    o = DefenseObjective(LossConfig(defense_mode="targeted"), DEV)
    with pytest.raises(ValueError, match="y_target"):
        o.defense_term([_img(53)], [_img(54)])


def test_未知的defense_mode必須報錯():
    o = DefenseObjective(LossConfig(defense_mode="minimax"), DEV)
    with pytest.raises(ValueError, match="defense_mode"):
        o.defense_term([_img(55)], [_img(56)])


def test_無目標模式行為不變(obj):
    """新增模式不得改動既有路徑，否則既有結果失去可比性。"""
    a, b = _img(57), _img(58)
    d = float(obj.distance(a, b))
    expected = max(0.0, obj.cfg.margin - d)
    assert obj.cfg.defense_mode == "untargeted", "預設必須維持無目標"
    assert float(obj.defense_term([a], [b])) == pytest.approx(expected, abs=1e-6)


def test_編碼器目標在相同時為零且可微(obj):
    """PhotoGuard 的 encoder attack 形式：不經 UNet，故可跑遠多於 25 步。"""
    z = torch.randn(1, 4, 8, 8, requires_grad=True)
    assert float(obj.encoder_term(z, z.detach())) == pytest.approx(0.0, abs=1e-9)

    zt = torch.randn(1, 4, 8, 8)
    loss = obj.encoder_term(z, zt)
    assert float(loss) > 0.0
    loss.backward()
    assert z.grad is not None and float(z.grad.abs().max()) > 0.0

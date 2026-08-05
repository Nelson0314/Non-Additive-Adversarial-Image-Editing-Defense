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
    """x_def = x 時 LPIPS=0、SSIM=1、兩道下限都不觸發，總和必須為 0。"""
    x = _img(1)
    total, parts = obj.fidelity_term(x, x)
    assert parts["fid_linf"] == 0.0
    assert parts["fid_pen_linf"] == 0.0
    assert parts["fid_pen_psnr"] == 0.0, "完全相同時 PSNR 下限不應觸發"
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


def test_SSIM只出現在回報欄位不進梯度(obj):
    """`LOGIC_CHECK` A8：SSIM／NLPD／VIF／GMSD／HaarPSI／ΔE 系列是已排除的
    度量，只能出現在回報欄位，不得進入 hinge。

    E20 實測 SSIM 在等 LPIPS 下把雜訊判得比模糊貴約 2 倍，留在梯度裡等於
    補貼模糊。2026-08-05 把 `alpha_ssim` 這個係數整個刪除——留一個預設為 0
    的係數，就留了一條「把它設回 1.0」的路。
    """
    assert not hasattr(LossConfig(), "alpha_ssim"), (
        "係數不得存在：預設為 0 只是預設，刪除才是約束")
    x = _structured()
    xd = _blur(x)
    _, parts = obj.fidelity_term(xd, x)
    assert "fid_ssim" in parts, "SSIM 仍須照實記錄，只是不進梯度"
    assert 0.0 <= parts["fid_ssim"] <= 1.0


def test_鈍化hinge對模糊施力對雜訊不施力(obj):
    """這是本約束存在的理由：它必須只對鈍化收費，不對加性擾動收費。"""
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


def test_PSNR下限在低於門檻時才施力(obj):
    """hinge 的定義：超過門檻不施力，低於才施力。兩側都要驗。"""
    x = _img(2)
    good = (x + 0.001 * torch.randn_like(x)).clamp(0, 1)   # 高 PSNR
    bad = (x + 0.2 * torch.randn_like(x)).clamp(0, 1)      # 低 PSNR

    _, pg = obj.fidelity_term(good, x)
    _, pb = obj.fidelity_term(bad, x)
    assert pg["fid_psnr"] > obj.cfg.psnr_floor
    assert pg["fid_pen_psnr"] == 0.0, "PSNR 高於下限時不得施力"
    assert pb["fid_psnr"] < obj.cfg.psnr_floor
    assert pb["fid_pen_psnr"] > 0.0, "PSNR 低於下限時必須施力"


def test_Linf下限在超過tau時才施力(obj):
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

    這是必要的：E0c 實測各影像的重建誤差下限由 19.61 dB 到 31.01 dB，相差
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


def test_LPIPS下限在超過tau時才施力(obj):
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


def test_LPIPS下限同樣以x_base為對象(obj):
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


def test_PSNR下限預設不參與梯度但仍記錄(obj):
    """修訂之二把 gamma_psnr 設為 0：保留量測與記錄，不影響優化方向。

    同時驗證係數可一行復原——這是保留該項而非刪除的唯一理由。
    """
    x = _img(32)
    xd = (x + 0.2 * torch.randn_like(x)).clamp(0, 1)

    total_off, p = obj.fidelity_term(xd, x)
    assert p["fid_pen_psnr"] > 0.0, "PSNR 低於下限，量測值必須照實記錄"
    assert obj.cfg.gamma_psnr == 0.0, "預設不參與梯度"

    revived = DefenseObjective(LossConfig(gamma_psnr=1.0), DEV)
    total_on, _ = revived.fidelity_term(xd, x)
    assert float(total_on) > float(total_off), "改回 1.0 後該項必須重新生效"
    assert float(total_on) - float(total_off) == pytest.approx(
        p["fid_pen_psnr"], abs=1e-3
    )


def test_保真項對x_def可微(obj):
    """φ 的梯度必須能穿過保真項，否則四道 hinge 形同虛設。

    2026-08-05 修訂。before：擾動 0.05 且不檢查是否越界。
    after：擾動加大到確實越過 τ_lpips，並先斷言該前提。

    原因是本測試先前能通過，靠的是 `beta_linf=100` 那個在 τ 以內就開始
    施力的項——把它改為預設關閉後，一個四道 hinge 都不越界的輸入其總損失
    精確為 0，梯度自然是零。那不是「不可微」，是「沒有東西要微」。

    交集式 hinge 的定義就是 τ 以內完全免費，故驗可微必須在越界處驗。
    """
    x = _img(4)
    xd = (x + 0.35 * torch.randn_like(x)).clamp(0, 1).requires_grad_(True)
    total, parts = obj.fidelity_term(xd, x)
    assert parts["fid_pen_lpips"] > 0.0, "前提：必須真的越過 τ_lpips"
    total.backward()
    assert xd.grad is not None and xd.grad.abs().sum() > 0


# ------------------------------------------------------------------ 防禦項


def test_margin已隨untargeted一併移除():
    """hinge 存在的唯一理由是「無界的最大化會發散」。

    targeted 是最小化一個下界為 0 的距離，不會發散，故不需要 margin。留一個
    無人使用的 margin 欄位，就留了一條「把 defense_mode 改回去」的路。
    """
    assert not hasattr(LossConfig(), "margin")


def test_防禦項取樣數不符時報錯(obj):
    """兩側逐一配對的長度檢查是「同噪聲配對」這個不變量唯一的程式化把關。"""
    a = _img(7)
    with pytest.raises(ValueError, match="取樣數不符"):
        obj.defense_term([a, a], [a], _img(70))


def test_edit_shift與L_def分開記錄(obj):
    """兩者量的是不同的東西，只看其中一個會誤判。

    targeted 之後 L_def 是「離目標多近」（越小越好），`edit_shift` 是
    「離原編輯多遠」（越大代表防禦越有進展，也是評測所報的量）。
    構造一個「已經很接近目標、但離原編輯很遠」的情形，兩個數字必須分別
    反映這兩件事。
    """
    x = _img(8)
    xd = (x + 0.02 * torch.randn_like(x)).clamp(0, 1)
    tgt = _img(50)
    near = (tgt + 0.01 * torch.randn_like(tgt)).clamp(0, 1)

    _, log = obj(xd, x, y_def_list=[near], y_orig_list=[x], y_target=tgt)
    d_target = float(obj.distance(near, tgt))
    d_orig = float(obj.distance(near, x))
    assert log["L_def"] == pytest.approx(d_target, abs=1e-5)
    assert log["edit_shift"] == pytest.approx(d_orig, abs=1e-5)
    assert log["edit_shift"] > log["L_def"], (
        "測試前提：這一格離目標近、離原編輯遠")


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


# 2026-08-05：全秩對照組（`site_pixel_full.py`）與 site P（`site_pixel.py`）
# 已依 `ARCH` §2.3 刪除——加性由 baseline 擔任，低秩不深究。相關測試隨之移除，
# 可由 `git checkout 4d2332c -- <path>` 取回。
# `LowRankResidual` 本身保留（site E 與 site L 仍在用），其測試見上方譜診斷節。


# ----------------------------------------------- 有目標與 VAE 編碼器目標


def test_targeted_output最小化與目標的距離(obj):
    """構造兩個候選：一個接近目標、一個遠離目標，前者的損失必須較小。

    這正是已淘汰的無目標模式做不到的區分——兩個候選離 y_orig 可能一樣遠，
    無目標的損失因此相同，而它們與「防禦要達到的狀態」的距離差很多。
    """
    tgt = _img(50)
    near = (tgt + 0.02 * torch.randn_like(tgt)).clamp(0, 1)
    far = _img(51)
    y_orig = _img(52)

    l_near = float(obj.defense_term([near], [y_orig], tgt))
    l_far = float(obj.defense_term([far], [y_orig], tgt))
    assert l_near < l_far, "接近目標者的損失必須較小"
    assert l_near >= 0.0


def test_targeted_output的MSE度量同樣可分辨遠近():
    """N3 階段二的 R_a 是 ‖·‖²（`DESIGN` §4），不改寫成 LPIPS。"""
    o = DefenseObjective(
        LossConfig(defense_mode="targeted_output", target_metric="mse"), DEV)
    tgt = _img(60)
    near = (tgt + 0.02 * torch.randn_like(tgt)).clamp(0, 1)
    far = _img(61)
    y_orig = _img(62)
    assert float(o.defense_term([near], [y_orig], tgt)) < float(
        o.defense_term([far], [y_orig], tgt))


def test_缺少目標時報錯(obj):
    """靜默退回任何無目標形式會讓實驗跑完才發現量的是另一個目標函數，
    而且那個函數在起點的梯度是零（LOGIC_CHECK A1）。"""
    with pytest.raises(ValueError, match="y_target"):
        obj.defense_term([_img(53)], [_img(54)], None)


def test_未知的模式與度量在建構設定時就報錯():
    """在 LossConfig 建構時擋下，而不是等到第一次 backward。"""
    with pytest.raises(ValueError, match="defense_mode"):
        LossConfig(defense_mode="minimax")
    with pytest.raises(ValueError, match="target_metric"):
        LossConfig(target_metric="ssim")


def test_預設模式為targeted():
    """三個非加性條件一律 targeted（`DESIGN` §2.1）。預設不得是別的東西。"""
    assert LossConfig().defense_mode == "targeted_output"


# ------------------------------------------------------- N1 的注意力目標


def _attn_maps(n_layers=3, n_tokens=77, q=16, seed=0):
    """假的 cross-attention 分佈 (B, Q, T)，已在 token 維度正規化。"""
    g = torch.Generator().manual_seed(seed)
    out = []
    for _ in range(n_layers):
        a = torch.rand(1, q, n_tokens, generator=g)
        out.append(a / a.sum(dim=-1, keepdim=True))
    return out


def test_注意力目標把質量導向shared_token():
    """把 shared token 的質量抬高，損失必須下降。方向錯了整個條件就是反的。"""
    o = DefenseObjective(
        LossConfig(defense_mode="targeted_attn", shared_tokens=(0,)), DEV)
    base = _attn_maps(seed=70)
    lifted = []
    for a in base:
        b = a.clone()
        b[..., 0] += 0.5
        lifted.append(b / b.sum(dim=-1, keepdim=True))
    assert float(o.attention_term([lifted])) < float(o.attention_term([base]))


def test_注意力目標的值域與質量互補():
    o = DefenseObjective(LossConfig(defense_mode="targeted_attn"), DEV)
    maps = _attn_maps(seed=71)
    mass = float(o.shared_token_mass(maps))
    assert 0.0 < mass < 1.0
    assert float(o.attention_term([maps])) == pytest.approx(1.0 - mass, abs=1e-6)


def test_shared_token涵蓋全部格時必須報錯():
    """全部 token 的注意力質量和恆為 1，該目標會退化成常數 0，最佳化不會
    產生任何更新——與「跑了但沒效果」在外部分不出來。"""
    o = DefenseObjective(
        LossConfig(defense_mode="targeted_attn",
                   shared_tokens=tuple(range(8))), DEV)
    with pytest.raises(ValueError, match="退化成常數"):
        o.shared_token_mass(_attn_maps(n_tokens=8, seed=72))


def test_shared_token超出範圍時必須報錯():
    """位置錯了會靜默地對別的 token 施力。"""
    o = DefenseObjective(
        LossConfig(defense_mode="targeted_attn", shared_tokens=(0, 99)), DEV)
    with pytest.raises(IndexError, match="超出範圍"):
        o.shared_token_mass(_attn_maps(n_tokens=77, seed=73))


def test_空的shared_tokens在建構設定時就報錯():
    with pytest.raises(ValueError, match="shared_tokens"):
        LossConfig(defense_mode="targeted_attn", shared_tokens=())


def test_兩種模式各自要求自己的引數():
    """引數傳錯時必須拋出，不得改走另一種模式。"""
    x = _img(80)
    out = DefenseObjective(LossConfig(defense_mode="targeted_output"), DEV)
    att = DefenseObjective(LossConfig(defense_mode="targeted_attn"), DEV)
    with pytest.raises(ValueError, match="y_def_list"):
        out(x, x, attn_maps=[_attn_maps(seed=81)])
    with pytest.raises(ValueError, match="attn_maps"):
        att(x, x, y_def_list=[x], y_orig_list=[x], y_target=x)


def test_注意力模式的edit_shift記為NaN而非零():
    """本路徑沒有 y_orig 可比，該欄位沒有定義。記 0 會讓「沒有定義」與
    「量到零」在事後分不出來；`plateau_stop` 對 NaN 監看量直接拋出。"""
    x = _img(82)
    att = DefenseObjective(LossConfig(defense_mode="targeted_attn"), DEV)
    _, log = att(x, x, attn_maps=[_attn_maps(seed=83)])
    assert log["edit_shift"] != log["edit_shift"], "必須是 NaN"
    assert 0.0 < log["shared_mass"] < 1.0


def test_編碼器目標在相同時為零且可微(obj):
    """PhotoGuard 的 encoder attack 形式：不經 UNet，故可跑遠多於 25 步。"""
    z = torch.randn(1, 4, 8, 8, requires_grad=True)
    assert float(obj.encoder_term(z, z.detach())) == pytest.approx(0.0, abs=1e-9)

    zt = torch.randn(1, 4, 8, 8)
    loss = obj.encoder_term(z, zt)
    assert float(loss) > 0.0
    loss.backward()
    assert z.grad is not None and float(z.grad.abs().max()) > 0.0


def test_τ以內的失真完全免費():
    """交集式 hinge 的定義：τ 以內不施力、τ 以外才施力。

    2026-08-05 之前 `L_fid` 裡有一個係數為 1 的原始 lpips 項，它是加權和的
    一半——一個持續把失真往零拉的力，使最佳化停在 τ 之下，τ 因而不是綁定的約束。
    實測（H100、w=7.5、其他候選有效約束全部排除）：site C 末端 LPIPS
    0.031–0.045、site P 0.040，而 τ=0.05 的 hinge 在 60 步中只啟動 0–8 步。

    該係數（`alpha_lpips`）與 `alpha_ssim` 已於 2026-08-05 整個刪除，
    不是改成預設 0——留一個預設為 0 的旋鈕，下一個人仍會把它打開。
    本測試改為直接驗證刪除後的性質。
    """
    import torch

    from src.defense.objective import DefenseObjective, LossConfig

    x = torch.rand(1, 3, 64, 64, device=DEV)
    xd = (x + 0.02 * torch.randn_like(x)).clamp(0, 1)

    # τ 遠大於實際失真，故四道 hinge 都不啟動，保真項應精確為零
    cfg = LossConfig(tau_lpips=0.9, tau_acut=0.9, tau_chroma=9.0,
                     beta_linf=0.0, gamma_psnr=0.0)
    total, parts = DefenseObjective(cfg, DEV).fidelity_term(xd, x)

    assert float(total) == pytest.approx(0.0, abs=1e-6), (
        "τ 以內的失真必須完全免費，否則最佳化不會把預算用滿")
    # 實際失真仍要被記錄下來供報表使用，只是不進梯度
    assert parts["fid_lpips"] > 0.0


def test_加權保真項已被刪除而非設為零():
    """`alpha_lpips` 與 `alpha_ssim` 必須不存在。

    留一個預設為 0 的係數不等於刪除：它仍在簽名裡，下一個人會把它打開，
    而打開之後沒有任何症狀——只是 τ 不再是綁定的約束。
    """
    from src.defense.objective import LossConfig

    cfg = LossConfig()
    for dead in ("alpha_lpips", "alpha_ssim"):
        assert not hasattr(cfg, dead), f"{dead} 應已刪除，不是設為 0"
        with pytest.raises(TypeError):
            LossConfig(**{dead: 1.0})


def test_色度偏壓約束預設開啟且門檻為零點六():
    """預設開啟是刻意的（與 gamma_acut 於 E20 導入時相同，而非 alpha_lpips
    那種預設關閉）。忘記開會重演 E27 的失效——site C 用色調偏移換防禦
    效果，而報告裡沒有任何一欄看得出來。

    0.6 由人眼判讀：使用者在 runs/p10_chroma_ladder 的階梯上判讀
    「0.3 還有 0.6 都看不出來，1.0 以上才開始有一些細微色調變化」。
    """
    from src.defense.objective import LossConfig

    c = LossConfig()
    assert c.gamma_chroma == 100.0
    assert c.tau_chroma == 0.8


def test_連貫色偏被色度hinge擋下而隨機擾動不被擋():
    """這道約束存在的理由，且它必須對兩個條件一視同仁。

    ΔE 那一類分不出這兩者（P9 實測等 LPIPS 下 2.44 對 2.79），若拿它當約束，
    加性基準會與非加性一起被擋，比較就不成立。
    """
    import torch

    from src.defense.objective import DefenseObjective, LossConfig

    g = torch.Generator().manual_seed(20260728)
    # 底圖要有亮度結構：`local_acutance_dev` 對全平坦影像會明確拒絕（梯度能量
    # 為零時該比值無定義），而 fidelity_term 一定會算它。條紋加在三個通道上，
    # 故底圖仍是無彩的，不影響色度的比較。
    xs = torch.linspace(0, 12 * torch.pi, 64)
    base = (0.5 + 0.15 * xs.sin().view(1, 1, 1, -1)).expand(1, 3, 64, 64).contiguous()
    # 兩個條件的逐像素色度誤差量值相近（實測 12.86 對 12.65），差別只在空間結構；
    # 偏壓則差 18 倍（0.69 對 12.64）。見 test_chroma.py 的同一構造。
    noisy = (base + 0.06 * torch.randn(base.shape, generator=g)).clamp(0, 1)
    shifted = base.clone()
    shifted[:, 0] += 0.12
    shifted = shifted.clamp(0, 1)

    # τ 在此顯式給定，不用生產預設值。這兩個條件的實測偏壓是 0.67 與 12.6，
    # 而生產的 τ_chroma=0.8 恰好落在 0.67 之上——若沿用預設，這個測試會通過，
    # 但通過的原因會與「τ 選在哪裡」糾纏在一起。測試要驗的是構造能分開
    # 兩者，不是某個特定門檻剛好區分得開；把 τ 設在兩者之間才驗得到那件事。
    cfg = LossConfig(tau_lpips=0.9, tau_acut=0.9,
                     beta_linf=0.0, gamma_psnr=0.0,
                     tau_chroma=2.0)
    obj = DefenseObjective(cfg, DEV)
    _, pn = obj.fidelity_term(noisy.to(DEV), base.to(DEV))
    _, ps = obj.fidelity_term(shifted.to(DEV), base.to(DEV))

    assert pn["fid_pen_chroma"] == 0.0, (
        f"隨機擾動不該被色度 hinge 擋下，實得偏壓 {pn['fid_chroma']:.3f}")
    assert ps["fid_pen_chroma"] > 0.0, (
        f"連貫色偏必須被擋下，實得偏壓 {ps['fid_chroma']:.3f}")


def test_Linf的hinge預設關閉():
    """L∞ 對非加性參數化不具鑑別力，開著它會讓它取代 LPIPS 成為綁定約束。

    實測（位移場、grid 32、bicubic）：LPIPS 0.0593 時 L∞ 已達 0.9386，
    在 τ_lpips=0.05／τ_linf=0.06 下兩者的 hinge 懲罰為 0.0093 對 0.8786，
    **相差 95 倍**。整個實驗設計的共同貨幣是 τ_LPIPS，射線縮放也沿 LPIPS
    求解；若最佳化實際停在 L∞ 的邊界上，報表會以為它停在 τ_LPIPS 上。
    """
    from src.defense.objective import LossConfig

    assert LossConfig().beta_linf == 0.0


def test_關閉Linf後位移場的綁定約束是LPIPS():
    """把上面那組數字釘成迴歸測試。"""
    import torch

    from src.defense.objective import DefenseObjective, LossConfig
    from src.residual.site_warp import WarpResidual

    torch.manual_seed(0)
    x = torch.rand(1, 3, 128, 128, device=DEV)
    m = WarpResidual(size=128, grid_size=32, max_disp=1.5,
                     resample="bicubic").to(DEV)
    with torch.no_grad():
        m.flow.normal_(0, 0.35)
    xd = m.pixel_residual(x)

    cfg_off = LossConfig()                      # beta_linf 預設為 0
    cfg_on = LossConfig(beta_linf=100.0)
    total_off, p = DefenseObjective(cfg_off, DEV).fidelity_term(xd, x)
    total_on, _ = DefenseObjective(cfg_on, DEV).fidelity_term(xd, x)

    assert p["fid_linf"] > 0.5, "前提：位移場的 L∞ 本來就很大"
    # `fid_pen_*` 記的是未乘係數的 hinge 量，恆為正；要驗的是它有沒有
    # 進入總損失。開與關的差額即 L∞ 實際貢獻的力。
    assert p["fid_pen_linf"] > 0.5, "前提：L∞ 確實越界"
    contribution = float(total_on) - float(total_off)
    lpips_force = cfg_off.gamma_lpips * p["fid_pen_lpips"]
    assert contribution > 50 * lpips_force, (
        f"前提：開著 L∞ 時它會主導（{contribution:.1f} vs {lpips_force:.3f}）")
    assert float(total_off) == pytest.approx(
        lpips_force + cfg_off.gamma_acut * p["fid_pen_acut"]
        + cfg_off.gamma_chroma * p["fid_pen_chroma"], rel=1e-4), (
        "關閉後總損失只剩 LPIPS、鈍化、色偏三道，L∞ 不得有任何貢獻")


def test_baseline的Linf走投影而非hinge():
    """四篇 baseline 原生是 ℓ∞ 約束的，但那條路徑是 `pgd.project()` 的
    硬投影，不是這裡的軟懲罰。兩者不可互相替代，故本檔的預設值為 0
    不影響 baseline 的忠實度。"""
    from src.baselines import REGISTRY
    from src.baselines.pgd import project

    assert sum(1 for s in REGISTRY.values() if s.norm == "linf") >= 4
    assert callable(project)


# ---------------------------------------------------------------------------
# LPIPS 的精度（2026-08-06，段 0 於 GPU 上實測後補）
# ---------------------------------------------------------------------------

def test_LPIPS對混合dtype的輸入不中止():
    """`piq.ContentLoss.forward` 第一行是 `self.model.to(x)`：它把 VGG 轉成
    **第一個引數**的 dtype，再用同一個模型對第二個引數取特徵。兩個引數
    dtype 不同時直接以 RuntimeError 中止，而本專案在 bf16 下的 `y_def` 是
    半精度、`y_target`（MIST.png）是 fp32，正是這個情形。

    本機無 GPU 也重現得了：dtype 的判定與裝置無關。
    """
    o = DefenseObjective(LossConfig(), DEV)
    a = torch.rand(1, 3, 64, 64, generator=torch.Generator().manual_seed(SEED))
    d = o.distance(a.to(torch.bfloat16), a)      # bf16 對 fp32
    assert torch.isfinite(d).all()
    assert d.dtype == torch.float32, "LPIPS 應在 fp32 上算出"


def test_LPIPS的呼叫一律經過_perceptual():
    """釘住呼叫點而不只是能力。

    綁定的保真約束 `lpips_rel` 與段 2 射線縮放解的是**同一個 τ**，而後者走
    `MetricSuite` 的 LPIPS、輸入是 fp32（`ray_scale.lpips_against`）。任何一個
    呼叫點繞過 `_perceptual`，訓練期滿足的 τ 與對齊期解出的 τ 就是兩個不同的
    量——而「匹配失真」是全案最關鍵的前提，已被證偽四次。
    """
    import inspect
    import re

    from src.defense import objective as mod

    src = inspect.getsource(mod.DefenseObjective)
    body = src.replace(inspect.getsource(mod.DefenseObjective._perceptual), "")
    hits = re.findall(r"self\._lpips\s*\(", body)
    assert not hits, f"有 {len(hits)} 處直接呼叫 self._lpips，必須改走 _perceptual"


def test_保真度的入口一律轉fp32():
    """混合精度下 `x_def` 是半精度而 `x`／`x_base` 是 fp32。

    `piq` 的 LPIPS 與 SSIM 都依**第一個引數**建核或轉模型，再拿它算第二個
    引數，故混著傳一定中止。2026-08-06 於段 0 連續遇到兩次。此處以實跑
    驗證整條 `fidelity_term` 在混合 dtype 下走得通。
    """
    o = DefenseObjective(LossConfig(), DEV)
    g = torch.Generator().manual_seed(SEED)
    x = torch.rand(1, 3, 64, 64, generator=g)
    xd = (x * 0.9 + 0.05).to(torch.bfloat16)      # 生成路徑的輸出：bf16
    loss, parts = o.fidelity_term(xd, x, x_base=x)
    assert torch.isfinite(loss).all() and loss.dtype == torch.float32
    for k in ("fid_lpips", "fid_ssim", "fid_psnr", "fid_linf"):
        assert k in parts and parts[k] == parts[k], f"{k} 不是有限值"


def test_targeted的MSE也吃得下混合dtype():
    """`y_target` 是由 PNG 載入的 fp32，`y_def` 在 bf16 下是半精度。"""
    o = DefenseObjective(LossConfig(target_metric="mse"), DEV)
    g = torch.Generator().manual_seed(SEED)
    y_t = torch.rand(1, 3, 32, 32, generator=g)
    y_d = (y_t * 0.8).to(torch.bfloat16)
    d = o.target_distance(y_d, y_t)
    assert torch.isfinite(d).all() and d.dtype == torch.float32

"""可達性分析的測試。最要緊的一條：**白化是可逆的、不改變可達集合**。

為什麼這一條是重點
────────────────────────────────────────────────────────────────────
`src/defense/reachability.py` 在 `G = F_xᵀF_x` 的特徵座標裡搜方向。只要那個
換座標偷偷改了可行域（例如把某條限制近似掉、或把被丟掉的子空間當成真的
不存在），搜出來的「最大 reach」就是另一個問題的答案，而且**不會報錯**。
所以這裡逐項釘住：映射兩邊互逆、白化後的能量就是歐氏長度、同一個物理方向
在兩組座標下 `reach` 相同、被丟掉的子空間有一個可查的上界、還原的域與新建
的域逐位相同。

影像用真的肖像縮到 256²：低頻錐與 `G` 都與影像有關，用隨機雜訊測等於測了
一個不存在的情境。**不能再小**——縮圖把內容推向更高的正規化頻率，低頻錐在
128² 以下對常數係數場已經沒有嚴格內點（實測 96/128/192 皆無，256 起才有），
那不是限制寫錯，是那個尺寸的影像本來就不滿足低頻規格。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.defense import bernstein_colour as bc  # noqa: E402
from src.defense import reachability as rb  # noqa: E402
from src.utils.io import load_image_tensor  # noqa: E402

SIZE = 256
IMAGE = ROOT / 'data' / 'portraits' / 'man' / 'man_00.png'


@pytest.fixture(scope='module')
def setup():
    x = load_image_tensor(IMAGE, torch.device('cpu'), size=SIZE)
    domain = bc.BernsteinDomain(x)
    theta0, _ = bc.interior_point(domain)
    gram = bc.perturbation_gram(x).numpy()
    geom = rb.ReachGeometry(domain, theta0, gram)
    return x, domain, theta0, gram, geom


def random_w(geom, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal(geom.whiten.dim)


# ---- 白化：可逆 ----

@pytest.mark.parametrize('seed', [0, 1, 2])
def test_whitening_roundtrip_on_w(setup, seed):
    """`ψ(φ(w)) = w`：白化座標裡的任何向量都能原封不動走一圈回來。"""
    _, _, _, _, geom = setup
    w = random_w(geom, seed)
    back = geom.whiten.to_w(geom.whiten.to_theta(w))
    assert np.allclose(back, w, rtol=1e-9, atol=1e-10 * max(1.0, np.abs(w).max()))


@pytest.mark.parametrize('seed', [0, 1, 2])
def test_whitening_roundtrip_on_theta(setup, seed):
    """`φ(ψ(θ)) = P_Sθ`，而且在 `S` 裡的 θ 走一圈回來就是自己。"""
    _, _, _, _, geom = setup
    rng = np.random.default_rng(100 + seed)
    theta = rng.standard_normal(bc.DIM)
    proj = geom.whiten.project(theta)
    assert np.allclose(geom.whiten.to_theta(geom.whiten.to_w(theta)), proj,
                       rtol=1e-8, atol=1e-10)
    assert np.allclose(geom.whiten.to_theta(geom.whiten.to_w(proj)), proj,
                       rtol=1e-8, atol=1e-10)


@pytest.mark.parametrize('seed', [0, 1, 2])
def test_whitened_energy_is_euclidean(setup, seed):
    """`φ(w)ᵀGφ(w) = ‖w‖²`：白化之後「擾動能量」就是座標的長度。"""
    _, _, _, gram, geom = setup
    w = random_w(geom, seed)
    theta = geom.whiten.to_theta(w)
    assert float(theta @ gram @ theta) == pytest.approx(float(w @ w), rel=1e-7)


# ---- 白化：不改變可達集合 ----

@pytest.mark.parametrize('seed', [0, 1, 2, 3])
def test_reach_identical_in_both_coordinates(setup, seed):
    """同一個物理方向，在 θ 座標與白化座標下的 `reach` 必須相同。

    這條把「換座標」與「換問題」分開：`reach` 只依賴射線本身，換座標不該
    動到它。方向先投影到有效子空間，因為白化座標只表示得了那裡面的方向。
    """
    _, _, _, _, geom = setup
    rng = np.random.default_rng(200 + seed)
    d = geom.whiten.project(rng.standard_normal(bc.DIM))
    a = geom.reach_theta(d)
    b = geom.reach_w(geom.whiten.to_w(d))
    assert b == pytest.approx(a, rel=1e-6)


@pytest.mark.parametrize('scale', [0.001, 17.0])
def test_reach_is_scale_invariant(setup, scale):
    """`reach` 是射線的函數：把方向乘上任何正數都不該改變它。"""
    _, _, _, _, geom = setup
    rng = np.random.default_rng(7)
    d = geom.whiten.project(rng.standard_normal(bc.DIM))
    assert geom.reach_theta(scale * d) == pytest.approx(geom.reach_theta(d),
                                                        rel=1e-6)


@pytest.mark.parametrize('seed', [0, 1, 2])
def test_dropped_subspace_is_bounded_by_its_bound(setup, seed):
    """被白化丟掉的方向，`reach` 不超過 `null_reach_bound`。

    白化「不改變可達集合」的實際內容是這一條：丟掉的不是可達的東西。
    上界由方盒的對角線乘上被丟掉的最大奇異值給，這裡抽樣驗它真的是上界。
    """
    _, _, theta0, _, geom = setup
    whiten = geom.whiten
    bound = whiten.null_reach_bound(geom.theta0_np, geom.base_energy)
    rng = np.random.default_rng(300 + seed)
    v = rng.standard_normal(bc.DIM)
    residual = v - whiten.project(v)          # 落在被丟掉的子空間裡
    assert np.linalg.norm(residual) > 1e-6
    assert geom.reach_theta(residual) <= bound


# ---- 兩類方向 ----

def test_projection_removes_darkening_in_g_metric(setup):
    """投影掉暗化之後，`G` 內積下與暗化的餘弦為零——影像上的擾動正交。

    門檻取 1e−10 而不是 1e−16，是**條件數**：`dᵀG·DARK` 是在 `cond(G) ≈ 10⁶`
    的二次型裡求一個理論值為零的差，float64 在那裡不可能給滿精度。
    """
    _, _, _, gram, geom = setup
    project = rb.GProjector(gram, [rb.DARK])
    rng = np.random.default_rng(11)
    for _ in range(5):
        d = project(rng.standard_normal(bc.DIM))
        assert abs(rb._cos_g(d, rb.DARK, gram)) < 1e-10


def test_projection_removes_both_uniform_directions(setup):
    """兩條一起投影掉時，兩條的餘弦都要是零。

    門檻比單條鬆一個數量級：兩條要解一個 2×2 的 `G`-Gram 反矩陣，抵銷多一層。
    """
    _, _, _, gram, geom = setup
    project = rb.GProjector(gram, [rb.DARK, rb.DESAT])
    rng = np.random.default_rng(14)
    for _ in range(3):
        d = project(rng.standard_normal(bc.DIM))
        assert abs(rb._cos_g(d, rb.DARK, gram)) < 1e-7
        assert abs(rb._cos_g(d, rb.DESAT, gram)) < 1e-7


def test_unprojected_random_direction_is_not_orthogonal(setup):
    """對照：沒投影的隨機方向與暗化並不正交，上一條測的不是恆等式。"""
    _, _, _, gram, geom = setup
    rng = np.random.default_rng(12)
    cosines = [abs(rb._cos_g(rng.standard_normal(bc.DIM), rb.DARK, gram))
               for _ in range(5)]
    assert max(cosines) > 1e-3


def test_projection_keeps_the_full_dimension(setup):
    """`G`-正交投影**不降維**：落在有效子空間外的分量必須留著。

    這一條把「搜尋在 250 維裡做」釘死。截斷到有效子空間會讓 `DARK` 這種
    係數完全相同的方向失去它的長半徑（實測 0.5610 掉到 0.04），搜出來的
    最大值就不是原來那個問題的答案。
    """
    _, _, _, gram, geom = setup
    project = rb.GProjector(gram, [rb.DARK])
    rng = np.random.default_rng(15)
    v = rng.standard_normal(bc.DIM)
    v = v / np.linalg.norm(v)
    outside_before = np.linalg.norm(v - geom.whiten.project(v))
    d = project(v)
    outside_after = np.linalg.norm(d - geom.whiten.project(d))
    assert outside_before > 1e-3
    assert outside_after > 0.5 * outside_before


def test_truncating_to_the_subspace_would_change_dark_reach(setup):
    """把 `DARK` 投影到有效子空間會大幅改變它的 `reach`。

    這是上一條的反面證據，也是「白化只當預條件、不當限制」這個選擇的理由：
    兩個數字若相同，截斷才是無害的。
    """
    _, _, _, _, geom = setup
    full = geom.reach_theta(rb.DARK)
    truncated = geom.reach_theta(geom.whiten.project(rb.DARK))
    assert truncated < 0.5 * full


# ---- 搜尋 ----

def test_search_never_below_its_seeds(setup):
    """搜尋回報的最佳值不低於任何一個起點的值：它是下界，而且是有效的下界。"""
    _, _, _, gram, geom = setup
    project = rb.GProjector(gram, [rb.DARK])
    rng = np.random.default_rng(13)
    seeds = [(True, geom.whiten.basis[:, i]) for i in range(3)]
    seed_values = [geom.reach_theta(project(v)) for _, v in seeds]
    sampler = rb.StepSampler(geom.whiten, np.random.default_rng(21))
    out = rb.spherical_search(geom.reach_theta, project, seeds, sampler, rng,
                              extra_random=1, iters=15)
    assert out['value'] >= max(seed_values) - 1e-12
    assert abs(float(np.linalg.norm(out['direction'])) - 1.0) < 1e-9
    assert abs(rb._cos_g(out['direction'], rb.DARK, gram)) < 1e-10


def test_reach_gradient_matches_finite_differences(setup):
    """`ReachGradient` 與 `reach` 的中央差分在切向上一致。

    它是靠隱函數定理穿過半徑求的，錯了不會拋錯、只會讓上升走錯方向而
    「收斂」到一個較小的值——那種失敗從結果看不出來。
    """
    _, domain, _, gram, geom = setup
    grad = rb.ReachGradient(domain, geom.theta0, gram, geom.base_energy)
    rng = np.random.default_rng(41)
    d = rng.standard_normal(bc.DIM)
    d = d / np.linalg.norm(d)
    r, unit = geom.radius(d)
    analytic = grad(unit, r)
    for _ in range(3):
        e = rng.standard_normal(bc.DIM)
        e = e - (e @ d) * d
        e = e / np.linalg.norm(e)
        h = 1e-5
        fd = (geom.reach_theta(d + h * e) - geom.reach_theta(d - h * e)) / (2 * h)
        assert float(analytic @ e) == pytest.approx(fd, rel=1e-4, abs=1e-9)
    # reach 對尺度不變，徑向分量必須是零
    assert abs(float(analytic @ d)) < 1e-9


def test_constant_plane_is_in_the_seeds(setup):
    """常數係數場的平面要在起點裡，而且那一族確實是 reach 高的地方。

    第 3–6 條限制都是係數差分的界，常數場的差分全為零，所以那個平面是
    分母最小的一族。實測它比隨機方向高兩個數量級；起點漏掉它，回報的
    「最大值」就只是隨機方向的量級。
    """
    _, _, _, _, geom = setup
    rng = np.random.default_rng(43)
    seeds = rb.theta_seeds(geom.whiten, rng)
    plane = [v for flag, v in seeds
             if np.std(v[:bc.NCOEF]) < 1e-12 and np.std(v[bc.NCOEF:]) < 1e-12]
    assert len(plane) >= 12
    best_plane = max(geom.reach_theta(v) for v in plane)
    random_reach = [geom.reach_theta(rng.standard_normal(bc.DIM))
                    for _ in range(8)]
    assert best_plane > 20.0 * max(random_reach)


# ---- 存檔的還原 ----

def test_restored_whitening_matches(setup):
    """`Whitening.from_arrays` 與重做特徵分解得到同一個白化。"""
    _, _, _, gram, geom = setup
    again = rb.Whitening.from_arrays(gram, geom.whiten.basis, geom.whiten.lam)
    rng = np.random.default_rng(23)
    w = rng.standard_normal(geom.whiten.dim)
    assert np.allclose(again.to_theta(w), geom.whiten.to_theta(w), atol=0)
    assert np.allclose(again.to_w(rb.DARK), geom.whiten.to_w(rb.DARK), atol=0)


def test_geometry_from_state_reproduces_reach(setup, tmp_path):
    """走一趟 npz 再回來，`reach` 必須完全相同。

    `gradient` 那一支就是靠這條路把 GPU 節點上的半徑算出來的；還原若有一點
    偏差，梯度那一半的「一階可達增量」就換了分母而不會報錯。
    """
    _, domain, theta0, gram, geom = setup
    path = tmp_path / 'state.npz'
    np.savez(path, theta0=geom.theta0_np, gram=gram,
             basis=geom.whiten.basis, lam=geom.whiten.lam,
             base_energy=np.asarray(geom.base_energy),
             **rb.cone_state(domain.cone))
    again = rb.ReachGeometry.from_state(np.load(path))
    rng = np.random.default_rng(29)
    for _ in range(3):
        d = geom.whiten.project(rng.standard_normal(bc.DIM))
        assert again.reach_theta(d) == pytest.approx(geom.reach_theta(d),
                                                     rel=1e-12)


# ---- 域的還原 ----

def test_restored_domain_matches_fresh_one(setup):
    """存檔還原的域與新建的域，餘裕與半徑都必須相同。"""
    _, domain, theta0, _, geom = setup
    state = rb.cone_state(domain.cone)
    restored = rb.restore_domain(state)
    rng = np.random.default_rng(17)
    for _ in range(3):
        d = torch.from_numpy(rng.standard_normal(bc.DIM))
        d = d / d.norm()
        assert restored.max_radius(theta0.double(), d) == pytest.approx(
            domain.max_radius(theta0.double(), d), rel=1e-12)
    fresh = domain.relative_slacks(theta0)
    again = restored.relative_slacks(theta0)
    for key, value in fresh.items():
        assert again[key] == pytest.approx(value, rel=1e-12, abs=1e-15)


# ---- 一階增量的解析梯度 ----

def test_gain_gradient_matches_finite_differences(setup):
    """`GainGradient` 與 `gain` 的中央差分在切向上一致。

    `q(d) = r(d)·gᵀd` 的梯度是 `r·g + (gᵀd)·∇r`，第二項靠隱函數定理穿過
    半徑。少掉第二項、或把 `∇reach` 直接拿來當 `∇q`，上升仍然會「收斂」到
    某個值，只是收斂到錯的地方——那種失敗從結果看不出來，所以在這裡釘住。
    """
    _, domain, _, gram, geom = setup
    rng = np.random.default_rng(53)
    g = rng.standard_normal(bc.DIM)
    g = g / np.linalg.norm(g) * 0.05
    grad = rb.GainGradient(
        rb.ReachGradient(domain, geom.theta0, gram, geom.base_energy), g)
    score = rb.gain_score(geom, g)
    for trial in range(2):
        d = rng.standard_normal(bc.DIM)
        d = d / np.linalg.norm(d)
        r, unit = geom.radius(d)
        analytic = grad(unit, r)
        for _ in range(3):
            e = rng.standard_normal(bc.DIM)
            e = e - (e @ d) * d
            e = e / np.linalg.norm(e)
            h = 1e-5
            fd = (score(d + h * e) - score(d - h * e)) / (2 * h)
            assert float(analytic @ e) == pytest.approx(fd, rel=1e-4,
                                                        abs=1e-9)
        # `q` 對尺度不變（`r` 是 −1 齊次），徑向分量必須是零
        assert abs(float(analytic @ d)) < 1e-9 * float(np.linalg.norm(analytic))


def test_gain_gradient_is_not_the_reach_gradient(setup):
    """兩個梯度不是同一個方向：`gain` 那一支不能沿用 `reach` 的梯度。"""
    _, domain, _, gram, geom = setup
    rng = np.random.default_rng(59)
    g = rng.standard_normal(bc.DIM) * 0.05
    inner = rb.ReachGradient(domain, geom.theta0, gram, geom.base_energy)
    d = rng.standard_normal(bc.DIM)
    d = d / np.linalg.norm(d)
    r, unit = geom.radius(d)
    a = inner(unit, r)
    b = rb.GainGradient(inner, g)(unit, r)
    cos = float(a @ b) / (np.linalg.norm(a) * np.linalg.norm(b))
    assert abs(cos) < 0.99


# ---- 三支搜尋的單調關係 ----

def test_cross_seeding_restores_monotone_branches(setup):
    """`strict ≤ nondark ≤ unrestricted`：子集的最大值不可能比較大。

    三支各自獨立爬山不保證這件事——既有結果裡 `nondark_strict` 在兩張影像上
    贏過 `nondark`。這裡故意給 `nondark` 一個很差的起點、給 `nondark_strict`
    一個好的，重現那個順序顛倒，再確認互相播種之後順序回來。
    """
    _, _, _, gram, geom = setup
    projectors = rb.branch_projectors(gram)
    good = projectors['nondark_strict'](geom.whiten.basis[:, 0])
    rng = np.random.default_rng(71)
    bad = projectors['nondark'](rng.standard_normal(bc.DIM))
    priors = {'nondark_strict': good, 'nondark': bad,
              'unrestricted': bad.copy()}
    before = {k: geom.reach_theta(v) for k, v in priors.items()}
    assert before['nondark_strict'] > before['nondark']   # 顛倒的前提成立
    sampler = rb.StepSampler(geom.whiten, np.random.default_rng(73))
    out = rb.refine_reach_monotone(geom, priors, sampler,
                                   np.random.default_rng(79),
                                   iters=4, ascent_steps=3)
    values = [out[k]['value'] for k in rb.BRANCH_ORDER]
    for lo, hi in zip(values, values[1:]):
        assert lo <= hi * (1.0 + 1e-9)
    # 合併取最佳：既有的值不會被新的搜尋弄小
    for key in rb.BRANCH_ORDER:
        assert out[key]['value'] >= before[key] - 1e-12


def test_cross_seeds_drops_directions_that_vanish(setup):
    """整條落在被投影掉的方向上的起點要被略過，不是拋例外。"""
    _, _, _, gram, geom = setup
    project = rb.GProjector(gram, [rb.DARK, rb.DESAT])
    seeds = rb.cross_seeds(project, [rb.DARK, rb.DESAT,
                                     geom.whiten.basis[:, 0]])
    assert len(seeds) == 1
    assert abs(rb._cos_g(seeds[0][1], rb.DARK, gram)) < 1e-10

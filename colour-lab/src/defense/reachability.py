"""250 維族的可達性分析：白化座標、暗化／非暗化的分解、最大 reach 的搜尋。

這一支回答什麼
────────────────────────────────────────────────────────────────────
`scripts/bernstein_domain_check.py` 報的是三族方向的 `reach` **中位數**：
沿方向走到邊界時，擾動能量相對於內點自身擾動能量的倍率

    reach(d) = r_max(θ₀, d) · √(dᵀGd) / √(θ₀ᵀGθ₀),   G = F_xᵀF_x

`F_x` 是「θ ↦ 擾動 `T_θx − x` 的線性部分」，行已去直流（見
`bernstein_colour.perturbation_gram`）。中位數回答「隨便抽一個方向平均
能走多遠」，**不是上限**。這裡求的是上限，而且把「整體暗化」那一類單獨
拿開，因為四維族已經走過它。

三個設計選擇（理由寫在各自的類別裡）
────────────────────────────────────────────────────────────────────
1. **搜尋在原來的 250 維裡做，白化只當預條件。** 白化 `φ(w) = U_kΛ_k^{−1/2}w`
   在有效子空間 `S` 上是可逆的換座標，但**把搜尋限制在 `S` 裡會改變答案**：
   `reach` 的分子只看能量，分母（半徑）卻由係數的硬界決定，而落在 `S` 外、
   幾乎不帶能量的分量可以大幅拉長半徑。實測 `DARK`（整體暗化）是最乾淨的
   例子：它的 125 個係數完全相同，所有的界同時綁；`P_S·DARK` 的係數不再相同，
   `reach` 從 0.5610 掉到 0.04。所以白化只用在三件事上——報有效維度、生成
   起點、塑造搜尋的步長分布——**可行性與半徑一律在原座標的 250 維裡求**。
2. **暗化與非暗化用 `G` 內積分。** `⟨d,e⟩_G = ⟨F_x d, F_x e⟩`，所以「與暗化
   正交」這句話問的是**影像上的擾動**正不正交，不是係數向量正不正交。用參數
   空間的歐氏正交會把「係數不同但畫面幾乎一樣」的方向判成非暗化。`GProjector`
   在 250 維裡直接做這個 `G`-正交投影，沒有任何截斷。
3. **最大 reach 是凸函數在凸集上的極大。** 可行域是凸的，`reach` 沿射線是
   正齊次的，求的是「可行體在哪個方向上伸得最遠」。這種問題沒有凸性可用，
   而且實測曲面極度多峰：純隨機起點的爬山只走到已知最大值的 35–43%，解析
   梯度上升從隨機起點更只到 7%。所以搜尋用**結構起點 ＋ 三段局部搜尋**
   （`ReachGradient` 的上升、`VertexMove` 的頂點提議、隨機爬山），回報的是
   **下界**，並且另外報「只從隨機起點能走到多少」當作曲面崎嶇程度的量測，
   不把結構起點的功勞算成搜尋器的能力。

   高 reach 的區域是**常數係數場**那一平面：第 3–6 條限制全是係數差分的界，
   常數場的差分為零，那些限制一條都不綁。分母因此最小，`reach` 比隨機方向
   高兩個數量級。
4. **一階增量 `q(d) = r(d)·gᵀd` 有自己的解析梯度。** `ReachGradient` 是
   `reach` 的梯度、不是 `q` 的：`reach` 只看能量，`q` 還要看防禦目標梯度
   `g = ∇_θJ` 在該方向上的投影，兩者的極大不在同一個地方。`GainGradient`
   把隱函數那一段（`ReachGradient.radius_grad`）接上 `∇q = r·g + (gᵀd)·∇r`，
   兩支因此用同一份半徑導數，而且不需要任何新的前向或反向——`g` 是存檔的。
5. **三支搜尋的最大值必須單調。** `nondark_strict ⊆ nondark ⊆ unrestricted`
   是集合的包含關係，最大值只能遞增。各自獨立爬山不保證這件事，違反時代表
   母集漏掉了一個已知更好的候選；`refine_reach_monotone` 由窄到寬依序補搜，
   每一支都拿到其他支的最佳方向（投影到自己的可行集合後）當起點。
"""
from __future__ import annotations

import math
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from src.defense import bernstein_colour as bc

#: 有效子空間的相對奇異值門檻，與 `bernstein_domain_check.py` 的 `rank_1e3`
#: 同一條線：兩份數字因此可以並排讀。
SV_THRESHOLD = 1e-3


# ────────────────────────────────────────────────────────────────────
# 白化：可逆的資料相依預條件化
# ────────────────────────────────────────────────────────────────────

def symmetrise(gram) -> np.ndarray:
    """`G` 的對稱化，所有讀 `G` 的地方都先走這裡。

    `G = F_xᵀF_x` 是用 float32 的 `F_x` 分塊累加出來的，兩半因此差到約
    `1e−9` 相對量級。只要不同的地方讀到不同的一半，「`G`-正交」就會在
    `1e−7` 的層級上前後不一致——例如把 `DARK` 投影掉之後，`DARK` 自己身上
    還留著那麼大的殘量，於是「整條落在被投影掉的方向上」判不出來。理論上
    `G` 是對稱的，所以統一取兩半的平均。
    """
    g = np.asarray(gram, dtype=np.float64)
    return 0.5 * (g + g.T)


class Whitening:
    """`G = UΛUᵀ` 的前 `k` 個特徵方向上的白化座標。

        φ(w) = U_k Λ_k^{−1/2} w        (w ∈ R^k → θ ∈ R²⁵⁰)
        ψ(θ) = Λ_k^{1/2} U_kᵀ θ        (θ ∈ R²⁵⁰ → w ∈ R^k)

    `ψ(φ(w)) = w` 對所有 `w` 成立（`U_k` 的行正交、`Λ_k` 可逆），所以在
    子空間 `S = range(U_k)` 上這是一個**可逆**的線性換座標；`φ(ψ(θ))` 則是
    θ 到 `S` 的正交投影。白化後擾動能量就是歐氏長度：

        φ(w)ᵀ G φ(w) = wᵀ Λ_k^{−1/2} U_kᵀ (UΛUᵀ) U_k Λ_k^{−1/2} w = ‖w‖²

    **可達集合不變**：這個類別只提供座標，不提供任何可行性判斷；所有的
    `max_radius` 都是在原座標的 θ 上、用原來的七條限制求的。
    """

    @classmethod
    def from_arrays(cls, gram, basis, lam, threshold: float = SV_THRESHOLD
                    ) -> 'Whitening':
        """由存檔的 `G`、`U_k`、`Λ_k` 還原，不重做特徵分解。

        `tests/test_reachability.py::test_restored_whitening_matches` 釘住
        還原的結果與重算的逐位相同，所以這不是另一條路徑。
        """
        obj = cls.__new__(cls)
        obj.gram = symmetrise(gram)
        obj.eigenvalues = np.asarray(lam, dtype=np.float64)
        obj.threshold = float(threshold)
        obj.basis = np.ascontiguousarray(np.asarray(basis, dtype=np.float64))
        obj.lam = np.asarray(lam, dtype=np.float64)
        obj.sqrt_lam = np.sqrt(obj.lam)
        obj.dim = int(obj.basis.shape[1])
        obj.dropped_lam_max = 0.0
        return obj

    def __init__(self, gram, threshold: float = SV_THRESHOLD):
        g = symmetrise(gram)                     # eigh 只讀一半，兩半要先一致
        lam, vec = np.linalg.eigh(g)
        order = np.argsort(lam)[::-1]
        lam = np.clip(lam[order], 0.0, None)
        vec = vec[:, order]
        sv = np.sqrt(lam)
        if sv[0] <= 0:
            raise ValueError('G 是零矩陣：這張影像激發不出任何係數')
        keep = sv > threshold * sv[0]
        self.gram = g
        self.eigenvalues = lam
        self.threshold = float(threshold)
        self.basis = np.ascontiguousarray(vec[:, keep])      # (250, k)
        self.lam = lam[keep]
        self.sqrt_lam = np.sqrt(self.lam)
        self.dim = int(self.basis.shape[1])
        #: 被丟掉的那些方向裡最大的特徵值，供 `null_reach_bound` 用。
        self.dropped_lam_max = float(lam[~keep].max()) if (~keep).any() else 0.0

    # ---- 兩個方向的映射 ----

    def to_theta(self, w) -> np.ndarray:
        return self.basis @ (np.asarray(w, dtype=np.float64) / self.sqrt_lam)

    def to_w(self, theta) -> np.ndarray:
        return self.sqrt_lam * (self.basis.T @ np.asarray(theta, dtype=np.float64))

    def project(self, theta) -> np.ndarray:
        """θ 在有效子空間上的正交投影 `φ(ψ(θ))`。"""
        t = np.asarray(theta, dtype=np.float64)
        return self.basis @ (self.basis.T @ t)

    def energy(self, theta) -> float:
        t = np.asarray(theta, dtype=np.float64)
        return float(t @ self.gram @ t)

    def null_reach_bound(self, theta0, base_energy: float) -> float:
        """被丟掉的子空間裡 `reach` 的上界。

        域被第 1 條限制成一個方盒（`A_ijk ∈ [0, 0.25]`、`B_ijk ∈ [0, 0.1]`），
        所以從 `θ₀` 出發的位移長度不超過盒子的對角線 `D`；而在被丟掉的
        子空間裡 `√(dᵀGd) ≤ √λ_drop`。兩者相乘即上界。

        **這只管「整條落在被丟掉的子空間裡」的方向。** 一個方向若同時有
        `S` 內與 `S` 外的分量，`S` 外那一段不進分子卻會改變半徑，那種混合
        不受這個上界約束——搜尋因此在完整的 250 維裡做，不是在 `S` 裡。
        """
        t0 = np.asarray(theta0, dtype=np.float64)
        hi = np.concatenate([np.full(bc.NCOEF, bc.A_CAP),
                             np.full(bc.NCOEF, bc.B_CAP)])
        span = float(np.linalg.norm(np.maximum(hi - t0, t0)))
        return span * math.sqrt(self.dropped_lam_max) / math.sqrt(base_energy)


# ────────────────────────────────────────────────────────────────────
# 兩類方向
# ────────────────────────────────────────────────────────────────────

def constant_direction(alpha: float, beta: float) -> np.ndarray:
    v = bc.constant_theta(alpha, beta).double().numpy()
    return v / np.linalg.norm(v)


#: 「整體暗化」：`a ≡ 常數`、`b ≡ 0`。四維族走過的就是這一條。
DARK = constant_direction(1.0, 0.0)
#: 「整體去飽和」：`a ≡ 0`、`b ≡ 常數`。與暗化一起構成**全域常數**那一平面。
DESAT = constant_direction(0.0, 1.0)


def orthonormal_block(whiten: Whitening, directions: Sequence[np.ndarray]
                      ) -> np.ndarray:
    """把幾個 θ 方向搬到白化座標並正交化，回傳 (k, m)。

    正交化在**白化座標**裡做，因此「正交」的意思是它們產生的影像擾動正交。
    """
    cols = []
    for d in directions:
        w = whiten.to_w(d)
        n = float(np.linalg.norm(w))
        if n <= 0:
            raise ValueError('這個方向在有效子空間裡是零，無法當作分類的基準')
        cols.append(w / n)
    q, _ = np.linalg.qr(np.stack(cols, axis=1))
    return q


class GProjector:
    """θ 空間（250 維）裡的 `G`-正交投影，再單位化。

        P(d) = d − U (UᵀGU)^{−1} UᵀG d

    `U` 是要投影掉的那幾個 θ 方向。投影後 `P(d)ᵀ G u = 0` 對每一個 `u` 成立，
    也就是 `F_x P(d)` 與 `F_x u` 在影像上正交。**沒有降維**：`P(d)` 仍是 250
    維，落在 `S` 外的分量原封不動留著（它們不帶能量卻會改變半徑，見模組說明
    的第 1 點）。`directions` 為空時只單位化。

    `residual` 與 `__call__` 分開，是因為兩種情形要分開處理：**搜尋途中**投影
    後長度為零代表程式錯了，該拋；**起點**投影後長度為零只代表那個起點整條
    落在被投影掉的方向上（例如把整體去飽和當起點、卻又要求與它正交），
    換一個起點即可，不是錯誤。用 `residual` 先量再決定，不用例外當控制流。
    """

    def __init__(self, gram: np.ndarray, directions: Sequence[np.ndarray] = ()):
        # `G = F_xᵀF_x` 是用 float32 的 `F_x` 累加出來的，兩半因此差到約
        # `1e-9` 相對量級。投影要的是一個**內積**：`P(d)` 與 `M = UᵀGU` 各
        # 讀 `G` 的不同一半，不對稱就會讓「投影掉 DARK」在 DARK 自己身上留下
        # 約 `1e-7` 的殘量。`Whitening` 在做特徵分解前已經對稱化，這裡用同一
        # 條線，兩支因此對同一個 `G` 說同一件事。
        self.gram = symmetrise(gram)
        self.dim = int(self.gram.shape[0])
        if len(directions) == 0:
            self.basis = None
            return
        u = np.stack([np.asarray(d, dtype=np.float64) for d in directions],
                     axis=1)
        gu = self.gram @ u
        m = u.T @ gu
        if np.linalg.matrix_rank(m, tol=1e-12 * float(np.abs(m).max())) < u.shape[1]:
            raise ValueError('要投影掉的方向在 G 內積下線性相關')
        self.basis = u
        self.gu = gu
        self.minv = np.linalg.inv(m)

    def residual(self, d: np.ndarray) -> np.ndarray:
        v = np.asarray(d, dtype=np.float64)
        if self.basis is None:
            return v
        return v - self.basis @ (self.minv @ (self.gu.T @ v))

    def __call__(self, d: np.ndarray) -> np.ndarray:
        v = self.residual(d)
        n = float(np.linalg.norm(v))
        if n < 1e-12:
            raise ValueError('投影後的向量長度為零，方向沒有定義')
        return v / n


class VertexMove:
    """Frank–Wolfe 式的提議步：往「係數盒的頂點」走。

    為什麼非有這一步不可
    ────────────────────────────────────────────────────────────────
    `reach(d) = ‖d‖_G / γ(d)`（`γ` 是可行體的 Minkowski 規範，凸且正齊次），
    求的是「凸函數在凸體上的極大」，最大值落在**端點**上。端點在 250 維裡是
    測度零的角落，等向的隨機爬山走不到：實測不設限的隨機爬山只能走到已知
    `DARK` 值的三分之一。

    把第 1 條（逐係數的上下界 `A ∈ [0, 0.25]`、`B ∈ [0, 0.1]`）單獨拿出來當
    外層鬆弛，線性化的子問題 `argmax_{v ∈ box} ⟨G·Δ, v − θ₀⟩` 就有封閉解：
    每個座標各自取上界或下界，看 `G·Δ` 那一格的正負。`DARK` 正是「A 區塊
    全取上界」那個頂點——`θ₀` 是常數場，所以 `hi − θ₀` 在 A 區塊上每一格
    相同，方向就是整體暗化。這一步因此走得到它。

    提議出來的頂點仍要經過投影與 `max_radius`，**可行性沒有被放鬆**：盒只是
    用來產生候選方向的鬆弛，不是用來判定可不可行。
    """

    def __init__(self, geom: 'ReachGeometry'):
        self.geom = geom
        caps = np.concatenate([np.full(bc.NCOEF, bc.A_CAP),
                               np.full(bc.NCOEF, bc.B_CAP)])
        self.hi = caps - geom.theta0_np
        self.lo = -geom.theta0_np

    def __call__(self, d: np.ndarray) -> np.ndarray:
        r, unit = self.geom.radius(d)
        c = self.geom.whiten.gram @ (r * unit)
        return np.where(c > 0.0, self.hi, self.lo)


class StepSampler:
    """搜尋的步長分布：一半走白化方向，一半走等向。

    250 維裡等向的隨機步幾乎全落在「影像激發不出來」的方向上，走不動能量；
    只走白化方向則永遠碰不到那些不帶能量、卻能換到更長半徑的分量。兩者
    各半，兩種進展都拿得到。`whitened_share` 是走白化那一半的機率。
    """

    def __init__(self, whiten: Whitening, rng: np.random.Generator,
                 whitened_share: float = 0.5):
        self.whiten = whiten
        self.rng = rng
        self.share = float(whitened_share)

    def __call__(self) -> np.ndarray:
        if self.rng.random() < self.share:
            step = self.whiten.to_theta(self.rng.standard_normal(self.whiten.dim))
        else:
            step = self.rng.standard_normal(bc.DIM)
        n = float(np.linalg.norm(step))
        if n <= 0:
            raise ValueError('步長取樣回傳零向量')
        return step / n


# ────────────────────────────────────────────────────────────────────
# reach 的評估與搜尋
# ────────────────────────────────────────────────────────────────────

class FastRadius:
    """`BernsteinDomain.max_radius` 的等價實作，每次求值的 numpy 呼叫數從
    約 50 降到 8。

    為什麼要自己寫一份
    ────────────────────────────────────────────────────────────────
    搜尋要呼叫半徑上萬次，而原版的 `worst(t)` 每次都重建 23 個小陣列的
    字典、再逐個取 `max`／`min`／`abs().max()`。那些陣列最多 125 個元素，
    時間幾乎全花在 numpy 的每次呼叫開銷上（實測一次半徑 40 ms，其中 `worst`
    佔 70%）。這裡把所有進入限制的線性泛函接成一條長向量，用三次
    `reduceat` 一次算完所有群的 max、min 與絕對值 max。

    **限制本身一個字都沒改**，二分的步數與原版相同，
    `tests/test_reachability.py::test_fast_radius_matches_domain` 釘住兩者
    在隨機方向上的結果相同到 1e−12。
    """

    def __init__(self, domain: bc.BernsteinDomain, theta0: torch.Tensor):
        self.domain = domain
        self.theta0 = theta0.detach().double()
        base = {k: v.detach().cpu().double().numpy()
                for k, v in bc.linear_pieces(self.theta0).items()}
        keys: List[str] = []
        for key in ('a', 'ra', 'b', 'rb', 'jac_row'):
            keys.append(key)
        for key in bc.JAC_ABS:
            if key not in keys:
                keys.append(key)
        for key, _ in bc.HESS_ABS:
            if key not in keys:
                keys.append(key)
        self.keys = keys
        index = {k: i for i, k in enumerate(keys)}
        sizes = [base[k].size for k in keys]
        self.starts = np.concatenate([[0], np.cumsum(sizes[:-1])]).astype(int)
        self.base_cat = np.concatenate([base[k] for k in keys])
        self.cap_idx = np.array([index[k] for k, _ in bc.CAP_KEYS])
        self.caps = np.array([c for _, c in bc.CAP_KEYS], dtype=np.float64)
        self.jac_row = index['jac_row']
        self.jac_idx = np.array([index[k] for k in bc.JAC_ABS])
        self.hess_idx = np.array([index[k] for k, _ in bc.HESS_ABS])
        self.hess_w = np.array([w for _, w in bc.HESS_ABS], dtype=np.float64)

    def __call__(self, direction: np.ndarray, iters: int = 40) -> float:
        d = torch.as_tensor(np.asarray(direction, dtype=np.float64),
                            dtype=torch.float64)
        rate = {k: v.detach().cpu().double().numpy()
                for k, v in bc.linear_pieces(d).items()}
        rate_cat = np.concatenate([rate[k] for k in self.keys])
        q0, q1, q2, r0, r1 = self.domain.cone.ray(self.theta0, d)
        scale = self.domain.cone.rhs_scale
        offset = self.domain.cone.offset
        base_cat, starts = self.base_cat, self.starts
        cap_idx, caps = self.cap_idx, self.caps

        def worst(t: float) -> float:
            p = base_cat + t * rate_cat
            gmax = np.maximum.reduceat(p, starts)
            gmin = np.minimum.reduceat(p, starts)
            gabs = np.maximum.reduceat(np.abs(p), starts)
            v = float(max(((gmax[cap_idx] - caps) / caps).max(),
                          (-gmin[cap_idx] / caps).max()))
            jac = float(gmax[self.jac_row] + gabs[self.jac_idx].sum())
            v = max(v, (jac - bc.JAC_CAP) / bc.JAC_CAP)
            hess = float(self.hess_w @ gabs[self.hess_idx])
            v = max(v, (hess - bc.HESS_CAP) / bc.HESS_CAP)
            lhs = math.sqrt(max(q0 + 2.0 * t * q1 + t * t * q2, 0.0)) + offset
            return max(v, (lhs - (r0 + t * r1)) / scale)

        if worst(0.0) >= 0.0:
            raise ValueError('起點不在域的內部，半徑沒有意義')
        hi = 1.0
        for _ in range(60):
            if worst(hi) >= 0.0:
                break
            hi *= 2.0
        else:
            raise ValueError('這個方向上所有限制都不綁；方向應為非零向量')
        lo = 0.0
        for _ in range(iters):
            mid = 0.5 * (lo + hi)
            if worst(mid) < 0.0:
                lo = mid
            else:
                hi = mid
        return lo


class ReachGeometry:
    """一張影像上的「域 ＋ 內點 ＋ 白化」，以及 `reach` 的求值。"""

    @classmethod
    def from_state(cls, state) -> 'ReachGeometry':
        """由 `geometry` 那一支存下的 npz 還原：域、內點、白化全部不重算。"""
        obj = cls.__new__(cls)
        obj.domain = restore_domain(state)
        obj.theta0 = torch.as_tensor(np.asarray(state['theta0']),
                                     dtype=torch.float64)
        obj.theta0_np = np.asarray(state['theta0'], dtype=np.float64)
        obj.whiten = Whitening.from_arrays(state['gram'], state['basis'],
                                           state['lam'])
        obj.base_energy = float(np.asarray(state['base_energy']))
        obj.fast = FastRadius(obj.domain, obj.theta0)
        obj.calls = 0
        return obj

    def __init__(self, domain: bc.BernsteinDomain, theta0: torch.Tensor,
                 gram: np.ndarray, threshold: float = SV_THRESHOLD):
        self.domain = domain
        self.theta0 = theta0.detach().double()
        self.theta0_np = self.theta0.cpu().numpy()
        self.whiten = Whitening(gram, threshold=threshold)
        self.base_energy = self.whiten.energy(self.theta0_np)
        if self.base_energy <= 0:
            raise ValueError('內點的擾動能量為零，reach 的分母沒有定義')
        self.fast = FastRadius(domain, self.theta0)
        self.calls = 0

    # ---- 求值 ----

    def radius(self, theta_dir: np.ndarray) -> Tuple[float, np.ndarray]:
        """單位化方向與它的邊界步長。單位化是為了讓二分的解析度與方向無關。"""
        d = np.asarray(theta_dir, dtype=np.float64)
        n = float(np.linalg.norm(d))
        if n <= 0:
            raise ValueError('零方向沒有半徑')
        unit = d / n
        r = self.fast(unit)
        self.calls += 1
        return float(r), unit

    def reach_theta(self, theta_dir: np.ndarray) -> float:
        r, unit = self.radius(theta_dir)
        energy = math.sqrt(max(float(unit @ self.whiten.gram @ unit), 0.0))
        return r * energy / math.sqrt(self.base_energy)

    def reach_w(self, w: np.ndarray) -> float:
        """白化座標下的 `reach`。**搜尋不走這條**，它只用來驗「換座標不改變
        `reach`」（`tests/test_reachability.py::test_reach_identical_in_both_
        coordinates`）。搜尋一律用 `reach_theta`，理由見模組說明的第 1 點。"""
        return self.reach_theta(self.whiten.to_theta(w))

    def boundary_theta(self, theta_dir: np.ndarray, frac: float = 0.9999
                       ) -> np.ndarray:
        r, unit = self.radius(theta_dir)
        return self.theta0_np + frac * r * unit

    def binding(self, theta_dir: np.ndarray, frac: float = 0.9999) -> str:
        theta = self.boundary_theta(theta_dir, frac)
        slacks = self.domain.relative_slacks(torch.from_numpy(theta))
        return min(slacks, key=slacks.get)


class ReachGradient:
    """`∇_d reach(d)`，用隱函數定理穿過半徑。

    `r(d)` 是 `W(r, d) = 0` 的解，`W` 是七條限制裡最緊的那一條的相對違反量、
    在 `r` 上嚴格遞增（可行集沿射線是含 0 的區間），所以

        ∂r/∂d = − (∂W/∂d) / (∂W/∂r)

    兩個偏導由 autograd 在同一個 `W` 上取得，限制的算式因此與求根用的是
    同一組（`bc.linear_pieces` ＋ 同一個低頻錐），不是另外寫一份近似。
    接著 `reach = r·‖d‖_G/√base` 的梯度是

        ∇reach = [ ‖d‖_G · ∂r/∂d + r · G d / ‖d‖_G ] / √base

    為什麼需要它
    ────────────────────────────────────────────────────────────────
    `reach` 的最大值落在可行體的端點上，而端點在 250 維裡是很薄的脊。
    隨機爬山與「盒頂點」的 Frank–Wolfe 步都走不上去：前者實測只到已知
    `DARK` 值的 35%，後者會停在一個 reach 只有 0.0045 的不動點（盒是太鬆
    的鬆弛，頂點在其他六條限制上違反得很厲害，拉回邊界之後幾乎不動）。
    沿著這個梯度走則是直接爬那道脊。

    `W` 在「哪一條限制最緊」換手的地方不可微，那是測度零的集合；爬山的
    隨機步負責處理卡在折點上的情形。
    """

    def __init__(self, domain: bc.BernsteinDomain, theta0: torch.Tensor,
                 gram: np.ndarray, base_energy: float):
        self.domain = domain
        self.theta0 = theta0.detach().double()
        self.gram_t = torch.as_tensor(symmetrise(gram))
        self.base_energy = float(base_energy)

    def worst(self, t: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        theta = self.theta0 + t * d
        p = bc.linear_pieces(theta)
        terms = []
        for key, cap in bc.CAP_KEYS:
            terms.append((p[key].max() - cap) / cap)
            terms.append(-p[key].min() / cap)
        jac = p['jac_row'].max()
        for key in bc.JAC_ABS:
            jac = jac + p[key].abs().max()
        terms.append((jac - bc.JAC_CAP) / bc.JAC_CAP)
        hess = torch.zeros((), dtype=theta.dtype)
        for key, weight in bc.HESS_ABS:
            hess = hess + weight * p[key].abs().max()
        terms.append((hess - bc.HESS_CAP) / bc.HESS_CAP)
        cone = self.domain.cone
        quad = theta @ (cone.H @ theta)
        lhs = torch.sqrt(quad.clamp_min(0.0)) + cone.offset
        rhs = cone.kappa * (cone.g @ theta)
        terms.append((lhs - rhs) / cone.rhs_scale)
        return torch.stack(terms).max()

    def radius_grad(self, direction: np.ndarray, radius: float) -> np.ndarray:
        """`∂r/∂d` 本身，不乘能量項。

        `reach` 與一階增量 `gain` 兩個目標都要穿過同一個半徑，差別只在外層
        那一項（見 `GainGradient`）。把隱函數那一步單獨拿出來，兩邊用的就
        是同一份算式，不會有一邊改了另一邊沒改的情形。
        """
        d = torch.as_tensor(np.asarray(direction, dtype=np.float64)
                            ).requires_grad_(True)
        t = torch.tensor(float(radius), dtype=torch.float64,
                         requires_grad=True)
        gt, gd = torch.autograd.grad(self.worst(t, d), [t, d])
        slope = float(gt)
        if abs(slope) < 1e-30:
            raise ValueError('違反量對半徑的導數為零，隱函數定理不適用')
        return (-gd / slope).numpy()

    def __call__(self, direction: np.ndarray, radius: float) -> np.ndarray:
        dv = np.asarray(direction, dtype=np.float64)
        dr = self.radius_grad(dv, radius)
        gd_vec = self.gram_t.numpy() @ dv
        energy = math.sqrt(max(float(dv @ gd_vec), 0.0))
        if energy <= 0:
            return dr * 0.0
        return (energy * dr + float(radius) * gd_vec / energy) / math.sqrt(
            self.base_energy)


class GainGradient:
    """一階可達增量 `q(d) = r(d)·gᵀd` 的解析梯度，`g = ∇_θJ` 是存檔的梯度。

        ∇q(d) = r(d)·g + (gᵀd)·∇r(d)

    `∇r` 就是 `ReachGradient.radius_grad`，所以這裡**不需要任何新的前向或
    反向**：`g` 是 `gradient` 那一支存下來的一次前向加反向的結果，半徑那一
    段的導數與 `reach` 共用同一份隱函數算式。

    為什麼要它
    ────────────────────────────────────────────────────────────────
    `q` 與 `reach` 的最大值不在同一個地方（`reach` 只看能量，`q` 還要看
    梯度的投影正負），所以 `ReachGradient` 不是 `q` 的梯度。沒有這一支時，
    `q` 的搜尋只剩頂點步與隨機爬山；幾何那一支有解析上升可用，兩邊的搜尋
    品質因此不在同一條線上，比較的是搜尋器而不是問題本身。

    尺度
    ────────────────────────────────────────────────────────────────
    `r` 沿射線是 −1 齊次（`r(cd) = r(d)/c`），所以 `q` 是 0 齊次：它只是
    射線的函數，`∇q` 自動落在球面的切平面上（`dᵀ∇q = r·gᵀd − (gᵀd)·r = 0`，
    因為 `dᵀ∇r = −r`）。呼叫端仍要傳**單位**方向與它自己的半徑，與
    `ReachGradient` 一致。
    """

    def __init__(self, radius_gradient: ReachGradient, grad_theta: np.ndarray):
        self.inner = radius_gradient
        self.g = np.asarray(grad_theta, dtype=np.float64)

    def __call__(self, direction: np.ndarray, radius: float) -> np.ndarray:
        d = np.asarray(direction, dtype=np.float64)
        n = float(np.linalg.norm(d))
        if n <= 0:
            raise ValueError('零方向沒有梯度')
        d = d / n
        return (float(radius) * self.g
                + float(self.g @ d) * self.inner.radius_grad(d, radius))


def gain_score(geom: 'ReachGeometry', grad_theta: np.ndarray
               ) -> Callable[[np.ndarray], float]:
    """`q(d) = r(d)·gᵀd` 的求值：一階可達增量 `ΔJ ≈ ⟨∇_θJ, r·d⟩`，沒有截斷。

    搜尋、測試與報表都從這裡取同一個定義，不各自寫一份閉包。
    """
    g = np.asarray(grad_theta, dtype=np.float64)

    def score(theta_dir: np.ndarray) -> float:
        r, unit = geom.radius(theta_dir)
        return float(r * (g @ unit))

    return score


def ascend_on_sphere(geom: 'ReachGeometry',
                     gradient: Callable[[np.ndarray, float], np.ndarray],
                     project: GProjector, start: np.ndarray, iters: int,
                     score: Optional[Callable[[np.ndarray], float]] = None,
                     step0: float = 0.3) -> Tuple[np.ndarray, float]:
    """沿解析梯度在球面上做帶回溯的上升。回傳 (方向, 目標值)。

    `gradient(unit, radius)` 回傳目標在該點的環境梯度，`score` 是對應的目標
    （不給就是 `reach`）。梯度先投影掉被排除的那幾個方向再走，所以整條軌跡
    都留在該類方向裡；只接受嚴格改善的候選，回傳值因此不低於起點的值。
    """
    measure = geom.reach_theta if score is None else score
    d = project(start)
    val = measure(d)
    step = step0
    for _ in range(iters):
        r, unit = geom.radius(d)
        g = project.residual(gradient(unit, r))
        n = float(np.linalg.norm(g))
        if n <= 0:
            break
        cand = project(d + step * g / n)
        v = measure(cand)
        if v > val:
            d, val = cand, v
            step = min(step * 1.4, 1.0)
        else:
            step *= 0.5
        if step < 1e-6:
            break
    return d, val


def gradient_ascent(geom: 'ReachGeometry', grad: ReachGradient,
                    project: GProjector, start: np.ndarray, iters: int,
                    step0: float = 0.3) -> Tuple[np.ndarray, float]:
    """`ascend_on_sphere` 在 `reach` 上的特例，保留原來的呼叫介面。"""
    return ascend_on_sphere(geom, grad, project, start, iters, step0=step0)


def spherical_search(score: Callable[[np.ndarray], float],
                     project: GProjector,
                     seeds: Sequence[Tuple[bool, np.ndarray]],
                     sampler: StepSampler, rng: np.random.Generator,
                     extra_random: int = 12, iters: int = 60,
                     sigma0: float = 0.5,
                     vertex: Optional['VertexMove'] = None,
                     vertex_rate: float = 0.3,
                     ascent: Optional[Callable] = None) -> Dict[str, object]:
    """多起點搜尋，變數是 250 維的 θ 方向。每個起點三段：

    1. **解析梯度上升**（`ascent`，可不給）：沿 `ReachGradient` 帶回溯地爬，
       爬的是那道脊本身。
    2. **頂點提議**（`vertex`）：跨到係數盒的角落，隨機步跨不過去。
    3. **(1+1) 自適應步長的隨機爬山**：處理前兩者卡住的折點。

    步的擾動取**單位方向**再乘 `sigma`，不是 `sigma · N(0, I)`：後者的長度
    是 `sigma·√250`，在球面上等於整個換一個隨機方向，前十幾步必定全部落空、
    只是在把 `sigma` 縮回來。

    **回傳的是下界。** 三個數字一起讀才知道它有多緊：`value`（最好的）、
    `within_1pct`（有幾個起點落在它的 1% 內）、`random_max`（只從隨機起點
    出發能走到多少）。最後一個是曲面崎嶇程度的量測，見 `theta_seeds`。
    """
    starts: List[Tuple[bool, np.ndarray]] = []
    dropped = 0
    for flag, seed in seeds:
        v = np.asarray(seed, dtype=np.float64)
        n0 = float(np.linalg.norm(v))
        if n0 <= 0 or float(np.linalg.norm(project.residual(v))) <= 1e-9 * n0:
            dropped += 1                       # 整條落在被投影掉的方向上
            continue
        starts.append((bool(flag), v))
    for _ in range(extra_random):
        starts.append((False, rng.standard_normal(bc.DIM)))
    if not starts:
        raise ValueError('沒有任何可用的起點')
    values = []
    structured_flags = []
    best_value = -np.inf
    best_dir = None
    for flag, start in starts:
        if ascent is not None:
            d, val = ascent(project, start)
        else:
            d = project(start)
            val = score(d)
        sigma = sigma0
        for _ in range(iters):
            cand = None
            if vertex is not None and rng.random() < vertex_rate:
                target = project.residual(vertex(d))
                if float(np.linalg.norm(target)) > 1e-12:
                    target = target / float(np.linalg.norm(target))
                    mix = target if rng.random() < 0.5 else (
                        d + rng.random() * (target - d))
                    if float(np.linalg.norm(project.residual(mix))) > 1e-12:
                        cand = project(mix)
            if cand is None:
                cand = project(d + sigma * sampler())
            v = score(cand)
            if v > val:
                d, val = cand, v
                sigma = min(sigma * 1.6, 1.0)
            else:
                sigma = max(sigma * 0.75, 1e-3)
        values.append(val)
        structured_flags.append(flag)
        if val > best_value:
            best_value, best_dir = val, d
    values = np.asarray(values, dtype=np.float64)
    flags = np.asarray(structured_flags, dtype=bool)
    top = np.sort(values)[::-1]
    return {
        'direction': best_dir,
        'value': float(best_value),
        'values': values,
        'second': float(top[1]) if len(top) > 1 else float('nan'),
        'within_1pct': int((values > 0.99 * best_value).sum()),
        'median': float(np.median(values)),
        'starts': int(len(starts)),
        'random_starts': int((~flags).sum()),
        'random_max': (float(values[~flags].max()) if (~flags).any()
                       else float('nan')),
        'dropped_seeds': int(dropped),
    }


def theta_seeds(whiten: Whitening, rng: np.random.Generator,
                axes: int = 10, structured: int = 8,
                plane: int = 24) -> List[Tuple[bool, np.ndarray]]:
    """起點（250 維的 θ 方向），回傳 `(是不是結構起點, 方向)`。

    四族
    ────────────────────────────────────────────────────────────────
    | 族 | 是什麼 | 為什麼放 |
    |---|---|---|
    | `G` 的主特徵方向 ± | 這張影像最容易被推動的顏色模式 | 分子（能量）最大的方向 |
    | **常數係數場的整個平面** | `a ≡ α`、`b ≡ β` 的所有 `(α, β)` 方向 | 分母最小的方向：第 3–6 條限制都是**係數差分**的界，常數場的差分全為零，這些限制一條都不綁 |
    | `smooth`（沿三軸線性） | 差分小但不為零 | 與 `bernstein_domain_check` 的族一致 |
    | 單一係數 | 差分最大 | 同上，另一個極端 |

    常數平面非放不可的理由是量出來的：`reach` 的曲面極度多峰，純隨機起點
    的爬山只走到常數平面最大值的 35–43%，解析梯度上升從隨機起點更只到 7%。
    **搜尋因此另外報「只從隨機起點出發能走到多少」**，那個比值就是這個
    曲面有多崎嶇的量測，不把結構起點的功勞算成搜尋器的能力。
    """
    seeds: List[Tuple[bool, np.ndarray]] = []
    for i in range(min(axes, whiten.dim)):
        vec = whiten.basis[:, i]
        seeds.append((True, vec.copy()))
        seeds.append((True, -vec))
    for j in range(plane):
        phi = 2.0 * math.pi * j / plane
        seeds.append((True, math.cos(phi) * DARK + math.sin(phi) * DESAT))
    idx = np.arange(bc.KNOTS) / bc.DEG - 0.5
    for _ in range(structured):
        w = rng.standard_normal(6)
        fa = (w[0] * idx[:, None, None] + w[1] * idx[None, :, None]
              + w[2] * idx[None, None, :])
        fb = (w[3] * idx[:, None, None] + w[4] * idx[None, :, None]
              + w[5] * idx[None, None, :])
        seeds.append((True, np.concatenate([fa.reshape(-1), fb.reshape(-1)])))
    for _ in range(structured):
        v = np.zeros(bc.DIM)
        v[int(rng.integers(bc.DIM))] = 1.0
        seeds.append((True, v))
    return [(flag, v) for flag, v in seeds if np.linalg.norm(v) > 1e-12]


def neighbourhood_seeds(centre: np.ndarray, sampler: StepSampler,
                        count: int,
                        sigmas: Sequence[float] = (0.03, 0.07, 0.15, 0.3, 0.6)
                        ) -> List[Tuple[bool, np.ndarray]]:
    """某個方向鄰域裡的起點，擾動幅度走幾何階梯。

    單一幅度會挑邊：太小全部落在同一個盆地裡、等於重跑一次局部搜尋，太大
    就退化成隨機起點。階梯讓同一批起點同時覆蓋「貼著 `centre` 的細擾動」與
    「跨得出這個盆地的粗擾動」，而擾動本身走 `StepSampler`，所以白化方向與
    等向方向的比例與主搜尋一致。
    """
    c = np.asarray(centre, dtype=np.float64)
    c = c / float(np.linalg.norm(c))
    out: List[Tuple[bool, np.ndarray]] = []
    for i in range(int(count)):
        out.append((True, c + float(sigmas[i % len(sigmas)]) * sampler()))
    return out


# ────────────────────────────────────────────────────────────────────
# 三支搜尋的單調關係
# ────────────────────────────────────────────────────────────────────

#: 三支搜尋由窄到寬。`nondark_strict` 的可行方向集合是 `nondark` 的子集、
#: `nondark` 又是 `unrestricted` 的子集，所以三者的最大值必須遞增。
BRANCH_ORDER = ('nondark_strict', 'nondark', 'unrestricted')


def branch_projectors(gram: np.ndarray) -> Dict[str, GProjector]:
    """三支搜尋各自的投影子，鍵與 `BRANCH_ORDER` 相同。"""
    return {
        'nondark_strict': GProjector(gram, [DARK, DESAT]),
        'nondark': GProjector(gram, [DARK]),
        'unrestricted': GProjector(gram),
    }


def cross_seeds(project: GProjector, directions: Sequence[np.ndarray]
                ) -> List[Tuple[bool, np.ndarray]]:
    """把別支找到的最佳方向投影到這一支的可行方向集合，當成起點。

    為什麼非有不可
    ────────────────────────────────────────────────────────────────
    三支各自獨立爬山時，**子集那一支可能贏過母集**——實測 `nondark_strict`
    在兩張影像上找到比 `nondark` 更大的值。子集的最大值不可能比較大，所以
    那代表母集的搜尋漏掉了一個**已知**更好的候選。把別支的最佳方向投影進來
    當起點，而搜尋只接受嚴格改善，母集的回傳值因此不低於子集的值。

    投影後長度為零的（整條落在被投影掉的方向上）直接略過，不是錯誤。
    """
    seeds: List[Tuple[bool, np.ndarray]] = []
    for d in directions:
        v = np.asarray(d, dtype=np.float64)
        n = float(np.linalg.norm(v))
        if n <= 0:
            continue
        res = project.residual(v)
        m = float(np.linalg.norm(res))
        if m <= 1e-9 * n:
            continue
        seeds.append((True, res / m))
    return seeds


def refine_reach_monotone(geom: 'ReachGeometry',
                          priors: Dict[str, np.ndarray],
                          sampler: StepSampler, rng: np.random.Generator,
                          iters: int = 60, ascent_steps: int = 30,
                          extra_seeds: Sequence[Tuple[bool, np.ndarray]] = (),
                          ) -> Dict[str, Dict[str, object]]:
    """由窄到寬依序補搜三支，每一支都拿到目前所有支的最佳方向當起點。

    順序是 `BRANCH_ORDER`：`nondark_strict` 先跑，它的結果進 `nondark` 的
    起點，`nondark` 的結果再進 `unrestricted` 的起點。因為搜尋只接受嚴格
    改善，跑完就有 `strict ≤ nondark ≤ unrestricted`（差別只到投影的數值
    誤差，約 `1e-12` 相對量級）。

    `priors` 是既有搜尋各支的最佳方向。回傳每支的 `direction`、`value`、
    `prior_value` 與這一次用了幾個起點；合併取最佳，既有的值不會被新的
    覆蓋掉。
    """
    gram = geom.whiten.gram
    projectors = branch_projectors(gram)
    grad = ReachGradient(geom.domain, geom.theta0, gram, geom.base_energy)
    vertex = VertexMove(geom)
    best: Dict[str, Tuple[np.ndarray, float]] = {}
    for key in BRANCH_ORDER:
        v = np.asarray(priors[key], dtype=np.float64)
        v = v / float(np.linalg.norm(v))
        best[key] = (v, geom.reach_theta(v))
    out: Dict[str, Dict[str, object]] = {}
    for key in BRANCH_ORDER:
        project = projectors[key]
        seeds = cross_seeds(project, [best[k][0] for k in BRANCH_ORDER])
        seeds = seeds + [(bool(f), np.asarray(s, dtype=np.float64))
                         for f, s in extra_seeds]
        res = spherical_search(
            geom.reach_theta, project, seeds, sampler, rng,
            extra_random=0, iters=iters, vertex=vertex,
            ascent=lambda p, s: gradient_ascent(geom, grad, p, s,
                                                ascent_steps))
        prior_dir, prior_val = best[key]
        if float(res['value']) > prior_val:
            best[key] = (np.asarray(res['direction'], dtype=np.float64),
                         float(res['value']))
        out[key] = {
            'direction': best[key][0],
            'value': best[key][1],
            'prior_value': prior_val,
            'search_value': float(res['value']),
            'starts': int(res['starts']),
            'within_1pct': int(res['within_1pct']),
        }
    return out


# ────────────────────────────────────────────────────────────────────
# 方向的描述
# ────────────────────────────────────────────────────────────────────

def describe_direction(theta_dir: np.ndarray, x01: torch.Tensor,
                       gram: np.ndarray, top: int = 8) -> Dict[str, object]:
    """一個方向「長什麼樣」：哪些 Bernstein 格權重大、對應到色彩空間哪一塊。

    Bernstein 格 `(i,j,k)` 的基底 `B_iB_jB_k` 在 `(i,j,k)/4` 這個 RGB 點上
    取到最大值，所以格的索引除以 4 就是它管的顏色。另外報這個方向在**這張
    影像的像素上**造成的 `a`、`b` 變化，因為係數大不等於影像上用得到。
    """
    d = np.asarray(theta_dir, dtype=np.float64)
    d = d / np.linalg.norm(d)
    a_part, b_part = d[:bc.NCOEF], d[bc.NCOEF:]
    cells = []
    for block, part in (('a', a_part), ('b', b_part)):
        for flat in np.argsort(-np.abs(part))[:top]:
            i = int(flat) // (bc.KNOTS * bc.KNOTS)
            j = (int(flat) // bc.KNOTS) % bc.KNOTS
            k = int(flat) % bc.KNOTS
            cells.append({'block': block, 'ijk': [i, j, k],
                          'rgb': [round(i / bc.DEG, 2), round(j / bc.DEG, 2),
                                  round(k / bc.DEG, 2)],
                          'weight': round(float(part[flat]), 6)})
    with torch.no_grad():
        basis = bc.basis_of(x01.double())
        a_coeff, b_coeff = bc.split(torch.from_numpy(d))
        fa = bc.evaluate_field(a_coeff, basis).flatten()
        fb = bc.evaluate_field(b_coeff, basis).flatten()
        y = bc.luma(x01.double()).flatten()

    def stats(v):
        # 常數係數場的 `a`、`b` 在整張圖上是同一個數，與亮度的相關係數
        # 沒有定義（分母為零）。那正是 reach 最大的那一族，所以這裡回
        # `None` 而不是硬給一個 0，也不寫成 NaN（JSON 裡不合法）。
        std = float(v.std())
        corr = (None if std <= 0
                else round(float(np.corrcoef(v.numpy(), y.numpy())[0, 1]), 4))
        return {'mean': round(float(v.mean()), 6), 'std': round(std, 6),
                'min': round(float(v.min()), 6), 'max': round(float(v.max()), 6),
                'corr_luma': corr}

    return {
        'a_energy_share': round(float(a_part @ a_part), 6),
        'b_energy_share': round(float(b_part @ b_part), 6),
        'cos_euclid_dark': round(float(d @ DARK), 6),
        'cos_g_dark': round(float(_cos_g(d, DARK, gram)), 6),
        'cos_g_desat': round(float(_cos_g(d, DESAT, gram)), 6),
        'field_a': stats(fa),
        'field_b': stats(fb),
        'top_cells': cells,
    }


def _cos_g(a: np.ndarray, b: np.ndarray, gram: np.ndarray) -> float:
    gram = symmetrise(gram)
    ga = float(a @ gram @ a)
    gb = float(b @ gram @ b)
    if ga <= 0 or gb <= 0:
        return 0.0
    return float(a @ gram @ b) / math.sqrt(ga * gb)


# ────────────────────────────────────────────────────────────────────
# 低頻錐的存檔與還原
# ────────────────────────────────────────────────────────────────────

def cone_state(cone: bc.LowFrequencyCone) -> Dict[str, np.ndarray]:
    return {
        'cone_H': cone.H.cpu().numpy(),
        'cone_g': cone.g.cpu().numpy(),
        'cone_eps_q': np.asarray(cone.eps_q),
        'cone_kappa': np.asarray(cone.kappa),
        'cone_offset': np.asarray(cone.offset),
        'cone_rhs_scale': np.asarray(cone.rhs_scale),
    }


def restore_domain(state) -> bc.BernsteinDomain:
    """由存下來的 `H`、`g` 與三個純量還原一個等價的域。

    `LowFrequencyCone` 建構一次要把 `250 × 3HW` 的 float32 展開（512² 約
    786 MB、十幾秒），而它只用到 `H`、`g`、`offset`、`kappa`、`rhs_scale`
    這五樣。梯度那一支要在 GPU 節點上重跑半徑，沒有理由把影像再走一次。
    **還原的是資料不是邏輯**：限制的求值仍然走 `LowFrequencyCone` 與
    `BernsteinDomain` 自己的程式，`tests/test_reachability.py` 釘住還原後的
    `relative_slacks` 與 `max_radius` 與新建的完全一致。
    """
    cone = bc.LowFrequencyCone.__new__(bc.LowFrequencyCone)
    cone.H = torch.as_tensor(np.asarray(state['cone_H']), dtype=torch.float64)
    cone.g = torch.as_tensor(np.asarray(state['cone_g']), dtype=torch.float64)
    cone.eps_q = float(np.asarray(state['cone_eps_q']))
    cone.kappa = float(np.asarray(state['cone_kappa']))
    cone.offset = float(np.asarray(state['cone_offset']))
    cone.rhs_scale = float(np.asarray(state['cone_rhs_scale']))
    domain = bc.BernsteinDomain.__new__(bc.BernsteinDomain)
    domain.device = torch.device('cpu')
    domain.dtype = torch.float32
    domain.lowfreq = True
    domain.cone = cone
    return domain

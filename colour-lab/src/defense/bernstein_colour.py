"""250 維保色相色彩載體：四階三變數 Bernstein 的暗化場與去飽和場。

這個族是什麼
────────────────────────────────────────────────────────────────────
對正規化 RGB `c`，令 `Y = w·c`（Rec.709 亮度權重）：

    T_θ(c) = [1 − a_θ(c) − b_θ(c)]·c + b_θ(c)·Y·1

`a` 控制暗化、`b` 控制去飽和，兩者都是四階三變數 Bernstein 多項式

    a(c) = Σ_{i,j,k=0..4} A_ijk B_i⁴(R) B_j⁴(G) B_k⁴(B)

各 125 個係數，θ = (A, B) 共 250 維。三個通道乘的是**同一個**比例
`1 − a − b`、加的是**同一個**灰量 `b·Y`，與 `tone_desat_param.py` 同一個
形式，差別只在 `a`、`b` 由「最大通道 `m` 的一元多項式」換成「整個 RGB
立方體上的三變數多項式」——自由度由 4 變成 250，色相不變與不加飽和這兩條
性質則由形式本身保住，不靠限制。

六條硬限制
────────────────────────────────────────────────────────────────────
全部用 **Bernstein 係數界**（凸包性質：多項式的值落在其係數的凸包內），
因此對整個立方體成立，不是抽樣檢查。

| # | 限制 | 擋住什麼 |
|---|---|---|
| 1 | `0 ≤ a ≤ 0.25`、`0 ≤ b ≤ 0.1` | 不加飽和、不提高最大通道、不出色域 |
| 2 | 三通道共用乘數與中性灰加量 | 不能偏色（已內建於 `T` 的形式） |
| 3 | `0 ≤ a + c·∇a ≤ 0.25` | 明暗映射沿色相射線的斜率維持 0.75–1 |
| 4 | `0 ≤ b + c·∇b ≤ 0.1` | 去飽和量不沿亮度方向反轉或暴增 |
| 5 | `sup‖J_{T−I}‖_∞ ≤ 0.35` | 相近 RGB 不被拉成色塊、也不折疊成同色 |
| 6 | `sup max_i Σ_{jk}|∂_jk(T−I)_i| ≤ 1` | 排除窄色階內的急轉 |

第 5 條等價於 `0.65‖c−d‖_∞ ≤ ‖T(c)−T(d)‖_∞ ≤ 1.35‖c−d‖_∞`。

第 3 條的係數形式
────────────────────────────────────────────────────────────────────
`R·d/dR B_i^n = i·B_i^n − (i+1)·B_{i+1}^n`，逐軸相加之後

    (L A)_ijk = (1+i+j+k)·A_ijk − i·A_{i−1,j,k} − j·A_{i,j−1,k} − k·A_{i,j,k−1}

就是 `a + c·∇a` 在同一組基底下的係數（次數仍是 (4,4,4)，不需要升階）。
常數係數場的像是它自己，所以 `a ≡ α` 同時滿足第 1 與第 3 條。

第 5、6 條的充分條件
────────────────────────────────────────────────────────────────────
令 `u = a + b`。`(T−I)_i = −u·c_i + b·Y`，故

    ∂_j(T−I)_i = −u·δ_ij + b·w_j + (Y·∂_j b − c_i·∂_j u)

前兩項的列和恰為 `a + 2b(1−w_i)`（因 `u ≥ b ≥ b·w_i`），最壞的一列是
藍色那列（`w` 最小），係數 `κ = 2(1−0.0722) = 1.8556`；而 `a + κb` 是
`A + κB` 這組係數的多項式，其上界即係數的最大值。導數項用
`|c_i| ≤ 1`、`|Y| ≤ 1` 放大成 `Σ_j (sup|∂_j u| + sup|∂_j b|)`，一階導數的
上界同樣由差分係數給：`sup|∂_R u| ≤ 4·max|Δ_R U|`。第 6 條同法，二階導數的
係數乘子是對角 `4·3 = 12`、混合 `4·4 = 16`。

兩條都是 θ 的**凸**分段線性函數（絕對值與最大值的和），所以與第 1–4 條的
線性限制、以及下面的低頻限制合起來仍是凸域。**充分條件比原式保守**，
測試釘的是原式（隨機 θ × 隨機影像上實際量 Jacobian 與 Hessian）。

低頻限制進參數域
────────────────────────────────────────────────────────────────────
擾動對係數是線性的：`r_θ = RemoveDC(T_θ x − x) = F_x θ`（`A_ijk` 那一行是
`−φ_ijk(c)·c`，`B_ijk` 那一行是 `φ_ijk(c)·(Y·1 − c)`）。規格是「**量化後**的
非直流擾動至少 97% 能量落在 `f ≤ 1/8` cycle/pixel」。量化誤差 `e` 滿足
`‖e‖ ≤ ε_q = √(3HW)/(2·255)`，於是

    ‖high(r+e)‖ ≤ √(θᵀH_xθ) + ε_q,    ‖(r+e)_nonDC‖ ≥ g_xᵀθ − ε_q

（`g_x = F_xᵀĝ`，`ĝ` 是固定的單位參照方向，故 `g_xᵀθ ≤ ‖F_xθ‖`），代入
`‖high‖ ≤ √0.03·‖nonDC‖` 得充分條件

    √(θᵀH_xθ) + (1+√0.03)·ε_q ≤ √0.03·g_xᵀθ

這是一個二階錐限制，仍是凸的。**它把恆等映射排除在外**：`θ = 0` 時左邊是
正的、右邊是零。量化誤差是固定大小的高頻雜訊，擾動太小就一定被它淹掉，
這不是實作的取捨而是規格本身的內容。

前向參數化
────────────────────────────────────────────────────────────────────
`BernsteinColourParam` 不做「更新後鉗回」也不加平滑懲罰：優化變數 `z ∈ R²⁵⁰`
經「內點 + 徑向映射」

    θ(z) = θ₀ + (1 − m)·tanh(‖z‖)·r_max(ẑ)·ẑ

送進域內。`r_max` 沿射線對每一條限制求根（線性的有封閉解、分段線性與
二階錐的用二分，都是凸的所以可行集是區間），在 `torch.no_grad()` 下算，
梯度只走 `tanh(‖z‖)·ẑ` 那一段：**域是精確的，被近似的只有映射的
Jacobian**。

`m = INTERIOR_MARGIN` 這個固定的相對退讓不是容差放寬，而是「嚴格在內部」
這句話在 float32 下唯一成立的寫法。`tanh` 在 `‖z‖ ≳ 8.6` 時就**恰好**等於
1.0（float32 的 eps 是 1.2e−7），單靠 `tanh < 1` 給的餘裕在那裡是零；而
`r_max` 本身是二分到邊界的值，θ 只要落在半徑上，θ 的 float32 表示誤差、
以及二階差分把係數誤差放大 12–16 倍之後的求值誤差（實測最壞 1.5e−6，
相對於上界 1）就足以讓限制的值超過上界。退讓 `m` 使餘裕不低於 `m`，
比這兩項誤差大兩個數量級，同時遠小於 `max_radius` 的契約精度。

`θ₀` 由 `interior_point` 在常數係數場 `(a ≡ α, b ≡ β)` 上格點搜出，取各條
限制的相對餘裕的最小值最大者；常數場的第 3、4、6 條餘裕最大，而低頻限制
要求振幅夠大，兩者的交集就是 α 接近上界、β 由第 5 條 `α + 1.8556β ≤ 0.35`
決定的那一塊。
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import torch

#: Bernstein 次數與每軸節點數。
DEG = 4
KNOTS = DEG + 1
#: 單一係數場的大小與 θ 的維度。
NCOEF = KNOTS ** 3
DIM = 2 * NCOEF

#: 第 1 條：暗化與去飽和的上界。
A_CAP = 0.25
B_CAP = 0.10
#: 第 5、6 條的上界。
JAC_CAP = 0.35
HESS_CAP = 1.0

#: 徑向映射對邊界的相對退讓（見模組說明的「前向參數化」）。float32 下的
#: 表示誤差與二階差分的求值誤差合計不超過 ~2e−6（相對於上界），取 1e−4
#: 留兩個數量級的餘裕；對載體的振幅只少 0.01%。
INTERIOR_MARGIN = 1e-4

#: Rec.709 亮度權重。
LUMA = (0.2126, 0.7152, 0.0722)
#: 第 5 條列和裡去飽和項的係數 `2(1 − min w)`。
KAPPA = 2.0 * (1.0 - min(LUMA))

#: 低頻的徑向門檻與能量佔比規格（與 `src/metrics/perturbation_band.py` 同一條線）。
LOW_BAND = 0.125
LOW_SHARE = 0.97
HIGH_SHARE = 1.0 - LOW_SHARE


# ────────────────────────────────────────────────────────────────────
# 基底與濾鏡本體
# ────────────────────────────────────────────────────────────────────

def bernstein(t: torch.Tensor) -> torch.Tensor:
    """四階 Bernstein 基底。`t` 為任意形狀，回傳在最前面多一軸（長度 5）。"""
    one = 1.0 - t
    return torch.stack([
        one ** 4,
        4.0 * t * one ** 3,
        6.0 * t ** 2 * one ** 2,
        4.0 * t ** 3 * one,
        t ** 4,
    ], dim=0)


def split(theta) -> Tuple[torch.Tensor, torch.Tensor]:
    """θ → `(A, B)`，兩個 (5,5,5) 的係數場。"""
    th = theta if isinstance(theta, torch.Tensor) else torch.as_tensor(theta)
    if th.numel() != DIM:
        raise ValueError(f'θ 要有 {DIM} 個元素，拿到 {th.numel()}')
    return th[:NCOEF].reshape(KNOTS, KNOTS, KNOTS), th[NCOEF:].reshape(
        KNOTS, KNOTS, KNOTS)


def join(a_coeff: torch.Tensor, b_coeff: torch.Tensor) -> torch.Tensor:
    return torch.cat([a_coeff.reshape(-1), b_coeff.reshape(-1)])


def evaluate_field(coeff: torch.Tensor, basis: Sequence[torch.Tensor]
                   ) -> torch.Tensor:
    """`Σ_ijk C_ijk B_i(R)B_j(G)B_k(B)`。`basis` 是三個 (5,N,H,W) 的張量。

    逐軸收縮而不是一次 einsum：中間量最大只有 (25,N,H,W)，整張 512² 的圖
    不會一次展開成 125 個通道。
    """
    br, bg, bb = basis
    t1 = torch.einsum('ijk,inhw->jknhw', coeff, br)
    t2 = torch.einsum('jknhw,jnhw->knhw', t1, bg)
    return torch.einsum('knhw,knhw->nhw', t2, bb).unsqueeze(1)


def basis_of(x01: torch.Tensor) -> List[torch.Tensor]:
    """三個通道各自的 Bernstein 基底，形狀 (5,N,H,W)。"""
    return [bernstein(x01[:, i]) for i in range(3)]


def luma(x01: torch.Tensor) -> torch.Tensor:
    w = torch.tensor(LUMA, device=x01.device, dtype=x01.dtype).view(1, 3, 1, 1)
    return (x01 * w).sum(dim=1, keepdim=True)


def apply_filter(x01: torch.Tensor, theta, basis=None) -> torch.Tensor:
    """`T_θ` 本體。`x01` 為 (N,3,H,W)、[0,1]；`theta` 為長度 250 的序列或 tensor。

    `theta` 是 tensor 時梯度接得回去。`basis` 可由呼叫端預先算好重用——
    它只依賴影像，求解的每一步都重算是純粹的浪費。
    """
    th = (theta.to(device=x01.device, dtype=x01.dtype)
          if isinstance(theta, torch.Tensor)
          else torch.tensor([float(v) for v in theta], device=x01.device,
                            dtype=x01.dtype))
    a_coeff, b_coeff = split(th)
    bas = basis if basis is not None else basis_of(x01)
    a = evaluate_field(a_coeff, bas)
    b = evaluate_field(b_coeff, bas)
    return (1.0 - a - b) * x01 + b * luma(x01)


# ────────────────────────────────────────────────────────────────────
# 係數上的線性算子
# ────────────────────────────────────────────────────────────────────

def radial_operator(coeff: torch.Tensor) -> torch.Tensor:
    """`c·∇` 之後再加回自己：`f + c·∇f` 在同一組基底下的係數。"""
    idx = torch.arange(KNOTS, device=coeff.device, dtype=coeff.dtype)
    weight = (1.0 + idx.view(-1, 1, 1) + idx.view(1, -1, 1)
              + idx.view(1, 1, -1))
    out = weight * coeff
    for axis in range(3):
        shifted = torch.zeros_like(coeff)
        sl_dst = [slice(None)] * 3
        sl_src = [slice(None)] * 3
        sl_dst[axis] = slice(1, None)
        sl_src[axis] = slice(0, KNOTS - 1)
        shifted[tuple(sl_dst)] = coeff[tuple(sl_src)]
        shape = [1, 1, 1]
        shape[axis] = KNOTS
        out = out - idx.reshape(shape) * shifted
    return out


def forward_diff(coeff: torch.Tensor, axis: int) -> torch.Tensor:
    """沿 `axis` 的一階前向差分，長度少一。可疊用求二階差分。"""
    n = coeff.shape[axis] - 1
    return coeff.narrow(axis, 1, n) - coeff.narrow(axis, 0, n)


def linear_pieces(theta: torch.Tensor) -> Dict[str, torch.Tensor]:
    """所有進入限制的**線性齊次**泛函的值。

    齊次是關鍵：沿射線 `θ₀ + t·d` 的值等於 `pieces(θ₀) + t·pieces(d)`，
    半徑的求解因此只要各算一次。
    """
    a_coeff, b_coeff = split(theta)
    u = a_coeff + b_coeff
    out: Dict[str, torch.Tensor] = {
        'a': a_coeff.reshape(-1),
        'b': b_coeff.reshape(-1),
        'ra': radial_operator(a_coeff).reshape(-1),
        'rb': radial_operator(b_coeff).reshape(-1),
        'jac_row': (a_coeff + KAPPA * b_coeff).reshape(-1),
    }
    for axis, name in enumerate('RGB'):
        out[f'du_{name}'] = (DEG * forward_diff(u, axis)).reshape(-1)
        out[f'db_{name}'] = (DEG * forward_diff(b_coeff, axis)).reshape(-1)
    for axis, name in enumerate('RGB'):
        out[f'd2u_{name}{name}'] = (
            DEG * (DEG - 1)
            * forward_diff(forward_diff(u, axis), axis)).reshape(-1)
        out[f'd2b_{name}{name}'] = (
            DEG * (DEG - 1)
            * forward_diff(forward_diff(b_coeff, axis), axis)).reshape(-1)
    for i, ni in enumerate('RGB'):
        for j, nj in enumerate('RGB'):
            if i >= j:
                continue
            out[f'd2u_{ni}{nj}'] = (
                DEG * DEG
                * forward_diff(forward_diff(u, i), j)).reshape(-1)
            out[f'd2b_{ni}{nj}'] = (
                DEG * DEG
                * forward_diff(forward_diff(b_coeff, i), j)).reshape(-1)
    return out


#: 四道逐係數的上下界：鍵、上界。下界一律 0。
CAP_KEYS: Tuple[Tuple[str, float], ...] = (
    ('a', A_CAP), ('ra', A_CAP), ('b', B_CAP), ('rb', B_CAP))
#: 第 5 條裡以絕對值最大值進來的群，權重皆 1。
JAC_ABS: Tuple[str, ...] = tuple(f'{p}_{d}' for p in ('du', 'db')
                                 for d in 'RGB')
#: 第 6 條裡以絕對值最大值進來的群與權重。混合的二階項在 `Σ_{j,k}` 的有序
#: 對裡各出現兩次；一階項的係數 2 來自 `Σ_k w_k = 1` 那兩項。
HESS_ABS: Tuple[Tuple[str, float], ...] = tuple(
    [(f'd2{p}_{d}{d}', 1.0) for p in ('u', 'b') for d in 'RGB']
    + [(f'd2{p}_{a}{b}', 2.0) for p in ('u', 'b')
       for a, b in (('R', 'G'), ('R', 'B'), ('G', 'B'))]
    + [(f'{p}_{d}', 2.0) for p in ('du', 'db') for d in 'RGB'])


def _absmax(v):
    return float(abs(v).max()) if hasattr(v, 'dtype') and not isinstance(
        v, torch.Tensor) else float(v.abs().max())


def _jac_bound(p) -> float:
    total = float(p['jac_row'].max())
    for key in JAC_ABS:
        total += _absmax(p[key])
    return total


def _hess_bound(p) -> float:
    total = 0.0
    for key, weight in HESS_ABS:
        total += weight * _absmax(p[key])
    return total


def structural_violations(theta: torch.Tensor, tol: float = 1e-9
                          ) -> Dict[str, float]:
    """六條結構限制（充分條件版）的違反量，未違反的不列。"""
    p = linear_pieces(theta.detach())
    out: Dict[str, float] = {}
    for key, cap in (('a', A_CAP), ('ra', A_CAP), ('b', B_CAP), ('rb', B_CAP)):
        hi = float(p[key].max())
        lo = float(p[key].min())
        if hi > cap + tol:
            out[f'{key}_above'] = hi - cap
        if lo < -tol:
            out[f'{key}_below'] = -lo
    jac = float(_jac_bound(p))
    if jac > JAC_CAP + tol:
        out['jacobian'] = jac - JAC_CAP
    hess = float(_hess_bound(p))
    if hess > HESS_CAP + tol:
        out['hessian'] = hess - HESS_CAP
    return out


def structural_report(theta: torch.Tensor) -> Dict[str, float]:
    """六條限制的**當前值**（不是違反量），逐欄寫進 CSV 用。"""
    p = linear_pieces(theta.detach())
    return {
        'a_min': float(p['a'].min()), 'a_max': float(p['a'].max()),
        'b_min': float(p['b'].min()), 'b_max': float(p['b'].max()),
        'radial_a_min': float(p['ra'].min()), 'radial_a_max': float(p['ra'].max()),
        'radial_b_min': float(p['rb'].min()), 'radial_b_max': float(p['rb'].max()),
        'jacobian_bound': float(_jac_bound(p)),
        'hessian_bound': float(_hess_bound(p)),
    }


# ────────────────────────────────────────────────────────────────────
# 低頻限制：影像相關的二階錐
# ────────────────────────────────────────────────────────────────────

def high_pass(field: torch.Tensor, band: float = LOW_BAND) -> torch.Tensor:
    """留下徑向頻率 `> band` 的部分。`field` 為 (...,H,W)。

    直流本來就在 `f = 0`，因此不在高頻帶裡；低頻佔比的分母另外處理。
    """
    h, w = field.shape[-2], field.shape[-1]
    spec = torch.fft.fft2(field.to(torch.float32), dim=(-2, -1))
    fy = torch.fft.fftfreq(h, device=field.device).view(-1, 1)
    fx = torch.fft.fftfreq(w, device=field.device).view(1, -1)
    keep = (fy ** 2 + fx ** 2).sqrt() > band
    return torch.fft.ifft2(spec * keep, dim=(-2, -1)).real


def remove_dc(field: torch.Tensor) -> torch.Tensor:
    return field - field.mean(dim=(-2, -1), keepdim=True)


def _columns(x01: torch.Tensor, basis, index0: int, index1: int
             ) -> torch.Tensor:
    """`F_x` 的第 `index0:index1` 行，形狀 (k,3,H,W)。

    `A_ijk` 那一行是 `−φ_ijk·c`，`B_ijk` 那一行是 `φ_ijk·(Y·1 − c)`。
    """
    y = luma(x01)
    cols = []
    for idx in range(index0, index1):
        which = 0 if idx < NCOEF else 1
        flat = idx % NCOEF
        i, j, k = flat // (KNOTS * KNOTS), (flat // KNOTS) % KNOTS, flat % KNOTS
        phi = basis[0][i] * basis[1][j] * basis[2][k]        # (N,H,W)
        phi = phi.unsqueeze(1)                                # (N,1,H,W)
        cols.append((-phi * x01) if which == 0 else (phi * (y - x01)))
    return torch.cat(cols, dim=0)


def perturbation_gram(x01: torch.Tensor, chunk: int = 25) -> torch.Tensor:
    """`F_xᵀF_x`（行已去直流），250×250。

    有它就不必再碰影像：任何一組 θ 產生的擾動之間的內積都是
    `θ_mᵀ G θ_n`，有效維度因此可以在 250 維裡算完。
    """
    if x01.shape[0] != 1:
        raise ValueError('一次一張影像')
    h, w = x01.shape[-2], x01.shape[-1]
    x32 = x01.to(torch.float32)
    basis = basis_of(x32)
    cols = torch.empty((DIM, 3 * h * w), device=x01.device, dtype=torch.float32)
    for start in range(0, DIM, chunk):
        stop = min(start + chunk, DIM)
        block = _columns(x32, basis, start, stop)
        cols[start:stop] = remove_dc(block).reshape(stop - start, -1)
        del block
    gram = torch.zeros((DIM, DIM), device=x01.device, dtype=torch.float64)
    step = 3 * h * w // 4 + 1
    for start in range(0, cols.shape[1], step):
        piece = cols[:, start:start + step]
        gram += (piece @ piece.T).double()
    del cols
    return gram


class LowFrequencyCone:
    """`√(θᵀHθ) + (1+√0.03)ε_q ≤ √0.03·gᵀθ` 這一條，以及它的組裝。

    `H` 與 `g` 只依賴影像，建一次重用。建的時候會暫時佔住
    `250 × 3HW` 的 float32（512² 約 786 MB），算完即釋放。
    """

    def __init__(self, x01: torch.Tensor, theta_ref: torch.Tensor,
                 band: float = LOW_BAND, chunk: int = 25):
        if x01.shape[0] != 1:
            raise ValueError('低頻限制是逐張影像的，一次只收一張')
        device = x01.device
        h, w = x01.shape[-2], x01.shape[-1]
        self.eps_q = math.sqrt(3.0 * h * w) / (2.0 * 255.0)
        self.kappa = math.sqrt(HIGH_SHARE)
        self.offset = (1.0 + self.kappa) * self.eps_q

        basis = [b.to(torch.float32) for b in basis_of(x01.to(torch.float32))]
        x32 = x01.to(torch.float32)
        ref = remove_dc(apply_filter(x32, theta_ref.to(torch.float32)) - x32)
        norm = float(ref.flatten().norm())
        if norm <= 0:
            raise ValueError('參照方向的擾動為零，單位方向沒有定義')
        ghat = (ref / norm).reshape(-1)

        highs = torch.empty((DIM, 3 * h * w), device=device, dtype=torch.float32)
        gvec = torch.zeros(DIM, device=device, dtype=torch.float64)
        for start in range(0, DIM, chunk):
            stop = min(start + chunk, DIM)
            cols = _columns(x32, basis, start, stop)          # (k,3,H,W)
            flat = cols.reshape(stop - start, -1)
            gvec[start:stop] = (flat.double() @ ghat.double())
            highs[start:stop] = high_pass(cols, band).reshape(stop - start, -1)
            del cols, flat
        hmat = torch.zeros((DIM, DIM), device=device, dtype=torch.float64)
        step = 3 * h * w // 4 + 1
        for start in range(0, highs.shape[1], step):
            block = highs[:, start:start + step]
            hmat += (block @ block.T).double()
        del highs
        self.H = hmat
        self.g = gvec
        self.rhs_scale = max(float(self.kappa * (gvec @ theta_ref.double())),
                             1e-6)

    def terms(self, theta: torch.Tensor) -> Tuple[float, float]:
        """回傳 `(左邊, 右邊)`。左 ≤ 右 即可行。"""
        th = theta.detach().to(self.H.device).double()
        quad = float(th @ (self.H @ th))
        return (math.sqrt(max(quad, 0.0)) + self.offset,
                float(self.kappa * (self.g @ th)))

    def ray(self, theta0: torch.Tensor, direction: torch.Tensor):
        """沿射線的三個二次係數與線性右邊，供半徑求解重用。"""
        t0 = theta0.detach().to(self.H.device).double()
        d = direction.detach().to(self.H.device).double()
        h0 = self.H @ t0
        hd = self.H @ d
        return (float(t0 @ h0), float(t0 @ hd), float(d @ hd),
                float(self.kappa * (self.g @ t0)),
                float(self.kappa * (self.g @ d)))


# ────────────────────────────────────────────────────────────────────
# 凸域：違反量、內點、徑向半徑
# ────────────────────────────────────────────────────────────────────

class _NullCone:
    """不加低頻限制時的替身：永不綁住。

    存在的理由是測試與診斷——六條結構限制與影像無關，要釘它們不必先算
    `H_x`（512² 要 8 秒與 786 MB）。求解路徑一律帶真的錐，`lowfreq=False`
    是呼叫端明寫的選擇，不是預設。
    """

    offset = 0.0
    rhs_scale = 1.0

    def terms(self, theta):
        return (0.0, 1.0)

    def ray(self, theta0, direction):
        return (0.0, 0.0, 0.0, 1.0, 0.0)


class BernsteinDomain:
    """六條結構限制 ＋ 低頻二階錐組成的凸域。"""

    def __init__(self, x01: torch.Tensor, band: float = LOW_BAND,
                 reference: Optional[torch.Tensor] = None,
                 lowfreq: bool = True):
        self.device = x01.device
        self.dtype = torch.float32
        self.lowfreq = bool(lowfreq)
        if not self.lowfreq:
            self.cone = _NullCone()
            return
        ref = (reference if reference is not None
               else constant_theta(0.24, 0.04, device=x01.device))
        self.cone = LowFrequencyCone(x01, ref, band=band)

    # ---- 違反量 ----

    def relative_slacks(self, theta: torch.Tensor) -> Dict[str, float]:
        """每一條限制的**相對餘裕**（正數即可行，越大越有餘裕）。"""
        p = linear_pieces(theta.detach())
        out: Dict[str, float] = {}
        for key, cap in (('a', A_CAP), ('ra', A_CAP), ('b', B_CAP),
                         ('rb', B_CAP)):
            out[key] = min(cap - float(p[key].max()), float(p[key].min())) / cap
        out['jacobian'] = (JAC_CAP - float(_jac_bound(p))) / JAC_CAP
        out['hessian'] = (HESS_CAP - float(_hess_bound(p))) / HESS_CAP
        lhs, rhs = self.cone.terms(theta)
        out['lowfreq'] = (rhs - lhs) / self.cone.rhs_scale
        return out

    def feasible(self, theta: torch.Tensor, tol: float = 0.0) -> bool:
        return min(self.relative_slacks(theta).values()) > tol

    # ---- 沿射線的最大步長 ----

    def max_radius(self, theta0: torch.Tensor, direction: torch.Tensor,
                   iters: int = 40) -> float:
        """從可行的 `theta0` 沿 `direction` 走到邊界的步長。

        每一條限制沿射線都是 `t` 的凸函數，可行集因此是一個含 0 的區間，
        二分收斂。線性的那些其實有封閉解，但混在同一個 `Φ(t)` 裡一起二分
        程式短很多，而每次求值只是幾個 250 維的向量運算。
        """
        import numpy as np

        base = {k: v.detach().cpu().double().numpy()
                for k, v in linear_pieces(theta0.detach()).items()}
        rate = {k: v.detach().cpu().double().numpy()
                for k, v in linear_pieces(direction.detach()).items()}
        q0, q1, q2, r0, r1 = self.cone.ray(theta0, direction)
        scale = self.cone.rhs_scale
        offset = self.cone.offset

        def worst(t: float) -> float:
            p = {k: base[k] + t * rate[k] for k in base}
            v = -1e9
            for key, cap in CAP_KEYS:
                arr = p[key]
                v = max(v, (arr.max() - cap) / cap, -arr.min() / cap)
            jac = p['jac_row'].max()
            for key in JAC_ABS:
                jac += np.abs(p[key]).max()
            v = max(v, (jac - JAC_CAP) / JAC_CAP)
            hess = 0.0
            for key, weight in HESS_ABS:
                hess += weight * np.abs(p[key]).max()
            v = max(v, (hess - HESS_CAP) / HESS_CAP)
            lhs = math.sqrt(max(q0 + 2.0 * t * q1 + t * t * q2, 0.0)) + offset
            v = max(v, (lhs - (r0 + t * r1)) / scale)
            return float(v)

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


def constant_theta(alpha: float, beta: float, device=None,
                   dtype=torch.float32) -> torch.Tensor:
    """常數係數場 `a ≡ α`、`b ≡ β` 的 θ。"""
    out = torch.empty(DIM, device=device, dtype=dtype)
    out[:NCOEF] = float(alpha)
    out[NCOEF:] = float(beta)
    return out


def interior_point(domain: BernsteinDomain, device=None,
                   grid: int = 24) -> Tuple[torch.Tensor, Dict[str, float]]:
    """在常數係數場上格點搜一個嚴格內點：取最小相對餘裕最大的 `(α, β)`。

    只搜常數場的理由：第 3、4、6 條對常數場的餘裕最大（差分全為零），
    而低頻限制要的是振幅，兩者的取捨只剩 `(α, β)` 兩個數。搜到的點會與
    整批一起寫進報告，不是寫死的魔術數字。
    """
    best = None
    for ai in range(grid):
        alpha = A_CAP * (ai + 1) / (grid + 1)
        for bi in range(grid):
            beta = B_CAP * (bi + 1) / (grid + 1)
            theta = constant_theta(alpha, beta, device=device)
            slacks = domain.relative_slacks(theta)
            worst = min(slacks.values())
            if best is None or worst > best[0]:
                best = (worst, alpha, beta, slacks)
    worst, alpha, beta, slacks = best
    if worst <= 0:
        raise ValueError(
            f'常數係數場上沒有嚴格內點：最好的 (α,β)=({alpha:.4f},{beta:.4f}) '
            f'最小相對餘裕 {worst:+.4f}，各條 {slacks}')
    theta = constant_theta(alpha, beta, device=device)
    return theta, {'alpha': alpha, 'beta': beta, 'worst_slack': worst, **slacks}


# ────────────────────────────────────────────────────────────────────
# 載體
# ────────────────────────────────────────────────────────────────────

class BernsteinColourParam:
    """250 維族的載體封裝。可學參數是 `z ∈ R²⁵⁰`，θ 由徑向映射給出。

    介面與 `src/defense/tone_desat_param.py::ToneDesatParam`、
    `src/defense/color_param.py::ColorCurveParam` 相同，
    `src/defense/immunise.py::optimise_carrier` 接得上。

    `project()` 是空的，這不是遺漏：`theta()` 的值恆在域內，沒有東西要投影。
    """

    def __init__(self, x01: torch.Tensor, band: float = LOW_BAND,
                 init_scale: float = 0.01, seed: int = 0,
                 domain: Optional[BernsteinDomain] = None):
        self.amplitude = 1.0
        self.device = x01.device
        self.init_scale = float(init_scale)
        self.domain = domain if domain is not None else BernsteinDomain(
            x01, band=band)
        self.theta0, self.interior = interior_point(self.domain,
                                                    device=x01.device)
        self._basis = None
        self.z = self._init_z(seed)

    def _init_z(self, seed: int) -> torch.Tensor:
        """`z` 從一個微小的固定方向出發，不從 0。

        徑向映射在 `z = 0` 沒有定義方向（`ẑ` 不存在），從那裡起步的梯度要
        靠除以 0 的極限。改成 `‖z‖ = init_scale` 的隨機單位方向：θ 與 θ₀ 的
        距離只有邊界的 1%，實質仍是從內點出發，而方向是良定的。
        """
        gen = torch.Generator(device='cpu').manual_seed(int(seed))
        v = torch.randn(DIM, generator=gen).to(self.device)
        v = v / v.norm()
        return (self.init_scale * v).requires_grad_(True)

    # ---- 載體介面 ----

    def reset(self, x01: torch.Tensor, seed: int) -> None:
        self.device = x01.device
        self._basis = None
        self.z = self._init_z(seed)

    def theta(self) -> torch.Tensor:
        norm = self.z.norm()
        with torch.no_grad():
            unit = (self.z / norm.clamp_min(1e-12)).detach()
            radius = self.domain.max_radius(self.theta0, unit)
        scale = (radius * (1.0 - INTERIOR_MARGIN) * torch.tanh(norm)
                 / norm.clamp_min(1e-12))
        return self.theta0 + scale * self.z

    def render(self, x01: torch.Tensor) -> torch.Tensor:
        if self._basis is None or self._basis[0].shape[-2:] != x01.shape[-2:]:
            self._basis = basis_of(x01)
        return apply_filter(x01, self.theta(), basis=self._basis)

    def params(self) -> List[torch.Tensor]:
        return [self.z]

    def state_dict(self):
        return {'z': self.z.detach().clone(),
                'theta0': self.theta0.detach().clone()}

    def load_state_dict(self, state) -> None:
        self.theta0 = state['theta0'].detach().clone()
        self.z = state['z'].detach().clone().requires_grad_(True)

    @property
    def stages(self):
        return [self]

    def set_amplitude(self, a) -> None:
        """只接受「維持原樣」。

        `optimise_carrier` 取回最佳 checkpoint 之後會把各段幅度設回
        `base_amp`（本載體恆為 1.0），那是一個 no-op，要放行。其他值代表
        呼叫端想用整體縮放收斂到上限之內——這個族沒有可縮放的幅度段，
        θ 本身就是幅度，靜默忽略會讓那條退路無聲失效。
        """
        values = a if isinstance(a, (list, tuple)) else [a]
        if all(abs(float(v) - 1.0) < 1e-12 for v in values):
            return
        raise NotImplementedError(
            'BernsteinColourParam 沒有幅度段；要縮小解請沿 z 縮短半徑')

    def project(self) -> None:
        """前向參數化已保證在域內，沒有東西要投影。"""
        return

    # ---- 量測 ----

    def report(self) -> Dict[str, float]:
        theta = self.theta().detach()
        lhs, rhs = self.domain.cone.terms(theta)
        return {**structural_report(theta),
                'lowfreq_lhs': lhs, 'lowfreq_rhs': rhs,
                'lowfreq_slack': rhs - lhs,
                'z_norm': float(self.z.detach().norm()),
                'theta_shift': float((theta - self.theta0).norm())}

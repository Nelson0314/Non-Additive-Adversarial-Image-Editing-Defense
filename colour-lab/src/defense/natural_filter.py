"""封閉的自然濾鏡集合：求解變數只有**一個離散索引**。

與本專案其餘載體的差別
────────────────────────────────────────────────────────────────────
其餘載體（`color_param.ColorCurveParam`、`lab_offset_field`、`material_patch`）
都是連續參數加梯度下降，自然度靠**事後的門檻或正則項**去擋。本專案已經量過
三種門檻（NIQE、平均色差、CVaR 尾端）各自被鑽過的形狀，所以這一支換一個
構造：**不自然的解在參數空間裡根本表達不出來**，門檻因此無事可做，也不再需要
任何自然度指標。

濾鏡
────────────────────────────────────────────────────────────────────
對一個像素 ``c ∈ [0,1]³``：

    Y = 0.2126 R + 0.7152 G + 0.0722 B      （Rec.709 相對亮度）
    u = Y·1 + s (c − Y·1)                    （向灰軸收縮，s ≤ 1）
    m = max_k u_k                            （逐像素的通道極大值）
    T(c) = α · u · [ 1 + a(1−m) + b(1−m)(2m−1) ]

中括號裡是一個**逐像素的純量**，三個通道共用，所以 ``T(c)`` 與 ``u`` 同方向：
RGB 的色相方向被保留。``s ≤ 1`` 讓飽和度只能維持或下降。中括號是亮度曲線的
斜率調節，``a`` 給線性項、``b`` 給二次項，兩者都只作用在 ``1−m`` 上，即
**亮部固定、暗部可調**。

封閉集合
────────────────────────────────────────────────────────────────────
``α ∈ {0.85, 0.95}``、``a ∈ {−0.05, 0.05}``、``b ∈ {−0.05, 0.05}``、
``s ∈ {0.95, 1}``，共 16 個**完整濾鏡**。不插值、不混合、不微調，也沒有梯度
下降——求解端能動的只有「挑哪一個」這個離散索引。

值域為什麼不必夾
────────────────────────────────────────────────────────────────────
``u`` 的每個分量都落在 ``[0, 1]``（``s ≤ 1`` 時 ``u`` 是 ``c`` 與 ``Y`` 的
凸組合），中括號的最大值是 ``1 + |a| + |b| = 1.1``，而它只在 ``m → 0`` 時
達到；乘上 ``u_k ≤ m`` 與 ``α ≤ 0.95`` 之後，``T`` 的上界是
``max_m α·m·(1 + 0.1(1−m)²) = α ≤ 0.95``。所以**不會有夾到 0 或 1 的像素**，
色斑與死白在構造上不會出現。`tests/test_natural_filter.py::test_stays_in_gamut`
釘住這一條；本模組因此刻意**不 clamp**，越界要看得見而不是被吃掉。

亮度曲線的斜率
────────────────────────────────────────────────────────────────────
灰階輸入（``u = m·1``）下 ``T(m)/m = α[1 + a(1−m) + b(1−m)(2m−1)]``。
``m = 1`` 時中括號恰為 1，故該端點的增益就是 ``α``；``m = 0`` 時中括號是
``1 + a − b ∈ {0.9, 1.1}``。16 個濾鏡的增益因此全部落在
``[0.85×0.9, 0.95×1.1] = [0.765, 1.045]``，這是集合共用的斜率帶。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import torch

#: Rec.709 相對亮度權重。與 `src/defense/color_param.py` 的灰階換算同一組，
#: 兩邊用不同權重會讓「保留色相」在兩個模組裡指到不同的東西。
LUMA = (0.2126, 0.7152, 0.0722)

#: 封閉集合的四個軸。**改這四行就是換一個方法**，不是調參數。
ALPHA_VALUES: Tuple[float, ...] = (0.85, 0.95)
A_VALUES: Tuple[float, ...] = (-0.05, 0.05)
B_VALUES: Tuple[float, ...] = (-0.05, 0.05)
S_VALUES: Tuple[float, ...] = (0.95, 1.0)

#: 亮度增益的上下界，由上面四個軸推出來（見模組 docstring）。
SLOPE_FLOOR = min(ALPHA_VALUES) * (1.0 + min(A_VALUES) - max(B_VALUES))
SLOPE_CEIL = max(ALPHA_VALUES) * (1.0 + max(A_VALUES) - min(B_VALUES))


@dataclass(frozen=True)
class Filter:
    """一個完整濾鏡。四個欄位一起決定它，沒有可以單獨微調的部分。"""

    alpha: float
    a: float
    b: float
    s: float

    @property
    def name(self) -> str:
        """檔名與 CSV 的鍵。**參數值本身**，不是流水號——看到名字就知道是哪一個。"""
        return (f"alpha{self.alpha:g}_a{self.a:+g}"
                f"_b{self.b:+g}_s{self.s:g}").replace("+", "p").replace("-", "m")


def catalogue() -> List[Filter]:
    """16 個候選，順序固定（α 最慢、s 最快）。

    順序固定是為了讓 CSV 的列序在任何一次重跑裡都一樣；排名不看順序。
    """
    out = []
    for alpha in ALPHA_VALUES:
        for a in A_VALUES:
            for b in B_VALUES:
                for s in S_VALUES:
                    out.append(Filter(alpha, a, b, s))
    return out


def by_name(name: str) -> Filter:
    """名字查回濾鏡。查不到就拋錯，不回傳 `None`——拿 `None` 去套濾鏡會
    在很遠的地方才炸，而且中間那幾張圖已經寫出去了。"""
    table = {f.name: f for f in catalogue()}
    if name not in table:
        raise KeyError(f"{name!r} 不在封閉集合裡；有的是 {sorted(table)}")
    return table[name]


def luma(x: torch.Tensor) -> torch.Tensor:
    """(N,3,H,W) → (N,1,H,W) 的 Rec.709 相對亮度。"""
    if x.dim() != 4 or x.shape[1] != 3:
        raise ValueError(f"需要 (N,3,H,W) 的 RGB，收到 {tuple(x.shape)}")
    w = torch.tensor(LUMA, dtype=x.dtype, device=x.device).view(1, 3, 1, 1)
    return (x * w).sum(dim=1, keepdim=True)


def apply_filter(x: torch.Tensor, f: Filter) -> torch.Tensor:
    """把濾鏡套在 [0,1] 的 (N,3,H,W) 上。**不 clamp**，理由見模組 docstring。"""
    y = luma(x)
    u = y + f.s * (x - y)
    m = u.max(dim=1, keepdim=True).values
    gain = 1.0 + f.a * (1.0 - m) + f.b * (1.0 - m) * (2.0 * m - 1.0)
    return f.alpha * u * gain


def quantise8(x: torch.Tensor) -> torch.Tensor:
    """量化到 8 bit 再回到 [0,1]。

    三道資格檢查全部在量化**之後**量。理由：交付出去的是 PNG，未量化的浮點
    擾動裡有一部分在寫檔那一步就消失了，用浮點量到的低頻佔比與模糊殘存會比
    攻擊方真正拿到的那張圖樂觀。
    """
    return torch.round(x.clamp(0.0, 1.0) * 255.0) / 255.0


def gamut_excursion(y: torch.Tensor) -> float:
    """輸出越出 [0,1] 的最大幅度。構造上應該是 0。"""
    return float(torch.maximum(-y.min(), y.max() - 1.0).clamp(min=0.0))


# ---------------------------------------------------------------- 資格檢查

def radial_frequency(height: int, width: int, device=None,
                     dtype=torch.float32) -> torch.Tensor:
    """每個 `rfft2` 頻格的徑向頻率，單位 cycle/pixel。

    `fftfreq` 的單位本來就是 cycle/sample，兩軸各自取再合成半徑即可；
    **不要用格子索引當半徑**，那樣非方形影像的兩軸尺度不同，門檻會歪掉。
    """
    fy = torch.fft.fftfreq(height, d=1.0, device=device, dtype=dtype).view(-1, 1)
    fx = torch.fft.rfftfreq(width, d=1.0, device=device, dtype=dtype).view(1, -1)
    return torch.sqrt(fy ** 2 + fx ** 2)


def lowfreq_fraction(delta: torch.Tensor, cutoff: float = 0.125) -> float:
    """去掉直流之後，徑向頻率 ≤ `cutoff` 的頻譜能量佔比。

    直流分量是整張圖的平均位移，任何淨化都動不到它，把它算進來會讓每一個
    候選的低頻佔比都趨近 1，這道檢查就什麼都擋不住了。故**先扣掉 DC**
    （`rfft2` 的 (0,0) 格），分母也是扣掉之後的總能量。
    """
    if delta.dim() != 4:
        raise ValueError(f"需要 (N,3,H,W)，收到 {tuple(delta.shape)}")
    d = delta.to(torch.float32)
    spec = torch.fft.rfft2(d, norm="ortho")
    power = (spec.real ** 2 + spec.imag ** 2)
    power[..., 0, 0] = 0.0                      # 直流
    radius = radial_frequency(d.shape[-2], d.shape[-1], d.device)
    total = float(power.sum())
    if total <= 0.0:
        raise ValueError("擾動去掉直流之後全為零，低頻佔比沒有定義")
    return float(power[..., radius <= cutoff].sum()) / total


def remove_dc(delta: torch.Tensor) -> torch.Tensor:
    """逐通道扣掉空間平均。模糊殘存與低頻佔比用的是同一個「非直流」定義。"""
    return delta - delta.mean(dim=(-2, -1), keepdim=True)


def blur_retention(delta: torch.Tensor, sigma: float = 2.0) -> float:
    """blur σ 之後，**非直流**擾動的 L2 範數保留率。

    模糊是線性算子，所以 `B(T(x)) − B(x) = B(δ)`：直接對擾動做模糊，與對
    兩張圖各做一次再相減逐位元相同，省一次卷積。

    **這一項不設門檻**，照實記錄。
    """
    from src.purify.ops import gaussian_blur

    ac = remove_dc(delta)
    base = float(ac.norm())
    if base <= 0.0:
        raise ValueError("擾動去掉直流之後全為零，保留率沒有定義")
    blurred = remove_dc(gaussian_blur(delta, sigma))
    return float(blurred.norm()) / base


def qualify(x01: torch.Tensor, f: Filter, cutoff: float = 0.125,
            blur_sigma: float = 2.0) -> dict:
    """單一 (濾鏡, 影像) 的三道資格檢查 ＋ 產物的值域統計。

    回傳的是**數**，不含通過與否——門檻由呼叫端套，這樣改門檻不必重算。
    """
    from src.defense.color_amplitude import delta_e00

    y = apply_filter(x01, f)
    excursion = gamut_excursion(y)
    yq = quantise8(y)
    delta = yq - x01
    return {
        "lowfreq_fraction": lowfreq_fraction(delta, cutoff),
        "deltaE00": delta_e00(x01, yq),
        "blur_retention": blur_retention(delta, blur_sigma),
        "gamut_excursion": excursion,
        "delta_linf": float(delta.abs().max()),
        "delta_rms": float(delta.pow(2).mean().sqrt()),
        "out_min": float(yq.min()),
        "out_max": float(yq.max()),
        "clipped_fraction": float(((yq <= 0.0) | (yq >= 1.0)).float().mean()),
        "luma_mean_before": float(luma(x01).mean()),
        "luma_mean_after": float(luma(yq).mean()),
        "chroma_rms_before": float((x01 - luma(x01)).pow(2).mean().sqrt()),
        "chroma_rms_after": float((yq - luma(yq)).pow(2).mean().sqrt()),
    }


#: 兩道**有門檻**的資格。第三道（模糊殘存）刻意不在這裡。
LOWFREQ_FLOOR = 0.95
DELTAE_CAP = 16.0


def passes(row: dict, lowfreq_floor: float = LOWFREQ_FLOOR,
           deltae_cap: float = DELTAE_CAP) -> bool:
    """單列是否同時滿足低頻資格與失真上限。"""
    return (float(row["lowfreq_fraction"]) >= lowfreq_floor
            and float(row["deltaE00"]) <= deltae_cap)


def qualified_names(rows: Sequence[dict], images: Sequence[str],
                    lowfreq_floor: float = LOWFREQ_FLOOR,
                    deltae_cap: float = DELTAE_CAP) -> List[str]:
    """通過資格的濾鏡名。

    **一張影像不合格，整個濾鏡就出局**——候選集合是全資料集共用的，留下一個
    「只在六張上合格」的濾鏡，等於讓另外兩張用一個超出預算的防禦，而 CSV 上
    看不出來。
    """
    seen = {}
    for row in rows:
        seen.setdefault(row["filter"], {})[row["image"]] = passes(
            row, lowfreq_floor, deltae_cap)
    need = set(images)
    out = []
    for name in [f.name for f in catalogue()]:
        got = seen.get(name, {})
        if set(got) != need:
            raise ValueError(
                f"{name} 只量到 {sorted(got)}，資料集是 {sorted(need)}："
                "資格檢查沒跑完就不能決定候選集合")
        if all(got.values()):
            out.append(name)
    return out

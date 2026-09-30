"""AdvDrop（Duan et al., ICCV 2021）— 唯一另一個明確的「非加性頻域」方法。

為什麼需要它
────────────────────────────────────────────────────────────────────
本輪的對照組要回答三個「會不會只是……」：

| 問題 | 對照組 | 狀態 |
|---|---|---|
| 只是頻域加性？ | DCT-Shield | 已完成（FND-058／061） |
| 只是頻譜比較低頻？ | BlurGuard | 已實作（`blurguard.py`） |
| **只是「非加性頻域操作」這個類別本身？** | **AdvDrop** | 本檔 |
| 只是同失真的隨機相位？ | `phase_rand` | 既有 |

AdvDrop 與紋理重相位同屬非加性頻域，但**方向相反**：

| | AdvDrop | 紋理重相位 |
|---|---|---|
| 對能量做什麼 | **移除**（量化丟掉一部分） | **一點都不移除**，只重排相位 |
| 可逆嗎 | 不可逆 | `θ` 轉回去即還原 |
| 變數是什麼 | 每個區塊每個頻格的**量化步長** | 每個區塊每個頻格的**相位角** |
| 零點 | **沒有**——最小步長仍會丟資訊 | `θ=0` 逐位元等於原圖 |

它同時修正了本專案的新穎性主張：「非加性的頻域操作」不是首創
（FND-060 的來源 survey §1.7）。

出處
────────────────────────────────────────────────────────────────────
arXiv:2108.09034。程式碼 https://github.com/RjDuan/AdvDrop ，本檔逐行對應
`infod_sample.py` 的 `InfoDrop.forward` 與 `utils.phi_diff`
（2026-08-19 由 raw 檔核對）。

演算法（原始碼的實際運算）
────────────────────────────────────────────────────────────────────
1. 每個通道各有一張 **(區塊數, 8, 8) 的量化表**，初始值全部是 `q_size`。
   注意這是**逐區塊逐頻格**的，不是 JPEG 那種全圖共用一張 8×8 表。
2. 逐通道：切 8×8 區塊 → 減 128 → DCT → **除以量化表 → 軟四捨五入 →
   乘回量化表** → IDCT → 拼回。
3. 軟四捨五入 `phi_diff(x, α)`：

       α ← min(α, 2)
       s = 1/(1−α)
       k = log(2/α − 1)
       φ(x) = tanh((x − (⌊x⌋ + 0.5)) · k) · s
       x' = (φ(x) + 1)/2 + ⌊x⌋

   `α → 0` 時 `k → ∞`、`s → 1`，`tanh` 飽和到 ±1，`x' → round(x)`；
   `α` 大時是平滑的。α 由 **0.1 線性退火到 1e-20**，即「先軟後硬」。
4. 更新：**手寫的 sign 步、步長 1**，再夾回 `[5, q_size]`。

**軟四捨五入的實測精度（2026-08-19 本機量測）**：`α = 1e-20` 給出
`k ≈ 46.7`，`tanh` 只在 `|f − 0.5| ≫ 1/46.7` 才飽和。離中點 0.05 以外，與真正
的 `round` 最大差 **0.0091**；正中點附近最大差 **0.495**。**它逼近 `round`
但不等於 `round`**，報表不可寫成「硬量化」。

**`α ≥ 2` 會在正中點產生 nan**：`clamp(α, max=2)` 之後 `k = log(2/2 − 1)
= log(0) = −∞`，而中點的 `(f − 0.5) = 0`，`0 × (−∞) = nan`。那個 clamp 不是
安全防護，它正是 nan 的來源。本專案的 α 由 0.1 起始不會走到 2，故實務上碰不到。

**兩個必須寫下來的原始碼事實**：

- **`Adam` 物件建了但沒被用來更新**。程式建立 `optimizer = Adam(...)`，
  但只呼叫 `optimizer.zero_grad()`；真正的更新是
  `q = q.detach() − sign(q.grad)` 再 clamp。照抄 Adam 會量到不同的東西。
- **沒有 RGB→YCbCr 轉換**。變數命名成 `y`／`cb`／`cr`，餵進去的卻是
  `images[:,:,:,0..2]`，也就是 R、G、B 三個通道。本檔照原樣在 RGB 上做，
  並在此註明命名與實際不符。

定案超參數（`InfoDrop.__init__` 的簽章預設）
────────────────────────────────────────────────────────────────────
`block_size = 8`、`q_size = 10`、`factor_range = [5, q_size]`、`steps = 40`、
`alpha_range = [0.1, 1e-20]`、更新步長 1。影像值域 `[0,255]`。

DCT 的等價性
────────────────────────────────────────────────────────────────────
AdvDrop 的 `dct_8x8` 用 `outer([1/√2,1,…],[1/√2,1,…]) × 0.25` 當縮放，
即 `C(u)C(v)/4`——與 `jpeg_codec.dct_matrix` 是同一個正交歸一 DCT-II。
本檔直接沿用後者，因為它有「往返逐位元可逆」的測試（誤差 < 1e-12），
而複製一份可能與其逆不精確互逆的實作只會多一個無症狀的錯誤來源。
"""

from __future__ import annotations


from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import torch

from src.baselines.jpeg_codec import (
    block_dct, block_idct, dct_matrix, rgb_to_ycbcr, ycbcr_to_rgb,
)

PAPER_BLOCK = 8
PAPER_Q_SIZE = 10.0
PAPER_Q_MIN = 5.0
PAPER_STEPS = 40
PAPER_ALPHA_HI = 0.1
PAPER_ALPHA_LO = 1e-20
PAPER_STEP_SIZE = 1.0
CHANNELS = ("r", "g", "b")          # 原始碼命名為 y/cb/cr，實際是 RGB

# **色彩空間是 2026-08-21 新增的檢定變因，不是論文的旗標。**
# 官方 `infod_sample.py` 把變數命名成 y/cb/cr，餵進去的卻是 images[:,:,:,0..2]
# 也就是 R、G、B（08-19 逐行核對）。但論文 Figure 4 的管線圖與那組命名都指向
# YCbCr。這個差別在 JPEG 防禦那一欄可能是決定性的——JPEG 對色度做 4:2:0
# 下採樣，擾動若平均分布在 RGB 三通道，轉到 YCbCr 之後有一部分落在會被
# 下採樣抹掉的色度上；若本來就只在 Y 上，就躲得掉。
COLOR_SPACES = ("rgb", "ycbcr")

# ---- 論文與官方程式碼不一致的地方（2026-08-21 由論文全文核出）----
#
# 論文正文 §3.1 與式 (7)：`q_init = 1`、約束是 `‖q − q_init‖∞ < ε`，
# §4.3 掃 `ε ∈ {20, 60, 100}`，也就是量化表由 1 **往上長**到最多 101。
# 官方 `infod_sample.py` 的簽章預設卻是 `q_size = 10`、`factor_range = [5, 10]`、
# 初始值 `q_size`，即由 10 **往下走**到 5。兩者的可動區間差了一個數量級。
#
# 這不是移植錯誤，是那份 repo 的 demo 設定與論文實驗設定不同。故本檔同時支援：
#
#   程式碼模式（預設）  q ∈ [q_min, q_size]，初始 q_size —— 沿用 08-19 的逐行對照
#   論文模式            q ∈ [q_init, q_init + eps]，初始 q_init —— 重現 Table 1／2
#
# 兩者的更新規則相同（sign 下降、步長 1、夾取），差別只有初始值與夾取範圍。
# **報表必須寫明用的是哪一個**，兩者的數字不可混放同一欄。
PAPER_Q_INIT = 1.0                  # 論文 §3.1「We set q_init = 1」
PAPER_EPS_SWEEP = (20.0, 60.0, 100.0)   # 論文 §4.3 Table 1
PAPER_UNTARGETED_STEPS = 50         # 論文 §4.3「untargeted ... 50」
PAPER_TARGETED_STEPS = 500          # 同上，targeted


def phi_diff(x: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    """`utils.phi_diff`：以 `tanh` 平滑的四捨五入，銳利度由 `alpha` 控制。

    `alpha` 越小越接近真正的 `round`。`alpha ≥ 2` 會被夾到 2（原始碼同此），
    因為 `k = log(2/α − 1)` 在 `α = 2` 時是 `log(0) = −∞`。
    """
    a = torch.clamp(alpha, max=2.0)
    s = 1.0 / (1.0 - a)
    k = torch.log(2.0 / a - 1.0)
    fl = torch.floor(x)
    return (torch.tanh((x - (fl + 0.5)) * k) * s + 1.0) / 2.0 + fl


def alpha_at(step: int, steps: int, *, hi: float = PAPER_ALPHA_HI,
             lo: float = PAPER_ALPHA_LO) -> float:
    """第 `step` 步的軟化程度。原始碼是 `alpha += (lo − hi)/steps` 每步一次，
    即由 `hi` 線性走到 `hi + (lo − hi)·(steps−1)/steps`——**注意終點不是 `lo`**，
    最後一步之後才會到。本檔照原樣。"""
    if steps <= 0:
        raise ValueError(f"steps 必須為正，收到 {steps}")
    if not 0 <= step < steps:
        raise ValueError(f"step={step} 超出 [0, {steps})")
    return hi + (lo - hi) / steps * step


def quantize_drop(coef: torch.Tensor, q_table: torch.Tensor,
                  alpha: torch.Tensor, hard_round: bool = False) -> torch.Tensor:
    """`compression.quantize` ＋ `decompression.dequantize`：除、捨入、乘回。

    `hard_round=True` 用真正的 `torch.round`，**梯度處處為零**。這是論文
    §4.5 的消融（它報 5.00±0.98%）：量化表完全不會動，剩下的成功率全部來自
    `q_init` 本身造成的損傷。

    注意這與「把 alpha 設得很小」**不是同一件事**——alpha=1e-20 時
    `phi_diff` 的 k 約 46.7，逼近 round 但仍可微，梯度還在，攻擊照樣成立
    （本專案實測成功率 1.000）。要重現那個 5% 必須用真正的 round。
    """
    if hard_round:
        return torch.round(coef / q_table) * q_table
    return phi_diff(coef / q_table, alpha) * q_table


def _channel_roundtrip(plane255: torch.Tensor, q_table: torch.Tensor,
                       alpha: torch.Tensor, d: torch.Tensor,
                       hard_round: bool = False) -> torch.Tensor:
    """單通道的 切塊 → −128 → DCT → 量化丟資訊 → IDCT → 拼回。"""
    coef = block_dct(plane255 - 128.0, d)
    return block_idct(quantize_drop(coef, q_table, alpha, hard_round), d) + 128.0


def render_advdrop(x01: torch.Tensor, q_tables: Dict[str, torch.Tensor],
                   alpha: torch.Tensor, color: str = "rgb",
                   hard_round: bool = False) -> torch.Tensor:
    """完整的前向。輸入輸出皆 `[0,1]`；內部在 `[0,255]` 上工作。

    `color="ycbcr"` 是檢定用的變體：先轉 YCbCr 再逐通道量化，**不做 4:2:0
    下採樣**（AdvDrop 本來就沒有那一步，加了會變成別的方法）。
    """
    if x01.dim() != 4 or x01.shape[1] != 3:
        raise ValueError(f"需要 (N,3,H,W)，收到 {tuple(x01.shape)}")
    h, w = x01.shape[-2:]
    if h % PAPER_BLOCK or w % PAPER_BLOCK:
        raise ValueError(f"高寬必須是 {PAPER_BLOCK} 的倍數，收到 {h}×{w}")
    d = dct_matrix(x01.device, x01.dtype)
    x255 = x01 * 255.0
    if color == "ycbcr":
        x255 = rgb_to_ycbcr(x255)
    out = [
        _channel_roundtrip(x255[:, i:i + 1], q_tables[c], alpha, d, hard_round)
        for i, c in enumerate(CHANNELS)
    ]
    y = torch.cat(out, dim=1)
    if color == "ycbcr":
        y = ycbcr_to_rgb(y)
    return (y / 255.0).clamp(0.0, 1.0)


def init_q_tables(x01: torch.Tensor, q_size: float = PAPER_Q_SIZE
                  ) -> Dict[str, torch.Tensor]:
    """逐區塊逐頻格的量化表，全部初始化為 `q_size`。形狀 (N, hb, wb, 8, 8)。"""
    h, w = x01.shape[-2:]
    shape = (x01.shape[0], h // PAPER_BLOCK, w // PAPER_BLOCK,
             PAPER_BLOCK, PAPER_BLOCK)
    return {c: torch.full(shape, q_size, device=x01.device, dtype=x01.dtype,
                          requires_grad=True) for c in CHANNELS}


@dataclass(frozen=True)
class AdvDropSpec:
    name: str = "advdrop"
    q_size: float = PAPER_Q_SIZE
    q_min: float = PAPER_Q_MIN
    steps: int = PAPER_STEPS
    step_size: float = PAPER_STEP_SIZE
    alpha_hi: float = PAPER_ALPHA_HI
    alpha_lo: float = PAPER_ALPHA_LO
    # "rgb" = 官方程式碼實際做的；"ycbcr" = 變數命名與 Figure 4 指向的。
    # 見 COLOR_SPACES 的說明。**ycbcr 必須標 modified_from_paper。**
    color: str = "rgb"
    # 論文 §4.5 的消融：用真正的 round（梯度為零）。**不是論文的方法本身**，
    # 必須標 modified_from_paper。
    hard_round: bool = False
    # 論文模式：兩個同時給定時，可動區間變成 [q_init, q_init + eps]，
    # 初始值是 q_init（見檔頭「論文與官方程式碼不一致的地方」）。
    q_init: Optional[float] = None
    eps: Optional[float] = None
    modified_from_paper: bool = False
    modification_note: str = ""
    source: str = "arXiv:2108.09034；超參數取自 infod_sample.py 的簽章預設"

    @property
    def paper_mode(self) -> bool:
        return self.q_init is not None and self.eps is not None

    def bounds(self) -> tuple:
        """回傳 `(lo, hi, init)`。兩種模式的唯一差別就在這三個數。"""
        if self.paper_mode:
            return self.q_init, self.q_init + self.eps, self.q_init
        return self.q_min, self.q_size, self.q_size

    def __post_init__(self):
        if self.color not in COLOR_SPACES:
            raise ValueError(f"未知色彩空間 {self.color}；可用 {COLOR_SPACES}")
        if self.hard_round and not self.modified_from_paper:
            raise ValueError(
                f"{self.name}: hard_round 是 §4.5 的消融，不是論文的方法本身，"
                "必須標 modified_from_paper 並寫明")
        if self.color != "rgb" and not self.modified_from_paper:
            raise ValueError(
                f"{self.name}: color={self.color} 不是官方程式碼實際做的"
                "（它在 RGB 上量化），必須標 modified_from_paper 並寫明")
        if (self.q_init is None) != (self.eps is None):
            raise ValueError(
                f"{self.name}: q_init 與 eps 必須同時給定（論文模式）或同時省略"
                "（程式碼模式）；只給一個時可動區間沒有定義")
        if self.paper_mode:
            if self.eps <= 0:
                raise ValueError(f"{self.name}: eps={self.eps} 必須為正，"
                                 "否則量化表沒有可動的區間")
            return
        if self.q_min >= self.q_size:
            raise ValueError(
                f"q_min={self.q_min} 不小於 q_size={self.q_size}，"
                "量化表沒有可動的區間")
        if self.modified_from_paper and not self.modification_note:
            raise ValueError(f"{self.name} 標了 modified_from_paper 卻沒寫改了什麼")


SPEC_PAPER = AdvDropSpec()


@dataclass
class AdvDropResult:
    x_def: torch.Tensor
    spec: AdvDropSpec
    history: List[Dict] = field(default_factory=list)


def run_advdrop(
    sd,
    x01: torch.Tensor,
    spec: AdvDropSpec = SPEC_PAPER,
    *,
    loss_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    log_every: int = 0,
) -> AdvDropResult:
    """`InfoDrop.forward`，逐行對應，但**損失換成本專案的目標**。

    原文的損失是分類的交叉熵（未定向時取負號）——本專案沒有分類器，威脅模型
    是擴散編輯。故預設改用與 DCT-Shield 相同的 `mean(latent²)`：這是**明確的
    改寫**，`modification_note` 必須寫出來，報表上不可寫成「原文設定」。

    更新規則、量化表形狀、α 退火、夾取範圍全部照原文。
    """
    if loss_fn is None:
        def loss_fn(x):
            return sd.encode_image(x).pow(2).mean()

    lo, hi, init = spec.bounds()
    q = init_q_tables(x01, init)
    history: List[Dict] = []

    for i in range(spec.steps):
        a = torch.tensor(alpha_at(i, spec.steps, hi=spec.alpha_hi,
                                  lo=spec.alpha_lo),
                         device=x01.device, dtype=x01.dtype)
        for t in q.values():
            t.requires_grad_(True)
        x_adv = render_advdrop(x01, q, a, spec.color, spec.hard_round)
        loss = loss_fn(x_adv)
        grads = torch.autograd.grad(loss, [q[c] for c in CHANNELS],
                                    allow_unused=True, materialize_grads=True)
        with torch.no_grad():
            for c, g in zip(CHANNELS, grads):
                q[c] = (q[c] - spec.step_size * torch.sign(g)).clamp(
                    lo, hi).detach().requires_grad_(True)
        if log_every and (i % log_every == 0 or i == spec.steps - 1):
            history.append({"step": i, "loss": float(loss.detach()),
                            "alpha": float(a)})
            print(f"    [advdrop] step {i:3d} loss {float(loss.detach()):.4f}",
                  flush=True)

    with torch.no_grad():
        hard = torch.tensor(spec.alpha_lo, device=x01.device, dtype=x01.dtype)
        x_def = render_advdrop(x01, q, hard, spec.color, spec.hard_round).detach()
    return AdvDropResult(x_def, spec, history)


class AdvDropParam:
    """`param_pgd.Parameterization` 的實作 —— 消融用。

    `radius` 是量化表的**上界** `q_size`：越大代表可以丟掉越多資訊，失真越大。
    下界固定在 `q_min = 5`（原文的 `factor_range[0]`）。

    **本參數化沒有零點**：即使量化表全部壓到 `q_min`，量化仍會丟資訊，
    輸出不等於原圖。這與 DCT-Shield 的 JPEG 地板同型，與紋理重相位
    `θ=0` 逐位元還原**不同**，`fit_to_budget` 的可達性要另行確認。

    `alpha` 固定在 `alpha_lo`（硬四捨五入）：共用迴圈沒有退火的位置，
    而退火是原文最佳化過程的一部分、不是參數化的一部分。
    """

    name = "advdrop"

    def __init__(self, radius: float = PAPER_Q_SIZE, q_min: float = PAPER_Q_MIN,
                 alpha: float = PAPER_ALPHA_LO):
        self.radius = radius
        self.q_min = q_min
        self.alpha = alpha
        self.q: Optional[Dict[str, torch.Tensor]] = None

    def reset(self, x01: torch.Tensor, seed: int) -> None:
        self.q = init_q_tables(x01, self.radius)

    def render(self, x01: torch.Tensor) -> torch.Tensor:
        a = torch.tensor(self.alpha, device=x01.device, dtype=x01.dtype)
        return render_advdrop(x01, self.q, a)

    def params(self) -> List[torch.Tensor]:
        return [self.q[c] for c in CHANNELS]

    @torch.no_grad()
    def project(self) -> None:
        for c in CHANNELS:
            self.q[c].clamp_(self.q_min, self.radius)

    def set_radius(self, r: float) -> None:
        if r <= self.q_min:
            raise ValueError(
                f"radius={r} 不大於 q_min={self.q_min}；AdvDrop 的半徑是量化表"
                "的上界，必須高於下界才有可動區間")
        self.radius = r

"""DCT-Shield（Bala et al., ICCV 2025 Highlight）— 頻域加性 baseline。

**這是別人論文的方法，不是我們的。** 官方 repo `SamsungLabs/dct-shield` 在
2026-08-18 查證時是空的（GitHub API 回 `This repository is empty.`，零分支），
故本檔由論文與補充材料重寫。每一個超參數都標出處。

出處
────────────────────────────────────────────────────────────────────
arXiv:2504.17894／ICCV 2025。補充材料 Algorithm 1 是唯一完整的偽碼，本檔
逐行對應：

    δ ← 0                                   Algorithm 1 行 1
    α ← JPEG_E(x; Q_alg)                    行 2（式 7）
    for i = 0 … N−1:                        行 5
        η ← (1 − i/N)·γ                     行 6   ← 線性衰減到 0
        α'_c ← α_c + δ_c   (c ∈ C)          行 8
        x' ← JPEG_D(α'; Q_alg)              行 9（式 9）
        L ← ‖E(x')‖₂                        行 10（§4.2 末段）
        δ ← δ − sign(∇_δ L)·η               行 11
        δ ← clamp(δ, −ε, +ε)                行 12
        δ_c ← δ_c ⊙ M_c   （有遮罩時）        行 14

定案超參數（正文 §5.4）：`Q_alg = 0.95`、`ε = 1`、`γ = 0.1`、`N = 1000`、
512×512。inpainting 用 `Q_alg = 0.9`（§6.2）、JPEG 強健度圖用 `Q_alg = 0.85`
的 Y-only 變體（§6.3）。

**本專案沒有的步長排程**：`(1 − i/N)γ` 是線性衰減到 0，而 `pgd.py` 的
`STEP_SCHEDULES` 只有 `constant` 與 `linear_decay_1pct`。本檔自己實作，
不改動 `pgd.py`——那支骨幹是五篇像素加性 baseline 共用的，DCT-Shield 不是
像素加性，套進去會讓 `BaselineSpec` 的欄位語意失效。

四件必須寫進報表的事
────────────────────────────────────────────────────────────────────
1. **`ε ≥ 1` 是抗 JPEG 的必要條件，不是調參建議。** 論文 §4.2：擾動必須至少
   造成一個量化級的改變，否則攻擊方以相同品質重壓時會被四捨五入回原值。
   把 ε 降到 1 以下以對齊本專案的人眼門檻時，**該條件失效**，必須註明。

2. **原生 ε=1 在本專案的人眼門檻上偏大。** 2026-08-19 本機實測（七張平均、
   δ 撞滿 ±1 的隨機正負號、Q_alg=0.95）：LPIPS 0.4254、DISTS 0.0692、
   PSNR 28.56、L∞ 0.225。對照紋理重相位的人眼門檻是 LPIPS 0.1893、
   DISTS 0.0349。1000 步 × 步長線性衰減的累積量是 50，遠大於 ε=1，故實際
   PGD 幾乎必然也撞滿邊界，該估計貼近真實。

3. **δ=0 時輸出不是原圖**，而是 Q_alg 品質的 JPEG 壓縮圖（失真地板，
   `jpeg_codec.jpeg_roundtrip`）。與紋理重相位 θ=0 逐位等於原圖不同。

4. **免疫影像不能存成 JPEG。** δ 是連續值、加在整數係數上；存成 JPEG 會被
   重新四捨五入掉。一律存 PNG。

抗 JPEG 的保證是單向的
────────────────────────────────────────────────────────────────────
補充材料 D.4：`Q_alg = q` 產生的影像只在攻擊方壓縮品質 `q' ≥ q` 時有效。
預設的 `Q_alg = 0.95` 因此只擋得住品質 95 以上的壓縮。**要擋更重的壓縮就得
先把自己的圖壓糊**，這個取捨要在報表上寫出來。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import torch

from src.baselines.jpeg_codec import CHANNEL_NAMES, jpeg_decode, jpeg_encode

PAPER_DEFAULT_QUALITY = 0.95      # §5.4，編輯任務
PAPER_INPAINT_QUALITY = 0.90      # §6.2
PAPER_JPEG_FIG_QUALITY = 0.85     # §6.3，圖 6 的 Y-only 變體
PAPER_EPS = 1.0                   # §5.4
PAPER_GAMMA = 0.1                 # §5.4
PAPER_STEPS = 1000                # §5.4
PAPER_EPS_SWEEP = (0.8, 1.0, 1.2, 1.4)   # §6.1 圖 5 的取捨曲線


@dataclass(frozen=True)
class DCTShieldSpec:
    """一組 DCT-Shield 設定。`modified_from_paper` 為真者報表必須加註。"""

    name: str
    q_alg: float = PAPER_DEFAULT_QUALITY
    eps: float = PAPER_EPS
    gamma: float = PAPER_GAMMA
    steps: int = PAPER_STEPS
    channels: Tuple[str, ...] = CHANNEL_NAMES
    modified_from_paper: bool = False
    modification_note: str = ""
    source: str = ""

    def __post_init__(self):
        bad = set(self.channels) - set(CHANNEL_NAMES)
        if bad:
            raise ValueError(f"未知通道 {sorted(bad)}；可用的是 {CHANNEL_NAMES}")
        if not self.channels:
            raise ValueError("channels 不可為空——那樣 δ 沒有任何自由度")
        if self.modified_from_paper and not self.modification_note:
            raise ValueError(f"{self.name} 標了 modified_from_paper 卻沒寫改了什麼")
        if self.eps < 1.0 and not self.modified_from_paper:
            raise ValueError(
                f"{self.name}: eps={self.eps} < 1 會使論文 §4.2 的抗 JPEG 條件"
                "失效，必須標 modified_from_paper 並寫明")


SPEC_BASE = DCTShieldSpec(
    name="dct_shield",
    source="arXiv:2504.17894 §5.4：Q_alg=0.95、eps=1、gamma=0.1、1000 步、512²")

SPEC_Y = DCTShieldSpec(
    name="dct_shield_y",
    q_alg=PAPER_JPEG_FIG_QUALITY,
    channels=("Y",),
    source="arXiv:2504.17894 §4.3 與 §6.3：只擾動 Y 通道；圖 6 用 Q_alg=0.85")

REGISTRY: Dict[str, DCTShieldSpec] = {s.name: s for s in (SPEC_BASE, SPEC_Y)}


def make_latent_norm_loss(sd) -> Callable[[torch.Tensor], torch.Tensor]:
    """論文 §4.2 的目標：`L(δ) = ‖E(x')‖₂`。

    取 latent 全部元素的 L2 範數（不是均方、也不是平方），照論文寫法。
    這與本專案共用的 `encoder_target`（推向 `gray.png` 的 latent）不同——
    **跑 baseline 時必須用論文自己的損失**，否則量到的不是那篇。

    在 δ=0 處梯度非零（latent 範數不是極值），不受 FND-053 的零梯度陷阱
    影響，不需要 random start。
    """

    def loss(x01: torch.Tensor) -> torch.Tensor:
        return sd.encode_image(x01).flatten().norm(p=2)

    return loss


@dataclass
class DCTShieldResult:
    x_def: torch.Tensor
    spec: DCTShieldSpec
    history: List[Dict] = field(default_factory=list)


def _component_masks(mask: torch.Tensor,
                     alpha: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """把像素遮罩降到每個通道的「區塊 × 8×8」座標上。

    論文 Algorithm 1 行 4 的 `component_wise_masks`，細節未載——這裡取
    「區塊內有任一像素被遮罩就整塊保留」，因為 DCT 係數的支撐是整個 8×8
    區塊，無法只遮住區塊的一部分。**這是本檔的推斷，不是論文的規定。**
    """
    import torch.nn.functional as F

    out: Dict[str, torch.Tensor] = {}
    m = mask.float()
    for name, coef in alpha.items():
        hb, wb = coef.shape[1], coef.shape[2]
        down = F.adaptive_max_pool2d(m, output_size=(hb, wb))
        out[name] = down.view(1, hb, wb, 1, 1).to(coef)
    return out


def run_dct_shield(
    sd,
    x01: torch.Tensor,
    spec: DCTShieldSpec = SPEC_BASE,
    *,
    loss_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    mask: Optional[torch.Tensor] = None,
    log_every: int = 0,
) -> DCTShieldResult:
    """補充材料 Algorithm 1，逐行照抄。

    `mask` 是論文的遮罩變體（只給 inpainting）。本專案的威脅模型只有
    img2img（2026-08-15 起），故實務上一律為 None；保留是因為它在被重現的
    演算法裡，靜默省略等於改了那篇論文。

    回傳的 `x_def` 是 `[0,1]`、已夾取。**存檔一律用 PNG**。
    """
    if loss_fn is None:
        loss_fn = make_latent_norm_loss(sd)

    alpha = {k: v.detach() for k, v in jpeg_encode(x01, spec.q_alg).items()}   # 行 2
    delta = {c: torch.zeros_like(alpha[c], requires_grad=True)                 # 行 1
             for c in spec.channels}
    masks = _component_masks(mask, alpha) if mask is not None else None        # 行 4
    history: List[Dict] = []

    for i in range(spec.steps):                                    # 行 5
        eta = (1.0 - i / spec.steps) * spec.gamma                  # 行 6
        coef = {k: (alpha[k] + delta[k] if k in delta else alpha[k])
                for k in alpha}                                    # 行 8
        x_adv = jpeg_decode(coef, spec.q_alg)                      # 行 9
        loss = loss_fn(x_adv)                                      # 行 10
        grads = torch.autograd.grad(loss, [delta[c] for c in spec.channels])
        with torch.no_grad():
            for c, g in zip(spec.channels, grads):
                delta[c].sub_(torch.sign(g) * eta)                 # 行 11
                delta[c].clamp_(-spec.eps, spec.eps)               # 行 12
                if masks is not None:
                    delta[c].mul_(masks[c])                        # 行 14
        if log_every and (i % log_every == 0 or i == spec.steps - 1):
            history.append({"step": i, "loss": float(loss.detach()), "eta": eta})
            print(f"    [{spec.name}] step {i:4d} loss {float(loss.detach()):.4f}",
                  flush=True)

    with torch.no_grad():
        coef = {k: (alpha[k] + delta[k] if k in delta else alpha[k]) for k in alpha}
        x_def = jpeg_decode(coef, spec.q_alg).detach()

    history.append({"saturated_fraction": {
        c: float((delta[c].abs() >= spec.eps - 1e-6).double().mean())
        for c in spec.channels}})
    return DCTShieldResult(x_def, spec, history)


class DCTShieldParam:
    """`param_pgd.Parameterization` 的實作 —— 消融用，**不是** baseline 用。

    存在理由與 `AdditiveParam`／`PhaseParam` 相同：把損失、更新規則、步數、
    種子、預算對齊程序全部固定，唯一的變因是參數化。

    **與 `run_dct_shield` 的差別必須寫在報表上**：

    | | `run_dct_shield` | `DCTShieldParam` ＋ `run_param_pgd` |
    |---|---|---|
    | 損失 | `‖E(x')‖₂`（論文） | 本專案共用的 `encoder_target` |
    | 步長 | `(1−i/N)γ`，論文行 6 | `radius/(steps·saturate_at)`，固定 |
    | 步數 | 1000（論文） | 100（共用） |
    | 用途 | 重現該篇 | 與相位臂做同預算比較 |

    `radius` 即 ε。`set_radius` 讓 `fit_to_budget` 可以二分搜尋；**搜到 1
    以下時抗 JPEG 的保證失效**，呼叫端要標註。實測失真地板（δ=0）在
    Q_alg=0.95 時 DISTS 只有 0.0022，遠低於相位臂的 0.0349，故對齊可行。
    """

    name = "dct_shield"

    def __init__(self, q_alg: float = PAPER_DEFAULT_QUALITY,
                 radius: float = PAPER_EPS,
                 channels: Tuple[str, ...] = CHANNEL_NAMES):
        self.q_alg = q_alg
        self.radius = radius
        self.channels = channels
        self.alpha: Optional[Dict[str, torch.Tensor]] = None
        self.delta: Optional[Dict[str, torch.Tensor]] = None

    def reset(self, x01: torch.Tensor, seed: int) -> None:
        self.alpha = {k: v.detach()
                      for k, v in jpeg_encode(x01, self.q_alg).items()}
        self.delta = {c: torch.zeros_like(self.alpha[c], requires_grad=True)
                      for c in self.channels}

    def render(self, x01: torch.Tensor) -> torch.Tensor:
        coef = {k: (self.alpha[k] + self.delta[k] if k in self.delta
                    else self.alpha[k]) for k in self.alpha}
        return jpeg_decode(coef, self.q_alg)

    def params(self) -> List[torch.Tensor]:
        return [self.delta[c] for c in self.channels]

    @torch.no_grad()
    def project(self) -> None:
        for c in self.channels:
            self.delta[c].clamp_(-self.radius, self.radius)

    def set_radius(self, r: float) -> None:
        self.radius = r


class DCTShieldYParam(DCTShieldParam):
    """只擾動亮度通道的變體（論文 §4.3）。參數量 `O(HW)`。"""

    name = "dct_shield_y"

    def __init__(self, q_alg: float = PAPER_JPEG_FIG_QUALITY,
                 radius: float = PAPER_EPS):
        super().__init__(q_alg=q_alg, radius=radius, channels=("Y",))

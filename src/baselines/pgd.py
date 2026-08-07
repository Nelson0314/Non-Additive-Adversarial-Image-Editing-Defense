"""五篇 baseline 共用的 PGD 骨幹與 `BaselineSpec`。

設計原則（`docs/ARCH_2026-08-05.md` §4、`docs/CODE_2026-08-05.md` §1.4）

**換 baseline = 換一個損失函式，不是各寫一支腳本。** 但「共用骨幹」不等於
「硬套同一條更新式」：五篇的更新規則實際上不一樣（PhotoGuard-c 是歸一化
梯度，其餘四篇是 sign），投影也不一樣（PhotoGuard-c 是 L2 renorm，其餘是
L∞ 逐元素夾），隨機初始化更是三種都有。這些差異全部由 `BaselineSpec` 的
欄位帶著，骨幹依欄位分派——把它們硬統一成一種，量到的就不是那篇論文。

值域：每篇獨立，不共用一套
──────────────────────────────────────────────────────────────────────

本專案的張量介面是 `[0,1]`（`src/models/sd.py` 模組 docstring），而五篇
baseline 全部在 `[-1,1]` 上最佳化。`docs/SOURCE_AUDIT_2026-08-05.md` §10
指出：三篇的 eps 換算方式彼此不同，且**看起來都寫 `eps/255`**——

| 方法 | 程式碼的 eps 寫法 | `[0,1]` 的實際 eps |
|---|---|---|
| PhotoGuard-c | `maxnorm=16`（L2 半徑，非 L∞） | L2 8 |
| Mist | `epsilon/255.0 * (1-(-1))`，**有**乘 2 | 16/255 |
| DIA | `eps=0.05` 直接用 | 0.025 |
| AdvPaint | `eps=0.06` 直接用 | 0.03 |
| PromptFlare | `eps/=255`，**沒**乘 2 | 6/255 |

因此每個 spec 各自帶一個 `ValueRange` 實例，且 `eps`（投影用，論文值域）
與 `eps_pixel01`（`[0,1]` 等價值）兩個欄位都要寫出來，由 `__post_init__`
交叉檢查。任何一篇改了值域，只會影響那一篇。

`delta` 的值域：`run_pgd` 回傳 `[0,1]` 尺度的 delta，因為呼叫端（射線縮放、
失真量測、存檔）全部在 `[0,1]` 上工作。轉換發生在骨幹的頭尾各一次。
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import torch

# 允許的列舉值。字串而非 Enum，是為了讓 spec 的定義讀起來就是原始碼的樣子。
NORMS = ("linf", "l2")
UPDATE_RULES = ("sign", "normalized_grad")
STEP_SCHEDULES = ("constant", "linear_decay_1pct")
OBJECTIVES = ("minimize", "maximize")
INIT_RULES = ("none", "uniform_linf", "l1_ball")


@dataclass(frozen=True)
class ValueRange:
    """單篇論文最佳化所在的值域，以及本專案 `[0,1]` 介面的換算。

    `note` 必填且必須指出該篇的出處行號（前處理在哪裡把影像映到這個值域）。
    沒有出處的換算就是猜的，而猜錯不會有任何症狀——輸出仍是一張合理的
    防禦圖，只是強度差一倍。
    """

    lo: float
    hi: float
    note: str

    def __post_init__(self):
        if self.hi <= self.lo:
            raise ValueError(f"值域上界必須大於下界，收到 [{self.lo}, {self.hi}]")
        if not self.note.strip():
            raise ValueError("值域必須註明出處，見本模組 docstring")

    @property
    def scale(self) -> float:
        """`[0,1]` 的 1 個單位在本值域對應多少。"""
        return self.hi - self.lo

    def from01(self, x01: torch.Tensor) -> torch.Tensor:
        return x01 * self.scale + self.lo

    def to01(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.lo) / self.scale


@dataclass(frozen=True)
class BaselineSpec:
    """一篇 baseline 的完整可執行規格。

    每一欄都必須在 `docs/SOURCE_AUDIT_2026-08-05.md` 或 `docs/_audit_*.md`
    中有出處。**不得由程式猜測**：查不到的項目由 `prepare` 拋出
    `NotImplementedError` 並寫明缺什麼，不填一個看起來合理的值。

    `prepare(sd, x01, spec, **kw) -> ctx`
        建立與 δ 無關的常量（目標 latent、文字嵌入、注意力 hook…）。
    `loss_fn(sd, x_adv, ctx) -> Tensor`
        `x_adv` 在**該篇的值域**上，回傳一個純量。骨幹依 `objective`
        決定往哪個方向走。
    """

    name: str
    source: str                       # arXiv 編號 + repo commit
    value_range: ValueRange

    eps: float                        # 投影半徑，該篇值域
    eps_pixel01: float                # 同一個約束在 [0,1] 的等價值
    norm: str
    steps: int
    step_size: Optional[float]        # 該篇值域
    step_schedule: str
    update_rule: str
    objective: str
    init_rule: str
    grad_reps: int

    needs_target_image: bool
    needs_mask: bool

    modified_from_paper: bool
    modification_note: str

    prepare: Callable[..., Any]
    loss_fn: Callable[..., torch.Tensor]

    # 原始碼是否把梯度限制在遮罩**外**。
    #
    # 只有 PhotoGuard-c 為真：其 `attack_forward` 逐字為
    # `grad = grad * (1 - cur_mask)`，擾動只落在不會被重繪的區域。
    # **不可用 `needs_mask` 代替**——那一欄問的是「原作的攻擊需不需要遮罩」
    # （AdvPaint 與 PromptFlare 為真，本輪以全圖遮罩代入），與「梯度要不要
    # 被遮罩限制」是兩件事，兩者在本專案的五篇上取值恰好互補。
    grad_outside_mask: bool = False
    # 論文正文與原始碼的落差。報表在該列加註，避免把原始碼的行為寫成
    # 論文宣稱的設定（SOURCE_AUDIT §3.6 對 PhotoGuard-c 的要求）。
    discrepancy_note: str = ""
    # 其他固定超參數（rate、mode、loss_depth…），純記錄用。
    extras: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.modified_from_paper and not self.modification_note.strip():
            raise ValueError(
                f"{self.name}：modified_from_paper 為真時 modification_note 不得為空。"
                "改寫過的方法若不註記，報表會把它寫成原論文設定，"
                "而事後無法從結果反推改了什麼"
            )
        if not self.modified_from_paper and self.modification_note.strip():
            raise ValueError(
                f"{self.name}：未改寫卻填了 modification_note，兩者必須一致"
            )
        for value, allowed, label in (
            (self.norm, NORMS, "norm"),
            (self.update_rule, UPDATE_RULES, "update_rule"),
            (self.step_schedule, STEP_SCHEDULES, "step_schedule"),
            (self.objective, OBJECTIVES, "objective"),
            (self.init_rule, INIT_RULES, "init_rule"),
        ):
            if value not in allowed:
                raise ValueError(
                    f"{self.name}：{label}={value!r} 不在 {allowed} 之中"
                )
        if self.eps <= 0:
            raise ValueError(f"{self.name}：eps 必須為正，收到 {self.eps}")
        if self.steps <= 0:
            raise ValueError(f"{self.name}：steps 必須為正，收到 {self.steps}")
        if self.step_size is not None and self.step_size <= 0:
            raise ValueError(
                f"{self.name}：step_size 必須為正，收到 {self.step_size}"
            )
        if self.grad_reps <= 0:
            raise ValueError(
                f"{self.name}：grad_reps 必須為正，收到 {self.grad_reps}"
            )
        if not self.source.strip():
            raise ValueError(f"{self.name}：source 不得為空")
        # 值域換算的自我檢查。L∞ 與 L2 半徑都隨值域線性縮放，故同一式子成立。
        expect = self.eps / self.value_range.scale
        if abs(expect - self.eps_pixel01) > 1e-9:
            raise ValueError(
                f"{self.name}：eps_pixel01={self.eps_pixel01} 與 "
                f"eps/{self.value_range.scale}={expect} 不符。"
                "兩者不一致代表值域換算寫錯了，而這個錯誤在結果上沒有症狀"
            )

    @property
    def step_size_pixel01(self) -> Optional[float]:
        if self.step_size is None:
            return None
        return self.step_size / self.value_range.scale


# ---------------------------------------------------------------------------
# 初始化
# ---------------------------------------------------------------------------


def _l1_projection(x2: torch.Tensor, y2: torch.Tensor, eps1: float) -> torch.Tensor:
    """`L1_projection`：逐字取自 DIA 官方實作 `attack/DIA_PT.py:22-86`。

    回傳 delta 使得 `‖y2 + delta‖₁ ≤ eps1`。原始碼的 docstring 另稱同時保證
    `0 ≤ x2 + y2 + delta ≤ 1`，但 DIA 把它套用在 `[-1,1]` 的影像張量上
    （`DIA_PT.py:241-248`），故那個箱型約束在 DIA 的用法下並不成立。
    **此處照抄不改**：改成 `[-1,1]` 的箱型約束會得到另一個隨機初始化分佈，
    那就不是 DIA 了。函式出處為 Croce & Hein 的 AutoAttack L1 投影。
    """
    x = x2.clone().float().view(x2.shape[0], -1)
    y = y2.clone().float().view(y2.shape[0], -1)
    sigma = y.clone().sign()
    u = torch.min(1 - x - y, x + y)
    u = torch.min(torch.zeros_like(y), u)
    l = -torch.clone(y).abs()
    d = u.clone()

    bs, indbs = torch.sort(-torch.cat((u, l), 1), dim=1)
    bs2 = torch.cat((bs[:, 1:], torch.zeros(bs.shape[0], 1).to(bs.device)), 1)

    inu = 2 * (indbs < u.shape[1]).float() - 1
    size1 = inu.cumsum(dim=1)

    s1 = -u.sum(dim=1)

    c = eps1 - y.clone().abs().sum(dim=1)
    c5 = s1 + c < 0
    c2 = c5.nonzero().squeeze(1)

    s = s1.unsqueeze(-1) + torch.cumsum((bs2 - bs) * size1, dim=1)

    if c2.nelement != 0:
        lb = torch.zeros_like(c2).float()
        ub = torch.ones_like(lb) * (bs.shape[1] - 1)

        nitermax = torch.ceil(torch.log2(torch.tensor(bs.shape[1]).float()))
        counter2 = torch.zeros_like(lb).long()
        counter = 0

        while counter < nitermax:
            counter4 = torch.floor((lb + ub) / 2.0)
            counter2 = counter4.type(torch.LongTensor)

            c8 = s[c2, counter2] + c[c2] < 0
            ind3 = c8.nonzero().squeeze(1)
            ind32 = (~c8).nonzero().squeeze(1)
            if ind3.nelement != 0:
                lb[ind3] = counter4[ind3]
            if ind32.nelement != 0:
                ub[ind32] = counter4[ind32]

            counter += 1

        lb2 = lb.long()
        alpha = (-s[c2, lb2] - c[c2]) / size1[c2, lb2 + 1] + bs2[c2, lb2]
        d[c2] = -torch.min(torch.max(-u[c2], alpha.unsqueeze(-1)), -l[c2])

    return (sigma * d).view(x2.shape)


def initial_point(
    x_ref: torch.Tensor, spec: BaselineSpec, generator: Optional[torch.Generator]
) -> torch.Tensor:
    """依 spec 的 `init_rule` 產生 PGD 起點（該篇值域）。

    三篇有隨機初始化、兩篇沒有，且形式不同——這不是可以統一的細節：
    隨機起點會改變最終解落在球面的哪一側，逐位元不同。

    | 規則 | 誰 | 出處 |
    |---|---|---|
    | `none` | PhotoGuard-c、PromptFlare、DIA-R | `super_l2` 的 `X_adv = X.clone()`；`promptflare.py:71`；`DIA_R` 的 `norm: null` |
    | `uniform_linf` | Mist、AdvPaint | `Masked_PGD.py` 的 `rand_init=True`；`AdvPaint.py:80` 的 `X + (rand*2*eps - eps)` |
    | `l1_ball` | DIA-PT | `DIA_PT.py:241-248`，`norm: "L1"` |
    """
    if spec.init_rule == "none":
        return x_ref.clone()
    if spec.init_rule == "uniform_linf":
        noise = torch.rand(
            x_ref.shape, generator=generator, dtype=x_ref.dtype, device=x_ref.device
        )
        return x_ref + (noise * 2.0 * spec.eps - spec.eps)
    if spec.init_rule == "l1_ball":
        t = torch.randn(
            x_ref.shape, generator=generator, dtype=x_ref.dtype, device=x_ref.device
        )
        delta = _l1_projection(x_ref, t, spec.eps)
        return x_ref + t + delta
    raise ValueError(f"未知的 init_rule {spec.init_rule!r}")


# ---------------------------------------------------------------------------
# 更新、步長、投影
# ---------------------------------------------------------------------------


def update_direction(grad: torch.Tensor, spec: BaselineSpec) -> torch.Tensor:
    """把梯度轉成位移方向。

    `sign`：四篇共用的標準 sign-PGD。
    `normalized_grad`：**只有 PhotoGuard-c**。逐 batch 切片除以自身 L2 範數
    （`demo_complex_attack_inpainting.ipynb` cell 8）：

        grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=1).view(-1, 1, 1, 1)
        grad_normalized = grad / (grad_norm + 1e-10)

    兩者的幾何完全不同：sign 走的是超立方體的角，歸一化梯度走的是梯度
    本身的方向。SOURCE_AUDIT §3.6 的裁決正是「真正影響結果的是更新規則的
    幾何」，故此處不可互換。
    """
    if spec.update_rule == "sign":
        return grad.sign()
    if spec.update_rule == "normalized_grad":
        l = len(grad.shape) - 1
        grad_norm = torch.norm(
            grad.detach().reshape(grad.shape[0], -1), dim=1
        ).view(-1, *([1] * l))
        return grad.detach() / (grad_norm + 1e-10)
    raise ValueError(f"未知的 update_rule {spec.update_rule!r}")


def step_size_at(spec: BaselineSpec, iteration: int) -> float:
    """第 `iteration` 步的步長。

    `linear_decay_1pct` 只有 AdvPaint 用（`AdvPaint.py:192`）：

        actual_step_size = step_size - (step_size - step_size / 100) / iters * iter

    即由 `step_size` 線性降到其 1%。PhotoGuard-c 的 notebook 有同一行但
    **被註解掉**，實跑是定值（`_audit_promptflare_photoguard.md` §1.3）。
    """
    if spec.step_size is None:
        raise ValueError(f"{spec.name}：step_size 未給定，無法計算第 {iteration} 步")
    if spec.step_schedule == "constant":
        return spec.step_size
    if spec.step_schedule == "linear_decay_1pct":
        s = spec.step_size
        return s - (s - s / 100.0) / spec.steps * iteration
    raise ValueError(f"未知的 step_schedule {spec.step_schedule!r}")


def project(
    x_adv: torch.Tensor, x_ref: torch.Tensor, spec: BaselineSpec
) -> torch.Tensor:
    """把 x_adv 投影回以 x_ref 為心、半徑 eps 的球內，再夾回值域。

    `linf`：`torch.minimum(torch.maximum(x, ref - eps), ref + eps)`，
    四篇共用（AdvPaint、PromptFlare、DIA 逐字相同，Mist 的
    `batch_clamp` 等價）。

    `l2`：**只有 PhotoGuard-c**，`torch.renorm(d_x, p=2, dim=0, maxnorm=eps)`。
    約束作用在 `dim=0` 的每個切片展平後的單一 L2 範數上（batch=1 時即整張
    影像的 786,432 個元素），不是逐通道也不是逐像素
    （`_audit_promptflare_photoguard.md` 附錄 B.3 實測）。L2 球允許少數
    像素遠超過 L∞ 等價值，兩者不可互相換算。
    """
    if spec.norm == "linf":
        out = torch.minimum(
            torch.maximum(x_adv, x_ref - spec.eps), x_ref + spec.eps
        )
    elif spec.norm == "l2":
        d_x = x_adv - x_ref.detach()
        out = x_ref + torch.renorm(d_x, p=2, dim=0, maxnorm=spec.eps)
    else:
        raise ValueError(f"未知的 norm {spec.norm!r}")
    return out.clamp(spec.value_range.lo, spec.value_range.hi)


# ---------------------------------------------------------------------------
# 骨幹
# ---------------------------------------------------------------------------


@dataclass
class PGDResult:
    delta01: torch.Tensor          # [0,1] 尺度的擾動
    x_adv01: torch.Tensor          # [0,1] 尺度的防禦圖
    history: List[Dict]
    seconds: float = 0.0


def run_pgd(
    sd,
    x01: torch.Tensor,
    spec: BaselineSpec,
    *,
    seed: int = 0,
    log_every: int = 20,
    verbose: bool = True,
    grad_mask: Optional[torch.Tensor] = None,
    **kw,
) -> PGDResult:
    """共用 PGD 骨幹。換 baseline = 換 `spec`。

    流程與各篇原始碼的對應：

        x_adv ← init(x)                              # spec.init_rule
        for i in 0..steps-1:
            g ← mean over grad_reps of ∇ L(x_adv)    # spec.grad_reps
            x_adv ← x_adv ∓ step(i) · u(g)           # spec.objective / update_rule
            x_adv ← clamp(project(x_adv), lo, hi)    # spec.norm / value_range

    `objective` 的符號：`minimize` 走 `x − step·u(g)`（PhotoGuard-c、Mist、
    AdvPaint、PromptFlare 四篇的原始碼都是減號，其中 AdvPaint 與 Mist 的
    損失本身已帶負號），`maximize` 走 `x + step·u(g)`（DIA 的
    `grad_normalize` 是加號）。**不要把符號吸收進損失去統一方向**——那樣
    報表上的損失曲線就不是論文那個量了。

    梯度以 `torch.autograd.grad` 對 `x_adv` 取，每個 rep 重建一次計算圖。
    """
    if x01.dim() != 4:
        raise ValueError(f"影像張量必須是 (B,C,H,W)，收到 {tuple(x01.shape)}")

    import time

    vr = spec.value_range
    device = x01.device
    x01 = x01.detach()
    x_ref = vr.from01(x01)

    # 產生器固定在 CPU：CUDA 的 randn 逐版本可能不同，而 baseline 的可重現性
    # 是報表要宣稱的東西。產生後再搬到目標裝置。
    gen = torch.Generator(device="cpu").manual_seed(seed)
    x_init_cpu = initial_point(x_ref.cpu(), spec, gen)
    x_adv = x_init_cpu.to(device=device, dtype=x_ref.dtype)
    x_adv = x_adv.clamp(vr.lo, vr.hi)

    # `seed` 必須顯式傳給 prepare。六篇的 prepare 都宣告了這個參數（用於
    # 評測噪聲、目標 latent 的取樣等），但它是 run_pgd 的 keyword-only 參數，
    # 不在 `**kw` 內，先前因此永遠傳不到。
    #
    # 症狀完全不存在：PGD 照跑、輸出一張合理的防禦圖。但對三篇
    # `init_rule="none"` 的（photoguard_c、dia_r、promptflare），起點也不隨
    # seed 改變，於是換 seed 得到**逐位元相同**的結果——多 seed 實驗的
    # 樣本標準差恆為 0。而 `|mean| > sd` 在 sd=0 時對任何非零均值自動成立，
    # 那正是先驗實驗產生 24 格假陽性的同一個機制。
    ctx = spec.prepare(sd, x01, spec, seed=seed, **kw)

    history: List[Dict] = []
    t0 = time.perf_counter()
    sign = -1.0 if spec.objective == "minimize" else 1.0

    for it in range(spec.steps):
        grads = []
        losses = []
        for _ in range(spec.grad_reps):
            probe = x_adv.detach().requires_grad_(True)
            loss = spec.loss_fn(sd, probe, ctx)
            if loss.dim() != 0:
                raise ValueError(
                    f"{spec.name} 的損失必須是純量，收到 {tuple(loss.shape)}"
                )
            g = torch.autograd.grad(loss, probe)[0]
            grads.append(g.detach())
            losses.append(float(loss.detach()))
            del loss, g, probe
        grad = torch.stack(grads).mean(0)
        if grad_mask is not None:
            # PhotoGuard-c 的 `attack_forward` 逐字為 `grad = grad * (1 - cur_mask)`
            # ——擾動只落在**不會被重繪**的區域。那是該篇在 inpainting 下的
            # 忠實項，也是它唯一能起作用的地方：遮罩內的像素攻擊方會整片
            # 覆寫掉，往那裡放擾動等於把預算丟進去換不到東西。
            #
            # 施加在**平均後的梯度**上而非逐 rep：兩者對線性運算等價，但
            # 前者少乘 `grad_reps` 次。遮罩為 1 的地方梯度歸零，於是那些像素
            # 的 `x_adv` 恆等於起點，`project` 也不會把它們拉開。
            grad = grad * (1.0 - grad_mask.to(grad))

        s = step_size_at(spec, it)
        x_adv = x_adv.detach() + sign * s * update_direction(grad, spec)
        x_adv = project(x_adv, x_ref, spec)

        rec = {
            "step": it,
            "loss": sum(losses) / len(losses),
            "step_size": s,
            "delta_linf01": float((vr.to01(x_adv) - x01).abs().max()),
            "delta_l2": float((x_adv - x_ref).norm()),
        }
        history.append(rec)
        if verbose and (it % log_every == 0 or it == spec.steps - 1):
            print(
                f"  [{spec.name}] step {it:>4d}  L={rec['loss']:.6e}  "
                f"Linf01={rec['delta_linf01']:.4f}  L2={rec['delta_l2']:.3f}",
                flush=True,
            )
        del grads, grad

    if hasattr(ctx, "close"):
        ctx.close()

    x_adv01 = vr.to01(x_adv).detach()
    return PGDResult(
        delta01=(x_adv01 - x01).detach(),
        x_adv01=x_adv01,
        history=history,
        seconds=time.perf_counter() - t0,
    )

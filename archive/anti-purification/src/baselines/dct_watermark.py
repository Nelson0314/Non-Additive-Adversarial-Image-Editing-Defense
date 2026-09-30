"""DJSMA —— DCT 域的 JSMA 攻擊（Chen et al., The Imaging Science Journal 2026）。

出處
────────────────────────────────────────────────────────────────────
Yangcheng Chen, Jiayi Liu, Yingkai Huang, Xiaolong Liu.
*JPEG compression-resistant adversarial attack with invisible watermark
embedding*. The Imaging Science Journal, 2026.
doi:10.1080/13682199.2026.2644653（Received 2025-12-17, Accepted 2026-03-09）。
College of Computer and Information Sciences, Fujian Agriculture and Forestry
University。**無公開程式碼**；本檔逐條對照論文 Algorithm 1（DJSMA）與式
(7)–(9) 實作，PDF 由使用者提供（掃描版，2026-08-21 逐頁判讀）。

**2026-08-21 的更正**：本檔的第一版是照摘要寫的，把方法猜成「顯著度 top-k
＋ sign-PGD」。取得全文後確認**全錯**：真正的方法是**逐次只改一個係數 ±1
的貪婪 JSMA**，攻擊區域是**固定的反對角帶**而非由梯度挑選，且**是定向攻擊**。
第一版已整份重寫。

論文的方法（三個階段，本檔只做第二階段）
────────────────────────────────────────────────────────────────────
    階段一  浮水印嵌入   訊息經 RS 碼 ＋ 三元 STC 編碼，以 J-UNIWARD 的失真
                         成本嵌入 8×8 區塊的**第 6–8 條反對角**（E678）
    階段二  對抗擾動     DJSMA，改的是**第 3–5 條反對角**（E345）  ← 本檔
    階段三  浮水印萃取   由 E678 取回位元

**階段一與三沒有實作。** 本專案的威脅模型不需要復原任何訊息，而 J-UNIWARD
＋STC＋RS 是一整套隱寫工具鏈。故本檔量到的失真是相對**原圖**，論文的 Table
2／3 則是相對**已嵌浮水印的影像**（PSNR 37.81、SSIM 0.981），兩者的參照點
不同，比較時必須註明。

為什麼分成兩個頻帶（論文 §Watermark embedding）
────────────────────────────────────────────────────────────────────
- **第 6–8 條反對角**：中偏高頻，對 JPEG 量化相對穩健、對視覺影響小，
  適合放要能被讀回來的浮水印。
- **第 3–5 條反對角**：對深度網路的預測影響較強，同時比低頻不顯眼，
  適合放對抗擾動。

反對角的編號是 1-based：位置 (i, j) 落在第 `i + j + 1` 條。故
E345 是 `i + j ∈ {2, 3, 4}`，E678 是 `i + j ∈ {5, 6, 7}`。

Algorithm 1（DJSMA）逐行
────────────────────────────────────────────────────────────────────
    輸入：含前處理層的目標模型 F、量化後的 DCT 係數 X、攻擊區 E_adv、
          迭代上限 τ、l∞ 約束 μ、目標標籤 t

     2  mask   = E_adv
     3  change = 0                      每個位置被改過幾次
     5  for i = 0..τ:
     6      if argmax F_j(X_i) == t: 成功，break
    10      算 S+(X_i, t) 與 S-(X_i, t)          式 (8)(9)
    11-12   S+ *= mask ; S- *= mask
    13-14   p+ = argmax S+ ; p- = argmax S-
    15-19   取兩者較大的那個位置 p 與方向 sign
    20      X[p] += sign                          **一次只動一個係數 ±1**
    21      change[p] += 1
    22-23   if change[p] > μ: mask[p] = 0

`τ` 限制的是 l0（改了幾個係數），`μ` 限制的是 l∞（同一個位置最多改幾階）。

式 (8)(9) 的顯著圖（標準 JSMA，只是定義域換成量化係數）：

    A = ∂F_t/∂x_p                     目標類別的偏導
    B = Σ_{j: F_j(X) > F_t(X)} ∂F_j/∂x_p    **只累加信心高於目標類別者**

    S+[p] = 0                 若 A < 0 或 B > 0 或 p ∉ E_adv
          = A × |B|           否則
    S-[p] = 0                 若 A ≥ 0 或 B ≤ 0 或 p ∉ E_adv
          = |A| × B           否則

「只累加信心高於目標類別的類」是論文明寫的省算手法（§Adversarial attack in
DCT domain 末段、§Robustness and efficiency 再述一次）。

定案超參數（論文 §Experimental results）
────────────────────────────────────────────────────────────────────
`τ = 1500`、`μ = 1`、評測時再壓一次 JPEG `Q = 75`（另報 `Q = 85`）。
資料集 ImageNet，六個模型：Inception_V3、SqueezeNet、ResNet101、VGG19、
ShuffleNet_V2、ResNet152。單張執行時間約 0.8–1.1 秒（RTX 4070）。

**論文未載、本專案指定**：目標標籤 `t` 怎麼選（本檔提供 `least_likely` 與
`random` 兩種，預設 `least_likely`）、嵌入端的 JPEG 品質（`q_embed`，
論文只說評測時再壓 Q=75）。兩者都寫進報表。

移植到本專案的威脅模型
────────────────────────────────────────────────────────────────────
DJSMA 的顯著圖需要**類別 logits**，而擴散編輯防護沒有分類器。故本檔另提供
`saliency="grad"`：把 S± 換成「本專案共用損失對該係數的偏導」，其餘（一次
一個係數、±1、E345、τ／μ 的意義）完全不變。**那不是論文的方法**，用到時
`modified_from_paper` 必須為真。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import torch

from src.baselines.jpeg_codec import CHANNEL_NAMES, jpeg_decode, jpeg_encode

PAPER_TAU = 1500          # 迭代上限＝l0 界
PAPER_MU = 1              # 同一位置最多改幾階＝l∞ 界
PAPER_EVAL_QUALITY = 75   # 評測時的第二次壓縮（另報 85）
PAPER_ADV_DIAGONALS = (3, 4, 5)    # E345，1-based 反對角
PAPER_WM_DIAGONALS = (6, 7, 8)     # E678，本檔未實作浮水印，僅記錄


def diagonal_mask(diagonals: Tuple[int, ...], device, dtype) -> torch.Tensor:
    """(8, 8) 的反對角遮罩。編號 1-based：位置 (i, j) 屬於第 `i + j + 1` 條。"""
    bad = [d for d in diagonals if not 1 <= d <= 15]
    if bad:
        raise ValueError(f"反對角編號必須落在 [1, 15]，收到 {bad}")
    i = torch.arange(8, device=device).view(8, 1)
    j = torch.arange(8, device=device).view(1, 8)
    d = i + j + 1
    m = torch.zeros(8, 8, device=device, dtype=dtype)
    for k in diagonals:
        m = m + (d == k).to(dtype)
    return m


@dataclass(frozen=True)
class DJSMASpec:
    """一組 DJSMA 設定。標「論文」者不可改動，否則量到的不是那一篇。"""

    name: str = "dct_wm"
    tau: int = PAPER_TAU                      # 論文
    mu: int = PAPER_MU                        # 論文
    diagonals: Tuple[int, ...] = PAPER_ADV_DIAGONALS   # 論文
    channels: Tuple[str, ...] = CHANNEL_NAMES
    # **論文未載**：嵌入端的 JPEG 品質。論文只說評測時再壓 Q=75。
    q_embed: float = 0.75
    # **論文未載**：目標標籤怎麼選。
    target: str = "least_likely"              # least_likely | random
    target_seed: int = 0
    # "jsma" = 論文的式 (8)(9)，需要類別 logits；
    # "grad" = 本專案移植用，換成損失對係數的偏導。
    saliency: str = "jsma"
    modified_from_paper: bool = False
    modification_note: str = ""
    source: str = "The Imaging Science Journal 2026, doi:10.1080/13682199.2026.2644653"

    def __post_init__(self):
        if self.saliency not in ("jsma", "grad"):
            raise ValueError(f"未知的 saliency {self.saliency!r}；可用 jsma／grad")
        if self.saliency == "grad" and not self.modified_from_paper:
            raise ValueError(
                f"{self.name}: saliency='grad' 把論文的顯著圖換掉了（那篇需要"
                "類別 logits），必須標 modified_from_paper 並寫明")
        if self.target not in ("least_likely", "random"):
            raise ValueError(f"未知的 target {self.target!r}")
        if self.tau <= 0:
            raise ValueError(f"tau={self.tau} 必須為正")
        if self.mu <= 0:
            raise ValueError(f"mu={self.mu} 必須為正")
        if self.modified_from_paper and not self.modification_note:
            raise ValueError(f"{self.name} 標了 modified_from_paper 卻沒寫改了什麼")


@dataclass
class DJSMAResult:
    x_def: torch.Tensor
    spec: DJSMASpec
    history: List[Dict] = field(default_factory=list)
    changed: Optional[torch.Tensor] = None      # 每張圖改了幾個係數（l0）
    # 最佳化出來的 δ 本體。**不可由防禦圖反推**——解碼含夾取與 4:2:0 重取樣，
    # 色度通道的改動會被抹到禁區去（與 DCT-Shield 的 skip_dc 同一個坑）。
    delta: Dict[str, torch.Tensor] = field(default_factory=dict)


def _pick_target(logits: torch.Tensor, spec: DJSMASpec) -> torch.Tensor:
    """目標標籤。**論文未載選法**，兩種都提供並寫進報表。"""
    if spec.target == "least_likely":
        return logits.argmin(dim=1)
    g = torch.Generator().manual_seed(int(spec.target_seed))
    n, k = logits.shape
    return torch.randint(k, (n,), generator=g).to(logits.device)


def jsma_saliency(logits: torch.Tensor, coef: Dict[str, torch.Tensor],
                  t: torch.Tensor) -> Tuple[Dict, Dict]:
    """式 (8)(9)。回傳 `(S_plus, S_minus)`，鍵與 `coef` 相同。

    兩次反向傳播：一次給 `F_t`，一次給 `Σ_{j: F_j > F_t} F_j`。後者的集合
    **依當前的 logits 決定**，論文明寫只累加信心高於目標類別者。
    """
    ps = [coef[c] for c in coef]
    f_t = logits.gather(1, t[:, None]).sum()
    ga = torch.autograd.grad(f_t, ps, retain_graph=True)

    higher = (logits > logits.gather(1, t[:, None])).to(logits.dtype)
    f_rest = (logits * higher).sum()
    gb = torch.autograd.grad(f_rest, ps)

    sp, sm = {}, {}
    for k, a, b in zip(coef.keys(), ga, gb):
        sp[k] = torch.where((a < 0) | (b > 0),
                            torch.zeros_like(a), a * b.abs())
        sm[k] = torch.where((a >= 0) | (b <= 0),
                            torch.zeros_like(a), a.abs() * b)
    return sp, sm


def grad_saliency(loss: torch.Tensor, coef: Dict[str, torch.Tensor]
                  ) -> Tuple[Dict, Dict]:
    """移植用的替代顯著圖：`S+ = max(-g, 0)`、`S- = max(g, 0)`，g = ∂L/∂x_p。

    語意與式 (8)(9) 一致——「往哪個方向動這個係數，對目標最有幫助」——但
    目標由「讓分類器選 t」換成「讓本專案的損失下降」。**這不是論文的方法。**
    """
    ps = [coef[c] for c in coef]
    g = torch.autograd.grad(loss, ps)
    sp = {k: torch.clamp(-gi, min=0.0) for k, gi in zip(coef.keys(), g)}
    sm = {k: torch.clamp(gi, min=0.0) for k, gi in zip(coef.keys(), g)}
    return sp, sm


@torch.enable_grad()
def run_djsma(
    x01: torch.Tensor,
    spec: DJSMASpec = DJSMASpec(),
    *,
    logits_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    loss_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    log_every: int = 0,
) -> DJSMAResult:
    """Algorithm 1。一次只改一個係數 ±1，直到成功或跑滿 `tau`。

    `logits_fn` 給 `saliency="jsma"`；`loss_fn` 給 `saliency="grad"`。
    兩者都吃 `[0,1]` 的像素影像（前處理層已在內部套用）。

    批次處理時每張圖各自挑自己的位置、各自早停；已成功的圖被凍結，不再更動。
    """
    if spec.saliency == "jsma" and logits_fn is None:
        raise ValueError("saliency='jsma' 需要 logits_fn")
    if spec.saliency == "grad" and loss_fn is None:
        raise ValueError("saliency='grad' 需要 loss_fn")

    dev, dt = x01.device, x01.dtype
    base = {k: v.detach() for k, v in jpeg_encode(x01, spec.q_embed).items()}
    delta = {c: torch.zeros_like(base[c]) for c in spec.channels}
    dmask = diagonal_mask(spec.diagonals, dev, dt)          # (8,8)
    mask = {c: dmask.expand_as(base[c]).clone() for c in spec.channels}
    change = {c: torch.zeros_like(base[c]) for c in spec.channels}

    n = x01.shape[0]
    alive = torch.ones(n, dtype=torch.bool, device=dev)
    target = None
    history: List[Dict] = []

    def decode(d):
        """式 (3)–(6)：量化係數 → 空間影像。可微，`round` 走直通。"""
        coef = {k: (base[k] + d[k] if k in d else base[k]) for k in base}
        return jpeg_decode(coef, spec.q_embed)

    for it in range(spec.tau):
        if not bool(alive.any()):
            break
        cur = {c: delta[c].detach().clone().requires_grad_(True)
               for c in spec.channels}
        x_adv = decode(cur)

        if spec.saliency == "jsma":
            lg = logits_fn(x_adv)
            if target is None:
                target = _pick_target(lg.detach(), spec)
            done = lg.detach().argmax(dim=1) == target
            alive = alive & ~done
            if not bool(alive.any()):
                break
            sp, sm = jsma_saliency(lg, cur, target)
        else:
            sp, sm = grad_saliency(loss_fn(x_adv), cur)

        with torch.no_grad():
            # 每張圖各自取兩個方向的最大值，攤平成 (N, -1) 再比
            best = {}
            for c in spec.channels:
                a = (sp[c] * mask[c]).reshape(n, -1)
                b = (sm[c] * mask[c]).reshape(n, -1)
                best[c] = (a.max(dim=1), b.max(dim=1))
            for i in range(n):
                if not bool(alive[i]):
                    continue
                pick = None
                for c in spec.channels:
                    (va, ia), (vb, ib) = best[c]
                    for val, idx, sgn in ((float(va[i]), int(ia[i]), 1.0),
                                          (float(vb[i]), int(ib[i]), -1.0)):
                        if val > 0 and (pick is None or val > pick[0]):
                            pick = (val, c, idx, sgn)
                if pick is None:            # 這張圖已經沒有可動的位置
                    alive[i] = False
                    continue
                _, c, idx, sgn = pick
                flat_d = delta[c].reshape(n, -1)
                flat_ch = change[c].reshape(n, -1)
                flat_m = mask[c].reshape(n, -1)
                flat_d[i, idx] += sgn
                flat_ch[i, idx] += 1
                if flat_ch[i, idx] > spec.mu:
                    flat_m[i, idx] = 0.0
        if log_every and (it % log_every == 0 or it == spec.tau - 1):
            history.append({"iter": it, "alive": int(alive.sum())})
            print(f"    [{spec.name}] iter {it:5d} 還在攻擊 {int(alive.sum())}/{n}",
                  flush=True)

    with torch.no_grad():
        x_def = decode(delta).clamp(0.0, 1.0).detach()
        l0 = torch.stack([(delta[c].reshape(n, -1) != 0).sum(dim=1)
                          for c in spec.channels]).sum(dim=0)
    history.append({"l0_mean": float(l0.double().mean())})
    return DJSMAResult(x_def, spec, history, l0,
                       {c: delta[c].detach() for c in spec.channels})

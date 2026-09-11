"""無導數搜尋：直接在真正的判準上找顏色參數。

PGD 在這個載體上是繼承來的，不是為它選的。四到六個參數、可行集合不是盒子、
梯度一半在抽樣之間互相抵銷、而且真正的判準（跑完取樣之後指令有沒有完成、
人還認不認得出來）不可微——PGD 因此只能打可微的代理，而診斷量到那些代理在
整個可達集合上只變動 1–6%。

這裡改成 (1+lambda) 演化策略：每代由目前最好的點抽 lambda 個子代，逐點評估
真正的判準，取最好的當下一代。評估可以是任何回傳純量的函式，包含要跑一次
編輯的那種。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Sequence, Tuple

import torch


@dataclass
class Knob:
    """一個可搜尋的參數：名字、下界、上界。"""

    name: str
    lo: float
    hi: float

    def clip(self, v: float) -> float:
        return min(self.hi, max(self.lo, float(v)))

    @property
    def span(self) -> float:
        return float(self.hi - self.lo)


@dataclass
class SearchResult:
    best: Dict[str, float]
    best_score: float
    evaluations: int
    history: List[Tuple[int, Dict[str, float], float]] = field(default_factory=list)


def evolution_search(knobs: Sequence[Knob],
                     evaluate: Callable[[Dict[str, float]], float],
                     *, budget: int = 40, children: int = 4,
                     sigma0: float = 0.3, seed: int = 0,
                     start: Dict[str, float] = None) -> SearchResult:
    """(1+lambda)-ES，最小化 `evaluate`。回傳最好的點與完整的評估紀錄。

    `sigma0` 是各旋鈕跨度的比例。步幅依 1/5 成功率規則調整：子代勝出就放大，
    連續失敗就縮小。
    """
    if budget < 1:
        raise ValueError('budget 必須至少為 1')
    if children < 1:
        raise ValueError('children 必須至少為 1')
    if not knobs:
        raise ValueError('沒有可搜尋的旋鈕')
    gen = torch.Generator().manual_seed(int(seed))
    parent = {k.name: (k.clip(start[k.name]) if start and k.name in start
                       else 0.5 * (k.lo + k.hi)) for k in knobs}
    score = float(evaluate(parent))
    used = 1
    history = [(0, dict(parent), score)]
    sigma = {k.name: float(sigma0) * k.span for k in knobs}
    step = 0
    while used < budget:
        step += 1
        wins = 0
        trials = 0
        for _ in range(children):
            if used >= budget:
                break
            child = {}
            for k in knobs:
                noise = float(torch.randn(1, generator=gen)) * sigma[k.name]
                child[k.name] = k.clip(parent[k.name] + noise)
            s = float(evaluate(child))
            used += 1
            trials += 1
            history.append((step, dict(child), s))
            if s < score:
                parent, score = child, s
                wins += 1
        if trials:
            factor = 1.5 if wins / trials > 0.2 else 0.75
            for k in knobs:
                sigma[k.name] = max(1e-4 * k.span,
                                    min(k.span, sigma[k.name] * factor))
    return SearchResult(best=parent, best_score=score, evaluations=used,
                        history=history)


def apply_knobs(param, values: Dict[str, float], x01: torch.Tensor,
                seed: int = 0) -> None:
    """把搜尋到的旋鈕值套到參數化上。

    `rotation_deg` 與 `amplitude` 要在 `reset` 前後分別處理：`reset` 會依
    `rotation_deg` 重建起點矩陣，而 `amplitude` 是事後的投影。
    """
    if 'rotation_deg' in values:
        if not getattr(param, 'isometric', False):
            raise ValueError('rotation_deg 只在等距臂上有定義')
        param.rotation_deg = float(values['rotation_deg'])
    param.reset(x01, seed)
    if 'amplitude' in values:
        param.set_amplitude(min(1.0, max(0.0, float(values['amplitude']))))
    if 'delta_scale' in values:
        with torch.no_grad():
            param.delta.mul_(float(values['delta_scale']))
        param.project()


def search_parameter(param, x01: torch.Tensor, loss: Callable,
                     *, budget: int = 200, children: int = 6,
                     sigma0: float = 0.3, seed: int = 0) -> SearchResult:
    """在參數化自己的可學張量上做無導數搜尋，取代 `run_param_pgd`。

    每個候選點都經過 `param.project()`，所以評估到的一律是可行點；不需要梯度、
    不需要步長與動量這些從相位線繼承過來、單位不相通的超參數。
    """
    ps = param.params()
    if not ps:
        raise ValueError('這個參數化沒有可學參數')
    if len(ps) != 1:
        raise ValueError(f'目前只支援單一參數張量，收到 {len(ps)} 個')
    base = ps[0].detach().clone()
    radius = float(getattr(param, 'radius', 1.0))
    flat = base.reshape(-1)
    knobs = [Knob(f'p{i}', -radius, radius) for i in range(flat.numel())]

    def evaluate(values):
        with torch.no_grad():
            v = torch.tensor([values[f'p{i}'] for i in range(flat.numel())],
                             dtype=base.dtype, device=base.device)
            ps[0].copy_(v.reshape(base.shape))
        param.project()
        with torch.no_grad():
            return float(loss(param.render(x01)))

    start = {f'p{i}': float(flat[i]) for i in range(flat.numel())}
    out = evolution_search(knobs, evaluate, budget=budget, children=children,
                           sigma0=sigma0, seed=seed, start=start)
    evaluate(out.best)
    return out

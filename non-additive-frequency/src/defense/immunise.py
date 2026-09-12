"""在色差上限**之內**解顏色載體，而不是解完再把解縮小。

先前的做法與它的缺陷
────────────────────────────────────────────────────────────────────
舊版跑 150 步無約束 Adam，結束後用 `fit_caps` 把三段幅度同乘一個係數縮回上限。
那不是解受約束的問題：`runs/immunise_both_budgets/` 的 30 列裡 28 列貼住整圖
上限、**0 列**貼住羽化臉上限，而選中解的身分框色差只有 8.4 到 14.3——預算沒有
用完。原因是整體縮放只能收縮，不能把預算從整圖搬到臉上，而身分住在臉上。

現行做法
────────────────────────────────────────────────────────────────────
色差以 **augmented Lagrangian** 進入損失：

    L = score + Σ_i [ λ_i g_i + (ρ/2)·max(0, g_i)² ],    g_i = measure_i / cap_i − 1

`λ` 每隔幾步依 `λ_i ← max(0, λ_i + ρ g_i)` 更新。這與「把色差當成一項加權重」
不同：乘子會自己長到剛好讓約束成立，不需要人去調一個換算率，而且可行時懲罰
為零，不會拿分數去換多餘的色差。

可微的色差來自 `delta_e_torch`；**量測仍走 skimage**。每隔 `check_every` 步用
量測那一份在**量化後**的影像上驗一次可行性，可行且分數更好就存 checkpoint。
最後交付的是最佳可行 checkpoint；一次都沒有可行過才退回 `fit_caps` 的縮放。

段與段之間的預算分配交給梯度：每段有自己的 `delta`，可行域的壓力會把預算推到
分數最敏感的那一段。**不需要可學的 amplitude**——幅度與 `delta` 的大小是同一個
自由度，先前它之所以關鍵，只是因為收尾的整體縮放只動得了它。
"""
from __future__ import annotations

from typing import Callable, Dict, List, NamedTuple, Optional

import torch


class Cap(NamedTuple):
    """一道色差上限。`soft` 可微、進損失；`hard` 是量測路徑、決定可行性。"""

    name: str
    soft: Callable[[torch.Tensor], torch.Tensor]
    hard: Callable[[torch.Tensor], float]
    value: float


def quantise(x):
    """交付的是 PNG，所以可行性要在量化後的影像上檢查。"""
    return (x.detach().clamp(0, 1) * 255).round() / 255


def fit_caps(carrier, x01, caps, tries: int = 14) -> float:
    """把各段幅度同乘一個係數，二分到所有上限同時成立，回傳那個係數。

    只在受約束最佳化一次都沒有走到可行點時當退路；正常情況下回傳 1.0。
    """
    live = [c for c in (caps or []) if c.value and c.value > 0]
    if not live:
        return 1.0
    base = [getattr(s, 'amplitude', 1.0) for s in carrier.stages]

    def feasible(mult: float) -> bool:
        carrier.set_amplitude([b * mult for b in base])
        y = quantise(carrier.render(x01))
        return all(c.hard(y) <= c.value for c in live)

    if feasible(1.0):
        return 1.0
    lo, hi = 0.0, 1.0
    for _ in range(tries):
        mid = 0.5 * (lo + hi)
        if feasible(mid):
            lo = mid
        else:
            hi = mid
    feasible(lo)
    return lo


def cap_measures(carrier, x01, caps) -> Dict[str, float]:
    y = quantise(carrier.render(x01))
    return {c.name: float(c.hard(y)) for c in (caps or [])}


def cap_violations(carrier, x01, caps) -> int:
    m = cap_measures(carrier, x01, caps)
    return sum(1 for c in (caps or [])
               if c.value and c.value > 0 and m[c.name] > c.value)


def optimise_carrier(carrier, x01, objective, *, steps: int = 150,
                     lr: float = 0.05, caps: Optional[List[Cap]] = None,
                     rho: float = 10.0, lam_every: int = 5,
                     check_every: int = 10,
                     log_every: int = 0) -> Dict[str, float]:
    """受約束地最小化 `objective.score(carrier.render(x01))`。

    回傳起點與終點的分數、三個項的終值、可行 checkpoint 來自第幾步、乘子終值、
    以及可行性檢查次數。這些逐列寫進 CSV，解有沒有落在可行域是可查的。
    """
    caps = [c for c in (caps or []) if c.value and c.value > 0]
    params = [p for p in carrier.params() if p.requires_grad]
    if not params:
        raise ValueError('這個載體沒有可學參數')
    lam = {c.name: 0.0 for c in caps}

    def gaps(y) -> Dict[str, torch.Tensor]:
        return {c.name: c.soft(y) / c.value - 1.0 for c in caps}

    with torch.no_grad():
        start = float(objective.score(carrier.render(x01)))

    opt = torch.optim.Adam(params, lr=lr)
    best = None
    checks = 0
    for k in range(steps):
        opt.zero_grad(set_to_none=True)
        y = carrier.render(x01)
        score = objective.score(y)
        g = gaps(y)
        loss = score
        for c in caps:
            gi = g[c.name]
            loss = loss + lam[c.name] * gi + 0.5 * rho * gi.clamp_min(0.0) ** 2
        loss.backward()
        if not all(torch.isfinite(p.grad).all()
                   for p in params if p.grad is not None):
            raise FloatingPointError(f'第 {k} 步的梯度不是有限值；不要靜默跳過')
        opt.step()
        carrier.project()

        if caps and (k + 1) % lam_every == 0:
            with torch.no_grad():
                gv = gaps(carrier.render(x01))
                for c in caps:
                    lam[c.name] = max(0.0, lam[c.name] + rho * float(gv[c.name]))

        if (k + 1) % check_every == 0 or k == steps - 1:
            with torch.no_grad():
                checks += 1
                m = cap_measures(carrier, x01, caps)
                if all(m[c.name] <= c.value for c in caps):
                    s = float(objective.score(quantise(carrier.render(x01))))
                    if best is None or s < best[0]:
                        best = (s, k + 1, carrier.state_dict())

        if log_every and k % log_every == 0:
            with torch.no_grad():
                worst = max([float(v) for v in gaps(carrier.render(x01)).values()]
                            or [0.0])
                print(f'    第 {k} 步 score {float(score):+.4f} '
                      f'最大違反 {worst:+.3f} '
                      f'λ {[round(v, 2) for v in lam.values()]}', flush=True)

    shrink = 1.0
    with torch.no_grad():
        unprojected = float(objective.score(carrier.render(x01)))
        pre = cap_measures(carrier, x01, caps)
        base_amp = [getattr(s, 'amplitude', 1.0) for s in carrier.stages]
        if best is not None:
            carrier.load_state_dict(best[2])
            carrier.set_amplitude(base_amp)
        else:
            shrink = fit_caps(carrier, x01, caps)
        end = float(objective.score(quantise(carrier.render(x01))))
        terms = {n: float(v) for n, v in
                 objective.terms(quantise(carrier.render(x01))).items()}
    return {'free_steps': steps, 'free_lr': lr, 'free_rho': rho,
            'free_score_start': round(start, 5),
            'free_score_unprojected': round(unprojected, 5),
            'free_score_end': round(end, 5),
            'free_amplitude_shrink': round(shrink, 5),
            'free_feasible_step': -1 if best is None else best[1],
            'free_feasible_checks': checks,
            'free_cap_violations': cap_violations(carrier, x01, caps),
            'free_caps_unprojected': '|'.join(f'{pre[c.name]:.3f}' for c in caps),
            'free_lambda': '|'.join(f'{lam[c.name]:.3f}' for c in caps),
            'free_amplitude_base': '|'.join(f'{b:.9f}' for b in base_amp),
            **{f'free_term_{n}': round(v, 5) for n, v in terms.items()}}

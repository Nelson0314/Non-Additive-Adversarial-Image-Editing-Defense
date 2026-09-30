"""在色差上限內以 augmented Lagrangian 最佳化顏色載體。

    L = score + Σ_i [ λ_i g_i + (ρ/2)·max(0, g_i)² ],    g_i = measure_i / cap_i − 1

乘子每 `lam_every` 步依 `λ_i ← max(0, λ_i + ρ g_i)` 更新。每道上限有可微的
`soft`（進損失）與量測用的 `hard`（決定可行性）；每 `check_every` 步在量化後
影像上以 `hard` 驗證可行性，保留分數最佳的可行 checkpoint。從未可行時以
`fit_caps` 將各段幅度同乘一係數二分回可行域，並回報該係數。回傳鍵以 `free_`
開頭，逐列寫入 CSV。
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


def quantize(x):
    """量化至 8 位元格點；可行性在量化後的影像上檢查。"""
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
        y = quantize(carrier.render(x01))
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
    y = quantize(carrier.render(x01))
    return {c.name: float(c.hard(y)) for c in (caps or [])}


def cap_violations(carrier, x01, caps) -> int:
    m = cap_measures(carrier, x01, caps)
    return sum(1 for c in (caps or [])
               if c.value and c.value > 0 and m[c.name] > c.value)


def optimize_carrier(carrier, x01, objective, *, steps: int = 150,
                     lr: float = 0.05, caps: Optional[List[Cap]] = None,
                     rho: float = 10.0, lam_every: int = 5,
                     check_every: int = 10,
                     log_every: int = 0, restarts: int = 1,
                     restart_seed: int = 0, lr_final_ratio: float = 1.0,
                     probe_every: int = 0) -> Dict[str, float]:
    """受約束地最小化 `objective.score(carrier.render(x01))`。

    回傳起點與終點的分數、三個項的終值、可行 checkpoint 來自第幾步、乘子終值、
    以及可行性檢查次數。這些逐列寫進 CSV，解有沒有落在可行域是可查的。

    選點與收斂都走 `objective.eval_score`（目標若有提供）而不是逐步的
    `score`。目標帶隨機重抽時逐步分數是一個樣本的估計，拿它挑 checkpoint 等於
    挑中運氣好的那一次抽樣；`eval_score` 是固定抽樣上的量，與訓練的隨機性無關。

    `restarts > 1` 時每一輪用不同的 `carrier.reset` 種子重跑，最後交付固定抽樣
    上分數最低的那一輪。`lr_final_ratio < 1` 時學習率依餘弦由 `lr` 降到
    `lr · lr_final_ratio`。
    """
    caps = [c for c in (caps or []) if c.value and c.value > 0]
    if restarts < 1:
        raise ValueError('restarts 至少為 1')
    evaluate = getattr(objective, 'eval_score', objective.score)
    eval_terms = getattr(objective, 'eval_terms', objective.terms)

    def gaps(y) -> Dict[str, torch.Tensor]:
        return {c.name: c.soft(y) / c.value - 1.0 for c in caps}

    def one_round(round_index: int):
        if restarts > 1:
            carrier.reset(x01, restart_seed + 1009 * round_index)
        params = [p for p in carrier.params() if p.requires_grad]
        if not params:
            raise ValueError('這個載體沒有可學參數')
        lam = {c.name: 0.0 for c in caps}
        with torch.no_grad():
            start = float(evaluate(carrier.render(x01)))
        opt = torch.optim.Adam(params, lr=lr)
        sched = None
        if lr_final_ratio != 1.0:
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=steps, eta_min=lr * float(lr_final_ratio))
        best = None
        checks = 0
        curve = []
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
            if sched is not None:
                sched.step()

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
                        v = float(evaluate(quantize(carrier.render(x01))))
                        if best is None or v < best[0]:
                            best = (v, k + 1, carrier.state_dict())

            if probe_every and ((k + 1) % probe_every == 0 or k == 0):
                with torch.no_grad():
                    curve.append((k + 1,
                                  float(evaluate(quantize(carrier.render(x01))))))

            if log_every and k % log_every == 0:
                with torch.no_grad():
                    worst = max([float(v) for v in gaps(carrier.render(x01)).values()]
                                or [0.0])
                    print(f'    第 {k} 步 score {float(score):+.4f} '
                          f'最大違反 {worst:+.3f} '
                          f'λ {[round(v, 2) for v in lam.values()]}', flush=True)

        shrink = 1.0
        with torch.no_grad():
            unprojected = float(evaluate(carrier.render(x01)))
            pre = cap_measures(carrier, x01, caps)
            base_amp = [getattr(s, 'amplitude', 1.0) for s in carrier.stages]
            if best is not None:
                carrier.load_state_dict(best[2])
                carrier.set_amplitude(base_amp)
            else:
                shrink = fit_caps(carrier, x01, caps)
            end = float(evaluate(quantize(carrier.render(x01))))
        return {
            'start': start, 'unprojected': unprojected, 'end': end,
            'shrink': shrink, 'checks': checks, 'lam': lam, 'pre': pre,
            'base_amp': base_amp, 'curve': curve,
            'feasible_step': -1 if best is None else best[1],
            'state': carrier.state_dict(),
        }

    rounds = [one_round(r) for r in range(restarts)]
    pick = min(range(len(rounds)), key=lambda i: rounds[i]['end'])
    carrier.load_state_dict(rounds[pick]['state'])
    r = rounds[pick]
    with torch.no_grad():
        terms = {n: float(v) for n, v in
                 eval_terms(quantize(carrier.render(x01))).items()}
    return {'free_steps': steps, 'free_lr': lr, 'free_rho': rho,
            'free_restarts': restarts, 'free_restart_pick': pick,
            'free_restart_scores': '|'.join(f'{q["end"]:.5f}' for q in rounds),
            'free_lr_final_ratio': lr_final_ratio,
            'free_score_start': round(r['start'], 5),
            'free_score_unprojected': round(r['unprojected'], 5),
            'free_score_end': round(r['end'], 5),
            'free_amplitude_shrink': round(r['shrink'], 5),
            'free_feasible_step': r['feasible_step'],
            'free_feasible_checks': r['checks'],
            'free_curve': '|'.join(f'{k}:{v:.5f}' for k, v in r['curve']),
            'free_cap_violations': cap_violations(carrier, x01, caps),
            'free_caps_unprojected': '|'.join(f'{r["pre"][c.name]:.3f}' for c in caps),
            'free_lambda': '|'.join(f'{r["lam"][c.name]:.3f}' for c in caps),
            'free_amplitude_base': '|'.join(f'{b:.9f}' for b in r['base_amp']),
            **{f'free_term_{n}': round(v, 5) for n, v in terms.items()}}


def randomize_carrier(carrier, x01, objective, *, seed: int = 0,
                      caps: Optional[List[Cap]] = None,
                      id_target: Optional[float] = None,
                      id_of=None, grow_tries: int = 40,
                      bisect_tries: int = 18) -> Dict[str, float]:
    """不經最佳化的對照：隨機參數，幅度縮放至與受約束解可比的位置。

    回傳鍵與 `optimize_carrier` 相同。受約束解同時包含該幅度的形變與對抗方向；
    本對照保留形變、移除對抗方向。

    `id_target` 給定時，二分幅度直到 `id_of(render)` 等於該值，對齊防禦圖的
    身分餘弦；此模式下上限只量測、不強制，越界記入 `free_cap_violations`。
    未給定時倍增 `amplitude` 至至少一道上限被違反，再以 `fit_caps` 二分回可行域；
    此模式對齊的是上限，兩臂的失真量可能不同。
    """
    import torch

    caps = [c for c in (caps or []) if c.value and c.value > 0]
    params = [p for p in carrier.params() if p.requires_grad]
    if not params:
        raise ValueError('這個載體沒有可學參數')

    with torch.no_grad():
        start = float(objective.score(carrier.render(x01)))
        gen = torch.Generator(device=params[0].device).manual_seed(int(seed))
        for p in params:
            p.copy_(torch.randn(p.shape, generator=gen, device=p.device,
                                dtype=p.dtype))
    carrier.project()

    stages = len(carrier.stages)
    shrink = 1.0

    if id_target is not None:
        if id_of is None:
            raise ValueError('給了 id_target 就要給 id_of')

        def measured(a):
            carrier.set_amplitude([a] * stages)
            return float(id_of(quantize(carrier.render(x01))))

        hi = 1.0
        with torch.no_grad():
            for _ in range(grow_tries):
                if measured(hi) <= id_target:
                    break
                hi *= 2.0
            else:
                raise ValueError(
                    f'倍增 {grow_tries} 次仍然沒有把身分壓到 {id_target}；'
                    '這個載體在這張圖上到不了那個目標')
            lo = 0.0
            for _ in range(bisect_tries):
                mid = 0.5 * (lo + hi)
                if measured(mid) > id_target:
                    lo = mid
                else:
                    hi = mid
            measured(hi)
    else:
        amp = 1.0
        if caps:
            with torch.no_grad():
                for _ in range(grow_tries):
                    carrier.set_amplitude([amp] * stages)
                    m = cap_measures(carrier, x01, caps)
                    if any(m[c.name] > c.value for c in caps):
                        break
                    amp *= 2.0
                else:
                    raise ValueError(f'倍增 {grow_tries} 次仍然沒有違反任何'
                                     '上限；這組上限對這個載體沒有約束力')
        carrier.set_amplitude([amp] * stages)
        with torch.no_grad():
            shrink = fit_caps(carrier, x01, caps)

    with torch.no_grad():
        base_amp = [getattr(s, 'amplitude', 1.0) for s in carrier.stages]
        y = quantize(carrier.render(x01))
        end = float(objective.score(y))
        terms = {n: float(v) for n, v in objective.terms(y).items()}
    return {'free_steps': 0, 'free_lr': 0.0, 'free_rho': 0.0,
            'free_restarts': 1, 'free_restart_pick': 0,
            'free_restart_scores': '', 'free_lr_final_ratio': 1.0,
            'free_curve': '',
            'free_score_start': round(start, 5),
            'free_score_unprojected': round(start, 5),
            'free_score_end': round(end, 5),
            'free_amplitude_shrink': round(shrink, 5),
            'free_feasible_step': 0,
            'free_feasible_checks': 0,
            'free_cap_violations': cap_violations(carrier, x01, caps),
            'free_caps_unprojected': '|'.join('nan' for _ in caps),
            'free_lambda': '|'.join('0.000' for _ in caps),
            'free_amplitude_base': '|'.join(f'{b:.9f}' for b in base_amp),
            **{f'free_term_{n}': round(v, 5) for n, v in terms.items()}}

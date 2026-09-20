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
                     log_every: int = 0, restarts: int = 1,
                     restart_seed: int = 0, lr_final_ratio: float = 1.0,
                     probe_every: int = 0,
                     patience: int = 0,
                     min_delta: float = 0.0) -> Dict[str, float]:
    """受約束地最小化 `objective.score(carrier.render(x01))`。

    回傳起點與終點的分數、三個項的終值、可行 checkpoint 來自第幾步、乘子終值、
    以及可行性檢查次數。這些逐列寫進 CSV，解有沒有落在可行域是可查的。

    選點與收斂都走 `objective.eval_score`（目標若有提供）而不是逐步的
    `score`。目標帶隨機重抽時逐步分數是一個樣本的估計，拿它挑 checkpoint 等於
    挑中運氣好的那一次抽樣；`eval_score` 是固定抽樣上的量，與訓練的隨機性無關。

    `restarts > 1` 時每一輪用不同的 `carrier.reset` 種子重跑，最後交付固定抽樣
    上分數最低的那一輪。`lr_final_ratio < 1` 時學習率依餘弦由 `lr` 降到
    `lr · lr_final_ratio`。

    早停（`patience > 0` 才啟用）
    ────────────────────────────────────────────────────────────────
    連續 `patience` 次**可行性檢查**（每 `check_every` 步一次）都沒有把固定
    抽樣上的 `eval_score` 改善超過 `min_delta` 就停。計數器盯的是
    `eval_score` 而不是逐步的 `score`：目標帶隨機重抽時逐步損失本來就會抖，
    拿它當停止訊號等於用雜訊決定步數。

    **還沒有可行解之前不累計。** 受約束最佳化的前段可能一路違反上限，那時候
    還沒有東西可以「改善」，讓計數器在那裡跑滿會在解進入可行域之前就停掉。
    第一次拿到可行 checkpoint 之後才開始數；之後每一次檢查（包含檢查時不可行
    的那些）都算一次沒有改善。

    實際跑到第幾步寫進 `free_stopped_step`，有沒有被早停停掉寫進
    `free_early_stopped`；`patience = 0` 時兩欄仍然照填，值為 `steps` 與 0。
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
        stale = 0
        stopped_step = steps
        early_stopped = False
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
                    improved = False
                    if all(m[c.name] <= c.value for c in caps):
                        v = float(evaluate(quantise(carrier.render(x01))))
                        if best is None or v < best[0] - min_delta:
                            improved = True
                            best = (v, k + 1, carrier.state_dict())
                    # 還沒有可行解之前不累計：那時候沒有東西可以改善。
                    if best is not None:
                        stale = 0 if improved else stale + 1

            if probe_every and ((k + 1) % probe_every == 0 or k == 0):
                with torch.no_grad():
                    curve.append((k + 1,
                                  float(evaluate(quantise(carrier.render(x01))))))

            if log_every and k % log_every == 0:
                with torch.no_grad():
                    worst = max([float(v) for v in gaps(carrier.render(x01)).values()]
                                or [0.0])
                    print(f'    第 {k} 步 score {float(score):+.4f} '
                          f'最大違反 {worst:+.3f} '
                          f'λ {[round(v, 2) for v in lam.values()]}', flush=True)

            if patience > 0 and stale >= patience:
                stopped_step = k + 1
                early_stopped = True
                print(f'    早停：連續 {stale} 次檢查沒有改善 eval_score，'
                      f'停在第 {stopped_step} 步（上限 {steps}）', flush=True)
                break

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
            end = float(evaluate(quantise(carrier.render(x01))))
        return {
            'start': start, 'unprojected': unprojected, 'end': end,
            'shrink': shrink, 'checks': checks, 'lam': lam, 'pre': pre,
            'base_amp': base_amp, 'curve': curve,
            'feasible_step': -1 if best is None else best[1],
            'stopped_step': stopped_step, 'early_stopped': early_stopped,
            'state': carrier.state_dict(),
        }

    rounds = [one_round(r) for r in range(restarts)]
    pick = min(range(len(rounds)), key=lambda i: rounds[i]['end'])
    carrier.load_state_dict(rounds[pick]['state'])
    r = rounds[pick]
    with torch.no_grad():
        terms = {n: float(v) for n, v in
                 eval_terms(quantise(carrier.render(x01))).items()}
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
            'free_stopped_step': r['stopped_step'],
            'free_early_stopped': int(r['early_stopped']),
            'free_patience': patience,
            'free_min_delta': min_delta,
            'free_curve': '|'.join(f'{k}:{v:.5f}' for k, v in r['curve']),
            'free_cap_violations': cap_violations(carrier, x01, caps),
            'free_caps_unprojected': '|'.join(f'{r["pre"][c.name]:.3f}' for c in caps),
            'free_lambda': '|'.join(f'{r["lam"][c.name]:.3f}' for c in caps),
            'free_amplitude_base': '|'.join(f'{b:.9f}' for b in r['base_amp']),
            **{f'free_term_{n}': round(v, 5) for n, v in terms.items()}}


def randomise_carrier(carrier, x01, objective, *, seed: int = 0,
                      caps: Optional[List[Cap]] = None,
                      id_target: Optional[float] = None,
                      id_of=None, grow_tries: int = 40,
                      bisect_tries: int = 18) -> Dict[str, float]:
    """不做最佳化的對照：隨機參數，幅度縮放到與對抗解可比的位置。

    回傳的鍵與 `optimise_carrier` 相同，兩者的列因此並列得起來。

    為什麼要有這一條
    ────────────────────────────────────────────────────────────────
    受約束解出來的場帶著兩件事：**這個幅度的形變**，以及**對抗方向**。
    只比較「有防禦」與「未防禦」分不開這兩件事——同樣的位移量隨便放一個平滑場
    上去，重取樣本身就會改變臉的嵌入。這一條把對抗方向拿掉、失真留著，
    差額才歸得到最佳化頭上。

    對齊在哪一個量上
    ────────────────────────────────────────────────────────────────
    `id_target` 給定時，二分幅度直到 `id_of(render)` 落在那個值上，對齊的是
    **發布圖的身分餘弦**。這是刻意的：要問的是「發布前身分掉了這麼多時，對抗
    方向還有沒有多買到東西」，所以那個量必須在兩臂之間相等，其餘的量放它們去。

    `id_target` 未給時退回貼住上限：倍增 `amplitude` 直到至少一道上限被違反，
    再讓 `fit_caps` 二分回可行域。**這個模式對齊的是上限，不是失真**——隨機場
    與對抗場綁住的往往不是同一道上限（隨機場的背景難以近似剛體，`flow_rigid`
    會先綁住，臉上的位移因此遠小於對抗場），兩臂的失真量會差很多，比較的前提
    就不成立。要做同失真對照時用 `id_target`。

    上限在 `id_target` 模式下只量測、不強制：把身分推到指定的位置本來就可能
    要越過某一道上限，靜默縮回去會讓對齊失效。越界照樣寫進 `cap_reached` 與
    `free_cap_violations`。
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
            return float(id_of(quantise(carrier.render(x01))))

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
        y = quantise(carrier.render(x01))
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

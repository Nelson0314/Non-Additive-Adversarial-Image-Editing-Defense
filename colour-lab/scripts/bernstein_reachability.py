"""250 維族的可達性分析：非暗化方向的**最大** reach，與防禦目標梯度的落點。

這一支與 `bernstein_domain_check.py` 的分工
────────────────────────────────────────────────────────────────────
`bernstein_domain_check.py` 報三族方向的 `reach` **中位數**——「隨便抽一個
方向平均走多遠」。這一支問兩件不同的事：

1. **上限。** 把「整體暗化」那一類拿開之後，剩下的方向裡 `reach` 最大是多少、
   最大的那個方向長什麼樣。四維族走過的正是整體暗化，所以只有拿開它之後的
   數字才是「還沒被走過的部分有多大」。
2. **梯度落在哪。** 在內點 `θ_c` 上算 `∇_θ J`（`J` 見
   `src/defense/vae_noncommute.py`），投影到暗化／非暗化兩類上，報能量佔比。
   這是**一次前向加反向**，不是優化。

三次搜尋
────────────────────────────────────────────────────────────────────
| 名稱 | 投影掉什麼 | 回答什麼 |
|---|---|---|
| `nondark` | 整體暗化 | 主問題：不走暗化時最遠能走多遠 |
| `nondark_strict` | 整體暗化 ＋ 整體去飽和 | 連「全域常數」那一平面都拿掉還剩多少 |
| `unrestricted` | 無 | 不設限時最大在哪 |

`unrestricted` 的起點**不含** `DARK`，所以它找不找得回已知的暗化 `reach`
就是搜尋器的自檢：找得回代表這個爬山有能力跨過整個球面。

四個 stage
────────────────────────────────────────────────────────────────────
| stage | 在哪跑 | 做什麼 | 產物 |
|---|---|---|---|
| `geometry` | CPU | 域、內點、`G`、白化、三次搜尋 | `per_image/<id>.npz`、`geometry.csv` |
| `gradient` | 一張 GPU | 讀 npz，算 `∇_θ J`，投影，搜一階可達增量 | `reachability.csv` |
| `refine` | CPU | 三支的單調修正 ＋ 用解析 `∇q` 補搜一階增量 | `refined/<id>.json`、`refined/<id>.npz` |
| `refine-merge` | CPU | 把逐張的補強結果併成一份 | `refined.csv`、`refined_detail.json` |

`geometry` 把低頻錐的 `H`、`g` 存進 npz，`gradient` 直接還原（見
`reachability.restore_domain`），不必把 `250 × 3HW` 的展開再做一次。
`refine` 再往下一層：`∇_θJ` 也已經存在 `gradient_detail.json` 裡，所以它
**一次前向反向都不用重算**，純 CPU 就跑得完。`refine` 不覆寫 `geometry.csv`
與 `reachability.csv`，補強前後的數字在 `refined.csv` 裡並排。

**這一支不下判準。** 它只把數擺出來。
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from src.defense import bernstein_colour as bc  # noqa: E402
from src.defense import reachability as rb  # noqa: E402
from src.defense.delta_e_torch import delta_e00_torch  # noqa: E402
from src.metrics.perturbation_band import (blur_retention,  # noqa: E402
                                           low_frequency_share)
from src.utils.io import load_image_tensor  # noqa: E402


# ────────────────────────────────────────────────────────────────────
# 共用
# ────────────────────────────────────────────────────────────────────

def image_entries(manifest: Path, wanted: str):
    data = json.loads(manifest.read_text(encoding='utf-8'))
    entries = {r['id']: r for r in data['images']}
    names = ([v.strip() for v in wanted.split(',') if v.strip()]
             or [r['id'] for r in data['images']])
    return [(n, ROOT / entries[n]['path']) for n in names]


def boundary_readout(geom: rb.ReachGeometry, x01: torch.Tensor,
                     theta_dir: np.ndarray, frac: float = 0.9999):
    """走到邊界那一點的影像端讀數：`ΔE00`、低頻佔比、模糊保留率。

    與 `bernstein_domain_check.py` 對內點報的三個是同一組，所以「最大 reach
    的方向」可以直接跟內點並排看，不必另外換算單位。
    """
    theta = torch.from_numpy(geom.boundary_theta(theta_dir, frac))
    with torch.no_grad():
        xd = x01.detach().cpu().double()
        y = bc.apply_filter(xd, theta)
        yq = (y.clamp(0, 1) * 255).round() / 255
        delta = (yq - xd)[0].numpy().astype(np.float64)
        de = float(delta_e00_torch(xd.float(), yq.float()))
        share = float(low_frequency_share(delta))
        retention = float(blur_retention(delta))
    return {'deltaE00': round(de, 3), 'low_freq_share': round(share, 6),
            'blur_retention': round(retention, 6)}


# ────────────────────────────────────────────────────────────────────
# stage geometry
# ────────────────────────────────────────────────────────────────────

def run_geometry(args) -> None:
    out = args.out
    (out / 'per_image').mkdir(parents=True, exist_ok=True)
    rows = []
    detail = {}
    for name, path in image_entries(args.manifest, args.images):
        t0 = time.time()
        x = load_image_tensor(path, torch.device('cpu'), size=args.size)
        domain = bc.BernsteinDomain(x)
        theta0, info = bc.interior_point(domain)
        gram = bc.perturbation_gram(x).numpy()
        geom = rb.ReachGeometry(domain, theta0, gram, threshold=args.threshold)
        rng = np.random.default_rng(args.seed + sum(ord(c) for c in name))

        # ---- 兩條結構方向的兩個符號 ----
        named = {}
        for label, vec in (('dark', rb.DARK), ('desat', rb.DESAT)):
            for sign, tag in ((1.0, 'pos'), (-1.0, 'neg')):
                d = sign * vec
                named[f'{label}_{tag}'] = {
                    'reach': geom.reach_theta(d),
                    'binding': geom.binding(d),
                    **boundary_readout(geom, x, d),
                }

        # ---- 三次搜尋 ----
        projectors = {
            'nondark': rb.GProjector(gram, [rb.DARK]),
            'nondark_strict': rb.GProjector(gram, [rb.DARK, rb.DESAT]),
            'unrestricted': rb.GProjector(gram),
        }
        # 主問題給全額迭代，另外兩個是對照，給一半。
        budgets = {'nondark': args.iters,
                   'nondark_strict': max(args.iters // 2, 10),
                   'unrestricted': max(args.iters // 2, 10)}
        seeds = rb.theta_seeds(geom.whiten, rng)
        sampler = rb.StepSampler(geom.whiten, rng)
        vertex = rb.VertexMove(geom)
        grad = rb.ReachGradient(domain, geom.theta0, gram, geom.base_energy)

        def ascent(project, start, steps=args.ascent):
            return rb.gradient_ascent(geom, grad, project, start, steps)

        searches = {}
        for key, project in projectors.items():
            res = rb.spherical_search(geom.reach_theta, project, seeds,
                                      sampler, rng,
                                      extra_random=args.extra_random,
                                      iters=budgets[key], vertex=vertex,
                                      ascent=ascent)
            theta_dir = res['direction']
            res['binding'] = geom.binding(theta_dir)
            res['describe'] = rb.describe_direction(theta_dir, x, gram)
            res['readout'] = boundary_readout(geom, x, theta_dir)
            searches[key] = res

        best = searches['nondark']
        dark_best = max(named['dark_pos']['reach'], named['dark_neg']['reach'])
        row = {
            'image': name, 'size': args.size,
            'alpha': round(info['alpha'], 6), 'beta': round(info['beta'], 6),
            'worst_slack': round(info['worst_slack'], 6),
            'eff_dim': geom.whiten.dim,
            'sv_threshold': args.threshold,
            'base_energy': round(geom.base_energy, 6),
            'null_reach_bound': round(geom.whiten.null_reach_bound(
                geom.theta0_np, geom.base_energy), 6),
            **{f'reach_{k}': round(v['reach'], 6) for k, v in named.items()},
            **{f'binding_{k}': v['binding'] for k, v in named.items()},
            'reach_dark_best': round(dark_best, 6),
            'reach_nondark_max': round(best['value'], 6),
            'reach_nondark_second': round(best['second'], 6),
            'reach_nondark_start_median': round(best['median'], 6),
            'nondark_starts': best['starts'],
            'nondark_within_1pct': best['within_1pct'],
            'reach_nondark_random_only_max': round(best['random_max'], 6),
            'nondark_random_starts': best['random_starts'],
            'binding_nondark_max': best['binding'],
            'reach_nondark_strict_max': round(
                searches['nondark_strict']['value'], 6),
            'reach_unrestricted_max': round(
                searches['unrestricted']['value'], 6),
            'reach_unrestricted_random_only_max': round(
                searches['unrestricted']['random_max'], 6),
            'unrestricted_over_dark': round(
                searches['unrestricted']['value'] / dark_best, 4),
            'nondark_over_dark': round(best['value'] / dark_best, 6),
            'nondark_cos_g_dark': best['describe']['cos_g_dark'],
            'nondark_cos_euclid_dark': best['describe']['cos_euclid_dark'],
            'nondark_a_energy_share': best['describe']['a_energy_share'],
            'nondark_b_energy_share': best['describe']['b_energy_share'],
            **{f'nondark_boundary_{k}': v for k, v in best['readout'].items()},
            **{f'dark_pos_boundary_{k}': v
               for k, v in named['dark_pos'].items()
               if k not in ('reach', 'binding')},
            'radius_calls': geom.calls,
            'seconds': round(time.time() - t0, 1),
        }
        rows.append(row)
        detail[name] = {
            'named': named,
            'searches': {k: {'value': float(v['value']),
                             'values': [float(t) for t in v['values']],
                             'binding': v['binding'],
                             'dropped_seeds': v['dropped_seeds'],
                             'random_max': float(v['random_max']),
                             'describe': v['describe'],
                             'readout': v['readout']}
                         for k, v in searches.items()},
            'eigenvalues': [float(v) for v in geom.whiten.eigenvalues[:60]],
        }
        np.savez(out / 'per_image' / f'{name}.npz',
                 theta0=geom.theta0_np, gram=gram,
                 basis=geom.whiten.basis, lam=geom.whiten.lam,
                 base_energy=np.asarray(geom.base_energy),
                 dir_best_nondark=best['direction'],
                 dir_best_strict=searches['nondark_strict']['direction'],
                 dir_best_unrestricted=searches['unrestricted']['direction'],
                 alpha=np.asarray(info['alpha']), beta=np.asarray(info['beta']),
                 **rb.cone_state(domain.cone))
        print(f'{name}: k={geom.whiten.dim} 暗化 +{named["dark_pos"]["reach"]:.4f}'
              f' -{named["dark_neg"]["reach"]:.4f} 非暗化最大 {best["value"]:.4f}'
              f'（前 1% 內 {best["within_1pct"]}/{best["starts"]} 起點）'
              f' 不設限 {searches["unrestricted"]["value"]:.4f}'
              f' {row["seconds"]:.0f}s', flush=True)

    write_csv(out / 'geometry.csv', rows)
    (out / 'geometry_detail.json').write_text(
        json.dumps(detail, ensure_ascii=False, indent=1, default=float),
        encoding='utf-8')


# ────────────────────────────────────────────────────────────────────
# stage gradient
# ────────────────────────────────────────────────────────────────────

class ThetaCarrier:
    """把一個 θ 張量包成 `VAENonCommutation` 要的載體介面。

    只有 `render` 會被用到，而且**必須與左邊用同一個 θ 張量**：右邊
    `T_θ(Vx)` 若寫成常數，梯度就只剩左邊那一半，量到的不是不交換量對 θ 的
    梯度（理由見 `src/defense/vae_noncommute.py` 的「兩邊都對 θ 微分」）。
    """

    def __init__(self, theta: torch.Tensor):
        self.theta = theta

    def render(self, x01: torch.Tensor) -> torch.Tensor:
        return bc.apply_filter(x01, self.theta)


def run_gradient(args) -> None:
    from src.defense.vae_noncommute import VAENonCommutation
    from src.metrics.suite import MetricSuite
    from src.models.ip2p import IP2PWrapper
    from src.utils.device import get_device

    device = get_device()
    if device.type != 'cuda':
        raise SystemExit('梯度這一支要 GPU：一次前向加反向，CPU 上沒有意義')
    ip2p = IP2PWrapper(dtype=torch.float32)
    metric = MetricSuite(device).lpips_module

    geometry_rows = {r['image']: r for r in read_csv(args.out / 'geometry.csv')}
    rows = []
    detail = {}
    for name, path in image_entries(args.manifest, args.images):
        t0 = time.time()
        state = np.load(args.out / 'per_image' / f'{name}.npz')
        x = load_image_tensor(path, device, size=args.size)
        theta0 = torch.from_numpy(state['theta0']).float().to(device)

        # ---- 一次前向加反向 ----
        theta = theta0.clone().requires_grad_(True)
        carrier = ThetaCarrier(theta)
        objective = VAENonCommutation(ip2p, x, carrier, metric, use_ckpt=True)
        value = objective.value(bc.apply_filter(x, theta))
        grad, = torch.autograd.grad(value, theta)
        j_value = float(value.detach())
        g = grad.detach().double().cpu().numpy()
        g_norm = float(np.linalg.norm(g))

        geom = rb.ReachGeometry.from_state(state)
        whiten = geom.whiten
        gram = whiten.gram

        # ---- 兩種分解 ----
        # (1) 白化座標：`∇_wJ = Λ^{-1/2}U_kᵀ∇_θJ`，量的是「每單位擾動能量
        #     能推動 J 多少」，與 reach 同一個單位系統；代價是要截掉有效
        #     子空間以外的部分，截掉多少由 `grad_outside_subspace_share` 報。
        # (2) θ 空間的歐氏分解：不需要截斷，但忽略資料幾何。兩個都報。
        g_proj = whiten.project(g)
        outside = 1.0 - float(g_proj @ g_proj) / float(g @ g)
        g_w = (whiten.basis.T @ g) / whiten.sqrt_lam
        norm_w = float(np.linalg.norm(g_w))
        w_dark = whiten.to_w(rb.DARK)
        w_dark = w_dark / np.linalg.norm(w_dark)
        dark_share = float(g_w @ w_dark) ** 2 / norm_w ** 2
        uniform_block = rb.orthonormal_block(whiten, [rb.DARK, rb.DESAT])
        uniform_share = (float(np.linalg.norm(uniform_block.T @ g_w)) ** 2
                         / norm_w ** 2)
        euclid_dark_share = float(g @ rb.DARK) ** 2 / g_norm ** 2

        # ---- 一階可達增量：`ΔJ ≈ ⟨∇_θJ, r·d⟩`，沒有任何截斷 ----
        def gain(theta_dir: np.ndarray) -> float:
            r, unit = geom.radius(theta_dir)
            return float(r * (g @ unit))

        rng = np.random.default_rng(args.seed + sum(ord(c) for c in name))
        project = rb.GProjector(gram, [rb.DARK])
        sampler = rb.StepSampler(whiten, rng)
        # 一階增量的搜尋沒有解析梯度（`ReachGradient` 是 `reach` 的，不是
        # `gain` 的），所以只有頂點步與隨機爬山，起點多給幾個：幾何那一支
        # 找到的最大 reach 方向、梯度自己的方向，再加上同一組結構起點。
        seeds = ([(True, state['dir_best_nondark']),
                  (True, -state['dir_best_nondark']),
                  (True, g / g_norm), (True, -g / g_norm)]
                 + rb.theta_seeds(whiten, rng))
        gain_search = rb.spherical_search(gain, project, seeds, sampler, rng,
                                          extra_random=args.extra_random,
                                          iters=args.iters,
                                          vertex=rb.VertexMove(geom))
        gain_dark = max(gain(rb.DARK), gain(-rb.DARK))
        rows.append({
            **geometry_rows[name],
            'J_at_interior': round(j_value, 6),
            'grad_norm_theta': round(g_norm, 6),
            'grad_norm_whitened': round(norm_w, 6),
            'grad_outside_subspace_share': round(outside, 8),
            'grad_dark_energy_share': round(dark_share, 6),
            'grad_nondark_energy_share': round(1.0 - dark_share, 6),
            'grad_uniform_energy_share': round(uniform_share, 6),
            'grad_euclid_dark_share': round(euclid_dark_share, 6),
            'gain_dark_pos': round(gain(rb.DARK), 6),
            'gain_dark_neg': round(gain(-rb.DARK), 6),
            'gain_dark_best': round(gain_dark, 6),
            'gain_nondark_best_reach_dir': round(
                max(gain(state['dir_best_nondark']),
                    gain(-state['dir_best_nondark'])), 6),
            'gain_nondark_max': round(gain_search['value'], 6),
            'gain_nondark_max_reach': round(
                geom.reach_theta(gain_search['direction']), 6),
            'gain_nondark_over_dark': (round(gain_search['value'] / gain_dark, 4)
                                       if gain_dark else float('nan')),
            'gain_nondark_within_1pct': gain_search['within_1pct'],
            'gain_nondark_starts': gain_search['starts'],
            'gradient_seconds': round(time.time() - t0, 1),
        })
        detail[name] = {
            'grad_theta': [float(v) for v in g],
            'grad_whitened': [float(v) for v in g_w],
            'gain_values': [float(v) for v in gain_search['values']],
            'gain_direction': rb.describe_direction(
                gain_search['direction'], x.detach().cpu(), gram),
        }
        print(f'{name}: J={j_value:.5f} grad={g_norm:.4f} '
              f'暗化佔比 {dark_share:.4f} 非暗化 {1 - dark_share:.4f} '
              f'子空間外 {outside:.2e} 一階增量 暗化 {gain_dark:+.5f} '
              f'非暗化 {gain_search["value"]:+.5f} '
              f'{time.time() - t0:.0f}s', flush=True)

    write_csv(args.out / 'reachability.csv', rows)
    (args.out / 'gradient_detail.json').write_text(
        json.dumps(detail, ensure_ascii=False, indent=1, default=float),
        encoding='utf-8')


# ────────────────────────────────────────────────────────────────────
# stage refine
# ────────────────────────────────────────────────────────────────────

def refined_dir(args) -> Path:
    return args.out / 'refined'


def run_refine(args) -> None:
    """補強已經跑完的搜尋，**不重算任何前向或反向**。

    兩件事
    ────────────────────────────────────────────────────────────────
    1. **三支 reach 的單調修正。** `nondark_strict` 的可行方向集合是
       `nondark` 的子集，最大值不可能比較大；既有結果在兩張影像上反過來，
       代表 `nondark` 漏掉了已知更好的候選。修法見
       `reachability.refine_reach_monotone`：由窄到寬依序補搜，每一支都拿
       到目前所有支的最佳方向（投影到自己的可行集合後）當起點。
    2. **一階增量 `gain` 的補強。** 既有的 `gain` 搜尋沒有解析梯度，只有
       頂點步與隨機爬山。`reachability.GainGradient` 用存檔的 `∇_θJ` 與
       `reach` 共用的隱函數半徑導數組出 `∇q`，不需要新的前向或反向。八張
       各加兩批起點，新舊的起點值合併取最佳。

    幾何、域、白化、`∇_θJ` 全部從 `per_image/<id>.npz` 與
    `gradient_detail.json` 還原，所以這一支是純 CPU。
    """
    out = refined_dir(args)
    out.mkdir(parents=True, exist_ok=True)
    geometry_rows = {r['image']: r
                     for r in read_csv(args.out / 'geometry.csv')}
    gradient_rows = {r['image']: r
                     for r in read_csv(args.out / 'reachability.csv')}
    grad_detail = json.loads(
        (args.out / 'gradient_detail.json').read_text(encoding='utf-8'))
    for name, path in image_entries(args.manifest, args.images):
        t0 = time.time()
        state = np.load(args.out / 'per_image' / f'{name}.npz')
        geom = rb.ReachGeometry.from_state(state)
        gram = geom.whiten.gram
        x = load_image_tensor(path, torch.device('cpu'), size=args.size)
        g = np.asarray(grad_detail[name]['grad_theta'], dtype=np.float64)
        g_norm = float(np.linalg.norm(g))
        rng = np.random.default_rng(args.seed + sum(ord(c) for c in name))
        sampler = rb.StepSampler(geom.whiten, rng)

        # ---- 1. 三支的單調修正 ----
        priors = {'nondark_strict': state['dir_best_strict'],
                  'nondark': state['dir_best_nondark'],
                  'unrestricted': state['dir_best_unrestricted']}
        reach = rb.refine_reach_monotone(geom, priors, sampler, rng,
                                         iters=args.reach_iters,
                                         ascent_steps=args.ascent)
        values = [reach[k]['value'] for k in rb.BRANCH_ORDER]
        monotone = all(values[i] <= values[i + 1] * (1.0 + 1e-9)
                       for i in range(len(values) - 1))

        # ---- 2. gain 的補強 ----
        score = rb.gain_score(geom, g)
        project = rb.GProjector(gram, [rb.DARK])
        radius_grad = rb.ReachGradient(geom.domain, geom.theta0, gram,
                                       geom.base_energy)
        gain_grad = rb.GainGradient(radius_grad, g)
        vertex = rb.VertexMove(geom)

        def ascent(project_, start):
            return rb.ascend_on_sphere(geom, gain_grad, project_, start,
                                       args.gain_ascent, score)

        # 非暗化的預條件梯度方向 `G⁺∇_θJ`：白化座標裡的最陡上升方向搬回
        # θ 空間。它與原始梯度方向不同（原始的被 `G` 的小特徵值方向主導），
        # 既有的起點只有原始梯度那一個。
        g_w = (geom.whiten.basis.T @ g) / geom.whiten.sqrt_lam
        precond = geom.whiten.to_theta(g_w)
        anchors = []
        for vec in (g, precond, np.asarray(state['dir_best_nondark'],
                                           dtype=np.float64),
                    np.asarray(reach['nondark']['direction'],
                               dtype=np.float64)):
            n = float(np.linalg.norm(vec))
            if n <= 0:
                continue
            anchors.extend([vec / n, -vec / n])
        anchors = rb.cross_seeds(project, anchors)

        # 兩批：第一批以既有最佳方向與梯度方向為錨，第二批把錨換成第一批
        # 找到的最佳方向。分兩批的理由是第二批的鄰域才問得到「第一批推進
        # 之後，那個新位置附近還有沒有更好的」。
        best_dir, best_value = None, -np.inf
        batches = []
        for batch in range(2):
            if batch == 0:
                centre = max((v for _, v in anchors),
                             key=lambda v: score(v))
                seeds = list(anchors) + rb.neighbourhood_seeds(
                    centre, sampler, max(args.gain_starts - len(anchors), 0))
            else:
                seeds = rb.neighbourhood_seeds(best_dir, sampler,
                                               args.gain_starts)
            res = rb.spherical_search(score, project, seeds, sampler, rng,
                                      extra_random=0, iters=args.gain_iters,
                                      vertex=vertex, ascent=ascent)
            if float(res['value']) > best_value:
                best_dir = np.asarray(res['direction'], dtype=np.float64)
                best_value = float(res['value'])
            batches.append({'starts': int(res['starts']),
                            'value': float(res['value']),
                            'values': [float(v) for v in res['values']]})

        prior_values = [float(v) for v in grad_detail[name]['gain_values']]
        prior_best = max(prior_values)
        merged = prior_values + [v for b in batches for v in b['values']]
        merged_best = max(merged + [best_value])
        if prior_best >= best_value:
            # 既有的最佳起點方向沒有存檔，只存了它的描述；新的搜尋沒有超過
            # 它時，方向欄留既有的值、方向本身標成 None，不假裝能取回。
            best_from = 'prior'
        else:
            best_from = 'refined'
        gain_dark_pos, gain_dark_neg = score(rb.DARK), score(-rb.DARK)
        gain_dark = max(gain_dark_pos, gain_dark_neg)
        row = {
            'image': name,
            'monotone_ok': bool(monotone),
            **{f'reach_{k}_before': round(float(reach[k]['prior_value']), 6)
               for k in rb.BRANCH_ORDER},
            **{f'reach_{k}_after': round(float(reach[k]['value']), 6)
               for k in rb.BRANCH_ORDER},
            'gain_dark_pos': round(gain_dark_pos, 6),
            'gain_dark_neg': round(gain_dark_neg, 6),
            'gain_dark_best': round(gain_dark, 6),
            'gain_nondark_max_before': round(prior_best, 6),
            'gain_nondark_max_after': round(merged_best, 6),
            'gain_nondark_over_dark_before': round(prior_best / gain_dark, 4),
            'gain_nondark_over_dark_after': round(merged_best / gain_dark, 4),
            'gain_nondark_exceeds_dark': bool(merged_best > gain_dark),
            'gain_within_1pct_before': int(
                sum(1 for v in prior_values if v > 0.99 * prior_best)),
            'gain_within_1pct_after': int(
                sum(1 for v in merged if v > 0.99 * merged_best)),
            'gain_starts_before': len(prior_values),
            'gain_starts_added': sum(b['starts'] for b in batches),
            'gain_best_from': best_from,
            # 新搜到的那個方向本身：值與它的 `reach`。既有搜尋的最佳方向
            # 沒有存檔（只存了描述），所以這兩欄一律指新的方向，不論它有沒有
            # 贏過既有的值。
            'gain_refined_value': round(best_value, 6),
            'gain_refined_dir_reach': round(geom.reach_theta(best_dir), 6),
            'gain_batch_first': round(batches[0]['value'], 6),
            'gain_batch_second': round(batches[1]['value'], 6),
            'grad_norm_theta': round(g_norm, 6),
            'radius_calls': geom.calls,
            'seconds': round(time.time() - t0, 1),
        }
        # 邊界讀數一律報新搜到的那個方向，不看它有沒有贏過既有的值：欄位
        # 指的是哪一個方向已經寫在名字裡，空欄只會讓表讀不成。
        row.update({f'gain_boundary_{k}': v for k, v in
                    boundary_readout(geom, x, best_dir).items()})
        detail = {
            'row': row,
            'reach': {k: {'value': float(reach[k]['value']),
                          'prior_value': float(reach[k]['prior_value']),
                          'search_value': float(reach[k]['search_value']),
                          'starts': int(reach[k]['starts'])}
                      for k in rb.BRANCH_ORDER},
            'gain_values_prior': prior_values,
            'gain_values_batches': [b['values'] for b in batches],
            'gain_direction': rb.describe_direction(best_dir, x, gram),
            'csv_gain_nondark_max': float(
                gradient_rows[name]['gain_nondark_max']),
            'csv_reach_nondark_max': float(
                geometry_rows[name]['reach_nondark_max']),
        }
        np.savez(out / f'{name}.npz',
                 dir_best_nondark=reach['nondark']['direction'],
                 dir_best_strict=reach['nondark_strict']['direction'],
                 dir_best_unrestricted=reach['unrestricted']['direction'],
                 dir_best_gain=best_dir)
        (out / f'{name}.json').write_text(
            json.dumps(detail, ensure_ascii=False, indent=1, default=float),
            encoding='utf-8')
        print(f'{name}: 單調 {monotone} reach strict {reach["nondark_strict"]["value"]:.5f}'
              f' ≤ nondark {reach["nondark"]["value"]:.5f}'
              f' ≤ 不設限 {reach["unrestricted"]["value"]:.5f}'
              f' | gain 非暗化 {prior_best:.6f} → {merged_best:.6f}'
              f'（暗化 {gain_dark:.6f}，比值 {merged_best / gain_dark:.4f}）'
              f' {row["seconds"]:.0f}s', flush=True)


def run_refine_merge(args) -> None:
    """把 `refined/<id>.json` 併成一份逐張表，缺的就說缺，不補零。"""
    out = refined_dir(args)
    rows, detail, missing = [], {}, []
    for name, _ in image_entries(args.manifest, args.images):
        path = out / f'{name}.json'
        if not path.exists():
            missing.append(name)
            continue
        item = json.loads(path.read_text(encoding='utf-8'))
        rows.append(item['row'])
        detail[name] = item
    if missing:
        raise SystemExit(f'這些影像還沒有補強結果，不併：{missing}')
    write_csv(args.out / 'refined.csv', rows)
    (args.out / 'refined_detail.json').write_text(
        json.dumps(detail, ensure_ascii=False, indent=1, default=float),
        encoding='utf-8')


# ────────────────────────────────────────────────────────────────────
# stage merge
# ────────────────────────────────────────────────────────────────────

def run_merge(args) -> None:
    """把 `<out>/parts/<id>/` 底下的逐張結果併成一份。

    `geometry` 是每張影像各自獨立的（域、內點、`G`、搜尋都只看一張圖），
    所以八張可以同時跑在不同的行程上，牆鐘時間從二十幾分鐘降到三分鐘。
    併的時候照 manifest 的順序排，不照檔案系統給的順序。
    """
    parts = args.out / 'parts'
    (args.out / 'per_image').mkdir(parents=True, exist_ok=True)
    rows = []
    detail = {}
    missing = []
    for name, _ in image_entries(args.manifest, args.images):
        part = parts / name
        csv_path = part / 'geometry.csv'
        npz_path = part / 'per_image' / f'{name}.npz'
        if not csv_path.exists() or not npz_path.exists():
            missing.append(name)
            continue
        rows.extend(read_csv(csv_path))
        detail.update(json.loads(
            (part / 'geometry_detail.json').read_text(encoding='utf-8')))
        (args.out / 'per_image' / f'{name}.npz').write_bytes(
            npz_path.read_bytes())
    if missing:
        raise SystemExit(f'這些影像沒有完整的結果，不併：{missing}')
    write_csv(args.out / 'geometry.csv', rows)
    (args.out / 'geometry_detail.json').write_text(
        json.dumps(detail, ensure_ascii=False, indent=1, default=float),
        encoding='utf-8')


# ────────────────────────────────────────────────────────────────────

def write_csv(path: Path, rows) -> None:
    keys = []
    for row in rows:
        for k in row:
            if k not in keys:
                keys.append(k)
    with path.open('w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    print(f'寫出 {path}（{len(rows)} 列）', flush=True)


def read_csv(path: Path):
    with path.open(newline='', encoding='utf-8') as fh:
        return list(csv.DictReader(fh))


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--stage',
                    choices=('geometry', 'merge', 'gradient', 'refine',
                             'refine-merge'),
                    default='geometry')
    ap.add_argument('--manifest', type=Path,
                    default=ROOT / 'data' / 'portraits_manifest.json')
    ap.add_argument('--images', default='')
    ap.add_argument('--out', type=Path,
                    default=ROOT / 'runs' / 'bernstein_reachability')
    ap.add_argument('--size', type=int, default=512)
    ap.add_argument('--threshold', type=float, default=rb.SV_THRESHOLD)
    ap.add_argument('--extra-random', dest='extra_random', type=int, default=12)
    ap.add_argument('--iters', type=int, default=60)
    ap.add_argument('--ascent', type=int, default=30)
    ap.add_argument('--seed', type=int, default=20260812)
    # refine 的預算：每個新起點最多 `gain_ascent + gain_iters` 次更新。
    ap.add_argument('--reach-iters', dest='reach_iters', type=int, default=60)
    ap.add_argument('--gain-starts', dest='gain_starts', type=int, default=35)
    ap.add_argument('--gain-ascent', dest='gain_ascent', type=int, default=60)
    ap.add_argument('--gain-iters', dest='gain_iters', type=int, default=240)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.stage == 'geometry':
        run_geometry(args)
    elif args.stage == 'merge':
        run_merge(args)
    elif args.stage == 'refine':
        run_refine(args)
    elif args.stage == 'refine-merge':
        run_refine_merge(args)
    else:
        run_gradient(args)


if __name__ == '__main__':
    main()

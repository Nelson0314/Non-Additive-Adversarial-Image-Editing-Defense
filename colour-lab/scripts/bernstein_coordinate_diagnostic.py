"""換座標能不能救得起來：三個臂各 60 次梯度更新的診斷。

這一支問什麼
────────────────────────────────────────────────────────────────────
`runs/bernstein_reachability/` 量的是**一階**的量：`∇_θJ` 落在哪、沿某個
方向走到邊界能換到多少一階增量。一階不夠回答「實際優化會怎樣」，因為
一階不含曲率、也不含「走了一步之後半徑與梯度都變了」這件事。這一支把同一
個域、同一個起點、同一個目標交給三個只差在座標的優化臂，各跑 60 次梯度
更新，看實際的 `ΔJ`。

三個臂的定義在 `src/defense/coordinate_arms.py`。

`J` 是 `src/defense/vae_noncommute.py` 的 `J(θ) = LPIPS(V(T_θ x), T_θ(V x))`，
**越大越好**，所以這裡是上升不是下降。

步長：每一臂自己找，量在同一個單位裡
────────────────────────────────────────────────────────────────────
A 與 B 的 `z` 不在同一個單位系統裡（B 的 `z` 帶著 `√λ` 的尺度），固定
learning rate 對兩者不是同一件事。反過來，把每一步的位移固定在某一個範數
裡也不中立：固定 `‖Δθ‖₂` 時一階最優的方向**就是** A，固定 `‖Δθ‖_G` 時
一階最優的方向**就是** B，選範數等於先選贏家。

所以兩件事分開處理：

1. **方向**由各臂自己的梯度給（C 再投影到限制的零空間）。
2. **步長**由**單調回溯**決定：`J` 沒有嚴格變大就把步長折半重試，最多
   `--backtrack` 次；第一次就成功則下一步的步長乘 `--step-grow`
   （上限 `--step-cap`）。每個臂因此跑的是它自己調出來的步長，不是外面塞
   給它的。

回溯的計量單位是 `‖Δv‖`（送進徑向映射的方向向量的位移長度），三臂共用同一
個初值 `--step-init` 與同一個上限。`v` 的尺度由參數化本身決定（`tanh(‖v‖)`
是半徑的佔比），與臂無關。

**一整步都失敗時步長要復位，不能沿用折半到底的值。** 沿用會讓步長以
`2^-backtrack` 衰減、下一步從更小的值起跳，於是整條軌跡在第一次失敗之後
就再也動不了，而 log 上看起來仍然「跑滿 60 步」——那是線搜尋的性質，不是
幾何的性質。失敗一步之後步長回到 `--step-init`；**連續** `--stall-patience`
步都失敗才認定這個臂在這個精度下停住，提前結束並把 `steps_taken` 照實寫。
提前結束對三臂是同一條規則。

**單調接受的代價要寫在讀數旁邊**：`ΔJ ≥ 0` 因此是構造上成立的，只要有任何
一步成功就 `ΔJ > 0`。停止規則的第 1 項在這個設定下幾乎必然通過，能分辨臂的
是**量值與比值**，不是符號。

起點
────────────────────────────────────────────────────────────────────
三臂共用同一個 `v_start = init_scale·û`（`û` 為逐張固定種子的隨機單位向量），
因此起點的 θ **逐位相同**。徑向映射在 `v = 0` 沒有定義方向，`init_scale`
取 `0.01`（與 `BernsteinColourParam._init_z` 同一個值），起點與 `θ_c` 的距離
只有半徑的 1%。`J_at_theta_c`（`θ_c` 本身）與 `J_start`（起點）兩個都報。

硬約束
────────────────────────────────────────────────────────────────────
六條結構限制與低頻錐全程由前向參數化保證，沒有任何更新後鉗回。每一步仍然
獨立量一次七條限制的相對餘裕並寫進曲線檔——保證與驗證分開，驗證才是證據。

兩個 stage
────────────────────────────────────────────────────────────────────
| stage | 在哪跑 | 產物 |
|---|---|---|
| `run` | 一張 GPU，逐張 | `curves/<image>__<arm>.csv`、`parts/<image>.json` |
| `merge` | CPU | `summary.csv`、`stop_rule.json` |

`run` 逐張寫檔，所以可以把不同影像丟到不同卡上平行跑。

**這一支不下判準。** 停止規則的逐項檢核照算照寫，要不要延長由使用者決定。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from src.defense import bernstein_colour as bc  # noqa: E402
from src.defense import coordinate_arms as ca  # noqa: E402
from src.defense import reachability as rb  # noqa: E402
from src.utils.io import load_image_tensor, write_csv  # noqa: E402

ARM_ORDER = ('theta', 'whitened', 'whitened_nondark')


def image_entries(manifest: Path, wanted: str):
    data = json.loads(manifest.read_text(encoding='utf-8'))
    entries = {r['id']: r for r in data['images']}
    names = ([v.strip() for v in wanted.split(',') if v.strip()]
             or [r['id'] for r in data['images']])
    return [(n, ROOT / entries[n]['path']) for n in names]


class MutableThetaCarrier:
    """`VAENonCommutation` 要的載體介面，θ 可以逐步換掉。

    兩邊（`T_θ x` 與 `T_θ(Vx)`）必須讀到**同一個** θ 張量，右邊寫成常數
    梯度就只剩一半（理由見 `src/defense/vae_noncommute.py`）。基底逐張快取：
    `x` 與 `Vx` 形狀相同但內容不同，**不能只用形狀當鍵**。
    """

    def __init__(self):
        self.theta = None
        self._cache = {}

    def render(self, x01: torch.Tensor) -> torch.Tensor:
        key = id(x01)
        entry = self._cache.get(key)
        if entry is None or entry[0] is not x01:
            entry = (x01, bc.basis_of(x01))
            self._cache[key] = entry
        return bc.apply_filter(x01, self.theta, basis=entry[1])


def g_cos(gram: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    """`G` 內積下的餘弦。兩個向量都必須非零。"""
    na = float(a @ gram @ a)
    nb = float(b @ gram @ b)
    if na <= 0 or nb <= 0:
        return float('nan')
    return float(a @ gram @ b) / float(np.sqrt(na * nb))


def run_arm(name, arm, radial, evaluate, v_start, gram, theta_start, args):
    """一個臂的 60 次梯度更新。回傳 (曲線列, 摘要)。"""
    z = arm.to_z(v_start.clone()).detach()
    scale = float(args.step_init)
    j_cur = evaluate(z, arm)[0]
    j_start = j_cur
    rows = []
    grad_evals = 0
    value_evals = 1
    accepted_steps = 0
    stalled_steps = 0
    backtracks = 0
    t_arm = time.time()
    worst_slack = float('inf')
    consecutive_stalls = 0
    steps_taken = 0
    for step in range(1, int(args.steps) + 1):
        t_step = time.time()
        j_here, grad, theta_here = evaluate(z, arm, grad=True)
        grad_evals += 1
        direction = arm.project(grad)
        dir_norm = float(direction.norm())
        trials = 0
        accepted = False
        j_new = j_here
        eta = 0.0
        dv_len = 0.0
        if dir_norm > 0:
            p = direction / dir_norm
            dv_unit = float(arm.to_v(p).norm())
            eta = scale / max(dv_unit, 1e-300)
            for attempt in range(int(args.backtrack) + 1):
                z_try = z + eta * p
                j_try, theta_try = evaluate(z_try, arm)[:2]
                trials += 1
                value_evals += 1
                if j_try > j_here:
                    accepted = True
                    j_new = j_try
                    z = z_try.detach()
                    theta_here = theta_try
                    dv_len = eta * dv_unit
                    break
                eta *= 0.5
                backtracks += 1
            if accepted:
                scale = (min(scale * float(args.step_grow),
                             float(args.step_cap))
                         if trials == 1 else eta * dv_unit)
            else:
                scale = float(args.step_init)
        else:
            scale = float(args.step_init)
        steps_taken = step
        if accepted:
            accepted_steps += 1
            consecutive_stalls = 0
        else:
            stalled_steps += 1
            consecutive_stalls += 1
        j_cur = j_new
        v_now = arm.to_v(z)
        slacks = ca.slack_report(radial.domain, theta_here)
        worst_slack = min(worst_slack, slacks['min_slack'])
        dz = (z - arm.to_z(v_start)).detach()
        rows.append({
            'step': step,
            'J': round(j_cur, 8),
            'J_before_step': round(j_here, 8),
            'accepted': int(accepted),
            'line_search_evals': trials,
            'step_scale_dv': float(scale),
            'moved_dv': float(dv_len),
            'v_norm': float(v_now.detach().norm()),
            'radius_fraction': float(torch.tanh(v_now.detach().norm())),
            'theta_shift': float((theta_here - radial.theta0).norm()),
            'grad_norm_z': dir_norm,
            'constraint_residual': arm.violation(dz),
            'min_slack': slacks['min_slack'],
            'slack_a': slacks['a'], 'slack_ra': slacks['ra'],
            'slack_b': slacks['b'], 'slack_rb': slacks['rb'],
            'slack_jacobian': slacks['jacobian'],
            'slack_hessian': slacks['hessian'],
            'slack_lowfreq': slacks['lowfreq'],
            'seconds': round(time.time() - t_step, 4),
        })
        if consecutive_stalls >= int(args.stall_patience):
            break
    v_end = arm.to_v(z).detach()
    theta_end = radial(v_end).detach()
    d_theta = (theta_end - theta_start).double().numpy()
    dz_end = (z - arm.to_z(v_start)).detach()
    summary = {
        'arm': name,
        'label': ca.ARM_LABELS[name],
        'J_start': round(j_start, 8),
        'J_final': round(j_cur, 8),
        'delta_J': round(j_cur - j_start, 8),
        'steps_requested': int(args.steps),
        'steps_taken': steps_taken,
        'stopped_early': int(steps_taken < int(args.steps)),
        'steps_accepted': accepted_steps,
        'steps_stalled': stalled_steps,
        'backtracks': backtracks,
        'grad_evals': grad_evals,
        'value_evals': value_evals,
        'final_v_norm': float(v_end.norm()),
        'final_radius_fraction': float(torch.tanh(v_end.norm())),
        'final_theta_shift': float((theta_end - radial.theta0).norm()),
        'travelled_theta': float(np.linalg.norm(d_theta)),
        'dtheta_dark_cos_g': (g_cos(gram, d_theta, rb.DARK)
                              if np.linalg.norm(d_theta) > 0 else float('nan')),
        'constraint_residual': arm.violation(dz_end),
        'worst_min_slack': worst_slack,
        'all_constraints_passed': int(worst_slack > 0.0),
        'seconds': round(time.time() - t_arm, 2),
    }
    return rows, summary


def run(args) -> None:
    from src.defense.vae_noncommute import VAENonCommutation
    from src.metrics.suite import MetricSuite
    from src.models.ip2p import IP2PWrapper
    from src.utils.device import get_device

    device = get_device()
    if device.type != 'cuda':
        raise SystemExit('這一支要 GPU：每一步都有一次 VAE 前向反向')
    ip2p = IP2PWrapper(dtype=torch.float32)
    metric = MetricSuite(device).lpips_module

    out = args.out
    (out / 'curves').mkdir(parents=True, exist_ok=True)
    (out / 'parts').mkdir(parents=True, exist_ok=True)

    arms = [a.strip() for a in args.arms.split(',') if a.strip()]
    for name, path in image_entries(args.manifest, args.images):
        t_image = time.time()
        state = np.load(args.state / f'{name}.npz')
        gram = rb.symmetrise(state['gram'])
        theta0 = torch.from_numpy(np.asarray(state['theta0'],
                                             dtype=np.float64))
        domain = rb.restore_domain(state)
        whiten = ca.RegularisedWhitening(gram, tau_rel=args.tau_rel)
        built = ca.build_arms(whiten)
        radial = ca.RadialTheta(domain, theta0)

        x = load_image_tensor(path, device, size=args.size)
        carrier = MutableThetaCarrier()
        objective = VAENonCommutation(ip2p, x, carrier, metric, use_ckpt=True)

        def evaluate(z, arm, grad: bool = False):
            """`J` 與（可選的）`∂J/∂z`。θ 在 CPU float64 上算，
            `apply_filter` 把它搬上卡並轉 float32——那一步是可微的轉型，
            梯度接得回 `z`。"""
            zz = z.detach().clone().requires_grad_(grad)
            theta = radial(arm.to_v(zz))
            carrier.theta = theta
            value = objective.value(carrier.render(x))
            if not grad:
                return float(value.detach()), theta.detach(), None
            g, = torch.autograd.grad(value, zz)
            return float(value.detach()), g.detach(), theta.detach()

        def evaluate_pair(z, arm, grad=False):
            if grad:
                j, g, th = evaluate(z, arm, grad=True)
                return j, g, th
            j, th, _ = evaluate(z, arm, grad=False)
            return j, th, None

        gen = torch.Generator(device='cpu').manual_seed(
            int(args.seed) + sum(ord(c) for c in name))
        u = torch.randn(bc.DIM, generator=gen, dtype=torch.float64)
        u = u / u.norm()
        v_start = args.init_scale * u
        theta_start = radial(v_start).detach()

        carrier.theta = theta0
        j_theta_c = float(objective.value(carrier.render(x)).detach())

        summaries = []
        for arm_name in arms:
            arm = built[arm_name]
            start_z = arm.to_z(v_start)
            check = float((arm.to_v(start_z) - v_start).norm())
            if check > 1e-8:
                raise ValueError(f'{arm_name} 的座標不可逆：起點還原誤差 {check:.3e}')
            rows, summary = run_arm(arm_name, arm, radial, evaluate_pair,
                                    v_start, gram, theta_start, args)
            summary['image'] = name
            write_csv(out / 'curves' / f'{name}__{arm_name}.csv', rows)
            summaries.append(summary)
            print(f'[{name}] {arm_name} ΔJ={summary["delta_J"]:+.6f} '
                  f'accepted={summary["steps_accepted"]}/'
                  f'{summary["steps_taken"]} '
                  f'{summary["seconds"]:.1f}s', flush=True)

        part = {
            'image': name,
            'size': int(args.size),
            'steps': int(args.steps),
            'seed': int(args.seed),
            'init_scale': float(args.init_scale),
            'step_init': float(args.step_init),
            'step_cap': float(args.step_cap),
            'step_grow': float(args.step_grow),
            'stall_patience': int(args.stall_patience),
            'backtrack': int(args.backtrack),
            'J_at_theta_c': round(j_theta_c, 8),
            'J_start': summaries[0]['J_start'] if summaries else None,
            'theta_start_shift': float((theta_start - theta0).norm()),
            'whitening': whiten.report(),
            'radius_calls': radial.radius_calls,
            'seconds': round(time.time() - t_image, 2),
            'arms': summaries,
        }
        (out / 'parts' / f'{name}.json').write_text(
            json.dumps(part, ensure_ascii=False, indent=1, default=float),
            encoding='utf-8')
        print(f'[{name}] 完成 {part["seconds"]:.1f}s '
              f'radius_calls={radial.radius_calls}', flush=True)


def merge(args) -> None:
    parts = sorted((args.out / 'parts').glob('*.json'))
    rows = []
    checks = []
    for p in parts:
        data = json.loads(p.read_text(encoding='utf-8'))
        by_arm = {a['arm']: a for a in data['arms']}
        for arm_name in ARM_ORDER:
            a = by_arm.get(arm_name)
            if a is None:
                continue
            rows.append({
                'image': data['image'],
                'label': a['label'],
                'arm': arm_name,
                'J_at_theta_c': data['J_at_theta_c'],
                'J_start': a['J_start'],
                'J_final': a['J_final'],
                'delta_J': a['delta_J'],
                'steps_taken': a['steps_taken'],
                'stopped_early': a['stopped_early'],
                'steps_accepted': a['steps_accepted'],
                'steps_stalled': a['steps_stalled'],
                'backtracks': a['backtracks'],
                'grad_evals': a['grad_evals'],
                'value_evals': a['value_evals'],
                'final_radius_fraction': round(a['final_radius_fraction'], 6),
                'travelled_theta': round(a['travelled_theta'], 6),
                'dtheta_dark_cos_g': round(a['dtheta_dark_cos_g'], 6),
                'constraint_residual': a['constraint_residual'],
                'worst_min_slack': round(a['worst_min_slack'], 8),
                'all_constraints_passed': a['all_constraints_passed'],
                'seconds': a['seconds'],
            })
        d = {k: by_arm[k]['delta_J'] for k in ARM_ORDER if k in by_arm}
        item = {
            'image': data['image'],
            'delta_J_A': d.get('theta'),
            'delta_J_B': d.get('whitened'),
            'delta_J_C': d.get('whitened_nondark'),
        }
        if item['delta_J_B'] is not None and item['delta_J_A'] is not None:
            item['B_over_A'] = (item['delta_J_B'] / item['delta_J_A']
                                if item['delta_J_A'] != 0 else None)
            item['criterion_B_positive'] = bool(item['delta_J_B'] > 0)
            item['criterion_B_beats_A'] = bool(item['delta_J_B']
                                               > item['delta_J_A'])
        if item['delta_J_C'] is not None and item['delta_J_B'] is not None:
            item['C_over_B'] = (item['delta_J_C'] / item['delta_J_B']
                                if item['delta_J_B'] != 0 else None)
            item['criterion_C_half_of_B'] = bool(
                item['delta_J_C'] >= 0.5 * item['delta_J_B'])
        item['constraints_all_passed'] = all(
            by_arm[k]['all_constraints_passed'] == 1 for k in by_arm)
        item['all_three'] = bool(item.get('criterion_B_positive')
                                 and item.get('criterion_B_beats_A')
                                 and item.get('criterion_C_half_of_B'))
        checks.append(item)
    write_csv(args.out / 'summary.csv', rows)
    passed = [c for c in checks if c['all_three']]
    verdict = {
        'images': len(checks),
        'images_meeting_all_three': len(passed),
        'images_meeting_all_three_names': [c['image'] for c in passed],
        'criterion_B_positive': sum(1 for c in checks
                                    if c.get('criterion_B_positive')),
        'criterion_B_beats_A': sum(1 for c in checks
                                   if c.get('criterion_B_beats_A')),
        'criterion_C_half_of_B': sum(1 for c in checks
                                     if c.get('criterion_C_half_of_B')),
        'constraints_all_passed': sum(1 for c in checks
                                      if c['constraints_all_passed']),
        'threshold_images': 6,
        'extension_allowed': bool(
            len(passed) >= 6
            and all(c['constraints_all_passed'] for c in checks)),
        'per_image': checks,
    }
    (args.out / 'stop_rule.json').write_text(
        json.dumps(verdict, ensure_ascii=False, indent=1, default=float),
        encoding='utf-8')
    print(json.dumps({k: v for k, v in verdict.items() if k != 'per_image'},
                     ensure_ascii=False, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--stage', choices=['run', 'merge'], default='run')
    ap.add_argument('--manifest', type=Path,
                    default=ROOT / 'data' / 'portraits_manifest.json')
    ap.add_argument('--images', default='')
    ap.add_argument('--arms', default=','.join(ARM_ORDER))
    ap.add_argument('--out', type=Path,
                    default=ROOT / 'runs' / 'bernstein_coordinate_diagnostic')
    ap.add_argument('--state', type=Path,
                    default=ROOT / 'runs' / 'bernstein_reachability'
                    / 'per_image')
    ap.add_argument('--size', type=int, default=512)
    ap.add_argument('--steps', type=int, default=60)
    ap.add_argument('--tau-rel', dest='tau_rel', type=float, default=ca.TAU_REL)
    ap.add_argument('--init-scale', dest='init_scale', type=float, default=0.01)
    ap.add_argument('--step-init', dest='step_init', type=float, default=0.05)
    ap.add_argument('--step-cap', dest='step_cap', type=float, default=0.25)
    ap.add_argument('--step-grow', dest='step_grow', type=float, default=1.5)
    ap.add_argument('--stall-patience', dest='stall_patience', type=int,
                    default=3)
    ap.add_argument('--backtrack', type=int, default=8)
    ap.add_argument('--seed', type=int, default=20260812)
    args = ap.parse_args()
    if args.stage == 'run':
        run(args)
    else:
        merge(args)


if __name__ == '__main__':
    main()

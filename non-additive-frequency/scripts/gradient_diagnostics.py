"""為什麼最佳化在顏色載體上不值錢：把三個互斥的解釋分開量。

`runs/objective_pilot/` 量到的事實是**最佳化幾乎不值錢**：整圖載體上隨機邊界
的位移 0.3626，最佳化後 0.3098。三個解釋各自會留下不同的指紋，這支腳本量的
就是那些指紋，不跑攻擊、不做判定。

    (a) 起點就是駐點       g_mean 與 g_rms 同時近零
    (b) 抽樣之間互相抵銷   g_mean 遠小於 g_rms（比值 cancel_ratio 接近 0）
    (c) 可達集合太小       同一個損失在放寬的集合上一階增益大得多（kappa 遠小於 1）
    (d) 目標對顏色不敏感   放寬的集合上目標值的散布也很小（value_spread 近零）

四者不互斥，所以四個量一起報，不挑選。

    g_mean      ‖E_ξ[∇_θ L]‖             抽樣平均之後還剩多少
    g_rms       sqrt(E_ξ‖∇_θ L‖²)        逐次抽樣的梯度有多大
    kappa       A_carrier(ρ) / A_relaxed(ρ)   同一個輸入失真下可達的一階下降之比
    value_spread  max L − min L over K 個隨機可達點

`A(ρ) = −∇_θL · h*`，`h*` 是把負梯度投影回該參數化可行集合、再縮放到輸入失真
`ρ` 的方向。放寬的集合就是同一個參數化把 `radius` 與 `epsilon_lab` 各放大
`--relax`（預設 10 倍）；**放寬之後的圖不一定自然，那不是這裡的問題**——這支
腳本問的是「損失看不看得見顏色」，不是「產物能不能用」。
"""
import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

LOSSES = ('latent_norm', 'cfg_shift', 'collision')

COLUMNS = [
    'image', 'class', 'instruction', 'carrier', 'loss', 'draws', 'seed',
    'g_mean', 'g_rms', 'cancel_ratio',
    'a_carrier', 'a_relaxed', 'kappa', 'relax',
    'moved_carrier', 'moved_relaxed', 'rho_reached_carrier', 'rho_reached_relaxed',
    'value_spread', 'value_min', 'value_max', 'value_at_start',
    'sampled', 'kappa_readable',
    'rho', 'radius', 'epsilon_lab', 'seconds',
]


def assert_free_cards():
    """在匯入 torch 或載入任何權重**之前**核對指定的卡。"""
    card = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if not card.isdecimal():
        raise SystemExit('CUDA_VISIBLE_DEVICES 必須指定單張實體卡號')
    subprocess.run(['bash', str(ROOT / 'scripts/free_cards.sh'), '--assert', card],
                   cwd=ROOT, check=True)
    uuid = subprocess.run(
        ['nvidia-smi', '-i', card, '--query-gpu=uuid', '--format=csv,noheader'],
        capture_output=True, text=True, check=True).stdout.strip()
    apps = subprocess.run(
        ['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader'],
        capture_output=True, text=True, check=True).stdout.splitlines()
    mine = str(os.getpid())
    for line in apps:
        if not line.strip() or uuid not in line:
            continue
        pid = line.split(',')[-1].strip()
        if pid == mine:
            continue
        seen = subprocess.run(['ps', '-p', pid], capture_output=True, text=True)
        raise SystemExit(
            f'卡 {card}（{uuid}）上有 pid {pid} 的 compute app，'
            f'{"是別人的" if seen.returncode else "不是本行程"}；換一張卡，不要擠。')


def load_image(path, device):
    import numpy as np
    import torch
    from PIL import Image
    arr = np.asarray(Image.open(path).convert('RGB'), dtype=np.float32) / 255.
    return torch.from_numpy(arr).permute(2, 0, 1)[None].to(device)


def grad_stats(param, loss_fn, x01, draws):
    """逐次抽樣取梯度，回傳 (g_mean, g_rms, 平均後的梯度張量清單)。

    **平均與範數的順序是這支腳本的重點**：先取範數再平均得到的是 `g_rms`，
    先平均再取範數得到的是 `g_mean`。兩者的落差就是抽樣之間的抵銷量，而逐步
    損失看起來抖不抖分不出這件事。
    """
    import torch
    sums = [torch.zeros_like(p) for p in param.params()]
    sq = 0.0
    for _ in range(draws):
        for p in param.params():
            if p.grad is not None:
                p.grad = None
        value = loss_fn(param.render(x01))
        value.backward()
        flat = []
        for acc, p in zip(sums, param.params()):
            g = torch.zeros_like(p) if p.grad is None else p.grad.detach()
            acc.add_(g)
            flat.append(g.reshape(-1))
        sq += float(torch.cat(flat).pow(2).sum())
    mean = [s / draws for s in sums]
    g_mean = float(torch.cat([m.reshape(-1) for m in mean]).norm())
    g_rms = (sq / draws) ** 0.5
    return g_mean, g_rms, mean


def first_order_gain(param, mean_grad, x01, rho):
    """沿**投影後的負梯度**走到影像位移 ρ 時的一階下降 `−∇L · h`。

    **這不是該位移下的最大一階下降。** 只搜尋一條 projected-gradient 路徑，
    沒有對所有可行步求上界；而且 `project()` 先逐元素 clamp 再逐列縮放、
    `render` 還含 SVD 裁切與非線性色彩轉換，位移對步長不保證單調，二分取到的
    是一個可行解而不是恰好落在 ρ 上的最大解。`kappa` 因此讀作「兩邊各自沿自己
    的負梯度、在同一個位移上拿到的一階下降之比」，不能推出集合的包含關係。

    縮放走影像位移而不是參數範數：兩個參數化的半徑在完全不同的座標裡，照參數
    範數縮等於比較兩個不同的東西。

    位移量的是 `‖render(θ+h) − render(θ)‖`，**不是** `‖render(θ+h) − x‖`。
    這一族的 `delta = 0` 已經整個套上 Monge–Kantorovitch 轉移，起點對原圖的
    失真本來就很大（單這一項在 512² 的人像上就給 ΔE00 24），拿它當基準的話
    任何正的步長都「超過 ρ」，二分會停在步長 0、一階增益恆為 0。
    """
    import torch
    with torch.no_grad():
        base = [p.detach().clone() for p in param.params()]
        start = param.render(x01).detach().clone()
        direction = [-g for g in mean_grad]
        norm = torch.cat([d.reshape(-1) for d in direction]).norm()
        if float(norm) == 0.:
            return 0., 0.
        unit = [d / norm for d in direction]
        # 二分一個步長，讓 render 的輸入失真剛好是 ρ。投影每步都做，所以走到的
        # 點一定在可行集合裡；投影把方向截掉多少，就反映在拿到的一階下降上。
        lo, hi = 0., 1.
        for _ in range(40):
            for p, b, u in zip(param.params(), base, unit):
                p.data.copy_(b + hi * u)
            param.project()
            if float((param.render(x01) - start).norm()) >= rho:
                break
            lo, hi = hi, hi * 2.
            if hi > 1e6:
                break
        for _ in range(30):
            mid = .5 * (lo + hi)
            for p, b, u in zip(param.params(), base, unit):
                p.data.copy_(b + mid * u)
            param.project()
            if float((param.render(x01) - start).norm()) < rho:
                lo = mid
            else:
                hi = mid
        for p, b, u in zip(param.params(), base, unit):
            p.data.copy_(b + lo * u)
        param.project()
        step = [p.detach() - b for p, b in zip(param.params(), base)]
        gain = -float(sum((g * s).sum() for g, s in zip(mean_grad, step)))
        moved = float((param.render(x01) - start).norm())
        for p, b in zip(param.params(), base):
            p.data.copy_(b)
        return gain, moved


def value_spread(param, loss_fn, x01, draws_points, seed, fixed=None):
    """K 個隨機可達點上目標值的散布。梯度找不到，不代表集合裡沒有。

    兩件事必須做對，否則量到的不是「集合上的散布」：

    **每個點都從起點出發。** `boundary_init` 是 `p.add_()`，不還原的話跑出來的
    是 `p_k = project(p_{k-1} + u_k)` 的相關隨機漫步，而不是可行集合上互相獨立
    的探測點；投影還會再改變分布。

    **用固定評估。** 帶抽樣的損失（`cfg_shift` 每次重抽 t、噪聲與指令）在不同
    點上會配到不同的隨機條件，量到的散布就混進抽樣雜訊，指認不了顏色敏感度。
    `fixed` 給了就走它（`StepwiseObjective.fixed`），沒有抽樣的損失兩者相同。
    """
    import torch
    from src.defense.carrier_objectives import boundary_init
    evaluate = fixed if fixed is not None else loss_fn
    with torch.no_grad():
        base = [p.detach().clone() for p in param.params()]
    values = []
    for k in range(draws_points):
        with torch.no_grad():
            for p, b in zip(param.params(), base):
                p.data.copy_(b)
        boundary_init(param, x01, seed + k, draw='uniform')
        with torch.no_grad():
            values.append(float(evaluate(param.render(x01))))
    with torch.no_grad():
        for p, b in zip(param.params(), base):
            p.data.copy_(b)
    return min(values), max(values)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--losses', default=','.join(LOSSES))
    ap.add_argument('--draws', type=int, default=32,
                    help='梯度抽樣次數；g_mean 與 g_rms 的落差要靠它才看得出來')
    ap.add_argument('--points', type=int, default=16,
                    help='value_spread 用的隨機可達點數')
    ap.add_argument('--relax', type=float, default=10.0,
                    help='放寬集合的倍率，同時乘在 radius 與 epsilon_lab 上')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()
    losses = [s.strip() for s in args.losses.split(',') if s.strip()]
    for name in losses:
        if name not in LOSSES:
            raise SystemExit(f'未知的損失 {name!r}；可用的是 {LOSSES}')
    if args.device != 'cpu':
        assert_free_cards()

    import torch
    from src.defense.carrier_objectives import boundary_init, make_objective
    from src.defense.cfg_shift_loss import instruction_embeddings
    from src.defense.collision_loss import make_collision_loss, ring_of
    from src.defense.ncf_library import sha256
    from src.defense.ncf_runner import ncf_support
    from scripts.color_ceiling import (build_param, cells_of, collision_region,
                                       palette_of)

    spec = json.loads(args.config.read_text(encoding='utf-8'))
    manifest = json.loads((ROOT / spec['manifest']).read_text(encoding='utf-8'))
    if args.out.exists() and any(args.out.iterdir()):
        raise SystemExit(f'{args.out} 已存在且非空；換一個輸出目錄')
    args.out.mkdir(parents=True, exist_ok=True)

    ip2p = None
    device = torch.device(args.device)
    if any(n in ('latent_norm', 'cfg_shift') for n in losses):
        from src.models.ip2p import IP2PWrapper
        ip2p = IP2PWrapper(dtype={'fp16': torch.float16, 'fp32': torch.float32,
                                  'bf16': torch.bfloat16}[spec['precision']])
        device = ip2p.device

    entries = {r['id']: r for r in manifest['images']}
    rows, t_start = [], time.time()
    todo = cells_of(spec)
    if args.limit:
        todo = todo[:args.limit]

    for cell in todo:
        entry = entries[cell['image']]
        src = ROOT / entry['path']
        if sha256(src) != entry['sha256']:
            raise ValueError(f"{cell['image']} 的輸入雜湊不符")
        x = load_image(src, device)
        clothes = ncf_support(x, 'clothes')
        support = clothes if cell['carrier'] == 'clothes' else ncf_support(x, 'frame')
        mean, cov = palette_of(spec, spec['palette_id'])

        for name in losses:
            if name == 'collision':
                region = collision_region(x, cell['class'], clothes, device=device)
                loss_fn = make_collision_loss(
                    region, ring_of(region, spec['collision_ring_width']))
            else:
                kw = dict(x_clean=x, seed=spec['seed'])
                if name == 'cfg_shift':
                    # `eval_draws = 1`：固定評估把 n 次抽樣**累加在同一張圖上**，
                    # 帶梯度算時 n 次的 UNet 圖同時留著。預設的 8 在 512² fp32
                    # 上會把 24 GB 的卡吃光（實測 OOM）。這裡只需要一個決定性的
                    # 方向，一次抽樣就夠。
                    kw.update(zt_mode=spec['zt_mode'], s_t=spec['s_t'],
                              s_i=spec['s_i'], normalise=spec['normalise'],
                              samples=1, eval_draws=1,
                              eval_seed=spec.get('eval_seed', 90001),
                              text_embeds=instruction_embeddings(
                                  ip2p, [c['instructions'][cell['image']]
                                         for c in spec['classes']]))
                loss_fn = make_objective(name, ip2p, **kw)

            made = []
            for label, scale in (('carrier', 1.0), ('relaxed', args.relax)):
                p = build_param('chroma_bounded', x, support=support,
                                target_mean=mean, target_cov=cov,
                                blur_sigma=spec['blur_sigma'],
                                radius=spec['radius'] * scale,
                                epsilon_lab=spec['epsilon_lab'] * scale)
                p.reset(x, spec['seed'])
                made.append((label, p))

            carrier, relaxed = made[0][1], made[1][1]
            fixed = getattr(loss_fn, 'fixed', None)
            sampled = int(fixed is not None and fixed is not loss_fn)
            evaluate = fixed if fixed is not None else loss_fn
            with torch.no_grad():
                start = float(evaluate(carrier.render(x)))
            # ρ 取原本那個載體從**起點**走到自己邊界的位移的一半：這樣兩邊都
            # 到得了，kappa 才是「等位移下的一階下降之比」。基準若取對原圖的
            # 失真，ρ 會超出載體一步能走的範圍，兩邊都在自己的盒邊被夾死，
            # kappa 就退化成 1/relax（盒子大小之比），量到的是另一件事。
            with torch.no_grad():
                start_render = carrier.render(x).detach().clone()
            boundary_init(carrier, x, spec['seed'], draw='corner')
            with torch.no_grad():
                rho = .5 * float((carrier.render(x) - start_render).norm())
            carrier.reset(x, spec['seed'])
            # g_mean 與 g_rms 要走**帶抽樣的**損失，抵銷才量得到。
            g_mean, g_rms, _ = grad_stats(carrier, loss_fn, x, args.draws)
            # 一階下降的方向兩邊都走**固定評估**：帶抽樣的損失其生成器會往前走，
            # 兩側拿到的是不同的抽樣，kappa 就混進抽樣差異而不只是集合差異。
            # 固定評估是決定性的，一次抽樣就夠。
            _, _, mean_grad = grad_stats(carrier, evaluate, x, 1)
            _, _, relaxed_grad = grad_stats(relaxed, evaluate, x, 1)
            a_c, moved_c = first_order_gain(carrier, mean_grad, x, rho)
            a_r, moved_r = first_order_gain(relaxed, relaxed_grad, x, rho)
            lo, hi = value_spread(relaxed, loss_fn, x, args.points, spec['seed'],
                                  fixed=fixed)

            row = {'image': cell['image'], 'class': cell['class'],
                   'instruction': cell['instruction'], 'carrier': cell['carrier'],
                   'loss': name, 'draws': args.draws, 'seed': spec['seed'],
                   'g_mean': g_mean, 'g_rms': g_rms,
                   'cancel_ratio': (g_mean / g_rms) if g_rms else '',
                   'a_carrier': a_c, 'a_relaxed': a_r,
                   'kappa': (a_c / a_r) if a_r else '',
                   'relax': args.relax,
                   'moved_carrier': moved_c, 'moved_relaxed': moved_r,
                   # 投影可能在走到 ρ 之前就把步長夾死。夾死的話 a 是在盒邊上
                   # 量到的，不是在等位移上量到的，kappa 讀起來就是另一件事。
                   'rho_reached_carrier': int(moved_c >= rho * .99),
                   'rho_reached_relaxed': int(moved_r >= rho * .99),
                   'value_min': lo, 'value_max': hi, 'value_spread': hi - lo,
                   'value_at_start': start, 'rho': rho,
                   # 抽樣抵銷這個指紋只對帶抽樣的損失有定義。沒有抽樣時
                   # cancel_ratio 恆為 1，那是構造，不是「量過而沒有抵銷」。
                   'sampled': sampled,
                   # 兩邊都沒走到 ρ 時，a_* 是各自盒邊上的值，比值讀的是盒子
                   # 大小之比而不是等位移下的一階下降之比。
                   'kappa_readable': int(moved_c >= rho * .99 and moved_r >= rho * .99),
                   'radius': spec['radius'], 'epsilon_lab': spec['epsilon_lab'],
                   'seconds': round(time.time() - t_start, 1)}
            rows.append(row)
            print(f"  {cell['image']}__{cell['class']} {name}: "
                  f"g_mean {g_mean:.4g} g_rms {g_rms:.4g} "
                  f"cancel {row['cancel_ratio']} kappa {row['kappa']} "
                  f"spread {row['value_spread']:.4g}", flush=True)

    write_csv(args.out / 'gradient_diagnostics.csv', rows)
    print(f'-> {args.out}  共 {len(rows)} 列，{time.time()-t_start:.0f} 秒', flush=True)


def write_csv(path, rows):
    if not rows:
        raise ValueError(f'{path} 沒有任何列可寫；不要留下空檔假裝跑過')
    fields = list(dict.fromkeys(k for r in rows for k in r))
    unknown = [f for f in fields if f not in COLUMNS]
    if unknown:
        raise ValueError(f'這些欄位不在 COLUMNS 的合約裡：{unknown}')
    with Path(path).open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


if __name__ == '__main__':
    main()

"""250 維族的參數域可行性檢查：內點、退化與否、有效維度。

為什麼要先做這一步
────────────────────────────────────────────────────────────────────
`src/defense/bernstein_colour.py` 把六條結構限制與一條低頻二階錐組成一個
凸域，並用「內點 ＋ 徑向映射」的前向參數化把優化變數送進去。這個作法
只有在域**有內部**的時候才有意義：如果所有方向的邊界距離都趨近 0，
那個參數化等於把 250 個自由度全部綁死在一個點上，跑出來的解與內點無異。
這一支在 CPU 上把三件事量出來，不改任何界：

1. **嚴格內點**存不存在（在常數係數場上格點搜，回報 α、β 與各條的相對餘裕）。
2. **退化與否**：從內點沿隨機方向量到邊界的距離 `r_max` 的分布，以及每個
   方向上先撞到哪一條限制。
3. **有效維度**：域內隨機取點，看它們產生的擾動 `F_xθ` 張成的空間。
   `F_xᵀF_x` 是 250×250，所以這一步不必反覆碰影像。

有效維度分兩個數
────────────────────────────────────────────────────────────────────
| 數 | 問的是 |
|---|---|
| `rank_image` | **影像**能不能激發 250 個基底：肖像只走過 RGB 立方體的一小塊，沒有像素落在某些 `B_i B_j B_k` 的支撐上時那一維就是零行 |
| `rank_domain` | **域**准不准走那些方向：限制 5、6 對係數差分的懲罰讓高頻方向的半徑遠小於常數方向 |

兩個都用能量佔比報（`dim_90`、`dim_99`）與參與比
`(Σλ)²/Σλ²`，不是用一個硬門檻下的秩。

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
from src.defense.delta_e_torch import delta_e00_torch  # noqa: E402
from src.metrics.perturbation_band import (blur_retention,  # noqa: E402
                                           low_frequency_share)
from src.utils.io import load_image_tensor  # noqa: E402


def energy_dims(eigenvalues: np.ndarray):
    """由特徵值（＝奇異值平方）算兩種維度。

    **兩種要分開看。** 能量維度（`dim_90`／`dim_99`／參與比）回答「擾動的
    能量集中在幾個方向」，它一定很小：125 個 Bernstein 基底加起來是 1，
    所有的行因此共用一個 `∝ −c` 的大分量。數值秩（`rank_*`）回答「有幾個
    方向根本到不了」，那才是「影像激發不出某些係數」與「域把某些方向壓成
    零」的量。單報前者會把「能量集中」誤讀成「維度塌掉」。
    """
    lam = np.clip(np.sort(eigenvalues)[::-1], 0.0, None)
    total = lam.sum()
    if total <= 0:
        return {'dim_90': 0, 'dim_99': 0, 'participation': 0.0,
                'rank_1e3': 0, 'rank_1e6': 0, 'sv_ratio_10': 0.0,
                'sv_ratio_50': 0.0, 'sv_ratio_last': 0.0}
    share = np.cumsum(lam) / total
    sv = np.sqrt(lam)
    n = len(sv)
    return {
        'dim_90': int(np.searchsorted(share, 0.90) + 1),
        'dim_99': int(np.searchsorted(share, 0.99) + 1),
        'participation': float(total ** 2 / (lam ** 2).sum()),
        'rank_1e3': int((sv > 1e-3 * sv[0]).sum()),
        'rank_1e6': int((sv > 1e-6 * sv[0]).sum()),
        'sv_ratio_10': float(sv[min(9, n - 1)] / sv[0]),
        'sv_ratio_50': float(sv[min(49, n - 1)] / sv[0]),
        'sv_ratio_last': float(sv[-1] / sv[0]),
    }


def unit(v: torch.Tensor) -> torch.Tensor:
    return v / v.norm()


def direction_family(name: str, gen: torch.Generator) -> torch.Tensor:
    """幾族方向。隨機方向只回答「平均長什麼樣」，結構方向回答「哪一類被擋」。"""
    if name == 'random':
        return unit(torch.randn(bc.DIM, generator=gen))
    if name == 'constant_a':
        return unit(bc.constant_theta(1.0, 0.0))
    if name == 'constant_b':
        return unit(bc.constant_theta(0.0, 1.0))
    if name == 'single_a':
        v = torch.zeros(bc.DIM)
        v[int(torch.randint(bc.NCOEF, (1,), generator=gen))] = 1.0
        return v
    if name == 'single_b':
        v = torch.zeros(bc.DIM)
        v[bc.NCOEF + int(torch.randint(bc.NCOEF, (1,), generator=gen))] = 1.0
        return v
    if name == 'smooth':
        # 係數場沿三軸只有最低階的變化：等價於 a、b 近似線性，差分小。
        idx = torch.arange(bc.KNOTS, dtype=torch.float32) / bc.DEG - 0.5
        w = torch.randn(6, generator=gen)
        field_a = (w[0] * idx.view(-1, 1, 1) + w[1] * idx.view(1, -1, 1)
                   + w[2] * idx.view(1, 1, -1))
        field_b = (w[3] * idx.view(-1, 1, 1) + w[4] * idx.view(1, -1, 1)
                   + w[5] * idx.view(1, 1, -1))
        return unit(bc.join(field_a, field_b))
    raise SystemExit(f'不認得的方向族 {name!r}')


FAMILIES = ('random', 'smooth', 'single_a', 'single_b',
            'constant_a', 'constant_b')


def binding_constraint(domain, theta0, direction, radius):
    """走到 `radius` 的 99.99% 處，哪一條限制的相對餘裕最小。"""
    theta = theta0 + 0.9999 * radius * direction
    slacks = domain.relative_slacks(theta)
    return min(slacks, key=slacks.get)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--manifest', type=Path,
                    default=ROOT / 'data' / 'portraits_manifest.json')
    ap.add_argument('--images', default='')
    ap.add_argument('--out', type=Path,
                    default=ROOT / 'runs' / 'bernstein_domain_check')
    ap.add_argument('--directions', type=int, default=64)
    ap.add_argument('--samples', type=int, default=250)
    ap.add_argument('--size', type=int, default=512)
    ap.add_argument('--seed', type=int, default=20260812)
    args = ap.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding='utf-8'))
    entries = {r['id']: r for r in manifest['images']}
    wanted = ([v.strip() for v in args.images.split(',') if v.strip()]
              or [r['id'] for r in manifest['images']])
    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device('cpu')

    rows = []
    detail = {}
    for name in wanted:
        t0 = time.time()
        x = load_image_tensor(ROOT / entries[name]['path'], device,
                              size=args.size)
        domain = bc.BernsteinDomain(x)
        theta0, info = bc.interior_point(domain, device=device)
        gram = bc.perturbation_gram(x)

        # ---- 內點本身的讀數 ----
        with torch.no_grad():
            y = bc.apply_filter(x, theta0)
            yq = (y.clamp(0, 1) * 255).round() / 255
            delta = (yq - x)[0].numpy().astype(np.float64)
            de = float(delta_e00_torch(x, yq))
            share = low_frequency_share(delta)
            retention = blur_retention(delta)
        lhs, rhs = domain.cone.terms(theta0)

        # ---- 半徑分布 ----
        # `hash(str)` 每個行程都不一樣（PYTHONHASHSEED），種子要自己算。
        gen = torch.Generator().manual_seed(
            args.seed + sum(ord(ch) for ch in name))
        gnp = gram.numpy()
        base_energy = float(theta0.numpy() @ gnp @ theta0.numpy())
        radii = {}
        binding = {}
        reaches = {}
        for family in FAMILIES:
            count = args.directions if family in ('random', 'smooth',
                                                  'single_a', 'single_b') else 1
            values = []
            hits = {}
            reach = []
            for _ in range(count):
                d = direction_family(family, gen).to(device)
                r = domain.max_radius(theta0, d)
                values.append(r)
                key = binding_constraint(domain, theta0, d, r)
                hits[key] = hits.get(key, 0) + 1
                dn = d.numpy()
                reach.append(r * float(np.sqrt(max(dn @ gnp @ dn, 0.0))
                                       / np.sqrt(base_energy)))
            radii[family] = values
            binding[family] = hits
            reaches[family] = reach

        # ---- 有效維度 ----
        eig_image = np.linalg.eigvalsh(gnp)
        dims_image = energy_dims(eig_image)
        deltas = []
        for _ in range(args.samples):
            d = unit(torch.randn(bc.DIM, generator=gen)).to(device)
            r = domain.max_radius(theta0, d)
            deltas.append((0.9 * r * d).numpy())
        dmat = np.asarray(deltas)
        spectrum = np.linalg.eigvalsh(dmat @ gnp @ dmat.T)
        dims_domain = energy_dims(spectrum)

        rnd = np.asarray(radii['random'])
        row = {
            'image': name,
            'alpha': round(info['alpha'], 6),
            'beta': round(info['beta'], 6),
            'worst_slack': round(info['worst_slack'], 6),
            **{f'slack_{k}': round(float(v), 6) for k, v in info.items()
               if k not in ('alpha', 'beta', 'worst_slack')},
            'interior_deltaE00': round(de, 3),
            'interior_low_freq_share': round(float(share), 6),
            'interior_blur_retention': round(float(retention), 6),
            'cone_lhs': round(lhs, 4), 'cone_rhs': round(rhs, 4),
            'identity_in_domain': int(domain.feasible(
                torch.zeros(bc.DIM, device=device))),
            **{f'rmax_{f}_median': round(float(np.median(radii[f])), 8)
               for f in FAMILIES},
            **{f'rmax_{f}_min': round(float(np.min(radii[f])), 8)
               for f in FAMILIES},
            **{f'rmax_{f}_max': round(float(np.max(radii[f])), 8)
               for f in FAMILIES},
            'rmax_random_p05': round(float(np.quantile(rnd, 0.05)), 8),
            'rmax_random_p95': round(float(np.quantile(rnd, 0.95)), 8),
            **{f'image_{k}': (v if isinstance(v, int) else round(v, 6))
               for k, v in dims_image.items()},
            **{f'domain_{k}': (v if isinstance(v, int) else round(v, 6))
               for k, v in dims_domain.items()},
            **{f'reach_{f}_median': round(float(np.median(reaches[f])), 6)
               for f in FAMILIES},
            **{f'reach_{f}_max': round(float(np.max(reaches[f])), 6)
               for f in FAMILIES},
            'domain_sample_count': args.samples,
            'seconds': round(time.time() - t0, 1),
        }
        rows.append(row)
        detail[name] = {
            'binding': binding,
            'reaches': {k: [float(v) for v in reaches[k]] for k in reaches},
            'radii': {k: [float(v) for v in radii[k]] for k in radii},
            'eig_image': [float(v) for v in np.sort(eig_image)[::-1]],
            'eig_domain': [float(v) for v in np.sort(spectrum)[::-1]],
        }
        print(f'{name}: α={info["alpha"]:.4f} β={info["beta"]:.4f} '
              f'最小餘裕 {info["worst_slack"]:+.4f} ΔE00 {de:.2f} '
              f'低頻 {share:.4f} r_max(random) 中位 '
              f'{np.median(rnd):.5f} 影像秩 {dims_image["rank_1e3"]}/'
              f'{dims_image["rank_1e6"]} 域秩 {dims_domain["rank_1e3"]}/'
              f'{dims_domain["rank_1e6"]} 能量維度 {dims_image["dim_99"]}→'
              f'{dims_domain["dim_99"]} {row["seconds"]:.0f}s', flush=True)

    csv_path = args.out / 'domain_check.csv'
    with csv_path.open('w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (args.out / 'domain_check_detail.json').write_text(
        json.dumps(detail, ensure_ascii=False, indent=1), encoding='utf-8')
    print(f'寫出 {csv_path}（{len(rows)} 列）', flush=True)


if __name__ == '__main__':
    main()

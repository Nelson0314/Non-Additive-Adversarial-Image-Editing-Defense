"""幾何印花貼片的未最佳化起點：建支撐、印上去、出圖給人看。

這一支不跑 IP2P、不最佳化任何東西、不含指令。目的只有兩個：
量出每種形狀在這張影像上**放得下多大**，以及印上去像不像衣服上的圖案。
自然度一律由使用者看圖判定，這裡只負責把圖擺出來。

支撐放不下要求的面積時**拋錯**並記成那個組合的適用性失敗，不自動縮小、
也不改放到臉上。
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

COLUMNS = ['image', 'variant', 'shape', 'count', 'pitch', 'radius',
           'archetype', 'colours', 'chroma_max', 'tau', 'patch_area',
           'deltaE00', 'deltaE00_patch', 'psnr', 'ssim', 'lpips', 'linf',
           'subject_id_defended', 'print_chroma_used', 'print_lightness_span',
           'print_share_min', 'note']


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', type=Path,
                    default=ROOT / 'data' / 'portrait_trio_manifest.json')
    ap.add_argument('--image', default='task_env_weather_121086')
    ap.add_argument('--shapes', nargs='+', default=['disc', 'square'])
    ap.add_argument('--counts', type=int, nargs='+', default=[1, 4])
    ap.add_argument('--areas', type=float, nargs='+', default=[0.025])
    ap.add_argument('--lattice', type=int, nargs='+', default=[16, 32],
                    help='點陣的 pitch；radius 取 pitch 的 0.45 倍')
    ap.add_argument('--archetypes', nargs='+',
                    default=['dots', 'checks', 'stripes', 'rings'])
    ap.add_argument('--colours', type=int, nargs='+', default=[3])
    ap.add_argument('--chroma-max', type=float, nargs='+', default=[28.0])
    ap.add_argument('--tau', type=float, default=0.15)
    ap.add_argument('--lightness-span', type=float, default=34.0)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', type=Path, default=ROOT / 'runs' / 'print_probe')
    args = ap.parse_args()

    import torch

    from src.defense.assets import load_image, save_png
    from src.defense.carrier_mask import lattice_support
    from src.defense.color_amplitude import delta_e00
    from src.defense.print_patch import (GeometricPrintParam, place_shapes,
                                         safe_mask)
    from src.metrics.identity import embed, similarity
    from src.metrics.suite import MetricSuite

    device = torch.device('cpu')
    manifest = json.loads(args.manifest.read_text(encoding='utf-8'))
    entry = next(e for e in manifest['images'] if e['id'] == args.image)
    args.out.mkdir(parents=True, exist_ok=True)
    suite = MetricSuite(device=device)

    x = load_image(ROOT / entry['path'], device)
    save_png(x, args.out / f'{args.image}__original.png')
    e_orig = embed(x, device=device)
    mask = safe_mask(x, device=device)
    save_png(mask.expand_as(x), args.out / f'{args.image}__safe_mask.png')
    print(f'安全區面積 {float((mask > 0.5).float().mean()):.4f}', flush=True)

    supports = []
    for shape in args.shapes:
        for count in args.counts:
            for area in args.areas:
                tag = f'{shape}{count}_a{area:g}'
                try:
                    sup = place_shapes(mask, shape=shape, area=area,
                                       count=count)
                except ValueError as exc:
                    print(f'{tag} 適用性失敗：{exc}', flush=True)
                    continue
                supports.append((tag, sup, dict(shape=shape, count=count,
                                                pitch='', radius='')))
    for pitch in args.lattice:
        radius = max(1.0, round(0.45 * pitch, 1))
        tag = f'lattice{pitch}'
        try:
            sup = lattice_support(mask, pitch, radius)
        except ValueError as exc:
            print(f'{tag} 適用性失敗：{exc}', flush=True)
            continue
        supports.append((tag, sup, dict(shape='lattice', count='',
                                        pitch=pitch, radius=radius)))

    rows = []
    for tag, sup, meta in supports:
        save_png(sup.expand_as(x), args.out / f'{args.image}__support_{tag}.png')
        area_got = float((sup > 0.5).float().mean())
        print(f'{tag}: 支撐面積 {area_got:.4f}', flush=True)
        for arch in args.archetypes:
            for colours in args.colours:
                for chroma in args.chroma_max:
                    variant = f'{tag}__{arch}_k{colours}_c{chroma:g}'
                    carrier = GeometricPrintParam(
                        sup, colours=colours, tau=args.tau,
                        chroma_max=chroma, archetype=arch,
                        lightness_span=args.lightness_span)
                    carrier.reset(x, args.seed)
                    with torch.no_grad():
                        y = (carrier.render(x).clamp(0, 1) * 255).round() / 255
                    save_png(y, args.out / f'{args.image}__{variant}.png')
                    m = suite.pairwise(x, y)
                    row = {c: '' for c in COLUMNS}
                    row.update({
                        'image': args.image, 'variant': variant,
                        'archetype': arch, 'colours': colours,
                        'chroma_max': chroma, 'tau': args.tau,
                        'patch_area': round(area_got, 5),
                        'deltaE00': round(float(delta_e00(x, y)), 4),
                        'deltaE00_patch': round(
                            float(delta_e00(x, y, support=sup)), 4),
                        'psnr': round(float(m['psnr']), 3),
                        'ssim': round(float(m['ssim']), 4),
                        'lpips': round(float(m['lpips']), 4),
                        'linf': round(float((y - x).abs().max()), 4),
                        'subject_id_defended': round(float(similarity(
                            e_orig, embed(y, device=device))), 4),
                    })
                    row.update({k: v for k, v in meta.items() if v != ''})
                    row.update({k: v for k, v in carrier.readout().items()
                                if k in COLUMNS})
                    rows.append(row)
                    print(f'  {variant}: ΔE00貼片 {row["deltaE00_patch"]}, '
                          f'身分 {row["subject_id_defended"]}', flush=True)

    path = args.out / 'print_probe.csv'
    with path.open('w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, '') for k in COLUMNS})
    print(f'{len(rows)} 列寫進 {path}', flush=True)


if __name__ == '__main__':
    main()

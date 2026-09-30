"""在三張人像上建固定支撐並算**未最佳化**的材質貼片，出圖給人看。

順序是刻意的：支撐與色盤的可用性、以及「印上去還像不像一件衣服」，都要在
花任何 GPU 之前先用眼睛判掉。這一支不跑 IP2P、不最佳化任何東西、不含指令。

支撐放不下要求的面積時**拋錯**並記成那張影像的適用性失敗，不自動縮小、
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

COLUMNS = ['image', 'variant', 'area', 'blobs', 'grid', 'palette',
           'palette_mode', 'chroma_gain', 'logit_cap', 'init_scale', 'patch_area', 'patch_weight_mean',
           'deltaE00', 'deltaE00_patch', 'psnr', 'ssim', 'lpips', 'linf',
           'subject_id_defended', 'note']


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', type=Path,
                    default=ROOT / 'data' / 'portrait_trio_manifest.json')
    ap.add_argument('--area', type=float, default=0.03)
    ap.add_argument('--blobs', type=int, nargs='+', default=[2])
    ap.add_argument('--grids', type=int, nargs='+', default=[8, 16])
    ap.add_argument('--palette', type=int, default=6)
    ap.add_argument('--chroma-gains', type=float, nargs='+', default=[1.0, 2.0])
    ap.add_argument('--palette-modes', nargs='+', default=['garment', 'print'])
    ap.add_argument('--chroma', type=float, default=40.0)
    ap.add_argument('--logit-caps', type=float, nargs='+', default=[6.0])
    ap.add_argument('--init-scales', type=float, nargs='+', default=[1.0])
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1])
    ap.add_argument('--out', type=Path, default=ROOT / 'runs' / 'patch_probe')
    args = ap.parse_args()

    import torch

    from src.defense.assets import load_image, save_png
    from src.defense.color_amplitude import delta_e00
    from src.defense.material_patch import (MaterialPatchParam, garment_palette,
                                            patch_support)
    from src.metrics.identity import embed, similarity
    from src.metrics.suite import MetricSuite

    device = torch.device('cpu')
    manifest = json.loads(args.manifest.read_text(encoding='utf-8'))
    args.out.mkdir(parents=True, exist_ok=True)
    suite = MetricSuite(device=device)

    rows = []
    for entry in manifest['images']:
        image = entry['id']
        x = load_image(ROOT / entry['path'], device)
        save_png(x, args.out / f'{image}__original.png')
        e_orig = embed(x, device=device)

        for blobs in args.blobs:
            try:
                sup = patch_support(x, area=args.area, blobs=blobs,
                                    device=device)
            except ValueError as exc:
                rows.append({'image': image, 'variant': f'b{blobs}',
                             'area': args.area, 'blobs': blobs,
                             'note': f'適用性失敗：{exc}'})
                print(f'{image} blobs={blobs} 適用性失敗：{exc}', flush=True)
                continue
            save_png(sup.expand_as(x),
                     args.out / f'{image}__support_b{blobs}.png')

            combos = [('garment', g) for g in args.chroma_gains
                      if 'garment' in args.palette_modes]
            combos += [('print', args.chroma)] if 'print' in args.palette_modes else []
            for pmode, gain in combos:
                pal = garment_palette(x, sup, args.palette, chroma_gain=gain,
                                      mode=pmode, chroma=args.chroma)
                for grid in args.grids:
                  for cap in args.logit_caps:
                   for scale in args.init_scales:
                    for seed in args.seeds:
                        p = MaterialPatchParam(sup, pal, grid=grid,
                                               logit_cap=cap,
                                               init_jitter=scale)
                        p.reset(x, seed=seed)
                        p.project()
                        y = p.render(x).detach()
                        tag = 'print' if pmode == 'print' else f'c{gain:g}'
                        name = (f'b{blobs}__g{grid}__{tag}__k{cap:g}'
                                f'__a{scale:g}__s{seed}')
                        save_png(y, args.out / f'{image}__{name}.png')
                        pw = suite.pairwise(x, y)
                        e_def = embed(y, device=device)
                        row = {'image': image, 'variant': name,
                               'area': args.area, 'blobs': blobs,
                               'grid': grid, 'palette': args.palette,
                               'chroma_gain': gain,
                               'palette_mode': pmode,
                               'logit_cap': cap, 'init_scale': scale,
                               'deltaE00': round(pw['deltaE00'], 4),
                               'deltaE00_patch': round(
                                   delta_e00(x, y, support=sup), 4),
                               'psnr': round(pw['psnr'], 4),
                               'ssim': round(pw['ssim'], 5),
                               'lpips': round(pw['lpips'], 5),
                               'linf': round(pw['linf'], 5),
                               'subject_id_defended': (
                                   '' if e_def is None
                                   else round(similarity(e_orig, e_def), 5)),
                               'note': ''}
                        row.update(p.readout())
                        rows.append(row)
                        print(json.dumps(
                            {k: row[k] for k in
                             ('image', 'variant', 'patch_area', 'deltaE00',
                              'deltaE00_patch', 'psnr', 'lpips',
                              'subject_id_defended')},
                            ensure_ascii=False), flush=True)

    with (args.out / 'readout.csv').open('w', newline='',
                                         encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)
    print(f'wrote {len(rows)} rows to {args.out / "readout.csv"}', flush=True)


if __name__ == '__main__':
    main()

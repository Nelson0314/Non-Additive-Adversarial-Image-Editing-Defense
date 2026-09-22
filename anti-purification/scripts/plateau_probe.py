"""單張人像上把亮度平台曲線掃一遍，出圖給人看。

**這一支不含任何指令，也不讀任何評估結果。** 它只做三件事：
用固定的柔邊主體遮罩、由原圖亮度分布固定平台中心、掃半寬與斜率，
把發布圖與讀數存下來。自然度由使用者看圖判定，不由這裡的任何欄位判定。

對照組是 `slope_floor`：同一個遮罩、同一組中心、同一個半寬，只把斜率的
下限從 0 換成 `ColorCurveParam(radius=5)` 的有效斜率下限 0.169，
也就是「同遮罩但保留正斜率」的壓縮版。兩者的差只有平台壓不壓得平。
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

COLUMNS = ['image', 'variant', 'mode', 'half_width', 'slope_floor',
           'flat_coverage', 'clip_fraction', 'mask_core_area', 'mask_mean',
           'deltaE00', 'deltaE00_subject', 'psnr', 'ssim', 'lpips',
           'linf', 'subject_id_defended', 'centres']

SLOPE_FLOOR_RADIUS5 = 0.169


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--image', default='task_env_weather_121086')
    ap.add_argument('--manifest', type=Path,
                    default=ROOT / 'data' / 'portrait_trio_manifest.json')
    ap.add_argument('--subject-text', default='a person')
    ap.add_argument('--half-widths', type=float, nargs='+',
                    default=[0.02, 0.04, 0.06, 0.09, 0.13])
    ap.add_argument('--modes', nargs='+',
                    default=['luma_gain', 'per_channel'])
    ap.add_argument('--pieces', type=int, default=256)
    ap.add_argument('--transition', type=float, default=0.03)
    ap.add_argument('--out', type=Path, default=ROOT / 'runs' / 'plateau_probe')
    args = ap.parse_args()

    import torch

    from src.defense.assets import load_image, save_png
    from src.defense.color_amplitude import delta_e00
    from src.defense.plateau_curve import PlateauCurveParam, centres_from_luma
    from src.defense.subject_mask import mask_stats, subject_mask
    from src.metrics.identity import embed, similarity
    from src.metrics.suite import MetricSuite

    device = torch.device('cpu')
    manifest = json.loads(args.manifest.read_text(encoding='utf-8'))
    entry = next(e for e in manifest['images'] if e['id'] == args.image)
    x = load_image(ROOT / entry['path'], device)

    mask = subject_mask(x, args.subject_text, device=device)
    stats = mask_stats(mask)
    centres = centres_from_luma(x, weight=mask)

    args.out.mkdir(parents=True, exist_ok=True)
    save_png(x, args.out / f'{args.image}__original.png')
    save_png(mask.expand_as(x), args.out / f'{args.image}__mask.png')

    suite = MetricSuite(device=device)
    e_orig = embed(x, device=device)

    rows = []
    for mode in args.modes:
        for hw in args.half_widths:
            for floor in (0.0, SLOPE_FLOOR_RADIUS5):
                p = PlateauCurveParam(centres, pieces=args.pieces,
                                      transition=args.transition,
                                      slope_floor=floor, mode=mode,
                                      apply_where=mask)
                p.reset(x, half_width=hw, slope=0.0)
                p.project()
                y = p.render(x).detach()
                tag = 'flat' if floor == 0.0 else 'sloped'
                name = f'{mode}__hw{hw:.2f}__{tag}'
                save_png(y, args.out / f'{args.image}__{name}.png')
                pw = suite.pairwise(x, y)
                e_def = embed(y, device=device)
                rows.append({
                    'image': args.image, 'variant': name, 'mode': mode,
                    'half_width': hw, 'slope_floor': floor,
                    'flat_coverage': round(p.flat_coverage(x, weight=mask), 5),
                    'clip_fraction': round(p.clip_fraction(x), 5),
                    'mask_core_area': stats['mask_core_area'],
                    'mask_mean': stats['mask_mean'],
                    'deltaE00': round(pw['deltaE00'], 4),
                    'deltaE00_subject': round(delta_e00(x, y, support=mask), 4),
                    'psnr': round(pw['psnr'], 4),
                    'ssim': round(pw['ssim'], 5),
                    'lpips': round(pw['lpips'], 5),
                    'linf': round(pw['linf'], 5),
                    'subject_id_defended': ('' if e_def is None
                                            else round(similarity(e_orig, e_def), 5)),
                    'centres': ' '.join(f'{c:.4f}' for c in centres),
                })
                print(json.dumps(rows[-1], ensure_ascii=False))

    with (args.out / 'readout.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    print(f'wrote {len(rows)} rows to {args.out / "readout.csv"}')


if __name__ == '__main__':
    main()

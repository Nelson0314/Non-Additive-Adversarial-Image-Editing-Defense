"""色盤預覽、ΔE00 可達下界與 PNG／頻域讀數，預設使用 CPU。

範例（在遠端 colour-lab 根目錄執行）：
    python scripts/bounded_cast_preview.py --steps 200 --restarts 3 --threads 4

預設讀取 manifest 的八張人像，以 512² 搜尋並交付 PNG。所有候選只依
全圖 ΔE00 選取；頻域與膚色帶涵蓋率均為事後讀數，不參與選解。
有限次梯度上升得到的是可達振幅的下界估計，不能稱為證明的天花板。
CSV 分開記錄浮點 render 與實際 PNG；結構證書不包含 8-bit 量化誤差。

--device 只改變搜尋張量所在裝置，不改變 dtype。prepare 的色彩轉換、
安全半徑與 render_prepared 的 B-spline / Lab 反轉換保留 float64；降精度
會削弱色域邊界的數值保證。skimage 的證書讀數與頻域統計也保留 float64。
讀圖、參數、render 回傳值、可微 ΔE00 與 PNG 原本沿用輸入 dtype
（CLI 讀圖為 float32），維持原樣；本次沒有新增任何 float32 降精度。
PNG、skimage 與頻域讀數留在 CPU，不參與裝置上的搜尋。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image
import torch

from src.defense.bounded_cast_curve import BoundedCastCurveParam, DEFAULT_BOUNDS
from src.defense.delta_e_torch import delta_e00_torch


def bound_pair(value):
    """解析 C_max:d_max，錯誤訊息交給 argparse。"""
    try:
        c, d = (float(v) for v in value.split(':'))
        BoundedCastCurveParam(C_max=c, d_max=d)
        return c, d
    except (ValueError, TypeError) as exc:
        raise argparse.ArgumentTypeError(f'界必須是可建構的 C_max:d_max：{exc}') from exc


def build_parser():
    """測試直接使用這些實際起點設定，避免另造一份測試專用設定。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=ROOT / 'data/portraits_manifest.json')
    parser.add_argument('--images', default='', help='以逗號指定 ID；預設為 manifest 全部八張')
    parser.add_argument('--out', type=Path, default=ROOT / 'runs/bounded_cast_preview')
    parser.add_argument('--bounds', nargs='+', type=bound_pair, default=list(DEFAULT_BOUNDS))
    parser.add_argument('--size', type=int, default=512)
    parser.add_argument('--steps', type=int, default=200)
    parser.add_argument('--restarts', type=int, default=3)
    parser.add_argument('--lr', type=float, default=0.08)
    parser.add_argument('--pieces', type=int, default=64)
    parser.add_argument('--skin-width', type=float, default=8.0)
    parser.add_argument('--init-jitter', type=float, default=0.0)
    parser.add_argument('--restart-jitter', type=float, default=0.5)
    parser.add_argument('--seed', type=int, default=11)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--device', type=torch.device, default=torch.device('cpu'),
                        help='搜尋裝置，例如 cpu、cuda 或 cuda:0；保留證書核心的 float64')
    return parser


def make_carrier(args, pair, restart):
    """第一輪從零參數內點出發，其餘輪使用可重現的不同起點。"""
    return BoundedCastCurveParam(
        C_max=pair[0], d_max=pair[1], pieces=args.pieces, skin_width=args.skin_width,
        init_jitter=args.init_jitter if restart == 0 else args.restart_jitter)


def load_image(path, size, device='cpu'):
    """先照原路徑讀取 float32 像素，再搬到指定裝置，不改變像素值。"""
    with Image.open(path) as source:
        im = source.convert('RGB')
        if im.size != (size, size):
            im = im.resize((size, size), Image.Resampling.LANCZOS)
        pixels = np.array(im, dtype=np.uint8, copy=True)
    x = torch.from_numpy(pixels).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    return x.to(device=device)


def save_png(path, image):
    """只做 PNG 所需量化；不偷偷裁切。回傳實際寫入的 8-bit 像素。"""
    x = image.detach().cpu()
    if not torch.isfinite(x).all() or float(x.min()) < 0 or float(x.max()) > 1:
        raise FloatingPointError('render 超出色域或不是有限值，拒絕用裁切修補')
    pixels = (x[0].permute(1, 2, 0) * 255).round().to(torch.uint8).numpy()
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pixels).save(path)
    return torch.from_numpy(pixels.copy()).permute(2, 0, 1).unsqueeze(0).float() / 255.0


def measured_maps(x, y):
    """量測走 skimage，與 repo 現行 ΔE00 讀數相同；沒有裁切路徑。"""
    from skimage.color import deltaE_ciede2000, rgb2lab
    a = x.detach().cpu().double().permute(0, 2, 3, 1).numpy()
    b = y.detach().cpu().double().permute(0, 2, 3, 1).numpy()
    la, lb = rgb2lab(a), rgb2lab(b)
    return deltaE_ciede2000(la, lb), np.linalg.norm(lb[..., 1:], axis=-1)


def ascend(x, args, pair, image_seed):
    """以 Adam 最小化 -ΔE00；保存每一步（含起點、最後一步）的最佳候選。"""
    best = None
    for restart in range(args.restarts):
        carrier = make_carrier(args, pair, restart)
        seed = image_seed + 1009 * restart
        carrier.reset(x, seed)
        prepared = carrier.prepare(x)
        # 球半徑為零的 RGB 邊界是固定端點。移除它們的反向圖，避免既有
        # CIEDE2000 在黑點 atan2(0,0) 的 NaN；全圖平均的分母仍保留它們。
        active = prepared.radius.reshape(-1) > 0
        active_x = x.reshape(1, 3, 1, -1)[..., active]
        active_prepared = carrier.prepare(active_x) if bool(active.any()) else None
        fraction = float(active.double().mean())
        optimizer = torch.optim.Adam(carrier.params(), lr=args.lr)
        restart_best = None
        for step in range(args.steps + 1):
            optimizer.zero_grad(set_to_none=True)
            if active_prepared is None:
                value = 0.0
            else:
                y = carrier.render_prepared(active_prepared)
                score = delta_e00_torch(active_x, y) * fraction
                value = float(score.detach())
                if not math.isfinite(value):
                    raise FloatingPointError(f'restart={restart}, step={step} 的 ΔE00 非有限值')
            if restart_best is None or value > restart_best[0]:
                restart_best = value, step, carrier.state_dict()
            if step == args.steps or active_prepared is None:
                break
            (-score).backward()
            for name, param in zip(('theta_l', 'theta_a', 'theta_b'), carrier.params()):
                if param.grad is None or not torch.isfinite(param.grad).all():
                    raise FloatingPointError(f'{name} 在 restart={restart}, step={step} 梯度異常')
            optimizer.step()
            carrier.project()
        carrier.load_state_dict(restart_best[2])
        with torch.no_grad():
            output = carrier.render_prepared(prepared)
        measured = float(measured_maps(x, output)[0].mean())
        print(f'    起點 {restart + 1}/{args.restarts} seed={seed}: '
              f'可達下界 {measured:.4f}，最佳步 {restart_best[1]}', flush=True)
        if best is None or measured > best['value']:
            best = dict(value=measured, carrier=carrier, image=output,
                        restart=restart, step=restart_best[1], seed=seed)
    return best


def readout(x, y, png, carrier):
    """頻域只在選解完成後量測，不參與 objective 或 checkpoint 選擇。"""
    from src.metrics.perturbation_band import blur_retention, low_frequency_share
    # PIL／skimage／NumPy 的讀數集中在 CPU，避免 PNG 與搜尋張量跨裝置相減。
    x, y, png = (t.detach().cpu() for t in (x, y, png))
    raw_de, raw_c = measured_maps(x, y)
    png_de, png_c = measured_maps(x, png)
    delta = png - x
    l2 = float(delta.norm())
    skin = carrier.skin_band(x).cpu().numpy()
    return {
        'raw_deltaE00_lower_bound': float(raw_de.mean()),
        'png_deltaE00': float(png_de.mean()),
        'reference_deltaE00': 15.7,
        'raw_minus_reference': float(raw_de.mean()) - 15.7,
        'png_L2': l2,
        'png_RGB_RMS': float(delta.square().mean().sqrt()),
        'raw_chroma_max': float(raw_c.max()), 'png_chroma_max': float(png_c.max()),
        'skin_pixels': int(skin.sum()), 'skin_fraction': float(skin.mean()),
        'raw_skin_deltaE00_max': float(raw_de[skin].max()) if skin.any() else float('nan'),
        'png_skin_deltaE00_max': float(png_de[skin].max()) if skin.any() else float('nan'),
        'low_frequency_share': low_frequency_share(delta[0].numpy()),
        'blur2_L2_retention': blur_retention(delta[0].numpy()),
    }


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if min(args.size, args.steps, args.restarts, args.threads) < 1:
        parser.error('size、steps、restarts、threads 必須為正整數')
    if not math.isfinite(args.lr) or args.lr <= 0:
        parser.error('lr 必須為正且有限')
    if args.restarts > 1 and args.restart_jitter <= 0:
        parser.error('多起點搜尋需要 restart-jitter > 0')
    for pair in args.bounds:
        for restart in (0, 1):
            try:
                make_carrier(args, pair, restart)
            except ValueError as exc:
                parser.error(str(exc))
    torch.set_num_threads(args.threads)
    data = json.loads(args.manifest.read_text(encoding='utf-8'))['images']
    entries = {row['id']: row for row in data}
    names = args.images.split(',') if args.images else list(entries)
    if len(set(names)) != len(names) or any(name not in entries for name in names):
        parser.error('images 必須是不重複且存在於 manifest 的 ID')
    args.out.mkdir(parents=True, exist_ok=True)
    print(f'{str(args.device).upper()} 預覽：數字是多起點搜尋的可達下界；自然度交由看圖裁定。',
          flush=True)
    print('結構界針對浮點 render；PNG 量化前後分欄。L2 是全部 RGB 元素的 Euclidean norm。',
          flush=True)
    print(f'膚色帶：固定暖色軌跡的 {args.skin_width:g} Lab 半徑管帶；不是所有人臉像素。',
          flush=True)
    rows = []
    for name in names:
        x = load_image(ROOT / entries[name]['path'], args.size, device=args.device)
        save_png(args.out / 'originals' / f'{name}.png', x)
        for pair in args.bounds:
            c_max, d_max = pair
            tag = f'C{c_max:g}_d{d_max:g}'
            print(f'{name} / {tag}', flush=True)
            start = time.monotonic()
            image_seed = args.seed + sum((i + 1) * ord(c) for i, c in enumerate(name))
            best = ascend(x, args, pair, image_seed)
            path = args.out / tag / f'{name}.png'
            png = save_png(path, best['image'])
            row = dict(image=name, C_max=c_max, d_max=d_max, skin_width=args.skin_width,
                       pieces=args.pieces, parameters=3 * args.pieces, size=args.size,
                       steps=args.steps, restarts=args.restarts, lr=args.lr,
                       init_jitter=args.init_jitter, restart_jitter=args.restart_jitter,
                       best_restart=best['restart'], best_step=best['step'], seed=best['seed'],
                       png=str(path.relative_to(args.out)),
                       **readout(x, best['image'], png, best['carrier']))
            row['seconds'] = time.monotonic() - start
            rows.append(row)
            # 每張完成就寫入，遠端長任務中斷時仍保留已完成的量測。
            with (args.out / 'results.csv').open('w', newline='', encoding='utf-8') as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            print(f"    PNG ΔE00={row['png_deltaE00']:.3f}；"
                  f"膚色帶涵蓋率={row['skin_fraction']:.1%}", flush=True)
    for pair in args.bounds:
        group = [r for r in rows if (r['C_max'], r['d_max']) == pair]
        print(f'{pair}: {len(group)} 張，可達下界中位 '
              f"{np.median([r['raw_deltaE00_lower_bound'] for r in group]):.3f} "
              f"（現行參考 15.7）", flush=True)
    print(f'CSV：{args.out / "results.csv"}', flush=True)


if __name__ == '__main__':
    main()

"""影像免疫任務常用的位移指標，逐格算在**已存下來的編輯圖**上。

三組比較
────────────────────────────────────────────────────────────────────
`before`  未淨化：`clean_edit` 對 `def_edit`
`after`   同一道淨化之後：`clean_edit(purified)` 對 `def_edit(purified)`
`gain`    `after − before`，也就是**淨化吃掉（或放大）了多少位移**

`before` 量的是「防禦把攻擊者的輸出推開多遠」。單看它會高估：攻擊者只要
做一次 JPEG 或裁切，推開的部分可能就還原了。`after` 是淨化之後還剩多少，
`gain` 是兩者的差。三個一起報，防禦的效果才有方向。

**這一支不重跑擴散。** `scripts/evaluate_defence.py` 已經把每一格的
`clean_edit` 與 `def_edit` 逐檔存下來（含各道淨化），這裡只讀檔算指標。
所以它可以在任何時候補跑，也不必佔用產圖的卡。

指標
────────────────────────────────────────────────────────────────────
`MetricSuite.pairwise` 的 psnr／ssim／lpips／dists／linf／rms，加
`MetricSuite.image_similarity` 的 clip／siglip 餘弦。前六項越大（psnr/ssim）
或越小（lpips/dists/linf/rms）代表兩張圖越接近；後兩項是語意空間的餘弦，
越小代表推得越開。**方向不一致，讀表時看欄名。**
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PAIR_KEYS = ('psnr', 'ssim', 'lpips', 'dists', 'linf', 'rms')
SEM_KEYS = ('clip', 'siglip')

def parse_stem(stem: str):
    """把檔名切成 (image, class, purifier, seed, kind)，切不出來回傳 None。

    **不要用正則的 `[a-z_]+` 去吃 purifier**：那個字元類含底線，會把
    `identity__immunised` 整段吞掉，於是 `def_edit` 解析出的 purifier 與
    `clean_edit` 的不同，兩者永遠配不成對——第一次跑就是這樣，864 組全部
    「缺一邊」而靜默寫出 0 列。依 `__` 切開再按位置取，沒有這個歧義。
    """
    parts = stem.split('__')
    if len(parts) < 5 or parts[-1] not in ('def_edit', 'clean_edit'):
        return None
    kind = 'def' if parts[-1] == 'def_edit' else 'clean'
    rest = parts[:-1]
    if not rest[-1].startswith('s') or not rest[-1][1:].isdigit():
        return None
    seed = int(rest[-1][1:])
    rest = rest[:-1]
    if rest and rest[-1] == 'immunised':
        rest = rest[:-1]
    if len(rest) < 3:
        return None
    pur, cls = rest[-1], rest[-2]
    image = '__'.join(rest[:-2])
    return image, cls, pur, seed, kind


def index_edits(root: Path):
    """回傳 {(variant, image, class, purifier, seed): {'clean': p, 'def': p}}。"""
    out = {}
    for p in root.rglob('*_edit.png'):
        parsed = parse_stem(p.stem)
        if not parsed:
            continue
        image, cls, pur, seed, kind = parsed
        variant = p.relative_to(root).parts[0]
        out.setdefault((variant, image, cls, pur, seed), {})[kind] = p
    return out


def main():
    import torch

    from src.defense.assets import load_image
    from src.metrics.suite import MetricSuite

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--edits', type=Path, required=True,
                    help='scripts/evaluate_defence.py 的輸出根目錄')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--shard', default='1/1')
    args = ap.parse_args()

    pairs = index_edits(args.edits)
    keys = sorted(k for k, v in pairs.items() if 'clean' in v and 'def' in v)
    dropped = len(pairs) - len(keys)
    i, n = (int(v) for v in args.shard.split('/'))
    keys = [k for j, k in enumerate(keys) if j % n == i - 1]
    args.out.mkdir(parents=True, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    suite = MetricSuite(device=device)
    print(f'分片 {i}/{n}：{len(keys)} 對'
          f'{f"（另有 {dropped} 組缺一邊，跳過）" if dropped else ""}', flush=True)

    rows = []
    for k, (variant, image, cls, pur, seed) in enumerate(keys):
        a = load_image(pairs[(variant, image, cls, pur, seed)]['clean'], device)
        b = load_image(pairs[(variant, image, cls, pur, seed)]['def'], device)
        with torch.no_grad():
            pw = suite.pairwise(a, b)
            sem = suite.image_similarity(a, b)
        row = {'variant': variant, 'image': image, 'class': cls,
               'purifier': pur, 'seed': seed}
        row.update({m: round(float(pw[m]), 5) for m in PAIR_KEYS if m in pw})
        row.update({m: round(float(sem[m]), 5) for m in SEM_KEYS if m in sem})
        rows.append(row)
        if (k + 1) % 25 == 0 or k + 1 == len(keys):
            print(f'  {k + 1}/{len(keys)}', flush=True)

    suffix = '' if n == 1 else f'_shard{i}of{n}'
    fields = ['variant', 'image', 'class', 'purifier', 'seed',
              *PAIR_KEYS, *SEM_KEYS]
    out_csv = args.out / f'edit_distance{suffix}.csv'
    with open(out_csv, 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)
    print(f'寫出 {out_csv}（{len(rows)} 列）', flush=True)


if __name__ == '__main__':
    main()

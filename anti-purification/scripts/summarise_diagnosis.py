"""把診斷兩批的產圖列與真編輯列接起來，逐臂報四個讀數。

回答的問題
────────────────────────────────────────────────────────────────────
既有 43 個設定的過半穿透與**防禦圖本身的身分餘弦**，Pearson r 是 −0.902，
逐張的 `id_norm ÷ 防禦圖身分` 中位是 0.981。那有兩種解釋：編輯把輸入的臉近乎
無損搬到輸出（於是任何把臉改成同樣程度的重取樣都會得到同樣的讀數），或者對抗
方向真的有貢獻、只是量級剛好相同。這支腳本把非對抗隨機臂與對抗臂放在同一張
「發布圖身分 → 過半穿透」的圖上，落點的高低差就是分得開的那一段。

四個讀數
────────────────────────────────────────────────────────────────────
`pub_id`     防禦圖本身的身分餘弦，錨在原圖主體框的固定座標上，偵測器不參與。
`majority`   過半規則下的穿透格數（每格三顆種子至少兩顆 criterion < 0.5）。
`extra`      `id_norm ÷ pub_id`。1.0 表示編輯完全沒有再多傷身分。
失真         `lpips`、`psnr`、`flow_px_max`、`deltaE00`，兩臂在同一個身分上的
             失真量不會相等，那個差本身就是結果的一部分，照報不省。

**不下「成立／不成立」的結論。** 這支腳本只把數字與影像路徑擺出來。
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

FIELDS = ('pub_id', 'majority', 'cells', 'extra', 'id_norm', 'use_norm',
          'lpips', 'psnr', 'flow_px_max', 'deltaE00', 'cap_reached',
          'cap_values', 'violations', 'rejected')


def _rows(pattern):
    for p in glob.glob(pattern, recursive=True):
        with open(p, encoding='utf-8') as fh:
            for r in csv.DictReader(fh):
                yield p, r


def _f(r, key):
    v = r.get(key)
    if v in (None, '', 'nan'):
        return None
    try:
        return float(v)
    except ValueError:
        return None


def immunise_table(batch: Path):
    """逐（變體, 影像）讀防禦圖那一側。"""
    out = {}
    for _, r in _rows(str(batch / 'shard*' / '*.csv')):
        out[(r['variant'], r['image'])] = r
    return out


def evaluate_table(batch_eval: Path):
    """逐（變體, 影像）收真編輯列，只取 immunised 臂。"""
    per = defaultdict(list)
    for p, r in _rows(str(batch_eval / '*' / 'shard*' / '*.csv')):
        if r.get('arm') != 'immunised':
            continue
        variant = os.path.basename(os.path.dirname(os.path.dirname(p)))
        per[(variant, r['image'])].append(r)
    return per


def majority_cells(rows):
    """過半規則：每個（影像, 指令類, 淨化）格裡至少兩顆種子 criterion < 0.5。"""
    cells = defaultdict(list)
    for r in rows:
        c = _f(r, 'criterion')
        if c is None:
            continue
        cells[(r['image'], r['class'], r['purifier'])].append(c)
    hit = sum(1 for v in cells.values() if sum(1 for c in v if c < 0.5) * 2 > len(v))
    return hit, len(cells)


def summarise(batch: Path, batch_eval: Path):
    imm = immunise_table(batch)
    ev = evaluate_table(batch_eval)
    variants = sorted({v for v, _ in imm} | {v for v, _ in ev})
    table = []
    for v in variants:
        pub, extra, idn, usen = [], [], [], []
        lp, ps, px, de = [], [], [], []
        caps_r, caps_v, viol, rej = '', '', 0, 0
        rows_all = []
        images = sorted({im for vv, im in imm if vv == v}
                        | {im for vv, im in ev if vv == v})
        for im in images:
            r = imm.get((v, im), {})
            vv = v
            p = _f(r, 'subject_id_defended')
            if p is not None:
                pub.append(p)
            for acc, key in ((lp, 'lpips_def'), (ps, 'psnr'),
                             (px, 'flow_px_max'), (de, 'face_deltaE00')):
                q = _f(r, key)
                if q is not None:
                    acc.append(q)
            caps_r = r.get('cap_reached', caps_r)
            caps_v = r.get('cap_values', caps_v)
            viol += int(_f(r, 'free_cap_violations') or 0)
            rej += int(_f(r, 'rejected') or 0)
            for e in ev.get((vv, im), ()):
                rows_all.append(e)
                d, c = _f(e, 'subject_id_edit_def'), _f(e, 'subject_id_edit_orig')
                if d is not None and c and c > 1e-3:
                    idn.append(d / c)
                    if p and p > 0.05:
                        extra.append((d / c) / p)
                u = _f(e, 'use_norm')
                if u is not None:
                    usen.append(u)
        hit, cells = majority_cells(rows_all)
        med = lambda xs: statistics.median(xs) if xs else float('nan')
        table.append({
            'variant': v, 'pub_id': med(pub), 'majority': hit, 'cells': cells,
            'extra': med(extra), 'id_norm': med(idn), 'use_norm': med(usen),
            'lpips': med(lp), 'psnr': med(ps), 'flow_px_max': med(px),
            'deltaE00': med(de), 'cap_reached': caps_r, 'cap_values': caps_v,
            'violations': viol, 'rejected': rej, 'rows': len(rows_all)})
    table.sort(key=lambda r: (-r['pub_id'] if r['pub_id'] == r['pub_id'] else 0))
    return table


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--batch', type=Path, required=True,
    ap.add_argument('--eval', dest='ev', type=Path, required=True,
    ap.add_argument('--out', type=Path, default=None)
    args = ap.parse_args()

    table = summarise(args.batch, args.ev)
    head = (f"{'variant':30s} {'pub_id':>7s} {'maj':>4s} {'/cells':>6s} "
            f"{'extra':>7s} {'id_norm':>8s} {'lpips':>7s} {'psnr':>7s} "
            f"{'flow_px':>8s} {'rows':>5s} {'越界':>5s}")
    print(head)
    for r in table:
        print(f"{r['variant']:30s} {r['pub_id']:7.3f} {r['majority']:4d} "
              f"{r['cells']:6d} {r['extra']:7.3f} {r['id_norm']:8.3f} "
              f"{r['lpips']:7.3f} {r['psnr']:7.2f} {r['flow_px_max']:8.2f} "
              f"{r['rows']:5d} {r['violations']:5d}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, 'w', newline='', encoding='utf-8') as fh:
            w = csv.DictWriter(fh, fieldnames=['variant', *FIELDS, 'rows'],
                               extrasaction='ignore')
            w.writeheader()
            w.writerows(table)
        print(f'\n寫出 {args.out}')


if __name__ == '__main__':
    main()

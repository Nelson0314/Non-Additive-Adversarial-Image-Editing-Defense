"""在**目視稽核後存活的格**上彙整評估結果。**唯讀。**

為什麼要這一支
────────────────────────────────────────────────────────────────────
判準的身分項是 `id_norm = 防禦後編輯的身分 ÷ 同一格未防禦編輯的身分`。
未防禦那一張本身就換成另一個人時，分母量的是別人，比值沒有意義；那一格的
攻擊已經失敗，防禦有沒有作用讀不出來。`scripts/summarise_screen.py` 用
`control_subject_lost` 與 `control_direction_flat` 擋掉「偵測不到臉」與
「指令沒往前走」，但擋不掉「臉還在、只是換了個人」——那兩個旗標在那種格上
都是 0。

`data/undefended_audit.json` 是逐張看過 81 張未防禦編輯之後列出的排除清單。
這一支把那些 (影像, 指令類, 種子) 從每一個臂裡拿掉，再算一次同一組讀數，
所以各臂仍然落在同一個分母上。

輸出每個 (變體, 臂, 淨化) 一列，欄位與 `summarise_screen.py` 對得起來。
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics as st
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load_excluded(path: Path):
    spec = json.loads(path.read_text(encoding='utf-8'))
    out = set()
    for e in spec['excluded']:
        for s in e['seeds']:
            out.add((e['image'], e['class'], str(s)))
    return out, spec


def number(row, key):
    v = row.get(key, '')
    if v in ('', 'nan', None):
        return None
    try:
        f = float(v)
    except ValueError:
        return None
    return None if f != f else f


def median(values):
    vs = [v for v in values if v is not None]
    return None if not vs else st.median(vs)


def crit(r):
    """缺值記 1.0（防禦拿不到分）。**不可以寫成 `number(...) or 1.0`**：
    `criterion` 真的會是 0.0，而 `0.0 or 1.0` 在 Python 裡等於 1.0。"""
    v = number(r, 'criterion')
    return 1.0 if v is None else v


def summarise(rows, threshold: float):
    keys = defaultdict(list)
    for r in rows:
        keys[(r['variant'], r['arm'], r['purifier'])].append(r)
    out = []
    for (variant, arm, purifier), group in sorted(keys.items()):
        cells = defaultdict(list)
        for r in group:
            cells[(r['image'], r['class'])].append(r)
        argmin = Counter()
        for r in group:
            i, u = number(r, 'id_norm'), number(r, 'use_norm')
            if i is None or u is None:
                continue
            argmin['id' if i <= u else 'use'] += 1
        out.append({
            'variant': variant, 'arm': arm, 'purifier': purifier,
            'rows': len(group), 'cells': len(cells),
            'penetrated': sum(1 for c in cells.values()
                              if any(crit(r) < threshold for r in c)),
            'majority': sum(1 for c in cells.values()
                            if sum(crit(r) < threshold for r in c) * 2 > len(c)),
            'all_seeds': sum(1 for c in cells.values()
                             if all(crit(r) < threshold for r in c)),
            'criterion': median(crit(r) for r in group),
            'id_norm': median(number(r, 'id_norm') for r in group),
            'dir_norm': median(number(r, 'dir_norm') for r in group),
            'use_norm': median(number(r, 'use_norm') for r in group),
            'argmin_id': argmin['id'], 'argmin_use': argmin['use'],
        })
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--audit', type=Path,
                    default=ROOT / 'data' / 'undefended_audit.json')
    ap.add_argument('--threshold', type=float, default=0.5)
    ap.add_argument('--out', type=Path, default=None)
    args = ap.parse_args()

    excluded, spec = load_excluded(args.audit)
    rows, dropped = [], 0
    for vdir in sorted(p for p in args.root.iterdir() if p.is_dir()):
        for csv_path in sorted(vdir.rglob('*.csv')):
            with open(csv_path, encoding='utf-8') as fh:
                for r in csv.DictReader(fh):
                    r['variant'] = vdir.name
                    if (r['image'], r['class'], r['eval_seed']) in excluded:
                        dropped += 1
                        continue
                    rows.append(r)
    if not rows:
        raise SystemExit(f'{args.root} 底下沒有存活的列')
    print(f'稽核排除 {dropped} 列，留下 {len(rows)} 列'
          f'（{spec["totals"]["rows_kept"]}/{spec["totals"]["rows_audited"]} '
          f'格·種子 × 臂 × 淨化）')

    table = summarise(rows, args.threshold)
    head = (f'{"變體":<26}{"臂":<12}{"淨化":<13}{"列":>4}{"格":>4}'
            f'{"穿透":>7}{"過半":>5}{"全種子":>7}{"criterion":>11}'
            f'{"id":>9}{"dir":>9}{"use":>9}')
    print(head)
    print('-' * len(head))
    for r in table:
        if r['arm'] != 'immunised':
            continue
        print(f'{r["variant"]:<26}{r["arm"]:<12}{r["purifier"]:<13}'
              f'{r["rows"]:>4}{r["cells"]:>4}'
              f'{r["penetrated"]:>5}/{r["cells"]:<2}{r["majority"]:>5}'
              f'{r["all_seeds"]:>6}'
              f'{r["criterion"]:>11.4f}{r["id_norm"]:>9.4f}'
              f'{r["dir_norm"]:>9.4f}{r["use_norm"]:>9.4f}')

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, 'w', newline='', encoding='utf-8') as fh:
            w = csv.DictWriter(fh, fieldnames=list(table[0]))
            w.writeheader()
            w.writerows(table)
        print(f'\n寫出 {args.out}（{len(table)} 列）')


if __name__ == '__main__':
    main()

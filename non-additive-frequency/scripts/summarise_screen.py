"""把逐變體的評估 CSV 彙整成一張可以並列的表。**唯讀，不改任何產物。**

三件事要一起看，缺一件就會誤讀：

**一、逐種子的未防禦對照。** 種子層的攻擊本身就可能失敗（臉找不到、指令沒往
前走），那時候整格的讀數是雜訊而不是防禦成功。`control_subject_lost` 與
`control_direction_flat` 逐列標記，這裡把它們分開計數，不混進穿透率。

**二、哪一項是軟極小的 argmin。** `criterion` 是連言的軟極小，數字本身不說
是哪一條腿垮了。`runs/purify_heldout/` 的 75 列裡 75 列都是 `id_norm`，
`use_norm` 從來沒有當過最小項——不分開報就看不出這件事。

**三、防禦圖本身的身分。** 由 `scripts/field_readout.py` 另外產出，這裡以
`--defended` 併進來。編輯輸出上的身分下降要扣掉防禦圖自己已經丟掉的那一段，
否則分不出「攻擊者拿不到同一個人」與「照片主人自己也不要這張圖」。
"""
import argparse
import csv
import statistics as st
from collections import Counter, defaultdict
from pathlib import Path


def read_rows(path: Path):
    rows = []
    for csv_path in sorted(path.rglob('*.csv')):
        with open(csv_path, encoding='utf-8') as fh:
            rows.extend(dict(r, _file=str(csv_path)) for r in csv.DictReader(fh))
    return rows


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


def fmt(v, nd=4):
    return '—' if v is None else f'{v:.{nd}f}'


def summarise(rows, threshold: float):
    """每個（變體, 臂, 淨化）一列。穿透格以**格**為單位，不是以列為單位。

    一格是（影像, 指令類）。三顆種子裡只要有一顆把 `criterion` 壓到門檻以下
    就算該格穿透——判準是連言，任一條垮掉算防禦成功，而攻擊者只需要一顆種子
    不成功就重抽，所以逐格取最好的那一顆對防禦方才是誠實的計法。
    `majority` 是三顆裡過半、`all_seeds` 是三顆全中，三個都報：任一顆是防禦方
    最寬鬆的計法，全中是最嚴的，過半在中間。哪一個才對取決於攻擊者願意重抽幾次，
    那是一個判斷，不是一個量測，所以三個都擺出來。

    **`runs/purify_heldout/` 的顏色線用的是過半那一欄。** 它的 README 記的
    18／18／15／16（identity／jpeg／blur／crop）與這裡的 `majority` 逐格相同，
    所以兩條線要並列時看這一欄，不要拿它跟 `penetrated` 比。
    """
    keys = defaultdict(list)
    for r in rows:
        keys[(r.get('variant', ''), r['arm'], r['purifier'])].append(r)

    out = []
    for (variant, arm, purifier), group in sorted(keys.items()):
        cells = defaultdict(list)
        for r in group:
            cells[(r['image'], r['class'])].append(r)
        def crit(r) -> float:
            """缺值記 1.0（防禦拿不到分）。**不可以寫成 `number(...) or 1.0`**：
            `criterion` 真的會是 0.0，而 `0.0 or 1.0` 在 Python 裡等於 1.0，
            於是防禦最成功的那些格會被算成完全沒穿透。"""
            v = number(r, 'criterion')
            return 1.0 if v is None else v

        penetrated = sum(1 for cell in cells.values()
                         if any(crit(r) < threshold for r in cell))
        all_seeds = sum(1 for cell in cells.values()
                        if all(crit(r) < threshold for r in cell))
        majority = sum(1 for cell in cells.values()
                       if sum(crit(r) < threshold for r in cell) * 2 > len(cell))
        argmin = Counter()
        for r in group:
            i, u = number(r, 'id_norm'), number(r, 'use_norm')
            if i is None or u is None:
                continue
            argmin['id' if i <= u else 'use'] += 1
        out.append({
            'variant': variant, 'arm': arm, 'purifier': purifier,
            'rows': len(group), 'cells': len(cells),
            'penetrated': penetrated, 'majority': majority,
            'all_seeds': all_seeds,
            'criterion': median(number(r, 'criterion') for r in group),
            'id_norm': median(number(r, 'id_norm') for r in group),
            'dir_norm': median(number(r, 'dir_norm') for r in group),
            'use_norm': median(number(r, 'use_norm') for r in group),
            'argmin_id': argmin['id'], 'argmin_use': argmin['use'],
            'control_lost': sum(int(r.get('control_subject_lost') or 0)
                                for r in group),
            'control_flat': sum(int(r.get('control_direction_flat') or 0)
                                for r in group),
            'subject_lost': sum(int(r.get('subject_lost') or 0) for r in group),
        })
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, required=True,
                    help='含 <variant>/ 子目錄的評估輸出根目錄')
    ap.add_argument('--defended', type=Path, default=None,
                    help='scripts/field_readout.py 的 CSV')
    ap.add_argument('--threshold', type=float, default=0.5,
                    help='criterion 低於此值算該格穿透')
    ap.add_argument('--sweep', default='0.3,0.4,0.5,0.6,0.7',
                    help='另外以這幾個門檻各報一次格數')
    ap.add_argument('--out', type=Path, default=None)
    args = ap.parse_args()

    rows = []
    for vdir in sorted(p for p in args.root.iterdir() if p.is_dir()):
        for r in read_rows(vdir):
            r['variant'] = vdir.name
            rows.append(r)
    if not rows:
        raise SystemExit(f'{args.root} 底下沒有任何 CSV')

    table = summarise(rows, args.threshold)
    head = (f'{"變體":<12}{"臂":<12}{"淨化":<12}{"列":>4}{"格":>4}'
            f'{"穿透":>6}{"過半":>5}{"全種子":>7}{"criterion":>11}{"id":>9}{"dir":>9}'
            f'{"use":>9}{"argmin":>12}{"對照失敗":>9}')
    print(head)
    print('-' * len(head))
    for r in table:
        print(f'{r["variant"]:<12}{r["arm"]:<12}{r["purifier"]:<12}'
              f'{r["rows"]:>4}{r["cells"]:>4}'
              f'{r["penetrated"]:>4}/{r["cells"]:<2}{r["majority"]:>5}'
              f'{r["all_seeds"]:>6}'
              f'{fmt(r["criterion"]):>11}{fmt(r["id_norm"]):>9}'
              f'{fmt(r["dir_norm"]):>9}{fmt(r["use_norm"]):>9}'
              f'{r["argmin_id"]:>7}id{r["argmin_use"]:>3}u'
              f'{r["control_lost"] + r["control_flat"]:>9}')

    if args.sweep:
        cuts = [float(v) for v in args.sweep.split(',') if v.strip()]
        print()
        print('門檻掃描（穿透格數）——格數對切點敏感，結論不可以只掛在一個門檻上')
        head = f'{"變體":<12}{"淨化":<13}' + ''.join(f'{c:>8.2f}' for c in cuts)
        print(head)
        print('-' * len(head))
        by_arm = [r for r in rows if r['arm'] != 'undefended']
        keys = sorted({(r.get('variant', ''), r['purifier']) for r in by_arm})
        for variant, purifier in keys:
            group = [r for r in by_arm if r.get('variant', '') == variant
                     and r['purifier'] == purifier]
            counts = []
            for cut in cuts:
                one = summarise(group, cut)
                counts.append(sum(r['penetrated'] for r in one))
            print(f'{variant:<12}{purifier:<13}'
                  + ''.join(f'{c:>8d}' for c in counts))

    if args.defended and args.defended.exists():
        print()
        print('防禦圖本身（攻擊尚未發生）')
        print(f'{"變體":<12}{"影像":<30}{"錨定身分":>10}{"PSNR":>8}'
              f'{"LPIPS":>9}{"ΔE00":>8}{"NIQE 比":>9}')
        with open(args.defended, encoding='utf-8') as fh:
            for r in csv.DictReader(fh):
                niqe = number(r, 'niqe_defended')
                base = number(r, 'niqe_orig')
                ratio = None if not base else niqe / base
                print(f'{r["variant"]:<12}{r["image"]:<30}'
                      f'{fmt(number(r, "subject_id_defended")):>10}'
                      f'{fmt(number(r, "psnr"), 2):>8}'
                      f'{fmt(number(r, "lpips")):>9}'
                      f'{fmt(number(r, "deltaE00"), 2):>8}'
                      f'{fmt(ratio, 3):>9}')

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, 'w', newline='', encoding='utf-8') as fh:
            w = csv.DictWriter(fh, fieldnames=list(table[0]))
            w.writeheader()
            w.writerows(table)
        print(f'\n寫出 {args.out}（{len(table)} 列）')


if __name__ == '__main__':
    main()

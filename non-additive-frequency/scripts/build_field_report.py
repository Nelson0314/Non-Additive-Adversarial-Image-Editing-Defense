"""把兩批位移場的產物與讀數整理成一份看圖用的資料檔。**唯讀。**

自然度只能由人眼判定——三道自動門檻都被最佳化鑽過——所以這一支的工作是把
「原圖、每一檔的防禦圖、以及該檔的四個讀數」湊在同一個地方，讓判斷可以逐張
做。它不下任何結論，也不挑圖。

輸出是一個 `data.js`（給報告頁用）與一份影像複本。每一檔的讀數分兩層：

- **防禦圖本身**（`scripts/field_readout.py`）——攻擊還沒發生，這一層回答
  「這張照片還是不是這個人、還像不像一張照片」。
- **編輯輸出**（`scripts/evaluate_defence.py` 經 `summarise_screen.py`）——
  這一層回答「攻擊者拿不拿得到可用的結果」。

兩層分開報：一個載體可能在第二層很強，只是因為第一層已經把照片毀了。
"""
import argparse
import csv
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_csv(path: Path):
    if not path or not path.exists():
        return []
    with open(path, encoding='utf-8') as fh:
        return list(csv.DictReader(fh))


def num(v):
    if v in ('', None, 'nan'):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if f != f else f


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--batch', action='append', required=True, metavar='NAME=DIR',
                    help='批次名稱=含 by_variant/ 的目錄，可給多次')
    ap.add_argument('--defended', action='append', default=[], metavar='CSV',
                    help='scripts/field_readout.py 的輸出，可給多次')
    ap.add_argument('--screen', action='append', default=[], metavar='CSV',
                    help='scripts/summarise_screen.py --out 的輸出，可給多次')
    ap.add_argument('--manifest', type=Path,
                    default=ROOT / 'data' / 'scaleup_manifest.json')
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding='utf-8'))
    paths = {r['id']: ROOT / r['path'] for r in manifest['images']}

    img_dir = args.out / 'img'
    img_dir.mkdir(parents=True, exist_ok=True)

    defended = {}
    for csv_path in args.defended:
        for r in read_csv(Path(csv_path)):
            defended[(r['variant'], r['image'])] = {
                k: num(r.get(k)) for k in
                ('subject_id_defended', 'subject_id_defended_detected',
                 'psnr', 'lpips', 'dists', 'deltaE00', 'ssim',
                 'niqe_orig', 'niqe_defended')}

    screen = {}
    for csv_path in args.screen:
        for r in read_csv(Path(csv_path)):
            screen.setdefault(r['variant'], {})[
                f"{r['arm']}|{r['purifier']}"] = {
                'cells': int(r['cells']), 'penetrated': int(r['penetrated']),
                'all_seeds': int(r['all_seeds']),
                'criterion': num(r.get('criterion')),
                'id_norm': num(r.get('id_norm')),
                'dir_norm': num(r.get('dir_norm')),
                'use_norm': num(r.get('use_norm')),
                'argmin_id': int(r['argmin_id']),
                'argmin_use': int(r['argmin_use'])}

    images, variants = [], []
    seen_images = set()
    for entry in args.batch:
        batch_name, _, directory = entry.partition('=')
        vroot = Path(directory)
        if not vroot.exists():
            raise SystemExit(f'{vroot} 不存在')
        for vdir in sorted(p for p in vroot.iterdir() if p.is_dir()):
            files = {}
            for png in sorted(vdir.glob('*__immunised.png')):
                image = png.name[:-len('__immunised.png')]
                dest = img_dir / f'{vdir.name}__{image}.png'
                shutil.copy2(png, dest)
                files[image] = f'img/{dest.name}'
                if image not in seen_images:
                    seen_images.add(image)
                    orig = img_dir / f'original__{image}.png'
                    shutil.copy2(paths[image], orig)
                    images.append({'id': image,
                                   'src': f'img/{orig.name}'})
            if not files:
                continue
            variants.append({
                'name': vdir.name, 'batch': batch_name, 'files': files,
                'defended': {im: defended.get((vdir.name, im))
                             for im in files},
                'screen': screen.get(vdir.name, {})})

    images.sort(key=lambda r: r['id'])
    payload = {'images': images,
               'variants': variants,
               'screen_keys': sorted({k for v in variants
                                      for k in v['screen']})}
    body = 'window.FIELD = ' + json.dumps(payload, ensure_ascii=False) + ';\n'
    (args.out / 'data.js').write_text(body, encoding='utf-8')
    print(f'寫出 {args.out / "data.js"}：'
          f'{len(images)} 張影像、{len(variants)} 個變體、'
          f'{len(list(img_dir.glob("*.png")))} 個影像檔')


if __name__ == '__main__':
    main()

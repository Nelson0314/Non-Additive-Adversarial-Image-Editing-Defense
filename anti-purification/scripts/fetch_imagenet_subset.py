"""取一小組 ImageNet 驗證影像，作為三篇顏色論文自己的受害資料。

為什麼需要
────────────────────────────────────────────────────────────────────
`scripts/paper_baseline.py` 的第一次煙霧測試在本專案的人像上跑，乾淨影像的
C&W margin 只有 **+0.017**——ResNet-50 對一張不屬於 ImageNet 任何類別的人像
本來就沒有信心，把那種預測「翻掉」證明不了攻擊有效。三篇報的成功率都是在
**分類器原本答對而且有信心**的 ImageNet 影像上算的，所以重現必須用同一種輸入。

挑選規則
────────────────────────────────────────────────────────────────────
只留 `top-1 == 標籤` 且 `p(top-1) ≥ min_prob` 的影像。這是三篇共同的前提：
攻擊成功率的分母是「原本分對的影像」。挑選用的分類器與後續攻擊用的同一個，
所以「原本分得對」這件事沒有跨模型的落差。

輸出與 `data/*_manifest.json` 同格式（`id`／`path`／`sha256`），
`scripts/paper_baseline.py` 因此不必特別處理。
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

REPO = 'mrm8488/ImageNet1K-val'
SHARD = 'data/train-00000-of-00014.parquet'
URL = f'https://huggingface.co/datasets/{REPO}/resolve/main/{SHARD}'


def main():
    import torch
    from PIL import Image

    from src.defense.victim_classifier import VictimClassifier

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, default=ROOT / 'data' / 'imagenet_subset')
    ap.add_argument('--manifest', type=Path,
                    default=ROOT / 'data' / 'imagenet_subset_manifest.json')
    ap.add_argument('--count', type=int, default=48)
    ap.add_argument('--min-prob', type=float, default=0.7)
    ap.add_argument('--victim', default='resnet50')
    ap.add_argument('--size', type=int, default=512,
                    help='存檔邊長；載體與 IP2P 都吃 512')
    ap.add_argument('--cache', type=Path, default=ROOT / 'data' / '_imagenet_shard.parquet')
    ap.add_argument('--stride', type=int, default=0,
                    help='每隔幾列取一張；0 表示依 count 自動算滿整個分片。'
                         '分片是**按標籤排序**的，不跨步取會拿到同兩三個類別的'
                         '幾十張圖，成功率就變成那幾類的成功率。')
    args = ap.parse_args()

    if not args.cache.exists():
        args.cache.parent.mkdir(parents=True, exist_ok=True)
        print(f'下載 {URL}', flush=True)
        urllib.request.urlretrieve(URL, args.cache)
    print(f'分片 {args.cache}（{args.cache.stat().st_size / 1e6:.0f} MB）', flush=True)

    import pyarrow.parquet as pq

    table = pq.read_table(args.cache, columns=['image', 'label'])
    print(f'{table.num_rows} 列', flush=True)

    victim = VictimClassifier(args.victim)
    args.out.mkdir(parents=True, exist_ok=True)
    stride = args.stride or max(1, table.num_rows // (args.count * 3))
    order = range(0, table.num_rows, stride)
    print(f'跨步 {stride}，掃 {len(order)} 列', flush=True)
    images, kept = [], 0
    for k in order:
        if kept >= args.count:
            break
        rec = table.slice(k, 1).to_pylist()[0]
        raw = rec['image']['bytes'] if isinstance(rec['image'], dict) else rec['image']
        try:
            im = Image.open(io.BytesIO(raw)).convert('RGB')
        except Exception:
            continue
        side = min(im.size)
        left, top = (im.width - side) // 2, (im.height - side) // 2
        im = im.crop((left, top, left + side, top + side)).resize(
            (args.size, args.size), Image.BICUBIC)
        x = torch.from_numpy(
            __import__('numpy').asarray(im, dtype='float32') / 255.0
        ).permute(2, 0, 1)[None].to(victim.device)
        anchor = victim.anchor(x)
        if anchor['clean_label'] != int(rec['label']):
            continue
        if anchor['clean_prob'] < args.min_prob:
            continue
        name = f'imagenet_{int(rec["label"]):04d}_{k:05d}'
        path = args.out / f'{name}.png'
        im.save(path)
        images.append({'id': name,
                       'path': str(path.relative_to(ROOT)).replace('\\', '/'),
                       'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                       'imagenet_label': int(rec['label']),
                       'clean_prob': round(anchor['clean_prob'], 4)})
        kept += 1
        print(f'  [{kept}/{args.count}] {name} p={anchor["clean_prob"]:.3f}',
              flush=True)

    if kept < args.count:
        raise SystemExit(f'只湊到 {kept} 張，少於要求的 {args.count}；'
                         '放寬 --min-prob 或換一個分片')

    args.manifest.write_text(json.dumps({
        'images': images,
        'selection_note': (
            f'{REPO} 的 {SHARD}，每 {stride} 列取一張（分片按標籤排序，'
            f'不跨步會拿到同兩三個類別的幾十張），'
            f'中央方形裁切後縮到 {args.size}×{args.size}。'
            f'只留 {args.victim} 分對且信心 ≥ {args.min_prob} 的影像——'
            '三篇論文的攻擊成功率分母就是「原本分對的影像」，'
            '在分類器沒有信心的輸入上談翻轉沒有意義。'),
        'victim_note': (
            f'挑選與後續攻擊用同一個 {args.victim}，'
            '所以「原本分得對」沒有跨模型的落差。'),
    }, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f'寫出 {args.manifest}（{kept} 張）', flush=True)


if __name__ == '__main__':
    main()

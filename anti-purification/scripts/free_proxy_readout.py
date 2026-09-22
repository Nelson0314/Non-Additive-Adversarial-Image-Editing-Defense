"""把每一張防禦圖的無指令目標三項，量在**同一組抽樣**上。**唯讀，不含指令。**

為什麼需要這一支
────────────────────────────────────────────────────────────────────
`scripts/paper_baseline.py` 寫出的 `free_term_*` 是**各臂自己那一組抽樣**上的值：
凍結抽樣的臂量在它最佳化時用的那一個 `(噪聲, 時刻)` 上，重抽的臂量在它自己的
固定驗證抽樣上。兩者不是同一個量，並列會誤讀。

這一支對每一張影像建**一組**目標，再把該影像所有臂的防禦圖丟進同一組抽樣裡，
所以跨臂的數字比得起來。兩組抽樣各報一次：

`train_*`  `noise_seed = 0` 的凍結抽樣——凍結那一臂就是對著它解出來的。
`probe_*`  另一組與訓練無關的抽樣。凍結的解若只在自己那一組上成立，
           兩欄的落差就會出現在這裡。

`score` 依 `src/defense/instruction_free.py` 的合成式算，權重由 `--weights` 給，
預設與 `DEFAULT_WEIGHTS` 相同。**越小代表代理目標越好**。

代理不是判準。判準仍然是 `scripts/evaluate_defence.py` 的真編輯。
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

COLUMNS = ['variant', 'image', 'train_enc', 'train_cond', 'train_id',
           'train_score', 'probe_enc', 'probe_cond', 'probe_id', 'probe_score']


def compose(terms, weights):
    import math
    s = (-weights['enc'] * math.tanh(terms['enc'])
         - weights['cond'] * math.tanh(terms['cond']))
    if 'id' in terms:
        s += weights['id'] * terms['id']
    return s


def main():
    import torch

    from src.defense.assets import load_image
    from src.defense.instruction_free import DEFAULT_WEIGHTS, FreeObjective
    from src.defense.ncf_library import sha256
    from src.metrics.identity import face_boxes
    from src.models.ip2p import IP2PWrapper

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--manifest', type=Path, required=True)
    ap.add_argument('--root', type=Path, required=True,
                    help='含 <variant>/<image>__immunised.png 的目錄')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--config', type=Path, required=True,
                    help='取 free 區塊的 timesteps／chain_steps／grad_steps')
    ap.add_argument('--probe-seed', type=int, default=5150)
    ap.add_argument('--weights', default='')
    args = ap.parse_args()

    spec = json.loads(args.config.read_text(encoding='utf-8'))
    free = spec['free']
    weights = dict(DEFAULT_WEIGHTS)
    if args.weights:
        weights.update(json.loads(args.weights))

    manifest = json.loads(args.manifest.read_text(encoding='utf-8'))
    entries = {r['id']: r for r in manifest['images']}
    ip2p = IP2PWrapper(dtype={'fp16': torch.float16, 'fp32': torch.float32,
                              'bf16': torch.bfloat16}[spec.get('precision',
                                                               'bf16')])
    device = ip2p.device

    by_image = {}
    for vdir in sorted(p for p in args.root.iterdir() if p.is_dir()):
        for png in sorted(vdir.glob('*__immunised.png')):
            image = png.name[:-len('__immunised.png')]
            by_image.setdefault(image, []).append((vdir.name, png))

    rows = []
    for image, items in sorted(by_image.items()):
        entry = entries[image]
        src = ROOT / entry['path']
        if sha256(src) != entry['sha256']:
            raise ValueError(f'{image} 的輸入雜湊不符')
        x = load_image(src, device)
        boxes = face_boxes(x, device)
        if not boxes:
            raise ValueError(f'{image} 偵測不到臉，id 項錨不住')
        box = max(boxes, key=lambda q: (q[2] - q[0]) * (q[3] - q[1]))
        common = dict(box=box, k=int(free['timesteps']),
                      steps=int(spec.get('attack_steps', 50)),
                      chain_steps=int(free.get('chain_steps', 6)),
                      grad_steps=int(free.get('grad_steps', 1)),
                      s_i=float(spec.get('s_i', 1.5)))
        train = FreeObjective(ip2p, x, seed=int(free['noise_seed']),
                              resample=False, **common)
        probe = FreeObjective(ip2p, x, seed=int(args.probe_seed),
                              resample=True, **common)
        for variant, png in items:
            y = load_image(png, device)
            with torch.no_grad():
                t = {k: float(v) for k, v in train.terms(y).items()}
                p = {k: float(v) for k, v in probe.eval_terms(y).items()}
            rows.append({
                'variant': variant, 'image': image,
                **{f'train_{k}': round(v, 5) for k, v in t.items()},
                'train_score': round(compose(t, weights), 5),
                **{f'probe_{k}': round(v, 5) for k, v in p.items()},
                'probe_score': round(compose(p, weights), 5)})
            print(f'  {image} · {variant}  train {rows[-1]["train_score"]:+.4f}'
                  f'  probe {rows[-1]["probe_score"]:+.4f}', flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)
    print(f'寫出 {args.out}（{len(rows)} 列）', flush=True)


if __name__ == '__main__':
    main()

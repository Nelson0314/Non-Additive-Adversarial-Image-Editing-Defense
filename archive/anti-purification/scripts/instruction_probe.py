"""未防禦的編輯樣本，給指令完成度讀數當校準集。

這一支不量任何東西，也不碰防禦。它只做一件事：把設定檔裡每一格的指令套在
**原圖**上跑一次真編輯，把圖存下來。

為什麼要單獨一支
────────────────────────────────────────────────────────────────────
`src/defense/criterion.py` 的 `dir_norm` 不在 `TERMS` 裡，因為 CLIP 的三種寫法
（方向分數、終點對齊、對齊增益）都在圖上被推翻過。要換一個量得準的做法，
第一件事不是換模型，是**先有一組有真值的圖**：未防禦的編輯本來就應該完成指令，
它是完成度讀數的上端錨點；原圖本來就不該完成，那是下端。兩端之間才談得上
「這個讀數對不對」。

真值由人看圖標註，不由任何自動讀數標註——三道自動化自然度門檻都被最佳化鑽過，
同一個錯誤不重犯。所以這支腳本只出圖與清單，`instruction_truth.csv` 的
`human_completed` 欄留空，等人填。

**輸出的圖是未防禦的。** 這裡沒有防禦圖，也不該有：校準要的是「編輯成功長什麼
樣子」，防禦有沒有效是後面的事。
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

COLUMNS = ['image', 'class', 'kind', 'instruction', 'seed', 'file',
           'check_question', 'check_expect', 'human_completed', 'seconds']


def main():
    import torch

    from src.defense.assets import load_image, save_png
    from src.defense.ncf_library import sha256
    from src.models.ip2p import IP2PWrapper

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--shard', default='1/1')
    args = ap.parse_args()

    spec = json.loads(args.config.read_text(encoding='utf-8'))
    manifest = json.loads((ROOT / spec['manifest']).read_text(encoding='utf-8'))
    entries = {r['id']: r for r in manifest['images']}
    if args.out.exists() and any(args.out.iterdir()):
        raise SystemExit(f'{args.out} 已存在且非空；換一個輸出目錄')
    args.out.mkdir(parents=True, exist_ok=True)

    i, n = (int(v) for v in args.shard.split('/'))
    if not 1 <= i <= n:
        raise SystemExit(f'--shard 要寫成 i/n，收到 {args.shard!r}')
    todo = [im for k, im in enumerate(spec['images']) if k % n == i - 1]

    ip2p = IP2PWrapper(dtype={'fp16': torch.float16, 'fp32': torch.float32,
                              'bf16': torch.bfloat16}[spec['precision']])
    seeds = [int(s) for s in spec['seeds']]
    classes = spec['classes']
    print(f'分片 {i}/{n}：{len(todo)} 張 × {len(classes)} 類 × {len(seeds)} 顆種子'
          f' = {len(todo) * len(classes) * len(seeds)} 張未防禦編輯', flush=True)

    rows = []
    for image in todo:
        entry = entries[image]
        src = ROOT / entry['path']
        if sha256(src) != entry['sha256']:
            raise ValueError(f'{image} 的輸入雜湊不符')
        x = load_image(src, ip2p.device)
        save_png(x, args.out / f'{image}__original.png')
        for cl in classes:
            instruction = cl['instructions'][image]
            expect = cl.get('expect', {}).get(image, cl.get('check_expect'))
            for seed in seeds:
                t0 = time.time()
                y = ip2p.edit(x, instruction, seed=seed,
                              steps=int(spec['attack_steps']),
                              s_t=float(spec['s_t']), s_i=float(spec['s_i']))
                name = f'{image}__{cl["name"]}__s{seed}__clean_edit.png'
                save_png(y, args.out / name)
                rows.append({
                    'image': image, 'class': cl['name'], 'kind': cl.get('kind', ''),
                    'instruction': instruction, 'seed': seed, 'file': name,
                    'check_question': cl.get('check_question', ''),
                    'check_expect': expect, 'human_completed': '',
                    'seconds': round(time.time() - t0, 1)})
                print(f'  {image} · {cl["name"]} · s{seed}  '
                      f'{rows[-1]["seconds"]:.1f}s', flush=True)

    suffix = '' if n == 1 else f'_shard{i}of{n}'
    out_csv = args.out / f'instruction_truth{suffix}.csv'
    with open(out_csv, 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)
    print(f'寫出 {out_csv}（{len(rows)} 列，human_completed 待人填）', flush=True)


if __name__ == '__main__':
    main()

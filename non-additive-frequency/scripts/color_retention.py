"""顏色載體的抗淨化：只讀已存的防禦圖，不重跑訓練。

`scripts/phase_retention.py` 綁在相位線的 `results.csv` schema 上
（`condition`／`budget_target`／`--data lo_aligned`），吃不了顏色批次的欄位，
所以這支獨立存在。**兩者的參照規則與扣地板的方式逐條相同**，只有輸入的
清單來源不同。

## 參照依算子分兩種

    effect(p) = LPIPS( 編輯(p(原圖)), 編輯(p(防禦圖)) )    幾何類
    effect(p) = LPIPS( 編輯(原圖),    編輯(p(防禦圖)) )    其餘

幾何類就是 `src/purify/ops.py` 的 `GEOMETRIC_KINDS`，判定走 `Purifier.kind`
而不是 label 字串（label 會把強度接在後面）。`reference` 欄逐列記下走的是哪
一種——兩種基準會出現在同一張表裡，沒有這一欄就分不出來。

## 空白地板

    floor(p) = LPIPS( 編輯(原圖), 編輯(p(原圖)) )

即算子自己造成的位移，與有沒有防禦無關。**幾何類的地板由構造為 0**（兩側
同算子、同輸入、同種子），故只有非幾何類要扣。逐列同時報

    total_gain = effect(p)
    net_gain   = effect(p) − floor(p)

兩欄並列、都報絕對值，不換算成佔某個範圍的百分比。
"""
import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

COLUMNS = [
    'image', 'class', 'instruction', 'carrier', 'arm', 'delta_e_target',
    'amplitude', 'eval_seed', 'operator', 'kind', 'strength', 'geometric',
    'reference', 'effect', 'floor', 'total_gain', 'net_gain',
    'def_png', 'seconds',
]


def assert_free_cards():
    card = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if not card.isdecimal():
        raise SystemExit('CUDA_VISIBLE_DEVICES 必須指定單張實體卡號')
    subprocess.run(['bash', str(ROOT / 'scripts/free_cards.sh'), '--assert', card],
                   cwd=ROOT, check=True)
    uuid = subprocess.run(
        ['nvidia-smi', '-i', card, '--query-gpu=uuid', '--format=csv,noheader'],
        capture_output=True, text=True, check=True).stdout.strip()
    apps = subprocess.run(
        ['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader'],
        capture_output=True, text=True, check=True).stdout.splitlines()
    mine = str(os.getpid())
    for line in apps:
        if not line.strip() or uuid not in line:
            continue
        pid = line.split(',')[-1].strip()
        if pid == mine:
            continue
        seen = subprocess.run(['ps', '-p', pid], capture_output=True, text=True)
        raise SystemExit(
            f'卡 {card}（{uuid}）上有 pid {pid} 的 compute app，'
            f'{"是別人的" if seen.returncode else "不是本行程"}；換一張卡，不要擠。')


def load_png(path, device):
    import numpy as np
    import torch
    from PIL import Image
    arr = np.asarray(Image.open(path).convert('RGB'), dtype=np.float32) / 255.
    return torch.from_numpy(arr).permute(2, 0, 1)[None].to(device)


def purifiers(spec):
    """設定裡的算子清單 → `Purifier` 物件。相依不齊的**回報並跳過**，不靜默。"""
    from src.purify.ops import GEOMETRIC_KINDS, Purifier
    out = []
    for item in spec['operators']:
        p = Purifier(item['kind'], strength=item.get('strength', 0.0),
                     seed=item.get('seed'), **item.get('options', {}))
        if not p.available:
            print(f"[purify] 相依不齊，跳過 {item['kind']}", flush=True)
            continue
        out.append((p, p.kind in GEOMETRIC_KINDS))
    if not out:
        raise SystemExit('沒有任何可用的淨化算子')
    return out


def rows_of(run_dir, wanted_arms, wanted_targets):
    """從 ceiling 批次的 CSV 還原要跑的格子清單，逐格對應一張已存的防禦圖。"""
    seen, cells = set(), []
    for csv_path in sorted(run_dir.glob('**/color_ceiling*.csv')):
        for r in csv.DictReader(csv_path.open(encoding='utf-8')):
            if r['arm'] == 'undefended':
                continue
            if wanted_arms and r['arm'] not in wanted_arms:
                continue
            if wanted_targets and float(r['delta_e_target']) not in wanted_targets:
                continue
            key = (r['image'], r['class'], r['arm'], r['delta_e_target'])
            if key in seen:
                continue
            seen.add(key)
            target = float(r['delta_e_target'])
            png = (csv_path.parent /
                   f"{r['image']}__{r['class']}__{r['arm']}__dE{target:g}__def.png")
            cells.append({'image': r['image'], 'class': r['class'],
                          'instruction': r['instruction'], 'carrier': r['carrier'],
                          'arm': r['arm'], 'delta_e_target': r['delta_e_target'],
                          'amplitude': r['amplitude'], 'png': png})
    return cells


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', type=Path, required=True)
    ap.add_argument('--run', type=Path, required=True,
                    help='ceiling 批次的目錄，底下要有 CSV 與 *__def.png')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--arms', default='')
    ap.add_argument('--targets', default='',
                    help='只跑這些 ΔE00 目標，逗號分隔')
    ap.add_argument('--images', default='')
    ap.add_argument('--classes', default='')
    ap.add_argument('--seeds', default='')
    ap.add_argument('--shard', default='1/1')
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()
    if args.device != 'cpu':
        assert_free_cards()

    import torch
    from src.defense.ncf_library import sha256
    from src.metrics.suite import MetricSuite
    from src.models.ip2p import IP2PWrapper

    spec = json.loads(args.config.read_text(encoding='utf-8'))
    manifest = json.loads((ROOT / spec['manifest']).read_text(encoding='utf-8'))
    if args.out.exists() and any(args.out.iterdir()):
        raise SystemExit(f'{args.out} 已存在且非空；換一個輸出目錄')
    args.out.mkdir(parents=True, exist_ok=True)

    arms = {s.strip() for s in args.arms.split(',') if s.strip()}
    targets = {float(s) for s in args.targets.split(',') if s.strip()}
    images = {s.strip() for s in args.images.split(',') if s.strip()}
    classes = {s.strip() for s in args.classes.split(',') if s.strip()}
    seeds = ([int(s) for s in args.seeds.split(',') if s.strip()]
             or spec['eval_seeds'])

    cells = rows_of(args.run, arms, targets)
    if images:
        cells = [c for c in cells if c['image'] in images]
    if classes:
        cells = [c for c in cells if c['class'] in classes]
    missing = [c for c in cells if not c['png'].exists()]
    if missing:
        raise SystemExit(f'{len(missing)} 張防禦圖不存在，第一張是 '
                         f'{missing[0]["png"]}；不要用缺圖的清單跑')
    i, n = (int(v) for v in args.shard.split('/'))
    cells = [c for k, c in enumerate(cells) if k % n == i - 1]

    ip2p = IP2PWrapper(dtype={'fp16': torch.float16, 'fp32': torch.float32,
                              'bf16': torch.bfloat16}[spec['precision']])
    suite = MetricSuite(device=ip2p.device)
    ops = purifiers(spec)
    entries = {r['id']: r for r in manifest['images']}
    rows, t_start = [], time.time()
    edit_kw = dict(steps=spec['attack_steps'], s_t=spec['s_t'], s_i=spec['s_i'])

    # 參照快取跨格重用：同一張影像、同一個種子、同一個算子的 `編輯(p(原圖))`
    # 與 `編輯(原圖)` 在多個臂與多個目標上都是同一份。
    clean_edit, purified_edit = {}, {}

    print(f'分片 {i}/{n}：{len(cells)} 格 × {len(ops)} 算子 × {len(seeds)} 種子',
          flush=True)

    for cell in cells:
        entry = entries[cell['image']]
        src = ROOT / entry['path']
        if sha256(src) != entry['sha256']:
            raise ValueError(f"{cell['image']} 的輸入雜湊不符")
        x = load_png(src, ip2p.device)
        x_def = load_png(cell['png'], ip2p.device)
        tag = (f"{cell['image']}__{cell['class']}__{cell['arm']}"
               f"__dE{float(cell['delta_e_target']):g}")

        for seed in seeds:
            ck = (cell['image'], cell['class'], seed)
            if ck not in clean_edit:
                clean_edit[ck] = ip2p.edit(x, cell['instruction'], seed=seed,
                                           **edit_kw)
            e_clean = clean_edit[ck]

            for op, geometric in ops:
                name = op.kind if not op.strength else f'{op.kind}{op.strength:g}'
                pd = op.evaluate(x_def)
                ed = ip2p.edit(pd, cell['instruction'], seed=seed, **edit_kw)

                if geometric:
                    pk = (cell['image'], cell['class'], seed, name)
                    if pk not in purified_edit:
                        purified_edit[pk] = ip2p.edit(
                            op.evaluate(x), cell['instruction'], seed=seed,
                            **edit_kw)
                    ref, reference = purified_edit[pk], 'purified_orig'
                    floor = 0.0
                else:
                    ref, reference = e_clean, 'orig'
                    pk = (cell['image'], cell['class'], seed, name)
                    if pk not in purified_edit:
                        purified_edit[pk] = ip2p.edit(
                            op.evaluate(x), cell['instruction'], seed=seed,
                            **edit_kw)
                    floor = float(suite.pairwise(
                        e_clean, purified_edit[pk])['lpips'])

                effect = float(suite.pairwise(ref, ed)['lpips'])
                rows.append({
                    'image': cell['image'], 'class': cell['class'],
                    'instruction': cell['instruction'], 'carrier': cell['carrier'],
                    'arm': cell['arm'], 'delta_e_target': cell['delta_e_target'],
                    'amplitude': cell['amplitude'], 'eval_seed': seed,
                    'operator': name, 'kind': op.kind, 'strength': op.strength,
                    'geometric': int(geometric), 'reference': reference,
                    'effect': round(effect, 5), 'floor': round(floor, 5),
                    'total_gain': round(effect, 5),
                    'net_gain': round(effect - floor, 5),
                    'def_png': cell['png'].name,
                    'seconds': round(time.time() - t_start, 1)})
                print(f"  {tag} s{seed} {name:22s} effect {effect:.4f} "
                      f"floor {floor:.4f} net {effect-floor:+.4f} [{reference}]",
                      flush=True)

    suffix = '' if n == 1 else f'_shard{i}of{n}'
    write_csv(args.out / f'color_retention{suffix}.csv', rows)
    print(f'-> {args.out}  共 {len(rows)} 列，{time.time()-t_start:.0f} 秒',
          flush=True)


def write_csv(path, rows):
    if not rows:
        raise ValueError(f'{path} 沒有任何列可寫；不要留下空檔假裝跑過')
    fields = list(dict.fromkeys(k for r in rows for k in r))
    unknown = [f for f in fields if f not in COLUMNS]
    if unknown:
        raise ValueError(f'這些欄位不在 COLUMNS 的合約裡：{unknown}')
    with Path(path).open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


if __name__ == '__main__':
    main()

"""攻擊編輯的吞吐量：批次大小 × 精度 → 每張秒數與峰值顯存。

為什麼要這支
────────────────────────────────────────────────────────────────────
每一批的機時幾乎全部花在攻擊方的取樣上（100 步擴散、512²）。批次為 1 時
量到的顯存只有 24 GB 的 29%，也就是卡大半閒著。批次大小與精度該取多少
**由量測決定**，不用猜：這支把 (batch, dtype) 掃過去，報每張的秒數與峰值
顯存，操作點從表上挑。

三件量測上的事
────────────────────────────────────────────────────────────────────
**先暖機再計時。** 第一次呼叫含 CUDA kernel 的編譯與權重搬移，把它算進去
會高估好幾倍。

**每次計時前 `torch.cuda.synchronize()`。** CUDA 是非同步的，不同步的話量到
的是「送出指令有多快」。

**峰值顯存用 `max_memory_allocated`，每個設定前重設。** `nvidia-smi` 看到的
是快取配置器保留的量，不是真正用到的量，兩者可以差很多。

輸出的 `seconds_per_image` 是拿來排序的量；`peak_gb` 決定批次的上限。
本支不做判定，操作點由使用者從表上挑。
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

COLUMNS = ['dtype', 'batch', 'steps', 'repeats', 'seconds_total',
           'seconds_per_call', 'seconds_per_image', 'peak_gb', 'speedup_vs_base',
           'max_abs_vs_fp32_single', 'note']


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


def load_image(path, device):
    import numpy as np
    import torch
    from PIL import Image
    arr = np.asarray(Image.open(path).convert('RGB'), dtype=np.float32) / 255.
    return torch.from_numpy(arr).permute(2, 0, 1)[None].to(device)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--batches', default='1,2,4,8,12',
                    help='要掃的批次大小，逗號分隔')
    ap.add_argument('--dtypes', default='fp32,bf16')
    ap.add_argument('--steps', type=int, default=0,
                    help='取樣步數，0 表示用設定裡的 attack_steps')
    ap.add_argument('--repeats', type=int, default=2,
                    help='暖機之後重複幾次取平均')
    args = ap.parse_args()
    assert_free_cards()

    import torch
    from src.defense.ncf_library import sha256
    from src.models.ip2p import IP2PWrapper

    spec = json.loads(args.config.read_text(encoding='utf-8'))
    manifest = json.loads((ROOT / spec['manifest']).read_text(encoding='utf-8'))
    steps = args.steps or spec['attack_steps']
    args.out.mkdir(parents=True, exist_ok=True)

    entry = manifest['images'][0]
    src = ROOT / entry['path']
    if sha256(src) != entry['sha256']:
        raise ValueError('輸入雜湊不符')
    instruction = spec['classes'][0]['instructions'][entry['id']]
    batches = [int(v) for v in args.batches.split(',') if v.strip()]
    dtypes = {'fp32': torch.float32, 'bf16': torch.bfloat16,
              'fp16': torch.float16}
    rows, baseline, reference = [], None, None

    for name in [d.strip() for d in args.dtypes.split(',') if d.strip()]:
        if name not in dtypes:
            raise SystemExit(f'未知的精度 {name!r}；可用的是 {sorted(dtypes)}')
        ip2p = IP2PWrapper(dtype=dtypes[name])
        x = load_image(src, ip2p.device)
        for b in batches:
            imgs = [x] * b
            texts = [instruction] * b
            seeds = [spec['eval_seeds'][0] + k for k in range(b)]
            note = ''
            try:
                # 暖機一次，不計時。
                out = ip2p.edit_batch(imgs, texts, seeds, steps=steps)
                if reference is None:
                    reference = out[:1].float().cpu().clone()
                diff = float((out[:1].float().cpu() - reference).abs().max())
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                t0 = time.time()
                for _ in range(args.repeats):
                    ip2p.edit_batch(imgs, texts, seeds, steps=steps)
                torch.cuda.synchronize()
                total = time.time() - t0
                peak = torch.cuda.max_memory_allocated() / 1024 ** 3
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache()
                rows.append({'dtype': name, 'batch': b, 'steps': steps,
                             'repeats': args.repeats, 'note': 'OOM'})
                print(f'  {name} batch {b}: OOM', flush=True)
                continue
            per_call = total / args.repeats
            per_image = per_call / b
            if baseline is None:
                baseline = per_image
            rows.append({
                'dtype': name, 'batch': b, 'steps': steps,
                'repeats': args.repeats,
                'seconds_total': round(total, 2),
                'seconds_per_call': round(per_call, 3),
                'seconds_per_image': round(per_image, 3),
                'peak_gb': round(peak, 2),
                'speedup_vs_base': round(baseline / per_image, 2),
                'max_abs_vs_fp32_single': round(diff, 6),
                'note': note})
            print(f"  {name} batch {b:>3d}: {per_image:6.2f} 秒/張  "
                  f"峰值 {peak:5.2f} GB  加速 {baseline/per_image:5.2f}x  "
                  f"與 fp32 單張的最大差 {diff:.5f}", flush=True)
        del ip2p
        torch.cuda.empty_cache()

    write_csv(args.out / 'throughput.csv', rows)
    print(f'-> {args.out}', flush=True)


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

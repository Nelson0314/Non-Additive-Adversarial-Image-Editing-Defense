"""每一個目標項實際推動載體多少：逐項的梯度範數。**不含任何指令。**

為什麼要量
────────────────────────────────────────────────────────────────────
`runs/advcf_texture_sds/` 的 CSV 讀到各項的值橫跨三個數量級：`id` ≈ 0.88、
`enc` ≈ 0.63、`enc_target` ≈ 1.6、`cond` ≈ 0.03、`diffusion` 與
`diffusion_target` ≈ 0.002。它們全部以權重 1.0 左右進分數，所以「這個目標
函數沒有用」與「這個目標函數的梯度比別項小三個數量級」在讀數上長得一模一樣。

值的尺度不等於梯度的尺度，所以這裡直接量後者：每一項單獨反傳到**載體參數**，
記下 `‖∂(w·f(term))/∂θ‖`，其中 `f` 是該項進分數時真正套用的函數（`tanh` 或
恆等），`w` 是設定檔給的權重。表上並列四欄——項的值、`f'` 在該值上的導數、
原始梯度範數、以及乘上權重與 `f'` 之後真正進到合梯度裡的範數。

**這一支不下判準。** 它只回報數字，哪一項該加權多少由使用者判。

起點不取恆等
────────────────────────────────────────────────────────────────────
`enc` 與 `cond` 是差向量的範數，載體在恆等起點上 `render(x) == x` 逐位元成立，
`torch.norm` 在零點回傳零次梯度，量到的會是假的零。載體一律帶 `init_jitter`，
與 `advcf_expect_noid_jitter` 同一個理由。
"""
import argparse
import csv
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

COLUMNS = ['image', 'at', 'term', 'weight', 'value', 'shape_derivative',
           'grad_norm_raw', 'grad_norm_effective', 'grad_absmax_raw',
           'seconds']


def main():
    import time

    import torch

    from src.defense.assets import load_image
    from src.defense.color_param import ColorCurveParam
    from src.defense.instruction_free import FreeObjective
    from src.defense.mainstream_terms import MainstreamTerms, build_target
    from src.defense.readout_terms import (OutputDisplacement,
                                           TargetedOutputDisplacement,
                                           ToneFlatness)
    from src.defense.sds_terms import SDSDiffusion
    from src.metrics.identity import face_boxes
    from src.metrics.suite import MetricSuite
    from src.models.ip2p import IP2PWrapper

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--target', default='texture')
    ap.add_argument('--jitter', type=float, default=0.02)
    ap.add_argument('--defended-root', type=Path, default=None,
                    help='含 <image>__immunised.png 的臂目錄。給了就在那張'
                         '收斂解上量，梯度對**影像**取而不是對載體參數取；'
                         '載體的狀態沒有存下來，而收斂點才是決定終局的地方。')
    args = ap.parse_args()

    spec = json.loads(args.config.read_text(encoding='utf-8'))
    manifest = json.loads(
        (ROOT / spec['manifest']).read_text(encoding='utf-8'))
    entries = {r['id']: r for r in manifest['images']}

    ip2p = IP2PWrapper(dtype={'fp16': torch.float16, 'fp32': torch.float32,
                              'bf16': torch.bfloat16}[spec.get('precision',
                                                               'bf16')])
    device = ip2p.device
    suite = MetricSuite(device=device)
    free = dict(spec['free'])
    rows = []

    for image in spec['images']:
        x = load_image(ROOT / entries[image]['path'], device)
        boxes = face_boxes(x, device)
        if not boxes:
            raise ValueError(f'{image} 偵測不到臉')
        box = max(boxes, key=lambda q: (q[2] - q[0]) * (q[3] - q[1]))
        carrier = ColorCurveParam(bound_mode='advcf', pieces=64, radius=5.0,
                                  init_jitter=float(args.jitter))
        carrier.reset(x, 0)
        params = [p for p in carrier.params() if p.requires_grad]

        objective = FreeObjective(
            ip2p, x, box=box, k=int(free['timesteps']),
            steps=int(spec.get('attack_steps', 50)),
            seed=int(free['noise_seed']), weights={'id': 1.0},
            chain_steps=int(free.get('chain_steps', 6)),
            grad_steps=int(free.get('grad_steps', 1)),
            resample=False, s_i=float(spec.get('s_i', 1.5)))
        target01 = build_target(args.target, x)
        mainstream = MainstreamTerms(objective, x, target=target01)
        sds = SDSDiffusion(objective)
        out_free = OutputDisplacement(objective, x, suite.lpips_module)
        out_tgt = TargetedOutputDisplacement(objective, target01,
                                             suite.lpips_module)
        tone = ToneFlatness(x, box=box, pieces=64)

        def base_term(name):
            return lambda y: objective.terms(y)[name]

        probes = [
            ('enc', 0.5, 'tanh', base_term('enc')),
            ('cond', 1.0, 'tanh', base_term('cond')),
            ('id', 1.0, 'identity', base_term('id')),
            ('enc_target', 1.0, 'tanh', mainstream.enc_target),
            ('diffusion', 1.0, 'tanh', mainstream.diffusion),
            ('diffusion_target', 1.0, 'tanh', mainstream.diffusion_target),
            ('sds', 1.0, 'tanh', sds),
            ('out', 2.0, 'identity', out_free),
            ('out_target', 2.0, 'identity', out_tgt),
            ('tone', 1.0, 'identity', lambda _y: tone(carrier)),
        ]
        if args.defended_root is not None:
            probes = [q for q in probes if q[0] != 'tone']

        defended = None
        if args.defended_root is not None:
            png = args.defended_root / f'{image}__immunised.png'
            if not png.exists():
                cands = sorted(args.defended_root.glob(
                    f'{image}__*__defended.png'))
                if len(cands) != 1:
                    raise SystemExit(
                        f'{args.defended_root} 底下找不到 {image} 唯一的防禦圖，'
                        f'候選 {[c.name for c in cands]}')
                png = cands[0]
            defended = load_image(png, device)
            print(f'{image}：在 {png.name} 上量', flush=True)

        for name, weight, shape, fn in probes:
            t0 = time.time()
            if defended is None:
                for p in params:
                    p.grad = None
                value = fn(carrier.render(x))
                value.backward()
                raw = torch.cat([p.grad.flatten() for p in params
                                 if p.grad is not None])
            else:
                y = defended.clone().detach().requires_grad_(True)
                value = fn(y)
                value.backward()
                raw = y.grad.detach().flatten()
            v = float(value.detach())
            deriv = (1.0 - math.tanh(v) ** 2) if shape == 'tanh' else 1.0
            rows.append({
                'image': image, 'term': name, 'weight': weight, 'value': v,
                'at': 'defended' if defended is not None else 'jitter_start',
                'shape_derivative': deriv,
                'grad_norm_raw': float(raw.norm()),
                'grad_norm_effective': float(raw.norm()) * weight * deriv,
                'grad_absmax_raw': float(raw.abs().amax()),
                'seconds': round(time.time() - t0, 1)})
            r = rows[-1]
            print(f'{image[:28]:28} {name:17} 值 {v:11.6f}  f\' {deriv:7.4f}  '
                  f'‖g‖ {r["grad_norm_raw"]:.4e}  '
                  f'有效 {r["grad_norm_effective"]:.4e}', flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)
    print(f'寫出 {args.out}（{len(rows)} 列）', flush=True)


if __name__ == '__main__':
    main()

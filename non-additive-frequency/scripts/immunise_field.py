"""產生防禦圖：載體是**直接參數化的位移場**，最佳化過程中沒有任何編輯指令。

與 `scripts/immunise.py` 的關係
────────────────────────────────────────────────────────────────────
目標函數、受約束求解、輸出合約、指令守門全部沿用，只換載體：

- `flow` — `geometry_field.FlowFieldParam`，取樣座標的低頻位移場。
- `lab_field` — `lab_offset_field.LabOffsetFieldParam`，逐通道的低頻 Lab 位移場。

兩者的均勻性都是參數化的性質而不是要對抗的約束：場活在低頻基底裡，做不出
高頻結構。`scripts/immunise.py` 一行都不動，兩條線的產物因此可以並列。

指令守門
────────────────────────────────────────────────────────────────────
`assert_no_instructions` 由 `scripts/immunise.py` 載入，不另外實作一份——
兩支腳本共用同一個檢查，`tests/test_field_carriers.py` 對本支的設定檔再釘一次。
"""
import argparse
import csv
import importlib.util
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

COLUMNS = [
    'image', 'variant', 'carrier', 'rejected', 'reject_reason',
    'free_score_start', 'free_score_unprojected', 'free_score_end',
    'free_score_q8', 'free_amplitude_shrink', 'free_cap_violations',
    'free_feasible_step', 'free_feasible_checks', 'free_rho', 'free_lambda',
    'free_caps_unprojected', 'free_amplitude_base',
    'free_term_enc', 'free_term_cond', 'free_term_id',
    'free_term_enc_target', 'free_term_diffusion', 'free_steps', 'free_lr',
    'cap_names', 'cap_values', 'cap_reached',
    'niqe_def', 'niqe_original',
    'support_deltaE00', 'face_deltaE00', 'face_box_deltaE00',
    'tv_frame', 'u16_frame', 'endpoint_new_frame',
    'tv_face_box', 'u16_face_box', 'endpoint_new_face_box',
    'tv_L', 'tv_a', 'tv_b', 'offset_std',
    'hf_rgb_total', 'hf_lab_L', 'hf_lab_a', 'hf_lab_b',
    'latent_l2', 'psnr', 'lpips_def', 'seconds',
    'grid', 'taper', 'flow_px_max', 'flow_px_mean',
    'flow_det_min', 'flow_det_mean', 'flow_theta_norm',
    'offset_L_max', 'offset_a_max', 'offset_b_max',
    'offset_L_mean', 'offset_a_mean', 'offset_b_mean',
    'offset_L_std', 'offset_a_std', 'offset_b_std',
    'clipping_fraction', 'clipping_max',
]


def load_guard():
    spec = importlib.util.spec_from_file_location(
        'immunise_script', ROOT / 'scripts' / 'immunise.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.assert_no_instructions, mod.box_mask


def main():
    import torch

    from src.defense.assets import load_image, save_png
    from src.defense.carrier_mask import face_subject_mask
    from src.defense.color_amplitude import delta_e00
    from src.defense.delta_e_torch import cvar_from_map, delta_e00_torch
    from src.defense.eot import EOTObjective
    from src.defense.geometry_field import (FlowFieldParam, affine_residual,
                                            det_jacobian)
    from src.defense.immunise import Cap, optimise_carrier
    from src.defense.instruction_free import FreeObjective
    from src.defense.lab_offset_field import CHANNELS, LabOffsetFieldParam
    from src.defense.lowfreq_color import highfreq_report
    from src.defense.mainstream_terms import (CombinedObjective,
                                              MainstreamTerms)
    from src.defense.ncf_library import sha256
    from src.defense.ncf_runner import ncf_support
    from src.defense.uniformity import (endpoint_ramp, new_endpoint_fraction,
                                        uniformity_row)
    from src.metrics.identity import face_boxes
    from src.metrics.suite import MetricSuite

    assert_no_instructions, box_mask = load_guard()

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--shard', default='1/1')
    ap.add_argument('--steps', type=int, default=0)
    ap.add_argument('--variants', default='',
                    help='逗號分隔的變體名稱，只跑這幾組')
    args = ap.parse_args()

    spec = json.loads(args.config.read_text(encoding='utf-8'))
    assert_no_instructions(spec)
    manifest = json.loads((ROOT / spec['manifest']).read_text(encoding='utf-8'))
    if args.out.exists() and any(args.out.iterdir()):
        raise SystemExit(f'{args.out} 已存在且非空；換一個輸出目錄')
    args.out.mkdir(parents=True, exist_ok=True)

    free = spec['free']
    steps = args.steps or int(free['steps'])
    variants = spec['variants']
    if args.variants:
        want = {v.strip() for v in args.variants.split(',')}
        variants = [v for v in variants if v['name'] in want]
        if not variants:
            raise SystemExit(f'設定檔裡沒有這些變體：{sorted(want)}')
    i, n = (int(v) for v in args.shard.split('/'))
    if not 1 <= i <= n:
        raise SystemExit(f'--shard 要寫成 i/n，收到 {args.shard!r}')
    todo = [im for k, im in enumerate(spec['images']) if k % n == i - 1]

    from src.models.ip2p import IP2PWrapper
    ip2p = IP2PWrapper(dtype={'fp16': torch.float16, 'fp32': torch.float32,
                              'bf16': torch.bfloat16}[spec['precision']])
    device = ip2p.device
    suite = MetricSuite(device=device)
    entries = {r['id']: r for r in manifest['images']}
    suffix = '' if n == 1 else f'_shard{i}of{n}'
    print(f'分片 {i}/{n}：{len(todo)} 張，變體 {len(variants)} 組，'
          f'梯度 {steps} 步，精度 {spec["precision"]}，指令：無', flush=True)

    rows = []
    for image in todo:
        entry = entries[image]
        src = ROOT / entry['path']
        if sha256(src) != entry['sha256']:
            raise ValueError(f'{image} 的輸入雜湊不符')
        x = load_image(src, device)
        frame = ncf_support(x, 'frame')
        face = face_subject_mask(x, device=device)
        boxes = face_boxes(x, device)
        if not boxes:
            raise ValueError(f'{image} 偵測不到臉，無法錨定身分項')
        box = max(boxes, key=lambda q: (q[2] - q[0]) * (q[3] - q[1]))
        anchor = box_mask(x, box)
        outside = (1.0 - anchor).to(anchor)
        niqe_x = suite.niqe(x)

        cache = {}

        def objective_for(variant):
            """依目標規格建目標，相同規格的變體共用一份。

            前置計算（原圖那一側的 latent、每個時刻的 ε、身分嵌入）佔一格裡
            不小的比例，而同一張影像上多數變體的目標規格是一樣的——只有掃目標
            函數的那一批不同。以規格為鍵快取，兩種批次都不會多付。
            """
            weights = dict(free.get('weights') or {}, **(variant.get('weights') or {}))
            chain = int(variant.get('chain_steps', free.get('chain_steps', 6)))
            grad = int(variant.get('grad_steps', free.get('grad_steps', 1)))
            eot_spec = dict(free.get('eot') or {}, **(variant.get('eot') or {}))
            key = (tuple(sorted(weights.items())), chain, grad,
                   tuple(sorted(eot_spec.get('kinds') or [])),
                   int(eot_spec.get('samples', 1)))
            if key in cache:
                return cache[key]
            obj = FreeObjective(
                ip2p, x, box=box, k=int(free['timesteps']),
                steps=int(spec['attack_steps']), seed=int(free['noise_seed']),
                weights=weights, chain_steps=chain, grad_steps=grad,
                s_i=float(spec.get('s_i', 1.5)))
            if any(weights.get(k) for k in ('enc_target', 'diffusion')):
                obj = CombinedObjective(obj, MainstreamTerms(obj, x),
                                        dict(obj.weights, **weights))
            if eot_spec.get('kinds'):
                obj = EOTObjective(
                    obj, kinds=list(eot_spec['kinds']),
                    samples=int(eot_spec.get('samples', 1)),
                    seed=int(eot_spec.get('seed', 0)), device=device,
                    include_identity=bool(eot_spec.get('include_identity', True)))
            cache[key] = obj
            return obj

        with torch.no_grad():
            z0 = ip2p.encode_image(x).float()

        for variant in variants:
            knobs = variant.get('knobs', {})
            caps_spec = variant.get('caps', {})
            kind = variant['carrier']
            objective = objective_for(variant)
            t0 = time.time()

            if kind == 'flow':
                carrier = FlowFieldParam(
                    x, grid=int(knobs.get('grid', 16)),
                    taper=float(knobs.get('taper', 16.0)),
                    box=float(knobs.get('box', 64.0)))
                carrier.reset(x)
                q = float(caps_spec.get('quantile', 0.99))
                cell_caps = [
                    Cap('flow_face',
                        (lambda _y, c=carrier: cvar_from_map(
                            c.magnitude(), anchor, q)),
                        (lambda _y, c=carrier: float(cvar_from_map(
                            c.magnitude(), anchor, q).detach())),
                        float(caps_spec['face_px'])),
                    Cap('flow_rigid',
                        (lambda _y, c=carrier: cvar_from_map(
                            affine_residual(c.flow(), outside).pow(2)
                            .sum(1, keepdim=True).clamp_min(1e-12).sqrt(),
                            outside, q)),
                        (lambda _y, c=carrier: float(cvar_from_map(
                            affine_residual(c.flow(), outside).pow(2)
                            .sum(1, keepdim=True).clamp_min(1e-12).sqrt(),
                            outside, q).detach())),
                        float(caps_spec['rigid_px'])),
                ]
                if float(caps_spec.get('face_rigid_px', 0) or 0) > 0:
                    cell_caps.append(Cap(
                        'flow_face_rigid',
                        (lambda _y, c=carrier: cvar_from_map(
                            affine_residual(c.flow(), anchor).pow(2)
                            .sum(1, keepdim=True).clamp_min(1e-12).sqrt(),
                            anchor, q)),
                        (lambda _y, c=carrier: float(cvar_from_map(
                            affine_residual(c.flow(), anchor).pow(2)
                            .sum(1, keepdim=True).clamp_min(1e-12).sqrt(),
                            anchor, q).detach())),
                        float(caps_spec['face_rigid_px'])))
                cell_caps += [
                    Cap('flow_fold',
                        (lambda _y, c=carrier: c.fold_measure(
                            floor=float(caps_spec.get('det_floor', 0.5)))),
                        (lambda _y, c=carrier: float(c.fold_measure(
                            floor=float(caps_spec.get('det_floor', 0.5)))
                            .detach())),
                        float(caps_spec.get('fold', 1e-3))),
                ]
            elif kind == 'lab_field':
                carrier = LabOffsetFieldParam(
                    x, grid=int(knobs.get('grid', 8)), support=frame,
                    box=tuple(knobs.get('box', (40.0, 80.0, 80.0))))
                carrier.reset(x)
                q = float(caps_spec.get('quantile', 0.99))
                cell_caps = []
                for k2, tag in enumerate(CHANNELS):
                    value = caps_spec.get(f'offset_{tag}')
                    if value is None or float(value) <= 0:
                        continue
                    cell_caps.append(Cap(
                        f'offset_{tag}',
                        (lambda _y, c=carrier, j=k2: cvar_from_map(
                            c.channel_magnitude(j), frame, q)),
                        (lambda _y, c=carrier, j=k2: float(cvar_from_map(
                            c.channel_magnitude(j), frame, q).detach())),
                        float(value)))
                if float(caps_spec.get('delta_e', 0) or 0) > 0:
                    de = float(caps_spec['delta_e'])
                    cell_caps += [
                        Cap('frame', lambda y: delta_e00_torch(x, y, frame),
                            lambda y: delta_e00(x, y, frame), de),
                        Cap('face_box', lambda y: delta_e00_torch(x, y, anchor),
                            lambda y: delta_e00(x, y, anchor), de)]
                if float(caps_spec.get('endpoint', 0) or 0) > 0:
                    cell_caps.append(Cap(
                        'frame_endpoint',
                        lambda y: endpoint_ramp(x, y, frame),
                        lambda y: new_endpoint_fraction(x, y, frame),
                        float(caps_spec['endpoint'])))
            else:
                raise SystemExit(f'不認得的載體 {kind!r}；只有 flow 與 lab_field')

            lr = float(variant.get('lr', free['lr']))
            stats = optimise_carrier(carrier, x, objective, steps=steps,
                                     lr=lr, caps=cell_caps,
                                     rho=float(free.get('rho', 10.0)),
                                     lam_every=int(free.get('lam_every', 5)),
                                     check_every=int(free.get('check_every', 10)),
                                     log_every=int(free.get('log_every', 0)))

            with torch.no_grad():
                x_def = (carrier.render(x).detach().clamp(0, 1) * 255).round() / 255
                niqe_def = suite.niqe(x_def)
                hf = highfreq_report(x, x_def)
                latent = float((ip2p.encode_image(x_def).float() - z0)
                               .flatten().norm())
                uni = uniformity_row(x, x_def, {'frame': frame,
                                                'face_box': anchor})
                diag = carrier.diagnostics(x)
                score_q8 = float(objective.score(x_def))
                reached = {c.name: float(c.hard(x_def)) for c in cell_caps}
                pair = suite.pairwise(x, x_def)
                psnr, lp = pair['psnr'], pair['lpips']
            over = [c.name for c in cell_caps if reached[c.name] > c.value]
            row = {
                'image': image, 'variant': variant['name'], 'carrier': kind,
                'rejected': int(bool(over)), 'reject_reason': '|'.join(over),
                'free_score_q8': round(score_q8, 5),
                'cap_names': '|'.join(c.name for c in cell_caps),
                'cap_values': '|'.join(f'{c.value:g}' for c in cell_caps),
                'cap_reached': '|'.join(f'{reached[c.name]:.4f}'
                                        for c in cell_caps),
                'niqe_def': round(niqe_def, 5),
                'niqe_original': round(niqe_x, 5),
                'support_deltaE00': round(float(delta_e00(x, x_def, frame)), 5),
                'face_deltaE00': round(float(delta_e00(x, x_def, face)), 5),
                'face_box_deltaE00': round(float(delta_e00(x, x_def, anchor)), 5),
                **uni,
                'hf_rgb_total': hf.get('hf_ratio_rgb_total'),
                'hf_lab_L': hf.get('hf_ratio_lab_L'),
                'hf_lab_a': hf.get('hf_ratio_lab_a'),
                'hf_lab_b': hf.get('hf_ratio_lab_b'),
                'latent_l2': round(latent, 5),
                'psnr': round(psnr, 4), 'lpips_def': round(lp, 5),
                'seconds': round(time.time() - t0, 1),
                **{k2: v for k2, v in diag.items() if k2 in COLUMNS},
                **stats,
            }
            rows.append(row)
            save_png(x_def, args.out / f'{image}__{variant["name"]}__def.png')
            print(f'  {image} · {variant["name"]}  score '
                  f'{row["free_score_start"]:.4f} → {row["free_score_end"]:.4f}  '
                  f'q8 {row["free_score_q8"]:.4f}  '
                  f'id {row.get("free_term_id")}  '
                  f'上限 {row["cap_reached"]} / {row["cap_values"]}  '
                  f'ΔE00 {row["support_deltaE00"]:.2f}  '
                  f'PSNR {row["psnr"]:.2f}  LPIPS {row["lpips_def"]:.4f}  '
                  f'niqe {niqe_def:.2f}/{niqe_x:.2f}  '
                  f'可行步 {row["free_feasible_step"]}'
                  f'{"  [違反]" if over else ""}', flush=True)

            if not over:
                save_png(x_def, args.out
                         / f'{image}__{variant["name"]}__immunised.png')
                (args.out / f'{image}__{variant["name"]}__immunised.json'
                 ).write_text(json.dumps(
                     {'image': image, 'variant': variant['name'],
                      'carrier': kind, 'knobs': knobs, 'caps': caps_spec,
                      'readouts': row, 'objective': 'instruction_free',
                      'weights': objective.weights,
                      'chain_steps': int(variant.get(
                          'chain_steps', free.get('chain_steps', 6))),
                      'grad_steps': int(variant.get(
                          'grad_steps', free.get('grad_steps', 1))),
                      'eot': dict(free.get('eot') or {},
                                  **(variant.get('eot') or {})),
                      'timesteps': [int(t) for t in objective.timesteps]},
                     ensure_ascii=False, indent=2), encoding='utf-8')

    target = args.out / f'immunise_field{suffix}.csv'
    with open(target, 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)
    print(f'寫出 {target}（{len(rows)} 列）', flush=True)


if __name__ == '__main__':
    main()

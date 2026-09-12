"""產生防禦圖：**最佳化過程中沒有任何編輯指令參與**。

這一支取代 `scripts/carrier_search.py` 的角色。舊路徑每評估一個候選點就用
那一格的攻擊指令跑一次真編輯，指令因此是最佳化的輸入；量到的是「對這句話
調出來的防禦」。新路徑只用 `src/defense/instruction_free.py` 的三個不含文字的
項（encoder 位移、空文字條件下的 UNet 預測位移、空指令取樣輸出上的身分餘弦），
所以一張照片只產生**一張**防禦圖，與攻擊者打算下什麼指令無關。

指令只在 `scripts/evaluate_defence.py` 出現，那是純粹的測試資料。本腳本讀到
設定檔裡有任何 `instruction` 欄位就拒絕啟動——這個保證要由程式擋，不靠自律。

分工
────────────────────────────────────────────────────────────────────
場的上千個參數由**梯度**解（Adam）。結構性的選擇（控制點密度、帶寬、配色、
亮度鎖不鎖）是離散的，梯度幫不上忙，改成在設定檔列出的一小組結構上逐一試，
取不含指令目標最好的那一個。因為每次評估不必跑 50 步取樣，成本比舊路徑低
兩個數量級，結構列得起。
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

COLUMNS = [
    'image', 'structure', 'selected', 'rejected', 'reject_reason',
    'free_score_start', 'free_score_unprojected', 'free_score_end',
    'free_score_q8', 'free_amplitude_shrink', 'free_cap_violations',
    'free_feasible_step', 'free_feasible_checks', 'free_rho', 'free_lambda',
    'free_caps_unprojected', 'free_amplitude_base',
    'free_term_enc', 'free_term_cond', 'free_term_id', 'free_steps', 'free_lr',
    'niqe_def', 'niqe_original', 'niqe_cap',
    'support_deltaE00', 'face_deltaE00', 'face_box_deltaE00',
    'clothes_deltaE00',
    'tail_quantile', 'tail_cap', 'tv_cap', 'endpoint_cap', 'raw_cap',
    'support_tail', 'face_tail', 'face_box_tail',
    'tv_frame', 'u16_frame', 'endpoint_new_frame',
    'tv_face_box', 'u16_face_box', 'endpoint_new_face_box',
    'tv_L', 'tv_a', 'tv_b', 'offset_std', 'raw_excursion',
    'clipping_fraction', 'clipping_max',
    'hf_rgb_total', 'hf_lab_L', 'hf_lab_a', 'hf_lab_b',
    'latent_l2', 'seconds',
    'knob_face_grid', 'knob_face_sigma', 'knob_face_scale',
    'knob_face_amplitude', 'knob_frame_grid', 'knob_frame_sigma',
    'knob_frame_scale', 'knob_frame_amplitude', 'knob_clothes_grid',
    'knob_clothes_sigma', 'knob_clothes_scale', 'knob_clothes_amplitude',
    'knob_face_palette', 'knob_clothes_palette', 'knob_lock_luminance',
    'knob_field_seed',
]


def box_mask(x, box):
    """主體框的硬遮罩，**與身分嵌入實際裁下來的範圍對齊**。

    `face_subject_mask` 是 ATR 的 Face 與 Hair 經膨脹與羽化，範圍比身分框大，
    多人照片裡還會把別人的臉算進平均。兩者都報，上限兩者都套。

    這裡照 `metrics.identity.embed_box` 的算式擴框：嵌入是在擴框之後的區域上取的，
    遮罩若只用原框，最佳化可以把高色差推到「嵌入看得到、上限沒算到」的邊緣。
    """
    import torch

    from src.metrics.identity import DETECTOR_IMAGE_SIZE, DETECTOR_MARGIN
    m = torch.zeros_like(x[:, :1])
    x0, y0, x1, y1 = (float(v) for v in box)
    mx = DETECTOR_MARGIN * (x1 - x0) / (DETECTOR_IMAGE_SIZE - DETECTOR_MARGIN)
    my = DETECTOR_MARGIN * (y1 - y0) / (DETECTOR_IMAGE_SIZE - DETECTOR_MARGIN)
    a, b = int(x0 - mx / 2), int(y0 - my / 2)
    c, d = int(x1 + mx / 2), int(y1 + my / 2)
    h, w = x.shape[-2:]
    m[..., max(0, b):min(h, d), max(0, a):min(w, c)] = 1.0
    if float(m.sum()) < 4:
        raise ValueError(f'主體框退化：{(a, b, c, d)}')
    return m


def assert_no_instructions(node, path='spec'):
    """設定檔裡不得出現指令。這個保證由程式擋，不靠自律。"""
    if isinstance(node, dict):
        for k, v in node.items():
            if 'instruction' in k.lower() or 'prompt' in k.lower():
                raise SystemExit(
                    f'{path}.{k} 帶有指令；免疫階段不得看到任何指令')
            assert_no_instructions(v, f'{path}.{k}')
    elif isinstance(node, list):
        for i, v in enumerate(node):
            assert_no_instructions(v, f'{path}[{i}]')


def main():
    import torch

    from src.defense.assets import load_image, palette_bank, palette_of, save_png
    from src.defense.carrier_search import START, build_carrier, describe
    from src.defense.carrier_mask import face_subject_mask
    from src.defense.color_amplitude import delta_e00
    from src.defense.color_amplitude import cvar_e00
    from src.defense.delta_e_torch import cvar_torch, delta_e00_torch
    from src.defense.immunise import Cap, fit_caps, optimise_carrier
    from src.defense.instruction_free import FreeObjective
    from src.defense.lowfreq_color import highfreq_report
    from src.defense.uniformity import (endpoint_ramp, new_endpoint_fraction,
                                        lab_offset, raw_excursion, tv_offset,
                                        u16_offset, uniformity_row)
    from src.defense.ncf_library import sha256
    from src.defense.ncf_runner import ncf_support
    from src.metrics.identity import face_boxes
    from src.metrics.suite import MetricSuite

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--shard', default='1/1')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--steps', type=int, default=0)
    ap.add_argument('--tail-cap', type=float, default=0.0,
                    help='尾端色差上限，覆寫設定檔的 tail_cap')
    ap.add_argument('--structures', default='',
                    help='逗號分隔的結構名稱，只跑這幾組')
    ap.add_argument('--tv-cap', type=float, default=0.0,
                    help='位移場全變差的上限，覆寫設定檔的 tv_cap')
    ap.add_argument('--endpoint-cap', type=float, default=0.0,
                    help='新增端點像素比例的上限')
    ap.add_argument('--raw-cap', type=float, default=0.0,
                    help='每段在 gamut 壓縮前的越界幅度上限')
    args = ap.parse_args()

    spec = json.loads(args.config.read_text(encoding='utf-8'))
    assert_no_instructions(spec)
    manifest = json.loads((ROOT / spec['manifest']).read_text(encoding='utf-8'))
    if args.out.exists() and any(args.out.iterdir()):
        raise SystemExit(f'{args.out} 已存在且非空；換一個輸出目錄')
    args.out.mkdir(parents=True, exist_ok=True)

    free = spec['free']
    steps = args.steps or int(free['steps'])
    tail_q = float(spec.get('tail_quantile', 0.95))
    tail_cap = args.tail_cap or float(spec.get('tail_cap', 0) or 0)
    tv_cap = args.tv_cap or float(spec.get('tv_cap', 0) or 0)
    end_cap = args.endpoint_cap or float(spec.get('endpoint_cap', 0) or 0)
    raw_cap = args.raw_cap or float(spec.get('raw_cap', 0) or 0)
    structures = spec['structures']
    if args.structures:
        want = {v.strip() for v in args.structures.split(',')}
        structures = [t for t in structures if t['name'] in want]
        if not structures:
            raise SystemExit(f'設定檔裡沒有這些結構：{sorted(want)}')
    i, n = (int(v) for v in args.shard.split('/'))
    if not 1 <= i <= n:
        raise SystemExit(f'--shard 要寫成 i/n，收到 {args.shard!r}')
    todo = [im for k, im in enumerate(spec['images']) if k % n == i - 1]
    if args.limit:
        todo = todo[:args.limit]

    from src.models.ip2p import IP2PWrapper
    ip2p = IP2PWrapper(dtype={'fp16': torch.float16, 'fp32': torch.float32,
                              'bf16': torch.bfloat16}[spec['precision']])
    device = ip2p.device
    suite = MetricSuite(device=device)
    entries = {r['id']: r for r in manifest['images']}
    suffix = '' if n == 1 else f'_shard{i}of{n}'
    print(f'分片 {i}/{n}：{len(todo)} 張，結構 {len(structures)} 組，'
          f'梯度 {steps} 步，尾端上限 {tail_cap or "無"}（q={tail_q}），'
          f'精度 {spec["precision"]}，指令：無', flush=True)

    rows = []
    for image in todo:
        entry = entries[image]
        src = ROOT / entry['path']
        if sha256(src) != entry['sha256']:
            raise ValueError(f'{image} 的輸入雜湊不符')
        x = load_image(src, device)
        frame = ncf_support(x, 'frame')
        clothes = ncf_support(x, 'clothes')
        face = face_subject_mask(x, device=device)
        frame_palette = palette_of(spec, spec['palette_id'])
        clothes_bank = palette_bank(spec, spec['clothes_palette_class'])
        face_bank = palette_bank(spec, spec['face_palette_class'])
        boxes = face_boxes(x, device)
        if not boxes:
            raise ValueError(f'{image} 偵測不到臉，無法錨定身分項')
        box = max(boxes, key=lambda q: (q[2] - q[0]) * (q[3] - q[1]))

        niqe_x = suite.niqe(x)
        niqe_cap = float(spec['naturalness_max_ratio']) * niqe_x
        de_cap = float(spec['delta_e_cap'])
        face_cap = float(spec['face_delta_e_cap'])
        anchor = box_mask(x, box)
        caps = [Cap('frame', lambda y: delta_e00_torch(x, y, frame),
                    lambda y: delta_e00(x, y, frame), de_cap),
                Cap('face', lambda y: delta_e00_torch(x, y, face),
                    lambda y: delta_e00(x, y, face), face_cap),
                Cap('face_box', lambda y: delta_e00_torch(x, y, anchor),
                    lambda y: delta_e00(x, y, anchor), face_cap)]
        if tail_cap > 0:
            caps += [
                Cap('frame_tail', lambda y: cvar_torch(x, y, frame, tail_q),
                    lambda y: cvar_e00(x, y, frame, tail_q), tail_cap),
                Cap('face_tail', lambda y: cvar_torch(x, y, face, tail_q),
                    lambda y: cvar_e00(x, y, face, tail_q), tail_cap),
                Cap('face_box_tail', lambda y: cvar_torch(x, y, anchor, tail_q),
                    lambda y: cvar_e00(x, y, anchor, tail_q), tail_cap)]
        if tv_cap > 0:
            caps += [
                Cap('frame_tv', lambda y: tv_offset(lab_offset(x, y), frame),
                    lambda y: float(tv_offset(lab_offset(x, y), frame)), tv_cap),
                Cap('face_box_tv', lambda y: tv_offset(lab_offset(x, y), anchor),
                    lambda y: float(tv_offset(lab_offset(x, y), anchor)), tv_cap)]
        if end_cap > 0:
            caps += [
                Cap('frame_endpoint', lambda y: endpoint_ramp(x, y, frame),
                    lambda y: new_endpoint_fraction(x, y, frame), end_cap),
                Cap('face_box_endpoint', lambda y: endpoint_ramp(x, y, anchor),
                    lambda y: new_endpoint_fraction(x, y, anchor), end_cap)]

        objective = FreeObjective(
            ip2p, x, box=box, k=int(free['timesteps']),
            steps=int(spec['attack_steps']), seed=int(free['noise_seed']),
            weights=free.get('weights'),
            chain_steps=int(free.get('chain_steps', 6)),
            grad_steps=int(free.get('grad_steps', 1)),
            s_i=float(spec.get('s_i', 1.5)))
        with torch.no_grad():
            z0 = ip2p.encode_image(x).float()

        best = None
        for structure in structures:
            values = dict(START, **structure['knobs'])
            t0 = time.time()
            carrier = build_carrier(
                values, x, frame_support=frame, clothes_support=clothes,
                face_support=face, frame_palette=frame_palette,
                clothes_palette=clothes_bank[int(values['clothes_palette'])
                                             % len(clothes_bank)],
                face_palette=face_bank[int(values['face_palette'])
                                       % len(face_bank)],
                radius=spec['radius'], max_gain=spec['max_gain'])
            cell_caps = list(caps)
            if raw_cap > 0:
                for k2, tag in enumerate(carrier.tags):
                    cell_caps.append(Cap(
                        f'raw_{tag}',
                        (lambda _y, c=carrier, j=k2: raw_excursion(c, x)[j]),
                        (lambda _y, c=carrier, j=k2:
                         float(raw_excursion(c, x)[j].detach())),
                        raw_cap))
            stats = optimise_carrier(carrier, x, objective, steps=steps,
                                     lr=float(free['lr']), caps=cell_caps,
                                     rho=float(free.get('rho', 10.0)),
                                     lam_every=int(free.get('lam_every', 5)),
                                     check_every=int(free.get('check_every', 10)),
                                     log_every=int(free.get('log_every', 0)))
            with torch.no_grad():
                x_def = (carrier.render(x).detach().clamp(0, 1)
                         * 255).round() / 255
                niqe_def = suite.niqe(x_def)
                hf = highfreq_report(x, x_def)
                latent = float((ip2p.encode_image(x_def).float() - z0)
                               .flatten().norm())
                de_frame = float(delta_e00(x, x_def, frame))
                de_face = float(delta_e00(x, x_def, face))
                de_box = float(delta_e00(x, x_def, anchor))
                tails = {c.name: float(c.hard(x_def)) for c in cell_caps
                         if c.name.endswith('_tail')}
                uni = uniformity_row(x, x_def, {'frame': frame,
                                                'face_box': anchor})
                raws = [float(v) for v in raw_excursion(carrier, x)]
                diag = carrier.diagnostics(x)
                score_q8 = float(objective.score(x_def))
            over = [c.name for c in cell_caps if float(c.hard(x_def)) > c.value]
            rejected = int(niqe_def > niqe_cap or bool(over))
            row = {
                'image': image, 'structure': structure['name'],
                'selected': 0, 'rejected': rejected,
                'reject_reason': ('niqe' if niqe_def > niqe_cap
                                  else '|'.join(over)),
                'free_score_q8': round(score_q8, 5),
                'niqe_def': round(niqe_def, 5),
                'niqe_original': round(niqe_x, 5),
                'niqe_cap': round(niqe_cap, 5),
                'support_deltaE00': round(de_frame, 5),
                'face_deltaE00': round(de_face, 5),
                'face_box_deltaE00': round(de_box, 5),
                'clothes_deltaE00': round(float(delta_e00(x, x_def, clothes)), 5),
                'tail_quantile': tail_q, 'tail_cap': tail_cap,
                'tv_cap': tv_cap, 'endpoint_cap': end_cap, 'raw_cap': raw_cap,
                'raw_excursion': '|'.join(f'{v:.4f}' for v in raws), **uni,
                'support_tail': round(tails.get('frame_tail', float('nan')), 5),
                'face_tail': round(tails.get('face_tail', float('nan')), 5),
                'face_box_tail': round(tails.get('face_box_tail', float('nan')), 5),
                'clipping_fraction': round(max(
                    v for k2, v in diag.items()
                    if k2.endswith('_clipping_fraction')), 6),
                'clipping_max': round(max(
                    v for k2, v in diag.items()
                    if k2.endswith('_clipping_max')), 6),
                'hf_rgb_total': hf.get('hf_ratio_rgb_total'),
                'hf_lab_L': hf.get('hf_ratio_lab_L'),
                'hf_lab_a': hf.get('hf_ratio_lab_a'),
                'hf_lab_b': hf.get('hf_ratio_lab_b'),
                'latent_l2': round(latent, 5),
                'seconds': round(time.time() - t0, 1),
                **stats, **describe(values),
            }
            rows.append(row)
            save_png(x_def, args.out / f'{image}__{structure["name"]}__def.png')
            print(f'  {image} · {structure["name"]}  score '
                  f'{row["free_score_start"]:.4f} → {row["free_score_end"]:.4f}  '
                  f'q8 {row["free_score_q8"]:.4f}  '
                  f'ΔE00 {row["support_deltaE00"]:.1f}/{row["face_deltaE00"]:.1f}/'
                  f'{row["face_box_deltaE00"]:.1f}  '
                  f'尾端 {row["support_tail"]:.1f}/{row["face_tail"]:.1f}/'
                  f'{row["face_box_tail"]:.1f}  '
                  f'TV {row["tv_frame"]:.3f} U16 {row["u16_frame"]:.1f} '
                  f'端點 {row["endpoint_new_frame"]*100:.1f}%  '
                  f'niqe {niqe_def:.2f}/{niqe_cap:.2f}  '
                  f'可行步 {row["free_feasible_step"]}'
                  f'{"  [退回]" if rejected else ""}', flush=True)
            if not rejected and (best is None
                                 or row['free_score_q8'] < best[0]['free_score_q8']):
                best = (row, values, x_def)

        if best is None:
            raise ValueError(f'{image} 的所有結構都被自然度門檻退回')
        row, values, x_def = best
        row['selected'] = 1
        save_png(x_def, args.out / f'{image}__immunised.png')
        (args.out / f'{image}__immunised.json').write_text(
            json.dumps({'image': image, 'structure': row['structure'],
                        'knobs': values, 'readouts': row,
                        'objective': 'instruction_free',
                        'weights': objective.weights,
                        'timesteps': [int(t) for t in objective.timesteps]},
                       ensure_ascii=False, indent=2), encoding='utf-8')

        start_values = dict(START)
        carrier = build_carrier(
            start_values, x, frame_support=frame, clothes_support=clothes,
            face_support=face, frame_palette=frame_palette,
            clothes_palette=clothes_bank[0], face_palette=face_bank[0],
            radius=spec['radius'], max_gain=spec['max_gain'])
        with torch.no_grad():
            start_shrink = fit_caps(carrier, x, caps)
            save_png(carrier.render(x).detach(), args.out / f'{image}__start.png')
        print(f'  {image} · start 幅度縮放 {start_shrink:.4f}', flush=True)

    target = args.out / f'immunise{suffix}.csv'
    with open(target, 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)
    print(f'寫出 {target}（{len(rows)} 列）', flush=True)


if __name__ == '__main__':
    main()

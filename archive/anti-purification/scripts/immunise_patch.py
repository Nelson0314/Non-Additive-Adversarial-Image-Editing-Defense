"""把衣物材質貼片對不含指令的目標最佳化，並存下防禦圖。

**這一支不含任何指令。** 設定檔遞迴檢查，出現 `instruction`／`prompt` 鍵就拒絕
啟動，與 `scripts/immunise.py`、`scripts/paper_baseline.py` 同一條規矩。

目標用既有的 `enc`／`cond`／`id`
────────────────────────────────────────────────────────────────────
不是新寫的文字條件反應比值。理由是 `runs/conditioning_probe.csv` 量到那個比值
在「照片還自然」的區間裡動不過換取樣種子的雜訊地板（中位相對全距 0.053），
而且同一張圖換 16 個文字方向比值散在 0.15–9.95。那個量沒有可用的最佳化訊號。

`enc`／`cond`／`id` 推得動讀數（`runs/advcf_*` 各批），而且**沿用它們才能讓貼片
與先前每一個臂頭對頭比**——變的只有載體，目標函數逐欄相同。

對照臂
────────────────────────────────────────────────────────────────────
`solver: random` 用**同一個支撐、同一組色盤、同一個起點振幅**但完全不最佳化。
這一族的隨機對照不能像曲線族那樣二分幅度：貼片的失真幾乎全部由支撐面積與
色盤決定，而那兩者在兩臂之間是同一份。所以對照的作法是固定其餘一切、
只拿掉最佳化本身。
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

COLUMNS = ['image', 'variant', 'solver', 'objective', 'steps', 'lr', 'restarts',
           'area', 'blobs', 'grid', 'palette', 'palette_mode', 'chroma_gain',
           'chroma', 'logit_cap', 'init_jitter', 'seed',
           'canvas_size', 'canvas_epsilon', 'canvas_keep_shading',
           'canvas_drift', 'patch_area', 'patch_weight_mean',
           'support_kind', 'support_shape', 'support_count', 'support_pitch',
           'print_colours', 'print_archetype', 'print_tau', 'print_chroma_max',
           'print_chroma_used', 'print_lightness_span', 'print_share_min',
           'print_patch_area',
           'target_mode', 'target_epsilon', 'target_drift', 'target_box',
           'target_keep_shading',
           'deltaE00', 'deltaE00_patch', 'psnr', 'ssim', 'lpips', 'linf',
           'subject_id_defended',
           'free_score_start', 'free_score_end', 'free_score_unprojected',
           'free_feasible_step', 'free_restart_pick', 'free_restart_scores',
           'free_curve', 'free_term_enc', 'free_term_cond', 'free_term_id',
           'free_term_outside',
           'outside_near', 'outside_far', 'outside_face', 'outside_outside',
           'seconds', 'note']


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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--variants', nargs='+', default=[])
    args = ap.parse_args()

    import torch

    from src.defense.assets import load_image, save_png
    from src.defense.color_amplitude import delta_e00
    from src.defense.immunise import optimise_carrier
    from src.defense.instruction_free import FreeObjective
    from src.defense.material_patch import (MaterialPatchParam, garment_palette,
                                            patch_support)
    from src.defense.carrier_mask import lattice_support
    from src.defense.outside_terms import OutsideDisplacement, PrintObjective
    from src.defense.patch_canvas import (PatchCanvasParam, SupportWeightedFree,
                                          load_pattern)
    from src.defense.print_patch import (GeometricPrintParam, place_shapes,
                                         safe_mask)
    from src.defense.target_patch import TargetImagePatch, load_target
    from src.defense.ncf_library import sha256
    from src.metrics.identity import embed, face_boxes, similarity
    from src.metrics.suite import MetricSuite
    from src.models.ip2p import IP2PWrapper

    spec = json.loads(args.config.read_text(encoding='utf-8'))
    assert_no_instructions(spec)
    manifest = json.loads((ROOT / spec['manifest']).read_text(encoding='utf-8'))
    entries = {r['id']: r for r in manifest['images']}
    args.out.mkdir(parents=True, exist_ok=True)

    variants = [v for v in spec['variants']
                if not args.variants or v['name'] in args.variants]
    if not variants:
        raise SystemExit('沒有要跑的臂')

    ip2p = IP2PWrapper(dtype=torch.bfloat16)
    device = ip2p.device
    suite = MetricSuite(device=device)
    patch = spec['patch']

    rows = []
    for image in spec['images']:
        entry = entries[image]
        src = ROOT / entry['path']
        if sha256(src) != entry['sha256']:
            raise ValueError(f'{image} 的輸入雜湊不符')
        x = load_image(src, device)
        boxes = face_boxes(x, device)
        if not boxes:
            raise ValueError(f'{image} 偵測不到臉，id 項錨不住')
        box = max(boxes, key=lambda q: (q[2] - q[0]) * (q[3] - q[1]))
        e_orig = embed(x, device=device)

        supports = {}

        def support_for(opt):
            """支撐依臂而定，同一種設定只建一次。

            `scatter` 走既有的散斑，`disc`／`square` 是整塊幾何形，
            `lattice` 是規則圓點。幾何形與點陣都放進安全區**內部**，
            所以邊界是完整的圓或方，不是與衣物輪廓相交後的殘形。
            """
            kind = opt.get('support', 'scatter')
            # **快取鍵要涵蓋每一個會改變支撐的欄位。** 漏掉任何一個都會讓
            # 後面的臂靜默重用前一個臂的支撐，兩臂讀數逐欄相同而報表上看不出
            # 來——環帶的內外半徑就漏過一次。
            key = (kind, opt.get('area'), opt.get('blobs'),
                   opt.get('support_count'), opt.get('support_pitch'),
                   opt.get('support_radius'), opt.get('ring_inner'),
                   opt.get('ring_outer'), opt.get('inset'),
                   opt.get('erode'), opt.get('feather'),
                   opt.get('face_margin'), opt.get('kind'),
                   opt.get('support_seed'))
            if key in supports:
                return supports[key]
            if kind == 'scatter':
                sup = patch_support(
                    x, kind=opt.get('kind', 'upper'), area=float(opt['area']),
                    blobs=int(opt['blobs']),
                    inset=float(opt.get('inset', 0.12)),
                    erode=int(opt.get('erode', 10)),
                    feather=int(opt.get('feather', 6)),
                    seed=int(opt.get('support_seed', 0)), device=device)
            else:
                m = safe_mask(x, kind=opt.get('kind', 'upper'),
                              inset=float(opt.get('inset', 0.12)),
                              erode=int(opt.get('erode', 10)),
                              face_margin=int(opt.get('face_margin', 12)),
                              device=device)
                if kind == 'ring':
                    # 緊貼主體外緣的環帶 ∩ 衣物。點陣賭的是「碰到所有 token」，
                    # 環帶賭的是「碰到對的 token」——`add_hat` 的帽子與
                    # `add_scarf` 的圍巾都長在主體邊界上，而整件衣物的版本
                    # 把同樣的預算攤到胸腹，那裡沒有東西長出來。
                    from src.defense.carrier_mask import (face_subject_mask,
                                                          feather_inward,
                                                          ring_support)
                    ring = ring_support(face_subject_mask(x),
                                        int(opt.get('ring_inner', 8)),
                                        int(opt.get('ring_outer', 120)))
                    sup = feather_inward((ring * m).clamp(0.0, 1.0),
                                         int(opt.get('feather', 4)))
                elif kind == 'full':
                    # 整件衣物都印上圖案。點陣是「圓點印花」，這一支是
                    # 「格紋襯衫」——面積最大的自然載體，而且邊界就是衣物
                    # 輪廓，沒有任何人造的硬邊。
                    from src.defense.carrier_mask import feather_inward
                    sup = feather_inward(m, int(opt.get('feather', 6)))
                elif kind == 'lattice':
                    pitch = int(opt['support_pitch'])
                    radius = float(opt.get('support_radius',
                                           round(0.45 * pitch, 1)))
                    sup = lattice_support(m, pitch, radius)
                elif kind in ('disc', 'square'):
                    sup = place_shapes(
                        m, shape=kind, area=float(opt['area']),
                        count=int(opt.get('support_count', 1)),
                        feather=int(opt.get('feather', 6)))
                else:
                    raise ValueError(
                        f'未知的支撐種類 {kind!r}；要是 scatter／disc／square'
                        '／lattice／full 之一')
            supports[key] = sup
            save_png(sup.expand_as(x), args.out / f'{image}__support_{kind}'
                     f'{opt.get("support_pitch", "")}.png')
            return sup

        for variant in variants:
            t0 = time.time()
            outside_term = None
            opt = dict(patch, **(variant.get('patch') or {}))
            sup = support_for(opt)
            if opt.get('carrier', 'material') == 'target':
                carrier = TargetImagePatch(
                    sup, load_target(ROOT / opt['target'], device),
                    mode=opt.get('target_mode', 'bounded'),
                    epsilon=float(opt.get('target_epsilon', 0.12)),
                    keep_shading=bool(opt.get('keep_shading', True)),
                    shade_radius=int(opt.get('shade_radius', 12)),
                    init_jitter=float(opt.get('init_jitter', 0.0)))
            elif opt.get('carrier', 'material') == 'print':
                carrier = GeometricPrintParam(
                    sup, colours=int(opt.get('print_colours', 3)),
                    tau=float(opt.get('print_tau', 0.15)),
                    chroma_max=float(opt.get('print_chroma_max', 28.0)),
                    archetype=opt.get('print_archetype', 'dots'),
                    keep_shading=bool(opt.get('keep_shading', True)),
                    shade_radius=int(opt.get('shade_radius', 12)),
                    lightness_span=float(opt.get('print_lightness_span', 34.0)),
                    palette_init=opt.get('palette_init', 'garment'))
            elif opt.get('carrier', 'material') == 'canvas':
                pattern = None
                if opt.get('pattern'):
                    pattern = load_pattern(ROOT / opt['pattern'],
                                           int(opt['canvas_size']), device)
                eps = opt.get('epsilon', 0.25)
                carrier = PatchCanvasParam(
                    sup, canvas=pattern,
                    canvas_size=int(opt['canvas_size']),
                    epsilon=None if eps in (None, -1, -1.0) else float(eps),
                    shade_radius=int(opt.get('shade_radius', 12)),
                    keep_shading=bool(opt.get('keep_shading', True)),
                    init_jitter=float(opt.get('init_jitter', 0.0)))
            else:
                pal = garment_palette(
                    x, sup, int(opt['palette']),
                    chroma_gain=float(opt.get('chroma_gain', 1.0)),
                    mode=opt.get('palette_mode', 'garment'),
                    chroma=float(opt.get('chroma', 40.0)))
                carrier = MaterialPatchParam(
                    sup, pal, grid=int(opt['grid']),
                    shade_radius=int(opt.get('shade_radius', 12)),
                    logit_cap=float(opt['logit_cap']),
                    init_jitter=float(opt['init_jitter']))
            seed = int(variant.get('seed', 0))
            carrier.reset(x, seed)
            carrier.project()

            if variant.get('solver') == 'random_matched':
                target = float(variant['match_deltaE'])
                lo, hi = 0.0, float(variant.get('match_max_amplitude', 1.0))
                with torch.no_grad():
                    carrier.set_amplitude(hi)
                    top = delta_e00(x, carrier.render(x).clamp(0, 1),
                                    support=sup)
                if top < target:
                    raise SystemExit(
                        f'{variant["name"]}：隨機色盤在幅度上限 {hi} 下只到 '
                        f'ΔE00 {top:.2f}，對不上要求的 {target:.2f}。'
                        '**不放寬目標**：換一顆 seed 或抬高 match_max_amplitude。')
                for _ in range(int(variant.get('match_iters', 24))):
                    mid = 0.5 * (lo + hi)
                    with torch.no_grad():
                        carrier.set_amplitude(mid)
                        got = delta_e00(x, carrier.render(x).clamp(0, 1),
                                        support=sup)
                    if got < target:
                        lo = mid
                    else:
                        hi = mid
                carrier.set_amplitude(hi)
                stats = {'free_score_start': '', 'free_score_end': '',
                         'free_score_unprojected': '',
                         'free_feasible_step': -1, 'free_restart_pick': 0,
                         'free_restart_scores': '', 'free_curve': '',
                         'free_term_enc': '', 'free_term_cond': '',
                         'free_term_id': '', 'free_term_outside': ''}
                with torch.no_grad():
                    final = delta_e00(x, carrier.render(x).clamp(0, 1),
                                      support=sup)
                print(f'  {image} · {variant["name"]}  失真對齊的隨機對照：'
                      f'目標 ΔE00 {target:.2f}、拿到 {final:.2f}、'
                      f'幅度 {hi:.4f}', flush=True)
            elif variant.get('solver') == 'random':
                stats = {'free_score_start': '', 'free_score_end': '',
                         'free_score_unprojected': '',
                         'free_feasible_step': -1, 'free_restart_pick': 0,
                         'free_restart_scores': '', 'free_curve': '',
                         'free_term_enc': '', 'free_term_cond': '',
                         'free_term_id': ''}
                print(f'  {image} · {variant["name"]}  隨機對照（同支撐、'
                      f'同色盤、同起點振幅，不最佳化）', flush=True)
            else:
                free = dict(spec['free'], **(variant.get('free') or {}))
                base_kw = dict(
                    box=box, k=int(free['timesteps']),
                    steps=int(spec.get('attack_steps', 50)),
                    seed=int(free['noise_seed']),
                    weights=dict(free.get('weights') or {}),
                    chain_steps=int(free.get('chain_steps', 6)),
                    grad_steps=int(free.get('grad_steps', 1)),
                    resample=bool(free.get('resample', False)),
                    face_weight=float(free.get('face_weight', 0.0)),
                    s_i=float(spec.get('s_i', 1.5)))
                sw = float(free.get('support_weight', 0.0))
                if sw > 0:
                    objective = SupportWeightedFree(
                        ip2p, x, support=sup, support_weight=sw, **base_kw)
                else:
                    objective = FreeObjective(ip2p, x, **base_kw)

                w_outside = float(free.get('weights', {}).get('outside', 0.0))
                outside_term = None
                if w_outside > 0:
                    outside_term = OutsideDisplacement(
                        objective, x, sup,
                        margin=int(free.get('outside_margin', 16)),
                        near_width=int(free.get('outside_near_width', 48)),
                        weight_far=float(free.get('outside_weight_far', 1.0)),
                        weight_face=float(free.get('outside_weight_face', 1.0)),
                        weight_near=float(free.get('outside_weight_near', 0.0)))
                    objective = PrintObjective(
                        objective, outside_term=outside_term,
                        weights=dict(free.get('weights') or {}))

                w_out = float(free.get('weights', {}).get('out', 0.0))
                w_main = max(float(free.get('weights', {}).get('enc_target', 0.0)),
                             float(free.get('weights', {}).get('diffusion', 0.0)))
                if w_outside > 0 and (w_out > 0 or w_main > 0):
                    raise SystemExit(
                        'outside 與 out／mainstream 不能同時開：兩者是不同的'
                        '外掛層，疊起來會有兩份 null_edit 的快取各自抽樣')
                if w_out > 0 or w_main > 0:
                    from src.defense.readout_terms import (CompositeObjective,
                                                           OutputDisplacement)
                    out_term = (OutputDisplacement(objective, x,
                                                   suite.lpips_module)
                                if w_out > 0 else None)
                    mainstream = None
                    if w_main > 0:
                        from src.defense.mainstream_terms import MainstreamTerms
                        mainstream = MainstreamTerms(objective, x)
                    objective = CompositeObjective(
                        objective, carrier=carrier, out_term=out_term,
                        mainstream=mainstream,
                        weights=dict(free.get('weights') or {}))
                stats = optimise_carrier(
                    carrier, x, objective,
                    steps=int(variant.get('steps', 100)),
                    lr=float(variant.get('lr', 0.05)), caps=[],
                    check_every=int(variant.get('check_every', 10)),
                    restarts=int(variant.get('restarts', 1)),
                    restart_seed=seed,
                    lr_final_ratio=float(variant.get('lr_final_ratio', 1.0)),
                    probe_every=int(variant.get('probe_every', 0)))

            with torch.no_grad():
                y = (carrier.render(x).detach().clamp(0, 1) * 255).round() / 255
            save_png(y, args.out / f'{image}__{variant["name"]}__defended.png')
            pw = suite.pairwise(x, y)
            e_def = embed(y, device=device)
            row = {'image': image, 'variant': variant['name'],
                   'solver': variant.get('solver', 'adam'),
                   'objective': variant.get('objective', 'free'),
                   'steps': variant.get('steps', 100),
                   'lr': variant.get('lr', 0.05),
                   'restarts': variant.get('restarts', 1),
                   'area': opt['area'], 'blobs': opt['blobs'],
                   'seed': seed,
                   'deltaE00': round(pw['deltaE00'], 4),
                   'deltaE00_patch': round(delta_e00(x, y, support=sup), 4),
                   'psnr': round(pw['psnr'], 4),
                   'ssim': round(pw['ssim'], 5),
                   'lpips': round(pw['lpips'], 5),
                   'linf': round(pw['linf'], 5),
                   'subject_id_defended': (
                       '' if e_def is None
                       else round(similarity(e_orig, e_def), 5)),
                   'seconds': round(time.time() - t0, 1), 'note': ''}
            row.update({k: opt[k] for k in
                        ('grid', 'palette', 'palette_mode', 'chroma_gain',
                         'chroma', 'logit_cap', 'init_jitter')
                        if k in opt})
            if spec.get('report_outside', True):
                # **每一個臂都用同一個報表用目標算支撐外的分帶**，包含對照臂。
                # 對照臂沒有讀數的話，最佳化臂那三個數字沒有基線；而最佳化臂
                # 若沿用訓練時的目標，它的取樣狀態與對照臂不同，跨臂也不可比。
                # 這裡固定 seed、關掉重抽，權重全零、不進任何梯度。
                free_cfg = dict(spec['free'], **(variant.get('free') or {}))
                probe = FreeObjective(
                    ip2p, x, box=box, k=int(free_cfg['timesteps']),
                    steps=int(spec.get('attack_steps', 50)),
                    seed=int(free_cfg['noise_seed']), weights={},
                    chain_steps=int(free_cfg.get('chain_steps', 6)),
                    grad_steps=1, resample=False,
                    s_i=float(spec.get('s_i', 1.5)))
                reporter = OutsideDisplacement(
                    probe, x, sup,
                    margin=int(free_cfg.get('outside_margin', 16)),
                    near_width=int(free_cfg.get('outside_near_width', 48)))
                row.update(reporter.report(y))
            row['support_kind'] = opt.get('support', 'scatter')
            row['support_shape'] = opt.get('support', 'scatter')
            row['support_count'] = opt.get('support_count', '')
            row['support_pitch'] = opt.get('support_pitch', '')
            row.update(carrier.readout())
            row.update({k: v for k, v in stats.items() if k in COLUMNS})
            rows.append(row)
            print(json.dumps(
                {k: row[k] for k in
                 ('image', 'variant', 'deltaE00_patch', 'psnr', 'lpips',
                  'subject_id_defended', 'free_score_start', 'free_score_end',
                  'seconds')}, ensure_ascii=False), flush=True)

    with (args.out / 'patch_readout.csv').open('w', newline='',
                                               encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)
    print(f'wrote {len(rows)} rows to {args.out / "patch_readout.csv"}',
          flush=True)


if __name__ == '__main__':
    main()

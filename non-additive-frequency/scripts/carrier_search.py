"""在使用者判準上直接搜尋載體：空間變化的顏色場 × 兩段串接。

自變數是**載體的結構**——控制點密度、帶寬、場的振幅、亮度鎖不鎖、兩段各自的
幅度——由 (1+lambda) 演化策略在真判準上搜尋，每個候選點跑一次真的編輯。

搜尋的起點是 `carrier_search.START`，也就是 `runs/color_scaleup_search/` 那批
的全域仿射；同一批裡另外跑 `start` 臂當原地對照，所以新舊在同一張表上。

三個與前一批不同的地方
────────────────────────────────────────────────────────────────────
1. **搜尋種子與回報種子分開。** 前一批兩者同為 17001，`search` 臂的數字是
   in-sample。這裡搜尋只看 `search_seeds`，回報只看 `report_seeds`，兩組不相交
   （程式會檢查）。
2. **目標函數是連言。** `criterion.py` 的三項各自對同一格的未防禦攻擊正規化，
   取軟極小：可用、認得出是同一個人、指令完成，任一條垮掉就算防禦成功。
3. **高頻不設限。** `sigma` 可以是 0，`radius` 與 `max_gain` 預設不設界。
   `hf_ratio_rgb_total` 仍逐列量、逐列報，但不擋任何候選點。

每個候選點的代理讀數（VAE latent 的位移）一併寫進 `search_trace.csv`，用來算
代理與判準的秩相關——代理有沒有用是量出來的，不是引用來的。
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

ARMS = ('searched', 'start', 'undefended')

COLUMNS = [
    'arm', 'image', 'class', 'instruction', 'carrier', 'region', 'seed',
    'eval_seed', 'search_seeds', 'report_seeds',
    'criterion', 'id_norm', 'dir_norm', 'use_norm',
    'subject_lost', 'control_subject_lost', 'control_direction_flat',
    'subject_id_edit_orig', 'subject_id_edit_def', 'subject_id_drop',
    'subject_id_edit_orig_detected', 'subject_id_edit_def_detected',
    'subject_box_iou_edit_orig', 'subject_box_iou_edit_def', 'subject_anchor',
    'align_orig', 'align_def', 'align_input_orig', 'align_input_def',
    'align_gain_orig', 'align_gain_def',
    'align_siglip_orig', 'align_siglip_def',
    'clip_s_orig', 'clip_s_def', 'clip_s_drop',
    'siglip_s_orig', 'siglip_s_def', 'siglip_s_drop',
    'niqe_orig', 'niqe_def', 'niqe_input', 'niqe_original',
    'edit_lpips', 'input_psnr', 'input_dists', 'input_linf', 'input_lpips',
    'support_deltaE00', 'face_deltaE00', 'clothes_deltaE00',
    'hf_rgb_total', 'hf_lab_L', 'hf_lab_a', 'hf_lab_b',
    'latent_l2', 'latent_norm_def', 'latent_norm_orig',
    'protected_max_abs', 'protected_pixels',
    'knob_face_grid', 'knob_face_sigma', 'knob_face_scale',
    'knob_face_amplitude',
    'knob_frame_grid', 'knob_frame_sigma', 'knob_frame_scale',
    'knob_frame_amplitude', 'knob_clothes_grid', 'knob_clothes_sigma',
    'knob_clothes_scale', 'knob_clothes_amplitude',
    'knob_face_palette', 'knob_clothes_palette', 'knob_lock_luminance',
    'knob_field_seed',
    'delta_e_cap', 'face_delta_e_cap',
    'inner_steps', 'inner_lr', 'inner_latent_start', 'inner_latent_end',
    'inner_amplitude_shrink',
    'search_budget', 'search_evaluations', 'search_score', 'search_children',
    'attack_steps', 'attack_batch', 'attack_precision', 'seconds',
]

TRACE_COLUMNS = [
    'image', 'class', 'generation', 'evaluation', 'score',
    'id_norm', 'dir_norm', 'use_norm', 'subject_lost',
    'subject_id_def', 'subject_box_iou_edit_def', 'align_def', 'align_gain_def',
    'clip_s_def',
    'niqe_def',
    'latent_l2', 'hf_rgb_total', 'niqe_input', 'niqe_original', 'rejected',
    'inner_steps', 'inner_lr', 'inner_latent_start', 'inner_latent_end',
    'inner_amplitude_shrink',
    'support_deltaE00', 'delta_e_cap', 'face_deltaE00', 'face_delta_e_cap',
    'knob_face_grid', 'knob_face_sigma', 'knob_face_scale',
    'knob_face_amplitude',
    'knob_frame_grid', 'knob_frame_sigma', 'knob_frame_scale',
    'knob_frame_amplitude', 'knob_clothes_grid', 'knob_clothes_sigma',
    'knob_clothes_scale', 'knob_clothes_amplitude',
    'knob_face_palette', 'knob_clothes_palette', 'knob_lock_luminance',
    'knob_field_seed',
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
            f'{"是別人的" if seen.returncode else "不是本行程"}；換一張卡。')


def load_image(path, device):
    import numpy as np
    import torch
    from PIL import Image
    arr = np.asarray(Image.open(path).convert('RGB'), dtype=np.float32) / 255.
    return torch.from_numpy(arr).permute(2, 0, 1)[None].to(device)


def save_png(x, path):
    import numpy as np
    from PIL import Image
    a = (x.detach().float().clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy() * 255)
    Image.fromarray(a.round().astype(np.uint8)).save(path)


def palette_of(spec, palette_id):
    rec = next(r for r in _library(spec) if r['id'] == palette_id)
    return rec['mean'], rec['covariance']


_LIB_CACHE = {}


def _library(spec):
    """整個雜湊驗過的色彩庫，不按類別過濾。

    `NCFLibrary` 的 `ade20k_classes` 是為「一個區域只配它自己的語意類別」而設，
    但這裡要讓搜尋在**多個類別**之間挑（臉配 `person`、衣物配 `apparel`），所以
    過濾改在取用端做，雜湊與 Lab 單位的檢查仍走同一條路徑。
    """
    key = spec['library']['sha256']
    if key not in _LIB_CACHE:
        from src.defense.ncf_library import NCFLibrary
        lib = spec['library']
        obj = NCFLibrary(ROOT / lib['path'], expected_sha256=lib['sha256'],
                         required_classes=lib['class_weights'],
                         ade20k_classes=None)
        _LIB_CACHE[key] = obj.records
    return _LIB_CACHE[key]


def palette_bank(spec, class_prefix):
    """某個 ADE20K 類別底下的所有真實色彩分布，順序固定。

    臉那一段要配的是 `ade013_person`：身分住在膚色上，而把膚色移到**另一個真實
    的膚色分布**既是往身分下手，也天然自然——那是別人身上量到的顏色，不是憑空
    造的。衣物配 `ade093_apparel`，與先前一致。
    """
    bank = [r for r in _library(spec) if r['id'].startswith(class_prefix)]
    if not bank:
        raise SystemExit(f'色彩庫裡沒有 {class_prefix} 開頭的分布')
    return [(r['mean'], r['covariance']) for r in bank]


def cells_of(spec):
    out = []
    for image in spec['images']:
        for cls in spec['classes']:
            out.append({'image': image, 'class': cls['name'],
                        'instruction': cls['instructions'][image],
                        'carrier': cls.get('carrier', 'composite'),
                        'region': cls.get('region', 'clothes'),
                        'seed': spec['seed']})
    return out


def control_of(suite, x, edit_orig, instruction):
    """未防禦那一張的讀數。逐格逐種子只算一次。

    先前版本在**每個候選點**上重算它，而它與候選點無關；200 次評估就是 200 次
    白算的 CLIP、SigLIP 與 NIQE。
    """
    d = suite.direction_similarity(x, edit_orig, instruction)
    s = suite.semantic(edit_orig, instruction)
    s0 = suite.semantic(x, instruction)
    return {'clip': float(d['clip']), 'siglip': float(d['siglip']),
            'align': float(s['clip']), 'align_siglip': float(s['siglip']),
            'align_input': float(s0['clip']),
            'gain': float(s['clip']) - float(s0['clip']),
            'niqe': suite.niqe(edit_orig)}


def readouts(suite, x, x_def, edit_orig, edit_def, instruction, device, control):
    """一張防禦後的編輯圖對上未防禦的那一張，回傳三項與原始讀數。

    **方向分數以各自的輸入為錨。** 防禦側量的是 `x_def → edit_def`，對照側量
    `x → edit_orig`。先前兩側都以原圖為錨，於是防禦本身的顏色位移被算進了
    「編輯往指令走了多少」那個差向量裡，色差夠大時分數就被沖成負的，看起來像
    指令沒完成——實際上圖上指令完成得好好的。第一批 18 列裡有 8 列是這樣拿到
    0 分的。
    """
    from src.defense.criterion import criterion_score, normalise_terms
    from src.metrics.identity import anchored_identity
    ident = anchored_identity(x, edit_orig, edit_def, device=device)
    d_def = suite.direction_similarity(x_def, edit_def, instruction)
    d_orig = {'clip': control['clip'], 'siglip': control['siglip']}
    a_def = suite.semantic(edit_def, instruction)
    a_input = suite.semantic(x_def, instruction)
    gain_def = float(a_def['clip']) - float(a_input['clip'])
    niqe_def, niqe_orig = suite.niqe(edit_def), control['niqe']
    terms = normalise_terms(
        id_def=ident['subject_id_edit_def'],
        id_control=ident['subject_id_edit_orig'],
        clip_def=gain_def, clip_control=control['gain'],
        quality_def=niqe_def, quality_control=niqe_orig,
        subject_iou=1.0)
    row = dict(ident)
    row.update({k: round(v, 6) if isinstance(v, float) else v
                for k, v in terms.items()})
    row.update({
        'align_def': round(float(a_def['clip']), 5),
        'align_orig': round(control['align'], 5),
        'align_input_def': round(float(a_input['clip']), 5),
        'align_input_orig': round(control['align_input'], 5),
        'align_gain_def': round(gain_def, 5),
        'align_gain_orig': round(control['gain'], 5),
        'align_siglip_def': round(float(a_def['siglip']), 5),
        'align_siglip_orig': round(control['align_siglip'], 5),
        'clip_s_def': round(float(d_def['clip']), 5),
        'clip_s_orig': round(float(d_orig['clip']), 5),
        'clip_s_drop': round(float(d_orig['clip']) - float(d_def['clip']), 5),
        'siglip_s_def': round(float(d_def['siglip']), 5),
        'siglip_s_orig': round(float(d_orig['siglip']), 5),
        'siglip_s_drop': round(float(d_orig['siglip']) - float(d_def['siglip']), 5),
        'niqe_def': round(niqe_def, 5), 'niqe_orig': round(niqe_orig, 5),
        'edit_lpips': round(float(suite.pairwise(edit_orig, edit_def)['lpips']), 5),
    })
    row['criterion'] = round(criterion_score(terms), 6)
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--shard', default='1/1')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--batch', type=int, default=0)
    ap.add_argument('--budget', type=int, default=0)
    ap.add_argument('--precision', default='')
    ap.add_argument('--delta-e-cap', type=float, default=0.0,
                    help='覆寫設定裡的 delta_e_cap：防禦圖相對原圖的支撐加權 '
                         'ΔE00 上限，超過的候選點直接退回。')
    args = ap.parse_args()
    if args.device != 'cpu':
        assert_free_cards()

    import torch
    from src.defense.carrier_search import (START, build_carrier, describe,
                                            knob_list, pick, spearman)
    from src.defense.color_amplitude import delta_e00
    from src.defense.color_search import evolution_search_batched
    from src.defense.criterion import criterion_score
    from src.defense.inner_optimise import optimise_field
    from src.defense.lowfreq_color import highfreq_report
    from src.defense.ncf_library import sha256
    from src.defense.carrier_mask import face_subject_mask
    from src.defense.ncf_runner import ncf_support
    from src.metrics.suite import MetricSuite

    spec = json.loads(args.config.read_text(encoding='utf-8'))
    manifest = json.loads((ROOT / spec['manifest']).read_text(encoding='utf-8'))
    search_seeds = list(spec['search_seeds'])
    report_seeds = list(spec['report_seeds'])
    if set(search_seeds) & set(report_seeds):
        raise SystemExit('search_seeds 與 report_seeds 不可相交，否則讀數是 in-sample')
    if args.out.exists() and any(args.out.iterdir()):
        raise SystemExit(f'{args.out} 已存在且非空；換一個輸出目錄')
    args.out.mkdir(parents=True, exist_ok=True)

    precision = args.precision or spec['precision']
    steps = spec['attack_steps']
    budget = args.budget or spec['search_budget']
    children = spec.get('search_children', 6)
    batch = args.batch or spec.get('attack_batch_default', 9)

    from src.models.ip2p import IP2PWrapper
    ip2p = IP2PWrapper(dtype={'fp16': torch.float16, 'fp32': torch.float32,
                              'bf16': torch.bfloat16}[precision])
    device = ip2p.device
    suite = MetricSuite(device=device)
    entries = {r['id']: r for r in manifest['images']}

    i, n = (int(v) for v in args.shard.split('/'))
    if not 1 <= i <= n:
        raise SystemExit(f'--shard 要寫成 i/n，收到 {args.shard!r}')
    todo = [c for k, c in enumerate(cells_of(spec)) if k % n == i - 1]
    if args.limit:
        todo = todo[:args.limit]
    suffix = '' if n == 1 else f'_shard{i}of{n}'
    print(f'分片 {i}/{n}：{len(todo)} 格，預算 {budget}，子代 {children}，'
          f'批次 {batch}，精度 {precision}，搜尋種子 {search_seeds}，'
          f'回報種子 {report_seeds}', flush=True)

    rows, trace, t_start = [], [], time.time()

    def edit_many(images, instruction, seeds):
        outs = []
        for k in range(0, len(images), batch):
            chunk_i = images[k:k + batch]
            chunk_s = seeds[k:k + batch]
            batched = ip2p.edit_batch(chunk_i, [instruction] * len(chunk_i),
                                      chunk_s, steps=steps, s_t=spec['s_t'],
                                      s_i=spec['s_i'])
            outs.extend(batched[k2:k2 + 1] for k2 in range(batched.shape[0]))
        return outs

    for cell in todo:
        entry = entries[cell['image']]
        src = ROOT / entry['path']
        if sha256(src) != entry['sha256']:
            raise ValueError(f"{cell['image']} 的輸入雜湊不符")
        x = load_image(src, device)
        tag = f"{cell['image']}__{cell['class']}"
        clothes = ncf_support(x, 'clothes')
        frame = ncf_support(x, 'frame')
        face = (face_subject_mask(x, device=device)
                if spec.get('face_stage', True) else None)
        frame_palette = palette_of(spec, spec['palette_id'])
        clothes_bank = palette_bank(spec, spec['clothes_palette_class'])
        face_bank = palette_bank(spec, spec['face_palette_class'])
        with torch.no_grad():
            z_orig = ip2p.encode_image(x).float()
        niqe_x = suite.niqe(x)
        natural_cap = float(spec.get('naturalness_max_ratio', 1.4)) * niqe_x
        de_cap = args.delta_e_cap or float(spec.get('delta_e_cap', 0) or 0)
        face_cap = float(spec.get('face_delta_e_cap', 0) or 0) or de_cap

        all_seeds = search_seeds + report_seeds
        clean = dict(zip(all_seeds,
                         edit_many([x] * len(all_seeds), cell['instruction'],
                                   all_seeds)))
        control = {s: control_of(suite, x, clean[s], cell['instruction'])
                   for s in all_seeds}
        for s in report_seeds:
            save_png(clean[s], args.out / f'{tag}__s{s}__clean_edit.png')

        inner = spec.get('inner', {})

        def caps_for():
            out = []
            if de_cap > 0:
                out.append((lambda y: delta_e00(x, y, frame), de_cap))
            if face is not None and face_cap > 0:
                out.append((lambda y: delta_e00(x, y, face), face_cap))
            return out

        def build(values):
            return build_carrier(values, x, frame_support=frame,
                                 clothes_support=clothes,
                                 face_support=face,
                                 frame_palette=frame_palette,
                                 clothes_palette=pick(clothes_bank,
                                                      values['clothes_palette']),
                                 face_palette=pick(face_bank,
                                                   values['face_palette']),
                                 radius=spec.get('radius'),
                                 max_gain=spec.get('max_gain'))

        state = {'evaluation': 0, 'generation': 0}

        def evaluate_many(candidates):
            state['generation'] += 1
            defended, inner_rows = [], []
            for v in candidates:
                carrier = build(v)
                if inner.get('steps'):
                    inner_rows.append(optimise_field(
                        carrier, x, ip2p, steps=int(inner['steps']),
                        lr=float(inner.get('lr', 0.05)), caps=caps_for()))
                else:
                    inner_rows.append({})
                defended.append(carrier.render(x).detach())
            images, seeds = [], []
            for y in defended:
                for s in search_seeds:
                    images.append(y)
                    seeds.append(s)
            edited = edit_many(images, cell['instruction'], seeds)
            scores = []
            for j, (v, y) in enumerate(zip(candidates, defended)):
                niqe_y = suite.niqe(y)
                de_y = delta_e00(x, y, frame)
                de_face = (delta_e00(x, y, face) if face is not None
                           else de_y)
                per_seed, terms_sum = [], None
                for m, s in enumerate(search_seeds):
                    ed = edited[j * len(search_seeds) + m]
                    read = readouts(suite, x, y, clean[s], ed,
                                    cell['instruction'], device, control[s])
                    per_seed.append(read)
                terms = {k: sum(float(r[k]) for r in per_seed) / len(per_seed)
                         for k in ('id_norm', 'dir_norm', 'use_norm')}
                rejected = int(niqe_y > natural_cap
                               or (de_cap > 0 and de_y > de_cap)
                               or (face_cap > 0 and de_face > face_cap))
                score = 1e3 if rejected else criterion_score(terms)
                scores.append(score)
                state['evaluation'] += 1
                with torch.no_grad():
                    z = ip2p.encode_image(y).float()
                hf = highfreq_report(x, y)
                trace.append({
                    'image': cell['image'], 'class': cell['class'],
                    'generation': state['generation'],
                    'evaluation': state['evaluation'],
                    'score': round(score, 6),
                    **{k: round(v, 6) for k, v in terms.items()},
                    'subject_lost': max(int(r['subject_lost']) for r in per_seed),
                    'subject_box_iou_edit_def': per_seed[0]['subject_box_iou_edit_def'],
                    'subject_id_def': per_seed[0]['subject_id_edit_def'],
                    'align_def': per_seed[0]['align_def'],
                    'align_gain_def': per_seed[0]['align_gain_def'],
                    'clip_s_def': per_seed[0]['clip_s_def'],
                    'niqe_def': per_seed[0]['niqe_def'],
                    'latent_l2': round(float((z - z_orig).norm()), 4),
                    'hf_rgb_total': round(hf['hf_ratio_rgb_total'], 5),
                    **inner_rows[j],
                    'niqe_input': round(niqe_y, 5),
                    'niqe_original': round(niqe_x, 5), 'rejected': rejected,
                    'support_deltaE00': round(de_y, 4), 'delta_e_cap': de_cap,
                    'face_deltaE00': round(de_face, 4),
                    'face_delta_e_cap': face_cap,
                    **describe(v)})
            return scores

        res = evolution_search_batched(knob_list(), evaluate_many,
                                       budget=budget, children=children,
                                       sigma0=spec.get('search_sigma0', 0.35),
                                       seed=cell['seed'], start=dict(START))
        print(f"  {tag} 搜尋 {res.evaluations} 次 最佳 {res.best_score:.4f} "
              f"{describe(res.best)}", flush=True)

        for arm, values in (('searched', res.best), ('start', dict(START))):
            carrier = build(values)
            inner_row = {}
            if inner.get('steps') and arm == 'searched':
                inner_row = optimise_field(
                    carrier, x, ip2p, steps=int(inner['steps']),
                    lr=float(inner.get('lr', 0.05)), caps=caps_for())
            x_def = carrier.render(x).detach()
            save_png(x_def, args.out / f'{tag}__{arm}__def.png')
            edited = edit_many([x_def] * len(report_seeds), cell['instruction'],
                               report_seeds)
            appearance = suite.pairwise(x, x_def)
            hf = highfreq_report(x, x_def)
            with torch.no_grad():
                z_def = ip2p.encode_image(x_def).float()
            protected = (x - x_def).abs() * (1.0 - clothes) * (1.0 - frame)
            for s, ed in zip(report_seeds, edited):
                save_png(ed, args.out / f'{tag}__{arm}__s{s}__def_edit.png')
                read = readouts(suite, x, x_def, clean[s], ed,
                                cell['instruction'], device, control[s])
                rows.append({
                    'arm': arm, 'image': cell['image'], 'class': cell['class'],
                    'instruction': cell['instruction'], 'carrier': cell['carrier'],
                    'region': cell['region'], 'seed': cell['seed'],
                    'eval_seed': s,
                    'search_seeds': '|'.join(str(v) for v in search_seeds),
                    'report_seeds': '|'.join(str(v) for v in report_seeds),
                    'input_psnr': round(appearance['psnr'], 5),
                    'input_dists': round(appearance['dists'], 5),
                    'input_linf': round(appearance['linf'], 5),
                    'input_lpips': round(appearance['lpips'], 5),
                    'support_deltaE00': round(delta_e00(x, x_def, frame), 5),
                    'face_deltaE00': ('' if face is None
                                      else round(delta_e00(x, x_def, face), 5)),
                    'clothes_deltaE00': round(delta_e00(x, x_def, clothes), 5),
                    'hf_rgb_total': round(hf['hf_ratio_rgb_total'], 5),
                    'hf_lab_L': round(hf['hf_ratio_lab_L'], 5),
                    'hf_lab_a': round(hf['hf_ratio_lab_a'], 5),
                    'hf_lab_b': round(hf['hf_ratio_lab_b'], 5),
                    'niqe_input': round(suite.niqe(x_def), 5),
                    'niqe_original': round(niqe_x, 5),
                    'latent_l2': round(float((z_def - z_orig).norm()), 4),
                    'latent_norm_def': round(float(z_def.norm()), 4),
                    'latent_norm_orig': round(float(z_orig.norm()), 4),
                    'protected_max_abs': round(float(protected.max()), 8),
                    'protected_pixels': int(((1.0 - clothes) * (1.0 - frame) > 0).sum()),
                    'delta_e_cap': de_cap, 'face_delta_e_cap': face_cap,
                    'search_budget': budget, 'search_children': children,
                    'search_evaluations': res.evaluations,
                    'search_score': round(res.best_score, 6),
                    'attack_steps': steps, 'attack_batch': batch,
                    'attack_precision': precision,
                    'seconds': round(time.time() - t_start, 1),
                    **inner_row, **describe(values), **read})

        for s in report_seeds:
            read = readouts(suite, x, x, clean[s], clean[s],
                            cell['instruction'], device, control[s])
            rows.append({
                'arm': 'undefended', 'image': cell['image'],
                'class': cell['class'], 'instruction': cell['instruction'],
                'carrier': cell['carrier'], 'region': cell['region'],
                'seed': cell['seed'], 'eval_seed': s,
                'search_seeds': '|'.join(str(v) for v in search_seeds),
                'report_seeds': '|'.join(str(v) for v in report_seeds),
                'attack_steps': steps, 'attack_batch': batch,
                'attack_precision': precision,
                'seconds': round(time.time() - t_start, 1), **read})

    write_csv(args.out / f'carrier_search{suffix}.csv', rows, COLUMNS)
    write_csv(args.out / f'search_trace{suffix}.csv', trace, TRACE_COLUMNS)
    for name in ('latent_l2', 'hf_rgb_total'):
        r = spearman([t[name] for t in trace], [t['score'] for t in trace])
        print(f'代理 {name} 與判準的秩相關 {r:.4f}（{len(trace)} 個候選點）',
              flush=True)
    print(f'-> {args.out}  共 {len(rows)} 列，{time.time()-t_start:.0f} 秒',
          flush=True)


def write_csv(path, rows, columns):
    with open(path, 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=columns, extrasaction='ignore')
        w.writeheader()
        for r in rows:
            w.writerow(r)


if __name__ == '__main__':
    main()

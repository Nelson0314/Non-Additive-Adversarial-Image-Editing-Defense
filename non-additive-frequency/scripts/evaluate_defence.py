"""用真編輯回報防禦：**指令只在這裡出現，而且只當測試資料**。

讀 `scripts/immunise.py` 產生的防禦圖，對每張圖跑一組指令 × 一組種子的真編輯，
算出使用者判準的三項與軟極小。防禦圖在這一步是**唯讀**的：沒有任何參數會被
這裡的讀數調整，所以指令與這批種子都是 held-out。

臂與淨化
────────────────────────────────────────────────────────────────────
`undefended` 未防禦、`start` 現行的全域仿射、`immunised` 不含指令目標解出來的；
由設定的 `arms` 決定跑哪幾個。`purifiers` 是攻擊者在編輯前的前處理，
**兩側配對**：`edit(P(x))` 當該淨化分支自己的對照，不拿未淨化的 clean 編輯
當所有分支的分母。每臂每個淨化每個指令每顆種子一列。

`dir_norm` 照算照報但不進判準，理由見 `src/defense/criterion.py`。
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

ARMS = ('undefended', 'start', 'immunised')


def purify(x, spec):
    """攻擊者在編輯前的前處理。`identity` 不可省，它是保留率的分母。

    三個算子的設定取自 `docs/EVALUATION.md`：JPEG 品質 75、高斯 σ=1.0、
    每邊裁 10% 後以 bicubic 升回。**兩側用同一個算子、同一顆種子配對比較**，
    不可拿未淨化的 clean 編輯當所有淨化分支的對照。
    """
    kind = spec['kind']
    if kind == 'identity':
        return x
    from src.purify.ops import crop_resize, gaussian_blur, jpeg_real
    if kind == 'jpeg':
        return jpeg_real(x, int(spec.get('quality', 75)))
    if kind == 'blur':
        return gaussian_blur(x, float(spec.get('sigma', 1.0)))
    if kind == 'crop_resize':
        return crop_resize(x, float(spec.get('fraction', 0.10)))
    raise SystemExit(f'不認得的淨化算子 {kind!r}')


def map_box(box, spec, size):
    """把原圖座標的框換算到淨化之後的座標。

    `crop_resize` 會裁掉每邊 `fraction` 再縮回原尺寸，輸出的像素座標因此與原圖
    不一致。身分讀數若沿用原框，量到的一部分是**座標錯位**而不是身分變化，
    而且配對的分母消不掉它——兩側的錯位一樣，但錯位造成的身分下降不是線性的。
    `jpeg` 與 `blur` 不改幾何，框照舊。
    """
    if spec['kind'] != 'crop_resize':
        return box
    h, w = size
    f = float(spec.get('fraction', 0.10))
    dh, dw = int(round(h * f)), int(round(w * f))
    sx, sy = w / (w - 2 * dw), h / (h - 2 * dh)
    x0, y0, x1, y1 = (float(v) for v in box)
    return [(x0 - dw) * sx, (y0 - dh) * sy, (x1 - dw) * sx, (y1 - dh) * sy]


COLUMNS = [
    'arm', 'purifier', 'image', 'class', 'instruction', 'eval_seed', 'seeds',
    'criterion', 'id_norm', 'dir_norm', 'use_norm',
    'subject_lost', 'control_subject_lost', 'control_direction_flat',
    'subject_id_edit_orig', 'subject_id_edit_def', 'subject_id_drop',
    'subject_id_edit_orig_detected', 'subject_id_edit_def_detected',
    'subject_box_iou_edit_orig', 'subject_box_iou_edit_def', 'subject_anchor',
    'subject_out_box',
    'align_orig', 'align_def', 'align_input_def', 'align_gain_def',
    'clip_s_def', 'siglip_s_def',
    'niqe_orig', 'niqe_def', 'niqe_input', 'niqe_original',
    'input_linf', 'input_lpips', 'edit_lpips',
    'support_deltaE00', 'face_deltaE00', 'seconds',
]


def main():
    import torch

    from src.defense.assets import load_image, save_png
    from src.defense.carrier_mask import face_subject_mask
    from src.defense.color_amplitude import delta_e00
    from src.defense.criterion import criterion_score, normalise_terms
    from src.defense.ncf_library import sha256
    from src.defense.ncf_runner import ncf_support
    from src.metrics.identity import anchored_identity, face_boxes
    from src.metrics.suite import MetricSuite

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', type=Path, required=True)
    ap.add_argument('--run', type=Path, required=True,
                    help='scripts/immunise.py 的輸出目錄')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--shard', default='1/1')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--batch', type=int, default=0)
    args = ap.parse_args()

    spec = json.loads(args.config.read_text(encoding='utf-8'))
    manifest = json.loads((ROOT / spec['manifest']).read_text(encoding='utf-8'))
    if args.out.exists() and any(args.out.iterdir()):
        raise SystemExit(f'{args.out} 已存在且非空；換一個輸出目錄')
    args.out.mkdir(parents=True, exist_ok=True)

    seeds = list(spec['seeds'])
    arms = tuple(spec.get('arms', ARMS))
    purifiers = spec.get('purifiers', [{'kind': 'identity'}])
    if not any(p['kind'] == 'identity' for p in purifiers):
        raise SystemExit('淨化清單必須含 identity，它是保留率的分母')
    steps = int(spec['attack_steps'])
    batch = args.batch or int(spec.get('attack_batch_default', 9))

    from src.models.ip2p import IP2PWrapper
    ip2p = IP2PWrapper(dtype={'fp16': torch.float16, 'fp32': torch.float32,
                              'bf16': torch.bfloat16}[spec['precision']])
    device = ip2p.device
    suite = MetricSuite(device=device)
    entries = {r['id']: r for r in manifest['images']}

    cells = [{'image': im, 'class': cls['name'],
              'instruction': cls['instructions'][im]}
             for im in spec['images'] for cls in spec['classes']]
    i, n = (int(v) for v in args.shard.split('/'))
    if not 1 <= i <= n:
        raise SystemExit(f'--shard 要寫成 i/n，收到 {args.shard!r}')
    todo = [c for k, c in enumerate(cells) if k % n == i - 1]
    if args.limit:
        todo = todo[:args.limit]
    suffix = '' if n == 1 else f'_shard{i}of{n}'
    print(f'分片 {i}/{n}：{len(todo)} 格 × {len(seeds)} 種子 × {len(arms)} 臂'
          f' × {len(purifiers)} 淨化', flush=True)

    def edit_many(images, instruction, evals):
        outs = []
        for k in range(0, len(images), batch):
            chunk_i, chunk_s = images[k:k + batch], evals[k:k + batch]
            b = ip2p.edit_batch(chunk_i, [instruction] * len(chunk_i), chunk_s,
                                steps=steps, s_t=spec['s_t'], s_i=spec['s_i'])
            outs.extend(b[j:j + 1] for j in range(b.shape[0]))
        return outs

    rows = []
    for cell in todo:
        t0 = time.time()
        image = cell['image']
        entry = entries[image]
        src = ROOT / entry['path']
        if sha256(src) != entry['sha256']:
            raise ValueError(f'{image} 的輸入雜湊不符')
        x = load_image(src, device)
        frame = ncf_support(x, 'frame')
        face = face_subject_mask(x, device=device)
        defended = {'undefended': x}
        if 'start' in arms:
            defended['start'] = load_image(
                args.run / f'{image}__start.png', device)
        if 'immunised' in arms:
            defended['immunised'] = load_image(
                args.run / f'{image}__immunised.png', device)

        niqe_x = suite.niqe(x)
        boxes = face_boxes(x, device)
        if not boxes:
            raise ValueError(f'{image} 偵測不到臉，無法錨定身分讀數')
        anchor = max(boxes, key=lambda q: (q[2] - q[0]) * (q[3] - q[1]))
        for pspec in purifiers:
            pname = pspec['kind']
            x_p = purify(x, pspec)
            out_box = map_box(anchor, pspec, x.shape[-2:])
            clean = dict(zip(seeds, edit_many([x_p] * len(seeds),
                                              cell['instruction'], seeds)))
            control = {}
            for s in seeds:
                d = suite.direction_similarity(x_p, clean[s], cell['instruction'])
                sem = suite.semantic(clean[s], cell['instruction'])
                control[s] = {'clip': float(d['clip']), 'align': float(sem['clip']),
                              'niqe': suite.niqe(clean[s])}
                save_png(clean[s], args.out / f'{image}__{cell["class"]}__'
                                              f'{pname}__s{s}__clean_edit.png')

            for arm in arms:
                x_def = purify(defended[arm], pspec)
                edits = dict(zip(seeds, edit_many([x_def] * len(seeds),
                                                  cell['instruction'], seeds)))
                niqe_input = suite.niqe(x_def)
                pair = suite.pairwise(x, x_def)
                de_frame = float(delta_e00(x, x_def, frame))
                de_face = float(delta_e00(x, x_def, face))
                for s in seeds:
                    y = edits[s]
                    ident = anchored_identity(x, clean[s], y, device=device,
                                              out_box=out_box)
                    d_def = suite.direction_similarity(x_def, y, cell['instruction'])
                    sem_def = suite.semantic(y, cell['instruction'])
                    sem_in = suite.semantic(x_def, cell['instruction'])
                    niqe_def = suite.niqe(y)
                    terms = normalise_terms(
                        id_def=ident['subject_id_edit_def'],
                        id_control=ident['subject_id_edit_orig'],
                        clip_def=float(d_def['clip']),
                        clip_control=control[s]['clip'],
                        quality_def=niqe_def, quality_control=control[s]['niqe'],
                        subject_iou=1.0)
                    rows.append({
                        'arm': arm, 'purifier': pname, 'image': image,
                        'class': cell['class'],
                        'instruction': cell['instruction'], 'eval_seed': s,
                        'seeds': '|'.join(str(v) for v in seeds),
                        'criterion': round(criterion_score(terms), 5),
                        **{k: round(v, 5) if isinstance(v, float) else v
                           for k, v in terms.items()},
                        **ident,
                        'align_orig': round(control[s]['align'], 5),
                        'align_def': round(float(sem_def['clip']), 5),
                        'align_input_def': round(float(sem_in['clip']), 5),
                        'align_gain_def': round(float(sem_def['clip'])
                                                - float(sem_in['clip']), 5),
                        'clip_s_def': round(float(d_def['clip']), 5),
                        'siglip_s_def': round(float(d_def['siglip']), 5),
                        'niqe_orig': round(control[s]['niqe'], 5),
                        'niqe_def': round(niqe_def, 5),
                        'niqe_input': round(niqe_input, 5),
                        'niqe_original': round(niqe_x, 5),
                        'input_linf': pair.get('linf'),
                        'input_lpips': pair.get('lpips'),
                        'edit_lpips': suite.pairwise(clean[s], y).get('lpips'),
                        'support_deltaE00': round(de_frame, 5),
                        'face_deltaE00': round(de_face, 5),
                        'seconds': round(time.time() - t0, 1),
                    })
                    if arm != 'undefended':
                        save_png(y, args.out / f'{image}__{cell["class"]}__'
                                               f'{pname}__{arm}__s{s}'
                                               f'__def_edit.png')
        print(f'  {image} · {cell["class"]}  '
              f'{round(time.time() - t0, 1)}s', flush=True)

    target = args.out / f'evaluate{suffix}.csv'
    with open(target, 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)
    print(f'寫出 {target}（{len(rows)} 列）', flush=True)


if __name__ == '__main__':
    main()

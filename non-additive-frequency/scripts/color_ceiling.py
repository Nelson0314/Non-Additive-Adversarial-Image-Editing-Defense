"""顏色族的天花板探針：把合法的顏色位移推到自然的極限，量編輯還會不會完成。

自變數是**臂 × 顏色位移（ΔE00）**，其餘一切固定。三個防禦臂都**不訓練**——
問的是「這個顏色載體能走到的地方，攻擊還成不成立」，訓練是下一批的事。
`runs/objective_pilot/` 已經量到最佳化在這個載體上幾乎不值錢（隨機邊界位移
0.3626 對最佳化後 0.3098），所以先把可達集合本身量清楚。

四個臂
────────────────────────────────────────────────────────────────────
`chroma_bounded`    `ChromaAffineParam(max_gain=1)`，奇異值上界，MK 的起點。
`chroma_isometric`  同上但投影到 O(2) 並指定色相旋轉角，奇異值恰為 1。
`region_palette`    `RegionPaletteParam`，衣物與其補集各一組配色，權重圖重度低通。
`undefended`        未防禦的攻擊。**每個（影像, 指令類, 種子）都要有**：種子層
                    的攻擊失敗會讓整批讀數變成雜訊，逐種子的對照不可省。

高頻約束不是構造保證
────────────────────────────────────────────────────────────────────
等距只在 Lab 的 (a,b) 平面上成立；`hf_ratio_rgb_total` 隨影像內容與旋轉角變動
（見 `tests/test_chroma_rotation.py`）。**每一列都照量照報**，不設門檻、不擋。
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

ARMS = ('chroma_bounded', 'chroma_isometric', 'region_palette', 'collision',
        'undefended')

# CSV 的合約。欄名沿用 `runs/objective_pilot/` 的字彙，兩批才並列得起來。
COLUMNS = [
    'arm', 'image', 'class', 'instruction', 'carrier', 'seed', 'eval_seed',
    'delta_e_target', 'region',
    'amplitude', 'delta_e_reached', 'support_deltaE00',
    'rotation_deg', 'max_gain', 'isometric', 'blur_sigma',
    'palette_id', 'palette_id_second',
    'effective_gain_max', 'effective_gain_min',
    'edit_lpips',
    'input_psnr', 'input_dists',
    'final_psnr', 'final_dists', 'final_deltaE00',
    'final_hf_rgb_total', 'final_hf_lab_L', 'final_hf_lab_a', 'final_hf_lab_b',
    'final_subject_identity', 'final_face_in_original',
    'final_outside_support_max_abs', 'final_support_exact',
    'final_hf_below_reference', 'final_id_above_reference',
    'protected_max_abs', 'protected_pixels',
    'collision_first', 'collision_last', 'collision_at_anchor',
    'collision_steps', 'collision_region_area', 'collision_ring_area',
    'subject_box_iou_edit_orig', 'subject_box_iou_edit_def',
    # 語意：編輯輸出對指令句的對齊。`_orig` 是未防禦的那張、`_def` 是防禦後
    # 的那張，`_drop` 是前者減後者。依 docs/EVALUATION.md，語意指標照報、
    # 不作判準。欄名沿用既有的 `edit_clip_*`／`edit_siglip_*`。
    'edit_clip_orig', 'edit_clip_def', 'edit_clip_drop',
    'edit_siglip_orig', 'edit_siglip_def', 'edit_siglip_drop',
    'subject_id_orig', 'subject_id_def', 'subject_id_drop',
    'id_embed_weights', 'attack_batch', 'attack_precision', 'seconds',
]


def assert_free_cards():
    """在匯入 torch 或載入任何權重**之前**核對指定的卡。

    與 `scripts/objective_pilot.py` 同一份檢查：`free_cards.sh` 兩個方向都會
    判錯，所以派工後自己再用 compute-apps 複驗一次。
    """
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


def palette_of(spec, palette_id):
    """從 NCF 的色彩分布庫取一組真實配色。自然度就是靠這個庫保證的。"""
    from src.defense.ncf_library import NCFLibrary
    lib = spec['library']
    obj = NCFLibrary(ROOT / lib['path'], expected_sha256=lib['sha256'],
                     required_classes=lib['class_weights'],
                     ade20k_classes=lib.get('ade20k_classes'))
    rec = next(r for r in obj.records if r['id'] == palette_id)
    return rec['mean'], rec['covariance']


def build_param(arm, x01, *, support, target_mean, target_cov, blur_sigma,
                rotation_deg=None, regions=None, radius=0.2, epsilon_lab=5):
    """依臂建參數化。**未知的臂拋錯，不靜默走預設。**

    `support` 是 (1,1,H,W)；整圖濾鏡傳全 1，衣物載體傳驗過的 `carrier_mask`
    路徑。`regions` 只有 `region_palette` 用，是 [(weight, mean, cov), ...]。
    """
    from src.defense.lowfreq_color import ChromaAffineParam, RegionPaletteParam
    common = dict(support=support, radius=radius, epsilon_lab=epsilon_lab,
                  max_gain=1.0, gamut='soft')
    if arm == 'chroma_bounded':
        return ChromaAffineParam(target_mean, target_cov, **common)
    if arm == 'chroma_isometric':
        return ChromaAffineParam(target_mean, target_cov, isometric=True,
                                 rotation_deg=rotation_deg, **common)
    if arm == 'region_palette':
        if not regions:
            raise ValueError('region_palette 需要 regions；缺了就不是這個臂')
        return RegionPaletteParam(regions, blur_sigma=blur_sigma, **common)
    if arm == 'collision':
        # **不是等距臂。** 同色碰撞要的是把兩塊區域的色度統計拉近，而全域仿射
        # 對任意兩塊區域的均值差作用是 M(mu_R − mu_E)：奇異值恰為 1 時那個差的
        # 長度被保住，梯度幾乎只剩 sigma 那一項；容許收縮（奇異值 ≤ 1）才推得動。
        # 與 `chroma_bounded` 用同一個參數化，兩者的差別因此只有目標函數。
        return ChromaAffineParam(target_mean, target_cov, **common)
    raise ValueError(f'未知的臂 {arm!r}；可用的是 {ARMS}')


def cells_of(spec):
    """(影像, 指令類別) 的完整清單，順序固定，不隨字典順序漂。"""
    out = []
    for image in spec['images']:
        for cls in spec['classes']:
            out.append({'image': image, 'class': cls['name'],
                        'instruction': cls['instructions'][image],
                        'carrier': cls['carrier'],
                        'region': cls.get('region', 'clothes'),
                        'seed': spec['seed']})
    return out


def collision_region(x01, region, clothes, device=None):
    """指令要改的那塊區域。`region` 由設定逐類指定，不由類名推。

    `collision` 那條臂問的是「把這塊區域的顏色統計與周邊拉近，指令還定位得到
    嗎」，所以區域必須對得上指令講的東西，不是對得上載體的支撐。指令類可以有
    很多個（帽子、太陽眼鏡、皇冠……）卻共用同一塊區域，所以對應寫在設定裡。

    `clothes`：ATR 衣物遮罩本身。
    `head_ring`：主體外緣的環帶——頭部配件會出現的地方。
    `outside_subject`：衣物與主體之外的一切。
    """
    import torch

    from src.defense.carrier_mask import face_subject_mask, ring_support
    if region == 'clothes':
        return clothes
    face = face_subject_mask(x01, device=device)
    if region == 'head_ring':
        return ring_support(face, inner=8, outer=48)
    if region == 'outside_subject':
        inside = torch.maximum(clothes, (face > 0.5).to(clothes.dtype))
        return (1.0 - inside).clamp(0.0, 1.0)
    raise ValueError(f'未知的區域 {region!r}；可用的是 clothes／head_ring／'
                     f'outside_subject')


def semantic_row(suite, edit_orig, edit_def, instruction):
    """兩張編輯輸出對**指令句**的語意對齊，CLIP 與 SigLIP 各一組。

    問的是「指令有沒有被執行」——那是使用者判準的三個條件之一，而位移與身分
    都答不了它：位移只說兩張圖差多少，身分只說主體還像不像。

    影像前向一次、文字逐 prompt 前向，走的是 `MetricSuite.semantic_multi`，
    與專案其他批次同一條路徑。依 `docs/EVALUATION.md`，語意指標**照報、
    不作判準**。
    """
    out = {}
    for name, y in (('orig', edit_orig), ('def', edit_def)):
        scores = suite.semantic_multi(y, [instruction])[instruction]
        for model in ('clip', 'siglip'):
            out[f'edit_{model}_{name}'] = round(float(scores[model]), 5)
    for model in ('clip', 'siglip'):
        out[f'edit_{model}_drop'] = round(
            out[f'edit_{model}_orig'] - out[f'edit_{model}_def'], 5)
    return out


def regions_of(x01, support_clothes):
    """衣物與其補集兩塊，供 `region_palette` 用。權重圖的邊由 blur_sigma 決定。"""
    return [support_clothes, (1.0 - support_clothes).clamp(0.0, 1.0)]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--arms', default=','.join(ARMS),
                    help='逗號分隔的臂清單，預設全部')
    ap.add_argument('--shard', default='1/1',
                    help='把格子切成 n 片只跑第 i 片，寫成 i/n。分片只切格子，'
                         '不改任何種子或參數，合併時直接串接 CSV。')
    ap.add_argument('--limit', type=int, default=0,
                    help='只跑前幾格，供冒煙測試')
    ap.add_argument('--device', default='cuda',
                    help="'cpu' 供冒煙測試；GPU 工作一律送遠端")
    ap.add_argument('--batch', type=int, default=1,
                    help='一次送幾張圖給攻擊模型。批次為 1 時 512² 的 UNet '
                         '餵不飽一張 3090（顯存只用到 24 GB 的 29%），加大就是'
                         '直接的吞吐量。每張圖各自一個 generator，所以輸出與'
                         '逐張跑相同；操作點由 scripts/throughput_probe.py 量。')
    ap.add_argument('--precision', default='',
                    help="覆寫設定裡的 precision（fp32／bf16／fp16）。"
                         "bf16 的骨幹與 VAE 同一種精度，不會有 fp16 那個 "
                         "Half/float 不符的問題。")
    ap.add_argument('--no-attack', action='store_true',
                    help='只渲染與量外觀，不跑編輯；冒煙測試用')
    args = ap.parse_args()
    arms = [a.strip() for a in args.arms.split(',') if a.strip()]
    for a in arms:
        if a not in ARMS:
            raise SystemExit(f'未知的臂 {a!r}；可用的是 {ARMS}')
    if args.device != 'cpu':
        assert_free_cards()

    import torch
    from src.defense.collision_loss import make_collision_loss, ring_of
    from src.defense.color_amplitude import delta_e00, solve_amplitude
    from src.defense.param_pgd import run_param_pgd
    from src.defense.naturalness_gate import gate_row
    from src.defense.ncf_library import sha256
    from src.defense.ncf_runner import ncf_support
    from src.metrics.identity import subject_identity_row
    from src.metrics.suite import MetricSuite

    spec = json.loads(args.config.read_text(encoding='utf-8'))
    manifest = json.loads((ROOT / spec['manifest']).read_text(encoding='utf-8'))
    if args.out.exists() and any(args.out.iterdir()):
        raise SystemExit(f'{args.out} 已存在且非空；換一個輸出目錄')
    args.out.mkdir(parents=True, exist_ok=True)

    precision = args.precision or spec['precision']
    if args.no_attack:
        ip2p, device = None, torch.device(args.device)
    else:
        from src.models.ip2p import IP2PWrapper
        ip2p = IP2PWrapper(dtype={'fp16': torch.float16, 'fp32': torch.float32,
                                  'bf16': torch.bfloat16}[precision])
        device = ip2p.device
    suite = MetricSuite(device=device)
    entries = {r['id']: r for r in manifest['images']}
    rows, t_start = [], time.time()

    i, n = (int(v) for v in args.shard.split('/'))
    if not 1 <= i <= n:
        raise SystemExit(f'--shard 要寫成 i/n 且 1 <= i <= n，收到 {args.shard!r}')
    todo = [c for k, c in enumerate(cells_of(spec)) if k % n == i - 1]
    if args.limit:
        todo = todo[:args.limit]
    print(f'分片 {i}/{n}：{len(todo)} 格，臂 {arms}，批次 {args.batch}'
          f'，精度 {args.precision or spec["precision"]}', flush=True)

    for cell in todo:
        entry = entries[cell['image']]
        src = ROOT / entry['path']
        if sha256(src) != entry['sha256']:
            raise ValueError(f"{cell['image']} 的輸入雜湊不符")
        x = load_image(src, device)
        tag = f"{cell['image']}__{cell['class']}"
        clothes = ncf_support(x, 'clothes')
        support = clothes if cell['carrier'] == 'clothes' else ncf_support(x, 'frame')
        whole_frame = cell['carrier'] != 'clothes'
        mean, cov = palette_of(spec, spec['palette_id'])
        second = palette_of(spec, spec['palette_id_second'])

        # ---- 第一階段：把這一格所有臂、所有目標的防禦圖先算完 ----
        # 攻擊編輯不在這裡跑。批次為 1 時 512² 的 UNet 餵不飽一張 3090
        # （實測顯存只用到 24 GB 的 29%），所以先把要編輯的東西收集起來，
        # 第二階段一次送一批。實驗設定完全不變，變的只有送進 GPU 的方式。
        pending = []
        for arm in [a for a in arms if a != 'undefended']:
            by_amplitude = {}
            regions = None
            if arm == 'region_palette':
                regions = [(w, m, c) for w, (m, c) in zip(
                    regions_of(x, clothes), [(mean, cov), second])]
            param = build_param(
                arm, x, support=support, target_mean=mean, target_cov=cov,
                blur_sigma=spec['blur_sigma'], regions=regions,
                rotation_deg=spec['rotation_deg'] if arm == 'chroma_isometric' else None,
                radius=spec['radius'], epsilon_lab=spec['epsilon_lab'])
            # `reset` 先跑：任何在它之前設好的東西都會被抹掉，而且不拋錯。
            param.reset(x, cell['seed'])
            trained, closs = {}, None
            if arm == 'collision':
                region = collision_region(x, cell['region'], clothes,
                                          device=device)
                ring = ring_of(region, spec['collision_ring_width'])
                closs = make_collision_loss(region, ring)
                param.set_amplitude(1.0)
                first = float(closs(param.render(x)))
                run_param_pgd(x, param, closs,
                              steps=spec['collision_steps'],
                              update=spec['collision_update'],
                              step_size=spec['step_size_ratio'] * float(param.radius),
                              seed=cell['seed'], momentum=spec['momentum'])
                last = float(closs(param.render(x)))
                trained = {'collision_first': round(first, 6),
                           'collision_last': round(last, 6),
                           'collision_steps': spec['collision_steps'],
                           'collision_region_area': round(float(region.mean()), 6),
                           'collision_ring_area': round(float(ring.mean()), 6)}
                print(f"  {tag} collision {first:.4f} -> {last:.4f}", flush=True)

            for target in sorted(spec['delta_e_targets'],
                                 key=lambda v: float('inf') if v == 'max_reach'
                                 else float(v)):
                if target == 'max_reach':
                    # 各臂自己走得到的最遠處。不二分——先前那批的 ΔE00 30 在
                    # 每一格都不可達，等於白跑一個目標；改成直接取幅度 1.0，
                    # 實際到了多遠由 `support_deltaE00` 記錄。**這個點不是
                    # 等失真錨點**，跨臂比較要用它旁邊那個 6.3 的點。
                    param.set_amplitude(1.0)
                    solved = {'amplitude': 1.0, 'reached': True,
                              'delta_e00': delta_e00(x, param.render(x), support)}
                else:
                    # 錨點取**支撐加權**的色差：全圖平均會被支撐面積稀釋，衣物
                    # 載體因此連 6.3 都到不了（量到 4.5–5.2），跟整圖濾鏡的 24
                    # 不可並列。
                    solved = solve_amplitude(param, x, float(target),
                                             support=support)
                if closs is not None:
                    trained['collision_at_anchor'] = round(
                        float(closs(param.render(x))), 6)
                x_def = param.render(x).detach()
                diag = param.diagnostics(x)
                # `isometric` 與 `rotation_deg` 記的是**插值前**的設定。幅度插值
                # 之後真正作用的矩陣是 (1-a)·I + a·M，`a < 1` 時奇異值小於 1
                # ——90 度旋轉在 a = 0.5 上是 0.707，那是收縮不是等距。
                eff = getattr(param, 'effective_chroma_matrix', None)
                if eff is None:
                    gain_max = gain_min = ''
                else:
                    sv = torch.linalg.svdvals(eff())
                    gain_max = round(float(sv.max()), 5)
                    gain_min = round(float(sv.min()), 5)
                final = {f'final{k[4:]}': v for k, v in
                         gate_row(x, x_def,
                                  support=None if whole_frame else support,
                                  suite=suite, device=device).items()}
                # 嚴格的受保護像素檢查：只看 w 恰為 0 的像素。
                # `gate_outside_support_max_abs` 用 (1 - w) 加權，羽化帶上
                # 0 < w < 1 就會非零，那是羽化不是違規。
                zero = (support <= 0).to(x_def.dtype)
                n_zero = int(zero.sum().item())
                protected = ('' if n_zero == 0 else
                             float(((x_def - x).abs() * zero).max()))
                d = suite.pairwise(x, x_def)
                label = target if isinstance(target, str) else f'{float(target):g}'
                name = f'{tag}__{arm}__dE{label}'
                save_png(x_def, args.out / f'{name}__def.png')
                base = {'arm': arm, **cell,
                        'delta_e_target': target,
                        'region': cell['region'],
                        'amplitude': round(solved['amplitude'], 5),
                        'delta_e_reached': int(solved['reached']),
                        'support_deltaE00': round(solved['delta_e00'], 4),
                        'rotation_deg': diag.get('rotation_deg', ''),
                        'max_gain': diag.get('max_gain', ''),
                        'isometric': diag.get('isometric', ''),
                        'blur_sigma': diag.get('blur_sigma', ''),
                        'palette_id': spec['palette_id'],
                        'palette_id_second': spec['palette_id_second'],
                        'effective_gain_max': gain_max,
                        'effective_gain_min': gain_min,
                        'input_psnr': round(float(d['psnr']), 4),
                        'input_dists': round(float(d['dists']), 5),
                        'protected_max_abs': protected,
                        'protected_pixels': n_zero,
                        # 批次與精度進 CSV：它們不該改變數字，但「不該」要能
                        # 被查核，而不是靠記憶。
                        'attack_batch': args.batch,
                        'attack_precision': precision,
                        **trained, **final}
                print(f"  {name} a {base['amplitude']} dE_support "
                      f"{base['support_deltaE00']}"
                      f"{'' if solved['reached'] else '(未達標)'} dE_frame "
                      f"{final['final_deltaE00']} hf {final['final_hf_rgb_total']} "
                      f"psnr {final['final_psnr']}", flush=True)
                if args.no_attack:
                    rows.append({**base, 'eval_seed': '',
                                 'seconds': round(time.time()-t_start, 1)})
                    continue
                # 不可達的目標會解到同一個幅度（1.0），渲染出**逐位元相同**的
                # 防禦圖。那些列的讀數必然一樣，重跑攻擊只是白燒機時。
                key = round(solved['amplitude'], 6)
                if key in by_amplitude:
                    pending.append({'base': base, 'x_def': None, 'name': name,
                                    'share': (arm, key)})
                    continue
                by_amplitude[key] = True
                pending.append({'base': base, 'x_def': x_def, 'name': name,
                                'share': (arm, key)})

        if args.no_attack or not pending:
            continue

        # ---- 第二階段：把這一格的所有編輯併成幾個批次 ----
        # 同一格的每一次編輯共用指令，只有影像與種子不同，正好是批次的形狀。
        work = []
        for eval_seed in spec['eval_seeds']:
            work.append({'kind': 'clean', 'seed': eval_seed, 'image': x})
        for item in pending:
            if item['x_def'] is None:
                continue
            for eval_seed in spec['eval_seeds']:
                work.append({'kind': 'def', 'seed': eval_seed,
                             'image': item['x_def'], 'item': item})
        done = []
        for k in range(0, len(work), args.batch):
            chunk = work[k:k + args.batch]
            outs = ip2p.edit_batch([w['image'] for w in chunk],
                                   [cell['instruction']] * len(chunk),
                                   [w['seed'] for w in chunk],
                                   steps=spec['attack_steps'],
                                   s_t=spec['s_t'], s_i=spec['s_i'])
            for w, y in zip(chunk, outs):
                done.append((w, y[None]))
        clean = {w['seed']: y for w, y in done if w['kind'] == 'clean'}
        for eval_seed, y in clean.items():
            save_png(y, args.out / f'{tag}__s{eval_seed}__clean_edit.png')

        # ---- 第三階段：讀數 ----
        if 'undefended' in arms:
            for eval_seed in spec['eval_seeds']:
                e0 = clean[eval_seed]
                read = subject_identity_row(x, e0, e0, device=device)
                for f in ('n_faces_orig', 'n_faces_edit_orig', 'n_faces_edit_def'):
                    read.pop(f, None)
                read.update(semantic_row(suite, e0, e0, cell['instruction']))
                rows.append({'arm': 'undefended', **cell, 'eval_seed': eval_seed,
                             **read, 'edit_lpips': 0.0,
                             'seconds': round(time.time()-t_start, 1)})
                print(f"  {tag} s{eval_seed} 未防禦：id {read['subject_id_orig']} "
                      f"siglip {read['edit_siglip_orig']}", flush=True)

        measured = {}
        for w, ed in done:
            if w['kind'] != 'def':
                continue
            item, eval_seed = w['item'], w['seed']
            e0 = clean[eval_seed]
            read = subject_identity_row(x, e0, ed, device=device)
            # 臉數不進報表：主體位置上有沒有臉已經由 `subject_box_iou_edit_def`
            # 與空的 `subject_id_def` 表達，而臉數會把旁人算進來。
            for f in ('n_faces_orig', 'n_faces_edit_orig', 'n_faces_edit_def'):
                read.pop(f, None)
            read.update(semantic_row(suite, e0, ed, cell['instruction']))
            lpips = round(float(suite.pairwise(e0, ed)['lpips']), 5)
            measured[(item['share'], eval_seed)] = (read, lpips)
            rows.append({**item['base'], 'eval_seed': eval_seed,
                         'edit_lpips': lpips, **read,
                         'seconds': round(time.time()-t_start, 1)})
            save_png(ed, args.out / f"{item['name']}__s{eval_seed}__def_edit.png")
            print(f"    {item['name']} s{eval_seed} id {read['subject_id_def']} "
                  f"siglip {read['edit_siglip_orig']}->{read['edit_siglip_def']} "
                  f"位移 {lpips}", flush=True)

        # 共用同一張防禦圖的那些列：沿用同一份攻擊讀數，不重跑。
        for item in pending:
            if item['x_def'] is not None:
                continue
            for eval_seed in spec['eval_seeds']:
                read, lpips = measured[(item['share'], eval_seed)]
                rows.append({**item['base'], 'eval_seed': eval_seed,
                             'edit_lpips': lpips, **read,
                             'seconds': round(time.time()-t_start, 1)})
    suffix = '' if n == 1 else f'_shard{i}of{n}'
    write_csv(args.out / f'color_ceiling{suffix}.csv', rows)
    print(f'-> {args.out}  共 {len(rows)} 列，{time.time()-t_start:.0f} 秒', flush=True)


def save_png(x, path):
    import numpy as np
    from PIL import Image
    arr = (x[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy()*255).astype('uint8')
    Image.fromarray(arr).save(path)


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

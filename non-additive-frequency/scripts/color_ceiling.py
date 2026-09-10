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
    'delta_e_target', 'amplitude', 'delta_e_reached', 'support_deltaE00',
    'rotation_deg', 'max_gain', 'isometric', 'blur_sigma', 'palette_id',
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
    'n_faces_orig', 'n_faces_edit_orig', 'n_faces_edit_def',
    'subject_box_iou_edit_orig', 'subject_box_iou_edit_def',
    'subject_id_orig', 'subject_id_def', 'subject_id_drop',
    'id_embed_weights', 'seconds',
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
                        'carrier': cls['carrier'], 'seed': spec['seed']})
    return out


def collision_region(x01, cls, clothes, device=None):
    """指令要改的那塊區域，逐指令類各一種取法。

    `collision` 那條臂問的是「把這塊區域的顏色統計與周邊拉近，指令還定位得到
    嗎」，所以區域必須對得上指令講的東西，不是對得上載體的支撐。

    衣物：ATR 衣物遮罩本身。
    配件：主體外緣的環帶——帽子會出現的地方。
    背景：衣物與主體之外的一切。
    """
    import torch

    from src.defense.carrier_mask import face_subject_mask, ring_support
    if cls == 'clothing':
        return clothes
    face = face_subject_mask(x01, device=device)
    if cls == 'accessory':
        return ring_support(face, inner=8, outer=48)
    if cls == 'background':
        inside = torch.maximum(clothes, (face > 0.5).to(clothes.dtype))
        return (1.0 - inside).clamp(0.0, 1.0)
    raise ValueError(f'未知的指令類 {cls!r}；沒有對應的區域取法')


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
    from src.defense.color_amplitude import solve_amplitude
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

    if args.no_attack:
        ip2p, device = None, torch.device(args.device)
    else:
        from src.models.ip2p import IP2PWrapper
        ip2p = IP2PWrapper(dtype={'fp16': torch.float16, 'fp32': torch.float32,
                                  'bf16': torch.bfloat16}[spec['precision']])
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
    print(f'分片 {i}/{n}：{len(todo)} 格，臂 {arms}', flush=True)

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

        # 未防禦的編輯：每個種子一次，所有臂與所有 ΔE00 共用同一張，省掉三分之二
        # 的攻擊機時。fp32 下單次 100 步是 21 秒。
        clean = {}
        if not args.no_attack:
            for eval_seed in spec['eval_seeds']:
                clean[eval_seed] = ip2p.edit(
                    x, cell['instruction'], seed=eval_seed,
                    steps=spec['attack_steps'], s_t=spec['s_t'], s_i=spec['s_i'])
                save_png(clean[eval_seed],
                         args.out / f'{tag}__s{eval_seed}__clean_edit.png')

        if 'undefended' in arms and not args.no_attack:
            for eval_seed in spec['eval_seeds']:
                e0 = clean[eval_seed]
                read = subject_identity_row(x, e0, e0, device=device)
                rows.append({'arm': 'undefended', **cell, 'eval_seed': eval_seed,
                             **read, 'edit_lpips': 0.0,
                             'seconds': round(time.time()-t_start, 1)})
                print(f"  {tag} s{eval_seed} 未防禦：臉數 "
                      f"{read['n_faces_edit_orig']} id {read['subject_id_orig']}",
                      flush=True)

        for arm in [a for a in arms if a != 'undefended']:
            # 不可達的目標會解到同一個幅度（1.0），渲染出**逐位元相同**的防禦圖。
            # 那些格子的讀數必然一樣，重跑攻擊只是白燒機時。按幅度快取，列還是
            # 逐目標各一列（`delta_e_reached` 標明可不可達），只是不重算。
            by_amplitude = {}

            # 建構、`reset` 與訓練都與 ΔE00 目標無關，所以只做一次；目標迴圈裡
            # 變的只有幅度，那是事後的空間常數投影。碰撞臂的 300 步因此不會被
            # 三個目標各跑一次。
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
                region = collision_region(x, cell['class'], clothes,
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

            for target in sorted(spec['delta_e_targets']):
                # 錨點取**支撐加權**的色差：全圖平均會被支撐面積稀釋，衣物載體
                # 因此連 6.3 都到不了（量到 4.5–5.2），跟整圖濾鏡的 24 不可並列。
                solved = solve_amplitude(param, x, target, support=support)
                if closs is not None:
                    trained['collision_at_anchor'] = round(
                        float(closs(param.render(x))), 6)
                x_def = param.render(x).detach()
                diag = param.diagnostics(x)
                final = {f'final{k[4:]}': v for k, v in
                         gate_row(x, x_def,
                                  support=None if whole_frame else support,
                                  suite=suite, device=device).items()}
                # 嚴格的受保護像素檢查：只看 w 恰為 0 的像素。
                # `gate_outside_support_max_abs` 用 (1 - w) 加權，羽化帶上
                # 0 < w < 1 就會非零，那是羽化不是違規；使用者的約束是
                # 「w = 0 的像素逐位元不動」，兩者要分開量。
                zero = (support <= 0).to(x_def.dtype)
                n_zero = int(zero.sum().item())
                protected = ('' if n_zero == 0 else
                             float(((x_def - x).abs() * zero).max()))
                d = suite.pairwise(x, x_def)
                name = f'{tag}__{arm}__dE{target:g}'
                save_png(x_def, args.out / f'{name}__def.png')
                base = {'arm': arm, **cell,
                        'delta_e_target': target,
                        'amplitude': round(solved['amplitude'], 5),
                        'delta_e_reached': int(solved['reached']),
                        'support_deltaE00': round(solved['delta_e00'], 4),
                        'rotation_deg': diag.get('rotation_deg', ''),
                        'max_gain': diag.get('max_gain', ''),
                        'isometric': diag.get('isometric', ''),
                        'blur_sigma': diag.get('blur_sigma', ''),
                        'palette_id': spec['palette_id'],
                        'input_psnr': round(float(d['psnr']), 4),
                        'input_dists': round(float(d['dists']), 5),
                        'protected_max_abs': protected,
                        'protected_pixels': n_zero,
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

                key = round(solved['amplitude'], 6)
                cached = by_amplitude.get(key)
                if cached is not None:
                    print(f"    幅度 {key} 與前一個目標相同，防禦圖逐位元一樣，"
                          f"沿用同一份攻擊讀數", flush=True)
                    for read, lpips, eval_seed in cached:
                        rows.append({**base, 'eval_seed': eval_seed,
                                     'edit_lpips': lpips, **read,
                                     'seconds': round(time.time()-t_start, 1)})
                    continue

                measured = []
                for eval_seed in spec['eval_seeds']:
                    ed = ip2p.edit(x_def, cell['instruction'], seed=eval_seed,
                                   steps=spec['attack_steps'], s_t=spec['s_t'],
                                   s_i=spec['s_i'])
                    read = subject_identity_row(x, clean[eval_seed], ed,
                                                device=device)
                    lpips = round(
                        float(suite.pairwise(clean[eval_seed], ed)['lpips']), 5)
                    measured.append((read, lpips, eval_seed))
                    rows.append({**base, 'eval_seed': eval_seed,
                                 'edit_lpips': lpips, **read,
                                 'seconds': round(time.time()-t_start, 1)})
                    save_png(ed, args.out / f'{name}__s{eval_seed}__def_edit.png')
                    print(f"    s{eval_seed} id {read['subject_id_def']} 臉 "
                          f"{read['n_faces_edit_orig']}->{read['n_faces_edit_def']}"
                          f" 位移 {lpips}", flush=True)
                by_amplitude[key] = measured

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

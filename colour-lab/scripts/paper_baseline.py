# 唯讀複製自主 repo：`non-additive-frequency/scripts/paper_baseline.py`
#（本機 `C:/image-immunization/non-additive-frequency/`、遠端
# `/nfs/home/nelson0314/WACV-s4/`）。主 repo 的那一份不得寫入。
# lab 這一份與來源的差異只有兩處，都是為了接上早停：
#   1. COLUMNS 多四欄 free_stopped_step / free_early_stopped /
#      free_patience / free_min_delta（DictWriter 是 extrasaction='ignore'，
#      不加欄位的話新讀數會被靜默丟掉）。
#   2. optimise_carrier 多傳 patience 與 min_delta，值取自 variant。
# 早停本身寫在 `src/defense/immunise.py::optimise_carrier`。
"""在論文自己的受害任務上重現三篇顏色攻擊，並把防禦圖存下來。

這一支回答的問題
────────────────────────────────────────────────────────────────────
**移植是對的嗎？** 三篇（ReColorAdv、NCF、AdvCF）打的都是 ImageNet 分類器，
損失是 C&W 的 logit margin。用它們自己的載體、自己的預算、自己的損失、自己的
受害模型跑一次，成功率要落在論文宣稱的量級。做不到就是移植有問題，那時候
「在擴散編輯上是零」這句話說不出口。

跑完之後，**同一批防禦圖**再交給 `scripts/evaluate_defence.py` 打 IP2P。
兩層用的是同一個檔案，所以「同一個方法、換一個受害模型」這句話成立。

為什麼影像一定要存
────────────────────────────────────────────────────────────────────
`runs/ncf_ip2p/`、`runs/chroma/` 這幾批只留了 CSV，**一張圖都沒有**，所以沒有
人看過那些產物長什麼樣；事後從 CSV 看到的是 ΔE00 25–30、PSNR 7–8，那不是
論文示例的樣子。要說「方法移植過來失效」，讀者必須看得到那些圖確實像論文的圖。

**這一支不含任何指令**，與 `scripts/immunise*.py` 同一條規矩：設定檔遞迴檢查，
出現 `instruction`／`prompt` 鍵就拒絕啟動。
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

COLUMNS = ['image', 'variant', 'carrier', 'victim', 'clean_label', 'clean_prob',
           'pred_label', 'pred_prob', 'true_prob', 'margin', 'flipped',
           'steps', 'lr', 'radius', 'smoothness_weight',
           'margin_start', 'margin_end', 'deltaE00', 'psnr', 'lpips',
           'linf', 'resample', 'face_weight', 'chain_steps', 'grad_steps',
           'timesteps', 'init_jitter', 'deltae_cap', 'deltae_floor',
           'w_out', 'w_tone',
           'free_cap_violations', 'free_caps_unprojected', 'free_lambda',
           'free_term_out', 'free_term_tone', 'free_term_enc_target',
           'free_term_diffusion', 'free_term_diffusion_target',
           'free_term_out_target', 'free_term_sds',
           'w_enc_target', 'w_diffusion', 'w_diffusion_target',
           'w_out_target', 'w_sds', 'target', 'enc_target_norm',
           'eot_kinds', 'eot_samples', 'eot_val_samples',
           'solver', 'free_amplitude_shrink',
           'free_restarts', 'free_restart_pick', 'free_restart_scores',
           'free_lr_final_ratio', 'free_score_start',
           'free_score_unprojected', 'free_score_end', 'free_feasible_step',
           'free_feasible_checks', 'free_stopped_step',
           'free_early_stopped', 'free_patience', 'free_min_delta',
           'free_term_enc', 'free_term_cond',
           'free_term_id', 'free_curve', 'seconds']


def load_guard():
    spec = importlib.util.spec_from_file_location(
        'immunise_script', ROOT / 'scripts' / 'immunise.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.assert_no_instructions


def build_carrier(kind, x01, opts, spec, device):
    """依論文名稱建載體。每一種的預設值就是該篇的操作點。

    NCF 需要一個來自 ADE20K 色彩庫的目標分布與一塊支撐。整圖色彩濾鏡的支撐
    是全 1（`NCFColorParam` 的 docstring 明寫這一點），而 `palette` 指定要用
    哪一組分布——**論文是逐分割類別各挑一組再按面積混合**，那需要它們的
    Swin-T UperNet。這裡由設定檔指名類別，是 `modified_from_paper`。
    """
    import torch

    if kind == 'recoloradv':
        from src.defense.recoloradv_param import ReColorAdvParam
        return ReColorAdvParam(
            x01, resolution=tuple(opts.get('resolution', (16, 32, 32))),
            radius=float(opts.get('radius', 0.06)))
    if kind == 'advcf_random':
        from src.defense.color_param import ColorCurveLowFreqRandom
        return ColorCurveLowFreqRandom(
            radius=float(opts.get('radius', 5.0)),
            pieces=int(opts.get('pieces', 64)),
            bound_mode='advcf',
            init_jitter=float(opts.get('init_jitter', 1.0)),
            modes=int(opts.get('modes', 3)))
    if kind == 'advcf':
        from src.defense.color_param import ColorCurveParam
        return ColorCurveParam(radius=float(opts.get('radius', 1.0)),
                               pieces=int(opts.get('pieces', 64)),
                               bound_mode='advcf',
                               init_jitter=float(opts.get('init_jitter', 0.0)))
    if kind == 'skin_locus_curve':
        from src.defense.skin_locus_curve import SkinLocusCurveParam
        return SkinLocusCurveParam(
            radius=float(opts.get('radius', 5.0)),
            pieces=int(opts.get('pieces', 64)),
            protect_scale=float(opts.get('protect_scale', 1.0)),
            hue_centre=float(opts.get('hue_centre', 50.0)),
            hue_inner=float(opts.get('hue_inner', 25.0)),
            hue_outer=float(opts.get('hue_outer', 55.0)),
            chroma_low=float(opts.get('chroma_low', 3.0)),
            chroma_high=float(opts.get('chroma_high', 12.0)),
            bound_mode='advcf',
            init_jitter=float(opts.get('init_jitter', 0.0)))
    if kind == 'graded_curve':
        from src.defense.graded_curve import GradedCurveParam
        return GradedCurveParam(radius=float(opts.get('radius', 5.0)),
                                pieces=int(opts.get('pieces', 64)),
                                span=float(opts.get('span', 0.25)),
                                lift_max=float(opts.get('lift_max', 0.25)),
                                drop_max=float(opts.get('drop_max', 0.25)),
                                init_jitter=float(opts.get('init_jitter', 0.0)))
    if kind == 'lifted_curve':
        from src.defense.lifted_curve_param import LiftedCurveParam
        return LiftedCurveParam(radius=float(opts.get('radius', 5.0)),
                                pieces=int(opts.get('pieces', 64)),
                                lift_max=float(opts.get('lift_max', 0.25)),
                                drop_max=float(opts.get('drop_max', 0.25)),
                                bound_mode='advcf',
                                init_jitter=float(opts.get('init_jitter', 0.0)))
    if kind == 'channel_span_curve':
        from src.defense.channel_span_curve import ChannelSpanCurveParam
        return ChannelSpanCurveParam(radius=float(opts.get('radius', 5.0)),
                                     pieces=int(opts.get('pieces', 64)),
                                     span=float(opts.get('span', 0.25)),
                                     init_jitter=float(opts.get('init_jitter', 0.0)))
    if kind == 'value_curve':
        from src.defense.value_curve_param import ValueCurveParam
        return ValueCurveParam(radius=float(opts.get('radius', 5.0)),
                               pieces=int(opts.get('pieces', 64)),
                               bound_mode='advcf',
                               init_jitter=float(opts.get('init_jitter', 0.0)))
    if kind == 'ncf':
        from src.defense.assets import palette_bank
        from src.defense.ncf_param import NCFColorParam
        bank = palette_bank(spec, opts.get('palette_class', 'person'))
        mean, cov = bank[int(opts.get('palette_index', 0)) % len(bank)]
        support = torch.ones_like(x01[:, :1])
        return NCFColorParam(mean, cov, support=support,
                             radius=float(opts.get('radius', 0.2)),
                             epsilon_lab=opts.get('epsilon_lab'),
                             whiten=bool(opts.get('whiten', False)))
    raise SystemExit(f'不認得的載體 {kind!r}；只有 recoloradv、advcf、'
                     f'advcf_random、value_curve、channel_span_curve、lifted_curve、graded_curve、skin_locus_curve、ncf')


def main():
    import torch

    from src.defense.assets import load_image, save_png
    from src.defense.color_amplitude import delta_e00
    from src.defense.immunise import optimise_carrier
    from src.defense.ncf_library import sha256
    from src.defense.victim_classifier import RegularisedVictim, VictimClassifier
    from src.metrics.suite import MetricSuite

    assert_no_instructions = load_guard()

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--shard', default='1/1')
    ap.add_argument('--variants', default='')
    ap.add_argument('--objective', default='classifier',
                    choices=('classifier', 'free'),
                    help="classifier＝三篇自己的 C&W margin（第一層）；"
                         "free＝本專案的無指令目標 enc／cond／id（第二層）。"
                         "換損失是消融，要與第一層分開報，見 "
                         "docs/reference/BIBLIOGRAPHY.md。")
    args = ap.parse_args()

    spec = json.loads(args.config.read_text(encoding='utf-8'))
    assert_no_instructions(spec)
    manifest = json.loads((ROOT / spec['manifest']).read_text(encoding='utf-8'))
    entries = {r['id']: r for r in manifest['images']}
    if args.out.exists() and any(args.out.iterdir()):
        raise SystemExit(f'{args.out} 已存在且非空；換一個輸出目錄')
    args.out.mkdir(parents=True, exist_ok=True)

    variants = spec['variants']
    if args.variants:
        want = {v.strip() for v in args.variants.split(',')}
        variants = [v for v in variants if v['name'] in want]
        if not variants:
            raise SystemExit(f'設定檔裡沒有這些變體：{sorted(want)}')
    i, n = (int(v) for v in args.shard.split('/'))
    listed = spec.get('images') or [r['id'] for r in manifest['images']]
    todo = [im for k, im in enumerate(listed) if k % n == i - 1]

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    suite = MetricSuite(device=device)
    victims = {}
    ip2p = None
    if args.objective == 'free':
        from src.models.ip2p import IP2PWrapper
        ip2p = IP2PWrapper(dtype={'fp16': torch.float16, 'fp32': torch.float32,
                                  'bf16': torch.bfloat16}[spec.get('precision',
                                                                   'bf16')])
        device = ip2p.device
        suite = MetricSuite(device=device)
    suffix = '' if n == 1 else f'_shard{i}of{n}'
    # 受害任務要照實印：`--objective free` 走的是不含文字的三個項，不是分類器的
    # logit margin。寫死成「分類器」會讓從 log 檢查派工的人以為跑錯了目標函數。
    print(f'分片 {i}/{n}：{len(todo)} 張 × {len(variants)} 組，'
          f'受害任務＝{"不含指令的三項目標" if args.objective == "free" else "分類器"}，'
          f'指令：無', flush=True)

    rows = []
    for image in todo:
        entry = entries[image]
        src = ROOT / entry['path']
        if sha256(src) != entry['sha256']:
            raise ValueError(f'{image} 的輸入雜湊不符')
        x = load_image(src, device)
        box = None
        if args.objective == 'free':
            from src.metrics.identity import face_boxes
            boxes = face_boxes(x, device)
            if not boxes:
                raise ValueError(f'{image} 偵測不到臉，無指令目標的 id 項錨不住')
            box = max(boxes, key=lambda q: (q[2] - q[0]) * (q[3] - q[1]))
        for variant in variants:
            t0 = time.time()
            vname = variant.get('victim', spec.get('victim', 'resnet50'))
            if vname not in victims:
                victims[vname] = VictimClassifier(vname, device=device)
            victim = victims[vname]
            anchor = victim.anchor(x)

            carrier = build_carrier(variant['carrier'], x,
                                    variant.get('options', {}), spec, device)
            if hasattr(carrier, 'reset'):
                carrier.reset(x, int(variant.get('seed', 0)))
            weight = float(variant.get('smoothness_weight', 0.0))
            if args.objective == 'free' and variant.get('solver') != 'random':
                from src.defense.instruction_free import FreeObjective
                free = dict(spec['free'], **(variant.get('free') or {}))
                target_name = str(free.get('target', 'grey'))
                objective = FreeObjective(
                    ip2p, x, box=box, k=int(free['timesteps']),
                    steps=int(spec.get('attack_steps', 50)),
                    seed=int(free['noise_seed']),
                    weights=dict(free.get('weights') or {}),
                    chain_steps=int(free.get('chain_steps', 6)),
                    grad_steps=int(free.get('grad_steps', 1)),
                    resample=bool(free.get('resample', False)),
                    face_weight=float(free.get('face_weight', 0.0)),
                    s_i=float(spec.get('s_i', 1.5)))
                wts = free.get('weights') or {}
                w_out = float(wts.get('out', 0.0))
                w_out_target = float(wts.get('out_target', 0.0))
                w_sds = float(wts.get('sds', 0.0))
                w_tone = float(wts.get('tone', 0.0))
                w_main = (float(wts.get('enc_target', 0.0))
                          + float(wts.get('diffusion', 0.0))
                          + float(wts.get('diffusion_target', 0.0)))
                if (w_out > 0 or w_tone > 0 or w_main > 0
                        or w_out_target > 0 or w_sds > 0):
                    from src.defense.readout_terms import (
                        CompositeObjective, OutputDisplacement,
                        TargetedOutputDisplacement, ToneFlatness)
                    out_term = (OutputDisplacement(objective, x,
                                                   suite.lpips_module)
                                if w_out > 0 else None)
                    tone_term = (ToneFlatness(
                        x, box=box,
                        pieces=int(variant.get('options', {}).get('pieces', 64)))
                        if w_tone > 0 else None)
                    target01 = None
                    if w_main > 0 or w_out_target > 0:
                        from src.defense.mainstream_terms import build_target
                        target01 = build_target(target_name, x)
                    mainstream = None
                    if w_main > 0:
                        from src.defense.mainstream_terms import MainstreamTerms
                        mainstream = MainstreamTerms(
                            objective, x, target=target01,
                            enc_target_norm=str(
                                free.get('enc_target_norm', 'target')))
                    out_target_term = (
                        TargetedOutputDisplacement(objective, target01,
                                                   suite.lpips_module)
                        if w_out_target > 0 else None)
                    sds_term = None
                    if w_sds > 0:
                        from src.defense.sds_terms import SDSDiffusion
                        sds_term = SDSDiffusion(objective)
                    objective = CompositeObjective(
                        objective, carrier=carrier, out_term=out_term,
                        tone_term=tone_term, mainstream=mainstream,
                        out_target_term=out_target_term, sds_term=sds_term,
                        weights=dict(wts))
                eot_spec = free.get('eot') or {}
                if eot_spec.get('kinds'):
                    from src.defense.eot import EOTObjective
                    objective = EOTObjective(
                        objective, kinds=list(eot_spec['kinds']),
                        samples=int(eot_spec.get('samples', 1)),
                        seed=int(eot_spec.get('seed', 0)), device=device,
                        include_identity=bool(
                            eot_spec.get('include_identity', True)),
                        val_samples=int(eot_spec.get('val_samples', 2)))
            elif variant.get('solver') == 'random':
                free = {}
                objective = None
            else:
                free = {}
                objective = RegularisedVictim(victim, carrier, weight)

            with torch.no_grad():
                margin_start = float(victim.margin(carrier.render(x)))
            if variant.get('solver') == 'random':
                from src.defense.color_amplitude import fit_curve_jitter
                target = float(variant['deltae_cap'])
                fit = fit_curve_jitter(carrier, x, target,
                                       seed=int(variant.get('seed', 0)))
                stats = {'free_steps': 0, 'free_lr': 0.0, 'free_rho': 0.0,
                         'free_restarts': 1, 'free_restart_pick': 0,
                         'free_restart_scores': '', 'free_lr_final_ratio': 1.0,
                         'free_curve': '', 'free_score_start': 0.0,
                         'free_score_unprojected': 0.0,
                         'free_score_end': 0.0,
                         'free_amplitude_shrink': round(fit['jitter'], 5),
                         'free_feasible_step': -1, 'free_feasible_checks': 0,
                         'free_cap_violations': int(
                             fit['delta_e00'] > target + 0.25),
                         'free_caps_unprojected': f'{fit["delta_e00"]:.3f}',
                         'free_lambda': '', 'free_amplitude_base': ''}
                print(f'  {image} · {variant["name"]}  隨機對照 '
                      f'jitter {fit["jitter"]:.4f} → ΔE00 '
                      f'{fit["delta_e00"]:.2f}（目標 {target}）'
                      f'{"" if fit["reached"] else " 未達標，半徑不足"}',
                      flush=True)
            else:
                stats = None
            caps = []
            cap_value = float(variant.get('deltae_cap', 0.0))
            floor_value = float(variant.get('deltae_floor', 0.0))
            if cap_value > 0 or floor_value > 0:
                from src.defense.delta_e_torch import delta_e00_torch
                from src.defense.immunise import Cap
            if cap_value > 0:
                caps.append(Cap('deltaE00',
                                lambda y, x=x: delta_e00_torch(x, y),
                                lambda y, x=x: float(delta_e00(x, y)),
                                cap_value))
            if floor_value > 0:
                # `Cap` 只表達得出上限（違反是 soft/value − 1 > 0）。下限改寫成
                # `floor / ΔE00 ≤ 1`：ΔE00 掉到 floor 以下時這個量就大於 1，
                # 同一套增廣拉格朗日與可行檢查照用，`immunise.py` 不必改。
                # 分母加一個小正數，ΔE00 = 0（恆等解）時不會變成 inf。
                caps.append(Cap('deltaE00_floor',
                                lambda y, x=x, f=floor_value:
                                    f / (delta_e00_torch(x, y).mean() + 1e-3),
                                lambda y, x=x, f=floor_value:
                                    f / (float(delta_e00(x, y)) + 1e-3),
                                1.0))
            stats = stats if stats is not None else optimise_carrier(
                carrier, x, objective, steps=int(variant.get('steps', 100)),
                lr=float(variant.get('lr', 0.01)), caps=caps,
                check_every=int(variant.get('check_every', 10)),
                log_every=int(variant.get('log_every', 0)),
                restarts=int(variant.get('restarts', 1)),
                restart_seed=int(variant.get('seed', 0)),
                lr_final_ratio=float(variant.get('lr_final_ratio', 1.0)),
                probe_every=int(variant.get('probe_every', 0)),
                patience=int(variant.get('patience', 0)),
                min_delta=float(variant.get('min_delta', 0.0)))

            with torch.no_grad():
                y = (carrier.render(x).detach().clamp(0, 1) * 255).round() / 255
                rep = victim.report(y)
                pair = suite.pairwise(x, y)
                de = delta_e00(x, y)
                linf = float((y - x).abs().max())

            save_png(y, args.out / f'{image}__{variant["name"]}__defended.png')
            rows.append({
                'image': image, 'variant': variant['name'],
                'carrier': variant['carrier'], 'victim': vname,
                **anchor, **rep, 'steps': variant.get('steps', 100),
                'lr': variant.get('lr', 0.01),
                'radius': variant.get('options', {}).get('radius'),
                'smoothness_weight': weight,
                'resample': bool(free.get('resample', False)),
                'face_weight': float(free.get('face_weight', 0.0)),
                'chain_steps': free.get('chain_steps'),
                'grad_steps': free.get('grad_steps'),
                'timesteps': free.get('timesteps'),
                'init_jitter': variant.get('options', {}).get('init_jitter', 0.0),
                'deltae_cap': variant.get('deltae_cap', 0.0),
                'deltae_floor': variant.get('deltae_floor', 0.0),
                'solver': variant.get('solver', 'adam'),
                'w_out': (free.get('weights') or {}).get('out', 0.0),
                'w_tone': (free.get('weights') or {}).get('tone', 0.0),
                'w_enc_target': (free.get('weights') or {}).get('enc_target', 0.0),
                'w_diffusion': (free.get('weights') or {}).get('diffusion', 0.0),
                'w_diffusion_target': (free.get('weights') or {}).get(
                    'diffusion_target', 0.0),
                'w_out_target': (free.get('weights') or {}).get(
                    'out_target', 0.0),
                'w_sds': (free.get('weights') or {}).get('sds', 0.0),
                'target': free.get('target', ''),
                'enc_target_norm': free.get('enc_target_norm', ''),
                'eot_kinds': '|'.join((free.get('eot') or {}).get('kinds', [])),
                'eot_samples': (free.get('eot') or {}).get('samples', 0),
                'eot_val_samples': (free.get('eot') or {}).get(
                    'val_samples', 0),
                **stats,
                'margin_start': round(margin_start, 5),
                'margin_end': round(rep['margin'], 5),
                'deltaE00': round(float(de), 3),
                'psnr': round(float(pair['psnr']), 2),
                'lpips': round(float(pair['lpips']), 4),
                'linf': round(linf, 4),
                'seconds': round(time.time() - t0, 1)})
            print(f'  {image} · {variant["name"]}  margin '
                  f'{margin_start:+.3f} → {rep["margin"]:+.3f}  '
                  f'翻轉 {rep["flipped"]}  ΔE00 {float(de):.2f}  '
                  f'PSNR {pair["psnr"]:.2f}  LPIPS {pair["lpips"]:.4f}',
                  flush=True)

    out_csv = args.out / f'paper_baseline{suffix}.csv'
    with open(out_csv, 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)
    hit = sum(r['flipped'] for r in rows)
    print(f'寫出 {out_csv}（{len(rows)} 列，翻轉 {hit}/{len(rows)}）', flush=True)


if __name__ == '__main__':
    main()

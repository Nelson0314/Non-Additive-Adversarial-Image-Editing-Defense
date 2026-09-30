"""量一批既有影像的文字條件反應比值。**唯讀，不最佳化任何東西。**

回答一個先決問題：影像側對 UNet 的**文字分支**有沒有控制權。若把照片改到
ΔE00 主體 1.5–15.9 這麼大的範圍，反應比值仍然落在數值地板之內，那麼任何
以這個比值為目標的載體（貼片或曲線）都沒有可用的梯度，方向在寫載體之前
就死了。

**這一支不含任何指令。** 文字方向由固定種子的隨機數產生，不查 tokenizer
詞表、不抽 token、不做最近詞句搜尋。唯一碰到 text encoder 的地方是取空字串
的 `e₀`。輸入只有影像路徑、數值超參與種子，不載入任何 manifest 說明欄位、
評估設定或語句資產。

三組讀數一次輸出
────────────────────────────────────────────────────────────────────
- **數值地板**：同一張圖、同一方向，換取樣鏈種子重跑，比值的散布。
- **可控性**：比值對 ΔE00 有沒有反應。
- **方向依賴**：多個獨立方向上的比值是否一致。
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

COLUMNS = ['image', 'purifier', 'step_index', 'timestep', 'h', 'direction',
           'chain_seed', 'energy_ref', 'energy_def', 'ratio', 'sigma_e']


def purify(x, spec):
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


PURIFIERS = [{'kind': 'identity'},
             {'kind': 'jpeg', 'quality': 75},
             {'kind': 'blur', 'sigma': 1.0},
             {'kind': 'crop_resize', 'fraction': 0.10}]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--reference', type=Path, required=True,
                    help='分母那一張：未防禦的原圖')
    ap.add_argument('--images', type=Path, required=True,
                    help='分子那些張所在的目錄，逐張掃 *.png')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--steps', type=int, default=50)
    ap.add_argument('--step-indices', type=int, nargs='+', default=[0, 16, 33, 49])
    ap.add_argument('--h', type=float, nargs='+', default=[0.0625, 0.125])
    ap.add_argument('--directions', type=int, default=16)
    ap.add_argument('--main-directions', type=int, default=1)
    ap.add_argument('--chain-seeds', type=int, nargs='+',
                    default=[20260812, 20260813, 20260814])
    ap.add_argument('--direction-seed', type=int, default=7311)
    ap.add_argument('--sweep-images', nargs='+', default=[],
                    help='另外做完整方向掃描的影像檔名（不含副檔名）')
    args = ap.parse_args()

    import torch

    from src.defense.assets import load_image
    from src.defense.conditioning_response import (finite_response,
                                                   null_trajectory,
                                                   probe_directions,
                                                   response_energy,
                                                   scale_latent)
    from src.defense.instruction_free import null_text_embedding
    from src.models.ip2p import IP2PWrapper

    if args.out.exists():
        raise SystemExit(f'{args.out} 已存在；換一個輸出檔名')
    args.out.parent.mkdir(parents=True, exist_ok=True)

    ip2p = IP2PWrapper(dtype=torch.bfloat16)
    device = ip2p.device
    ref = load_image(args.reference, device)
    targets = sorted(args.images.glob('*.png'))
    if not targets:
        raise SystemExit(f'{args.images} 底下沒有 PNG')

    e0 = null_text_embedding(ip2p)
    sigma_e = float(e0.float().pow(2).mean().sqrt())
    dirs = probe_directions(e0, args.directions, args.direction_seed)
    print(f'e₀ {tuple(e0.shape)} dtype {e0.dtype} σ_e {sigma_e:.5f}；'
          f'{len(targets)} 張待測、{len(PURIFIERS)} 道淨化、'
          f'{len(args.step_indices)} 個時刻、{len(args.h)} 個尺度', flush=True)

    sweep_set = {s for s in args.sweep_images}
    missing = sweep_set - {p.stem for p in targets}
    if missing:
        raise SystemExit(f'--sweep-images 指到不存在的影像：{sorted(missing)}')

    rows = []
    for chain_seed in args.chain_seeds:
        first_seed = chain_seed == args.chain_seeds[0]
        traj = null_trajectory(ip2p, ref, args.steps, chain_seed)
        picked = [(i, traj[i]) for i in args.step_indices]
        for pspec in PURIFIERS:
            pname = pspec['kind']
            cond_ref = ip2p.image_latents(purify(ref, pspec)).to(e0.dtype)
            conds = {p.stem: ip2p.image_latents(
                purify(load_image(p, device), pspec)).to(e0.dtype)
                for p in targets}
            extra_ok = first_seed and pname == 'identity' and bool(sweep_set)
            n_dirs = args.directions if extra_ok else args.main_directions
            for si, (t, z, sigma) in picked:
                scaled = scale_latent(z, sigma)
                for h in args.h:
                    for di in range(n_dirs):
                        v = dirs[di:di + 1]
                        e_ref = response_energy(finite_response(
                            ip2p, scaled, t, cond_ref, e0, v, h, sigma_e))
                        for name, cond_def in conds.items():
                            if di >= args.main_directions and name not in sweep_set:
                                continue
                            e_def = response_energy(finite_response(
                                ip2p, scaled, t, cond_def, e0, v, h, sigma_e))
                            rows.append({
                                'image': name, 'purifier': pname,
                                'step_index': si, 'timestep': int(t),
                                'h': h, 'direction': di,
                                'chain_seed': chain_seed,
                                'energy_ref': round(e_ref, 6),
                                'energy_def': round(e_def, 6),
                                'ratio': round(e_def / e_ref, 6),
                                'sigma_e': round(sigma_e, 6)})
                print(f'  seed {chain_seed} · {pname} · step {si} · '
                      f'{len(rows)} 列', flush=True)

    with args.out.open('w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    print(json.dumps({'rows': len(rows), 'out': str(args.out),
                      'sigma_e': sigma_e}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()

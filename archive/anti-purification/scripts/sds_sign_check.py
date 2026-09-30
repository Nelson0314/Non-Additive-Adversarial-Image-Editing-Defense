"""上卡前的 SDS 梯度守門：在真的 IP2P 上驗方向，不看代理。**不含任何指令。**

`src/defense/sds_terms.py` 把 `∂ε_θ/∂z_t` 整項丟掉。丟錯一個符號或一個縮放
不會報錯、不會出 NaN，只會讓最佳化往反方向走——等於在幫攻擊者把圖修回去，
而讀數上看起來只是「這個臂比較弱」。單元測試是在線性的假 UNet 上跑的，
真的 UNet 非線性，所以派工前要在真模型上再驗一次。

驗三件事
────────────────────────────────────────────────────────────────────
一、**兩側方向**：沿梯度上升一步，損失要變大；下降一步，要變小。只看單側會被
    「怎麼動都變大」蒙混過去。
二、**與完整反傳的夾角**：SDS 梯度與 `MainstreamTerms.diffusion` 一路反傳出來的
    梯度取餘弦。夾角為負就表示近似把方向翻掉了，這一臂不能上卡。
三、**值相同**：兩者回報的損失必須是同一個數，否則 CSV 裡兩欄不同量綱。

兩個對照
────────────────────────────────────────────────────────────────────
同一組步長也沿**完整反傳的梯度**與**一個隨機方向**各走一次。完整反傳那一組
是量測本身的對照：它如果也看不出方向，問題在步長或在 bf16 前向的數值雜訊，
不在 SDS。隨機方向那一組是雜訊地板。

回報原始數字，不回傳通過與否——判定留給使用者。
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    import torch

    from src.defense.assets import load_image
    from src.defense.instruction_free import FreeObjective
    from src.defense.mainstream_terms import MainstreamTerms
    from src.defense.sds_terms import SDSDiffusion, directional_check
    from src.models.ip2p import IP2PWrapper

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', type=Path, required=True)
    ap.add_argument('--image', type=str, default='')
    ap.add_argument('--out', type=Path, default=None)
    ap.add_argument('--steps', type=str, default='0.005,0.02,0.05')
    args = ap.parse_args()

    spec = json.loads(args.config.read_text(encoding='utf-8'))
    manifest = json.loads(
        (ROOT / spec['manifest']).read_text(encoding='utf-8'))
    entries = {r['id']: r for r in manifest['images']}
    image = args.image or spec['images'][0]
    x = None

    ip2p = IP2PWrapper(dtype={'fp16': torch.float16, 'fp32': torch.float32,
                              'bf16': torch.bfloat16}[spec.get('precision',
                                                               'bf16')])
    x = load_image(ROOT / entries[image]['path'], ip2p.device)
    free = dict(spec['free'])
    objective = FreeObjective(
        ip2p, x, box=None, k=int(free['timesteps']),
        steps=int(spec.get('attack_steps', 50)),
        seed=int(free['noise_seed']), weights={'id': 0.0},
        chain_steps=int(free.get('chain_steps', 6)),
        grad_steps=int(free.get('grad_steps', 1)),
        resample=False, s_i=float(spec.get('s_i', 1.5)))

    sds = SDSDiffusion(objective)
    exact = MainstreamTerms(objective, x)
    steps = tuple(float(v) for v in args.steps.split(','))

    y = x.clone().detach().requires_grad_(True)
    v_sds = sds(y)
    v_sds.backward()
    g_sds = y.grad.detach().clone().float()

    z = x.clone().detach().requires_grad_(True)
    v_exact = exact.diffusion(z)
    v_exact.backward()
    g_exact = z.grad.detach().clone().float()

    cos = float(torch.nn.functional.cosine_similarity(
        g_sds.flatten()[None], g_exact.flatten()[None]))
    report = {
        'image': image,
        'value_sds': float(v_sds.detach()),
        'value_exact': float(v_exact.detach()),
        'value_gap': float(v_sds.detach()) - float(v_exact.detach()),
        'cosine_sds_vs_exact': round(cos, 6),
        'rms_sds': float(g_sds.pow(2).mean().sqrt()),
        'rms_exact': float(g_exact.pow(2).mean().sqrt()),
        'absmax_sds': float(g_sds.abs().amax()),
        'absmax_exact': float(g_exact.abs().amax()),
        'along_sds': directional_check(sds, x, g_sds, steps=steps),
        'along_exact': directional_check(sds, x, g_exact, steps=steps),
        'along_random': directional_check(
            sds, x, torch.randn_like(g_sds), steps=steps),
    }
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text, flush=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + chr(10), encoding='utf-8')


if __name__ == '__main__':
    main()

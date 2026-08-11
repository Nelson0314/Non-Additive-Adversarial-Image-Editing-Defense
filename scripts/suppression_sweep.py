#!/usr/bin/env python
"""把既有的 φ 放大到指定的注意力抑制水準，看那個工作點的防禦圖與編輯圖。

    python scripts/suppression_sweep.py --phi-root runs/s3t20_pj_merged/apa_pj \\
        --out runs/suppress_sweep --images horse_00 horse_03 woman_03 \\
        --target-suppression 0.8 <run_stage 的旗標>

## 這支腳本要回答什麼

DAYN 在像素 + L∞ ≤ 0.06 上把遮罩內的 c_a 注意力壓掉 70–90%，`edit_lpips`
做到 0.51–0.56。本專案的 `apa_pj` 在 Δ=0.04 的預算下只壓下 5–10%，`edit_lpips`
0.101。兩者的差別**不在損失**（同一條式 (5)），而在失真預算：DAYN 的
`pert_lpips` 是 0.52–0.59，我們的 `fid_lpips` 是 0.13，差四倍以上。

那麼把我們的 φ 放大到同樣的抑制水準會怎樣？**這不需要重新訓練**：射線縮放
只是把方向參數乘上 k，而 `materialize` 已經實作了該參數化的正確縮放方式
（latent 偏移乘 k 後重跑生成，不是影像空間的線性內插）。

本腳本因此只做三件事：對 k 掃描並量抑制、在達標的 k 上渲染 `x_def`、用攻擊方
的設定編輯一次。**沒有任何最佳化**，成本是每張圖數十次前向。

## 為什麼要看圖而不是只看數字

批次 A（τ=0.50，≈ DAYN 的預算）實測的機制是：`x_def` 被破壞到一定程度之後，
img2img 把它當噪聲、直接照 prompt 重畫，於是編輯成功率反而由 67% 升到 100%。
那件事在指標上看不出來（位移很大），只有把圖並排才看得見。故本腳本一律輸出
`x_def` 與編輯結果，並附未防禦的編輯當基線。
"""

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.ca_attention_probe import _mask_and_span  # noqa: E402
from scripts.run_stage import (build_parser, build_resources,  # noqa: E402
                               resolve_thresholds)
from src.experiment.executors import (eval_noise_seed, load_phi,  # noqa: E402
                                      materialize, save_image)
from src.models.attention import (CrossAttentionRecorder,  # noqa: E402
                                  aggregate_token_attention,
                                  masked_attention_l1)


def masked_l1(res, entry, x, mask, span, emb, ts, abar):
    """遮罩內的 c_a 注意力 L1，對取樣到的 timestep 取平均。

    與訓練期、與 `ca_attention_probe` 用同一組遮罩與同一個 c_a 嵌入——換任何
    一項都會讓「抑制了多少」變成另一個量。
    """
    rec = CrossAttentionRecorder(res.sd.unet)
    lat = res.sd.latent_shape(*x.shape[-2:])
    side = int(mask.shape[-1])
    vals = []
    with torch.no_grad():
        z0 = res.sd.encode_image(x)
        for i, t in enumerate(ts):
            n = res.sd.sample_edit_noise(torch.empty(lat, device=res.device),
                                         seed=res.cfg.seed + i)
            zt = abar[t].sqrt() * z0 + (1 - abar[t]).sqrt() * n
            if res.sd.is_inpainting:
                m9, zm = res.sd.mask_latents(x, entry.mask)
                zt = torch.cat([zt, m9.to(zt.dtype), zm.to(zt.dtype)], dim=1)
            with rec:
                res.sd._eps(zt, t, emb)
            att = aggregate_token_attention(rec.maps, span, side=side,
                                            reduce="sum")
            rec.clear()
            vals.append(float(masked_attention_l1(att, mask)))
    return sum(vals) / len(vals)


def edit_once(res, entry, x, seed_idx=0):
    """攻擊方的一次編輯。prompt、噪聲、步數、guidance 全取評測期的值。"""
    emb = res.sd.encode_text(entry.attack_prompt).detach()
    emb_u = res.sd.uncond_prompt().detach()
    lat = res.sd.latent_shape(*x.shape[-2:])
    noise = res.sd.sample_edit_noise(torch.empty(lat, device=res.device),
                                     seed=eval_noise_seed(res, seed_idx))
    with torch.no_grad():
        return res.sd.edit(x, emb, noise, res.cfg.steps,
                           guidance_scale=res.cfg.guidance, emb_uncond=emb_u,
                           mask=entry.mask,
                           strength=(None if entry.mask is not None
                                     else res.cfg.strength))


def main(argv=None) -> int:
    ap = build_parser()
    ap.add_argument("--phi-root", type=Path, required=True,
                    help="條件目錄，其下是 <影像>/phi.pt")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--target-suppression", type=float, default=0.80,
                    help="要達到的抑制比例（0.80 = 壓掉 80%%）")
    ap.add_argument("--k-grid", default="1,2,4,8,16,32,64",
                    help="要掃的縮放係數。抑制隨 k 單調上升，故取最小的達標者")
    ap.add_argument("--n-timesteps", type=int, default=6)
    ap.add_argument("--render-k", default=None,
                    help="改為在明指的 k 上各渲染一次（逗號分隔），不做達標挑選。"
                         "抑制對 k **不是單調的**（實測 horse_00 在 k=16 達到"
                         "38%% 之後回落到 24%%），所以「最小的達標 k」在達不到"
                         "目標時沒有意義，要看的是整條曲線上的幾個點")
    argv = list(sys.argv[1:] if argv is None else argv)
    args, _ = ap.parse_known_args(["eval", "--batch", "sweep", *argv])
    resolve_thresholds(args, verbose=False)

    args.out.mkdir(parents=True, exist_ok=True)
    res = build_resources(args, args.out)
    ks = [float(v) for v in args.k_grid.split(",")]
    rows = []

    for image_id in args.images:
        entry = res.image(image_id)
        mask, span, emb, ts, abar = _mask_and_span(res, entry, args.n_timesteps)
        base = masked_l1(res, entry, entry.x01, mask, span, emb, ts, abar)
        print(f"\n[{image_id}] c_a={entry.content!r}  遮罩覆蓋 "
              f"{float(mask.mean()) * 100:.1f}%  未防禦 L1={base:.2f}", flush=True)

        payload = load_phi(args.phi_root / image_id / "phi.pt")
        if args.render_k:
            save_image(entry.x01, args.out / f"{image_id}__orig.png")
            save_image(edit_once(res, entry, entry.x01),
                       args.out / f"{image_id}__edit_undefended.png")
            for k in [float(v) for v in args.render_k.split(",")]:
                x = materialize(payload, res, entry, k)
                l1 = masked_l1(res, entry, x, mask, span, emb, ts, abar)
                fid = res.suite.pairwise(entry.x01, x)
                sup = 1 - l1 / base
                save_image(x, args.out / f"{image_id}__xdef_k{k:g}.png")
                save_image(edit_once(res, entry, x),
                           args.out / f"{image_id}__edit_k{k:g}.png")
                print(f"  k={k:>5.1f}  抑制 {sup * 100:>5.1f}%  "
                      f"lpips {fid['lpips']:.4f}  銳利度 {fid['acutance_ratio']:.3f}"
                      f"  → 已存", flush=True)
                rows.append({"image_id": image_id, "k": k, "suppression": sup,
                             "masked_l1": l1, "base_l1": base, "rendered": True,
                             **{f"fid_{a}": b for a, b in fid.items()}})
            continue
        picked = None
        for k in ks:
            x = materialize(payload, res, entry, k)
            l1 = masked_l1(res, entry, x, mask, span, emb, ts, abar)
            fid = res.suite.pairwise(entry.x01, x)
            sup = 1 - l1 / base
            print(f"  k={k:>5.1f}  抑制 {sup * 100:>5.1f}%  "
                  f"lpips {fid['lpips']:.4f}  dists {fid['dists']:.4f}  "
                  f"psnr {fid['psnr']:.2f}  銳利度 {fid['acutance_ratio']:.3f}",
                  flush=True)
            rows.append({"image_id": image_id, "k": k, "suppression": sup,
                         "masked_l1": l1, "base_l1": base,
                         **{f"fid_{a}": b for a, b in fid.items()}})
            if picked is None and sup >= args.target_suppression:
                picked = (k, x, fid, sup)
        if picked is None:
            k, x, fid, sup = ks[-1], x, fid, sup
            print(f"  **k 掃到 {ks[-1]:g} 仍未達 "
                  f"{args.target_suppression * 100:.0f}%**，取最大者如實輸出",
                  flush=True)
        else:
            k, x, fid, sup = picked

        save_image(x, args.out / f"{image_id}__xdef_k{k:g}.png")
        save_image(edit_once(res, entry, x), args.out / f"{image_id}__edit_k{k:g}.png")
        save_image(edit_once(res, entry, entry.x01),
                   args.out / f"{image_id}__edit_undefended.png")
        save_image(entry.x01, args.out / f"{image_id}__orig.png")
        print(f"  → 取 k={k:g}（抑制 {sup * 100:.1f}%，lpips {fid['lpips']:.4f}）"
              f"，已存防禦圖與編輯圖", flush=True)

    (args.out / "sweep.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n寫入 {args.out / 'sweep.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

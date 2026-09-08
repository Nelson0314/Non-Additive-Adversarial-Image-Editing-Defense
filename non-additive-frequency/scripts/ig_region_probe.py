"""影像引導的殘差落在畫面的哪一塊：主體 latent 對背景 latent。

問的是什麼
────────────────────────────────────────────────────────────────────
`runs/regional_displacement/` 量到一個形狀：**把主體逐位元凍結的防禦**
（遮罩色彩網格、以及把整個補集換成噪聲的補丁）主體內位移都只有 0.08–0.15，
而相位族與 DCT-Shield 是 0.41–0.67。`runs/ig_probe/` 同時量到那些
凍結主體的條件，`L_ig` 一律停在 0.41 附近——用完全不同機制達到的條件停在
同一個值，看起來像是**約束本身造成的天花板**。

若這個解釋成立，殘差 `‖ε(z_t, c_I, ∅) − ε(z_t, 0, ∅)‖²` 應該由**主體自己的
latent token** 主導；凍結那些 token 就等於凍結了殘差的大部分，於是不論在
背景做什麼都壓不下去。

本探針把同一組固定抽樣的殘差按遮罩拆成兩塊直接驗證。走
`make_fixed(...)(x, weight)` 的同一條路徑，故與 `runs/ig_probe/` 的全圖值
是同一條軸（`weight=None` 時逐位元相同）。

用法：

    python scripts/ig_region_probe.py --out runs/ig_probe/by_region.csv
        --ig-zt diffuse_src --images task_attr_mod_color_11699
        --entry color_masked_r020=runs/ip2p_color_masked/m_r020_plant
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402

from src.defense.image_guidance_loss import ZT_MODES, make_image_guidance_loss  # noqa: E402
from src.utils.io import load_image_tensor, write_csv  # noqa: E402

import ig_probe  # noqa: E402

RESOLUTION = 512


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--entry", action="append", required=True,
                    metavar="LABEL=DIR")
    ap.add_argument("--images", nargs="+", required=True)
    ap.add_argument("--data", type=Path, default=Path("data/omniedit150"))
    ap.add_argument("--catalogue", type=Path,
                    default=Path("data/decoy_catalogue.yaml"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--ig-zt", choices=ZT_MODES, default=None)
    ap.add_argument("--ig-t-min", type=int, default=1)
    ap.add_argument("--ig-t-max", type=int, default=1000)
    ap.add_argument("--eval-draws", type=int, default=8)
    ap.add_argument("--eval-seed", type=int, default=99991)
    ap.add_argument("--subject-mask-threshold", type=float, default=0.30)
    ap.add_argument("--subject-mask-dilate", type=int, default=16)
    ap.add_argument("--subject-mask-feather", type=int, default=24)
    return ap


def main() -> None:
    args = build_parser().parse_args()
    ig_probe.check_args(args)
    if not args.catalogue.exists():
        raise SystemExit(f"找不到主體名詞目錄 {args.catalogue}")

    import yaml

    from src.defense.subject_mask import mask_stats, subject_mask
    from src.models.ip2p import IP2PWrapper

    objects = (yaml.safe_load(
        args.catalogue.read_text(encoding="utf-8")) or {}).get("objects") or {}

    ip2p = IP2PWrapper(dtype=torch.float32)
    dev = ip2p.device

    rows = []
    for name in args.images:
        if name not in objects:
            raise SystemExit(f"{args.catalogue} 的 objects 裡沒有 {name}")
        x = load_image_tensor(args.data / name / f"{name}.png", dev,
                              size=RESOLUTION)
        texts = objects[name]
        texts = [texts] if isinstance(texts, str) else list(texts)
        m = subject_mask(x, texts, threshold=args.subject_mask_threshold,
                         dilate=args.subject_mask_dilate,
                         feather=args.subject_mask_feather)
        stats = mask_stats(m)

        loss = make_image_guidance_loss(
            ip2p, zt_mode=args.ig_zt, x_clean=x,
            t_min=args.ig_t_min, t_max=args.ig_t_max)
        fixed = loss.make_fixed(args.eval_draws, args.eval_seed)

        def three(img):
            with torch.no_grad():
                return (float(fixed(img)), float(fixed(img, m)),
                        float(fixed(img, 1.0 - m)))

        base = three(x)
        common = {"image": name, "subject_text": " | ".join(texts), **stats,
                  "ig_zt": args.ig_zt, "ig_t_min": args.ig_t_min,
                  "ig_t_max": args.ig_t_max, "eval_draws": args.eval_draws,
                  "eval_seed": args.eval_seed}

        def record(label, cond, vals, extra=None):
            full, sub, bg = vals
            rows.append({
                **common, "label": label, "condition": cond,
                "ig_full": round(full, 8), "ig_subject": round(sub, 8),
                "ig_background": round(bg, 8),
                "ig_full_ratio": round(full / base[0], 6),
                "ig_subject_ratio": round(sub / base[1], 6),
                "ig_background_ratio": round(bg / base[2], 6),
                # 主體那一塊占全圖殘差的比重。原圖上的值就是「約束綁住了
                # 多少」——凍結主體等於凍結這一部分。
                "subject_share": round(sub * stats["mask_mean"] / full, 6),
                **(extra or {}),
            })
            print(f"{name[15:]:24s}{label + '/' + cond:26s}"
                  f"全 {full:.6f} ({full / base[0]:.3f}×)  "
                  f"主體 {sub:.6f} ({sub / base[1]:.3f}×)  "
                  f"背景 {bg:.6f} ({bg / base[2]:.3f}×)", flush=True)
            write_csv(args.out, rows)

        record("original", "", base)
        for spec in args.entry:
            label, _, d = spec.partition("=")
            for cond, png, r in ig_probe.defense_images(Path(d), name):
                xd = load_image_tensor(png, dev, size=RESOLUTION)
                record(label, cond, three(xd),
                       {"fid_dists": r.get("fid_dists", ""),
                        "edit_lpips": r.get("edit_lpips", "")})

    write_csv(args.out, rows)
    print(f"\n寫出 {args.out}（{len(rows)} 列）")


if __name__ == "__main__":
    main()

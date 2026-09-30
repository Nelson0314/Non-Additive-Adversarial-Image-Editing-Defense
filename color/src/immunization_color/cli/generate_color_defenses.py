"""color：CIELAB 色度平面的全域映射，以等變殘差目標對 ip2p 最佳化。

載體（全域映射，輸出只依賴該像素自己的顏色，與位置無關）

    (a, b) → (a, b) + d(a, b)     d 為 7×7 個錨點位移的 Gaussian RBF 單位分割內插，|d| ≤ warp_radius
    L      → 100 · F(L/100)       F 為 16 段單調分段線性曲線

目標（等變殘差）

    comm(θ; ξ) = LPIPS( N_ξ(T_θ(x)), T_θ(N_ξ(x)) )
    score(θ)   = − w_comm · comm / c0

N_ξ 為無文字的 ip2p 短鏈（`FreeObjective.null_edit`，6 步、最後 1 步回傳梯度、s_i 1.5），噪聲每步重抽。
編輯器對 T 等變時 comm = 0；comm 量的是色調穿過編輯之後剩下的差。c0 為起點在固定驗證抽樣上的值，
w_comm 使起點梯度範數等於 FreeObjective。恆等起點的 comm 與梯度皆為 0，因此用非恆等起點。

上限（增廣 Lagrange，可行性在量化後的 PNG 上檢查）

| 上限 | 值 |
|---|---|
| 整圖 ΔE00 | ≤ 32 |
| 臉框 ΔE00、與原膚色同色像素的 ΔE00 | ≤ 16 |
| 彩度 p95 | ≤ 原圖 × 2.0 |
| 逐像素 Lab 位移（p95 與精確最大值） | a*＋ ≤ 4、a*－ ≤ 15、b*＋ ≤ 4、b*－ ≤ 25、\\|ΔL*\\| ≤ 15 |
| 對原圖的 LPIPS | ≤ `data/color_lpips_reference.csv` 的逐張值 ＋ 0.0025 |

產出：{out}/{image}__orig.png、{image}__color__def.png、{image}__warp.png、{image}__carrier.pt、results.csv
"""


from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import torch

from immunization_color import layout
from immunization_color.method import (
    ARM, CHANNEL_CAPS, RESOLUTION, ColorMap, CommObjective, XAttnScore, _grad_norm, align_weight,
    box_support, chroma_p95, expanded_box, skin_center, skin_color_support, skin_scaled_box,
    warp_picture,
)
from immunization_core.artifacts.images import save_image
from immunization_core.color.difference import delta_e00, delta_e00_torch
from immunization_core.color.shift import channel_shift_max, channel_shift_p95
from immunization_core.color.uniformity import lab_offset, tv_offset
from immunization_core.editors.instruct_pix2pix import IP2PWrapper
from immunization_core.io import load_dataset_images, load_image_tensor, write_sorted_csv
from immunization_core.metrics.identity import face_boxes
from immunization_core.optimization.carrier import Cap, optimize_carrier, quantize
from immunization_core.optimization.instruction_free import FreeObjective


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output-dir", dest="out", type=Path, required=True)
    ap.add_argument("--data-root", dest="data", type=Path, default=layout.PORTRAITS)
    ap.add_argument("--images", nargs="+", default=None)
    ap.add_argument("--lpips-ref-csv", dest="lpips_ref", type=Path, default=layout.DATA / "color_lpips_reference.csv")
    ap.add_argument("--lpips-tolerance", type=float, default=0.0025)
    ap.add_argument("--steps", type=int, default=900)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--noise-seed", type=int, default=0)
    ap.add_argument("--arm", default=ARM, help="產出檔名與 CSV 的臂名")
    ap.add_argument("--objective", default="comm", choices=("comm", "xattn"),
                    help="comm：等變殘差；xattn：降低攻擊端對類別詞（man／woman）的整圖交叉注意力")
    ap.add_argument("--lr-final-ratio", type=float, default=0.2)
    ap.add_argument("--lam-every", type=int, default=5)
    ap.add_argument("--caps", default="full", choices=("full", "simple"),
                    help="full：現行 15 道上限；simple：五個方向的色偏上限改為錨點方框（構造保證）、"
                         "亮度斜率範圍 1.35（|ΔL| ≤ 15），只保留膚色同色 ΔE00 ≤ 16 與暖色（a*＋、b*＋）逐像素最大值 3 道")
    ap.add_argument("--carrier", default="ab", choices=("ab",),
                    help="ab：(a,b) RBF 位移 ＋ 亮度曲線")
    ap.add_argument("--box", default="uniform", choices=("uniform", "skin"),
                    help="simple 的方框：uniform 各錨點相同；skin 冷色方向依離膚色中心距離放大到 2 倍")
    return ap


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)

    caps_cfg = {"frame": 32.0, "face": 16.0, "skin_radius": 12.0, "chroma_gain": 2.0,
                "shift": {"a_pos": 4, "a_neg": 15, "b_pos": 4, "b_neg": 25, "l_abs": 15}}
    lpips_ref = {r["image"]: float(r["lpips_ref"]) for r in csv.DictReader(args.lpips_ref.open(encoding="utf-8"))}

    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ip2p = IP2PWrapper(dtype=torch.float32)
    import piq
    lp = piq.LPIPS().to(device).eval()
    lp.requires_grad_(False)

    rows = []
    for item in load_dataset_images(args.data, set(args.images) if args.images else None):
        started = time.time()
        name = item["name"]
        if name not in lpips_ref:
            raise SystemExit(f"{args.lpips_ref} 沒有 {name} 的 LPIPS 參考值")
        x = load_image_tensor(item["path"], device, size=RESOLUTION)
        boxes = face_boxes(x, device)
        if not boxes:
            raise SystemExit(f"{name} 偵測不到臉；膚色群心定不出來")
        box = expanded_box(x, max(boxes, key=lambda q: (q[2] - q[0]) * (q[3] - q[1])))
        frame, face = torch.ones_like(x[:, :1]), box_support(x, box)
        skin = skin_color_support(x, face, caps_cfg["skin_radius"])
        c95 = float(chroma_p95(x))

        simple = args.caps == "simple"
        # simple：斜率範圍 1.35 使 |F(L) − L| ≤ 0.15（任意 16 段單調曲線的最壞情形）
        carrier = ColorMap(l_radius=0.35 if simple else 0.6)
        carrier.reset(x, args.noise_seed)
        g = torch.Generator(device="cpu").manual_seed(args.noise_seed)
        with torch.no_grad():  # 非恆等起點：恆等映射下 comm 與其梯度皆為 0
            carrier.w_raw.copy_((0.1 * torch.randn(carrier.w_raw.shape, generator=g)).to(device))
            carrier.th_raw.copy_(((torch.rand(carrier.th_raw.shape, generator=g) * 2 - 1) * 0.1).to(device))
        sh = caps_cfg["shift"]
        if simple:
            if args.box == "skin":
                carrier.set_box(*skin_scaled_box(carrier.anchors, skin_center(x, face), sh))
            else:
                carrier.set_box(-sh["a_neg"], sh["a_pos"], -sh["b_neg"], sh["b_pos"])
        carrier.project()

        caps = [
            Cap("frame", lambda y: delta_e00_torch(x, y, frame), lambda y: delta_e00(x, y, frame), caps_cfg["frame"]),
            Cap("face_box", lambda y: delta_e00_torch(x, y, face), lambda y: delta_e00(x, y, face), caps_cfg["face"]),
            Cap("skin_color", lambda y: delta_e00_torch(x, y, skin), lambda y: delta_e00(x, y, skin), caps_cfg["face"]),
            *[Cap(f"shift_{n}", lambda y, c=c, s=s: channel_shift_p95(x, y, c, s),
                  lambda y, c=c, s=s: float(channel_shift_p95(x, y, c, s)), sh[n]) for n, c, s in CHANNEL_CAPS],
            *[Cap(f"shift_{n}_max", lambda y, c=c, s=s: channel_shift_max(x, y, c, s),
                  lambda y, c=c, s=s: float(channel_shift_max(x, y, c, s)), sh[n]) for n, c, s in CHANNEL_CAPS],
            Cap("chroma_p95", lambda y: chroma_p95(y), lambda y: float(chroma_p95(y)), c95 * caps_cfg["chroma_gain"]),
            Cap("input_lpips", lambda y: lp(y, x).mean(), lambda y: float(lp(y, x).mean()),
                lpips_ref[name] + args.lpips_tolerance),
        ]
        if simple:
            # 方框只管映射本身；轉回 RGB 的色域裁切仍可能產生暖色位移，暖色兩個方向保留逐像素最大值
            caps = [c for c in caps if c.name in ("skin_color", "shift_a_pos_max", "shift_b_pos_max")]
            if args.objective == "xattn":
                # 注意力目標的權重把解推到暖色上限邊界，量化後超出 0.4–0.5；訓練端留 0.5 餘量，可行性仍以 4 檢查
                caps = [c._replace(soft=lambda y, f=c.soft: f(y) + 0.5) if c.name.startswith("shift_") else c
                        for c in caps]

        free = FreeObjective(ip2p, x, box=box, k=4, steps=50, seed=args.noise_seed,
                             weights={"id": 1.0, "enc": 0.5, "cond": 1.0},
                             chain_steps=6, grad_steps=1, s_i=1.5, resample=True)
        if args.objective == "comm":
            objective = CommObjective(free, carrier, x, lp)
            extra = {k: round(v, 6) for k, v in align_weight(objective, free, carrier, x).items()}
        else:
            objective = XAttnScore(ip2p, x, item["class"], args.noise_seed)
            g_obj = _grad_norm(carrier, lambda: objective.eval_score(carrier.render(x)))
            g_free = _grad_norm(carrier, lambda: free.eval_score(carrier.render(x)))
            objective.w = g_free / g_obj  # 起點梯度範數對齊 FreeObjective
            extra = {"xattn_weight": round(objective.w, 6), "class_word": item["class"]}
        term = args.objective
        extra.update(objective=args.objective, caps_mode=args.caps, box=args.box if simple else "", carrier=carrier.name)
        with torch.no_grad():
            extra[f"{term}_val_start"] = round(float(objective.eval_terms(carrier.render(x))[term]), 6)

        stats = optimize_carrier(carrier, x, objective, steps=args.steps, lr=args.lr, caps=caps,
                                 rho=10.0, lam_every=args.lam_every, check_every=10, log_every=100,
                                 lr_final_ratio=args.lr_final_ratio, probe_every=50)
        # optimize_carrier 的回傳鍵一律以 free_ 開頭
        if stats.get("free_feasible_step", -1) == -1:
            raise SystemExit(f"{name}：{args.steps} 步內找不到可行 checkpoint"
                             f"（違反：{stats.get('free_caps_unprojected')}），拒絕存檔")

        with torch.no_grad():
            y = quantize(carrier.render(x))
            off = lab_offset(x, y)
            row = {
                "image": name, "class": item["class"], "arm": args.arm, "solver_steps": args.steps, "lr": args.lr,
                "lpips_cap": round(lpips_ref[name] + args.lpips_tolerance, 6), "lpips_out": round(float(lp(y, x).mean()), 6),
                f"{term}_val_end": round(float(objective.eval_terms(y)[term]), 6),
                "chroma_p95_orig": round(c95, 4), "chroma_p95_out": round(float(chroma_p95(y)), 4),
                **{f"shift_{n}_p95": round(float(channel_shift_p95(x, y, c, s)), 4) for n, c, s in CHANNEL_CAPS},
                **{f"shift_{n}_max_out": round(float(channel_shift_max(x, y, c, s)), 4) for n, c, s in CHANNEL_CAPS},
                "delta_e00_frame": round(float(delta_e00(x, y, frame)), 4),
                "delta_e00_face_box": round(float(delta_e00(x, y, face)), 4),
                "delta_e00_skin_color": round(float(delta_e00(x, y, skin)), 4),
                "tv_frame": round(float(tv_offset(off, frame)), 5),
                "warp_max": round(float(carrier.w.norm(dim=-1).max()), 3),
                "psnr": round(float(10 * torch.log10(1.0 / (y - x).pow(2).mean())), 4),
                "seconds": round(time.time() - started, 1), **extra, **stats,
            }
        save_image(x, args.out / f"{name}__orig.png")
        save_image(y, args.out / f"{name}__{args.arm}__def.png")
        torch.save(carrier.state_dict(), args.out / f"{name}__carrier.pt")
        save_image(warp_picture(carrier, device), args.out / f"{name}__warp.png")
        rows.append(row)
        write_sorted_csv(args.out / "results.csv", rows)
        print(f"[{args.arm}] {name}  ΔE frame {row['delta_e00_frame']} face {row['delta_e00_face_box']} "
              f"LPIPS {row['lpips_out']}/{row['lpips_cap']}  {row['seconds']}s", flush=True)
    print(f"完成：{len(rows)} 張 → {args.out}", flush=True)


if __name__ == "__main__":
    main()

"""ip2p 風格編輯當載體（SPA，Wang et al., Neurocomputing 2026 的移植）。

x_ref = G(x, e_txt)、x_def = G(x, e_adv; z_off)。G 與攻擊端同一個 ip2p 權重，DDIM `steps` 步：
前 `warm` 步 eta=1 且不回傳梯度，其後 eta=0 並回傳梯度；兩條軌跡共用起始噪聲與暖身噪聲。
可調的是指令 token（含第一個 EOS）的 embedding 與暖身結束時的 latent 位移。

訓練目標一律不使用任何攻擊指令（評估指令或自選指令皆不用），文字只用空字串或類別詞：
`free`（空指令 ip2p 代理，錨在 x_ref）、`enc_gray`（攻擊端影像條件 E(y) 推向灰圖 latent）、
`attn`（攻擊端 UNet 自注意力圖相對 x_ref 的偏離，最大化）、`xattn`（攻擊端對類別詞的交叉注意力，
最小化）、`chaos`（攻擊端以類別詞為指令的短程編輯輸出，最大化其與輸入的結構差異）、
`classifier`（SPA 原損失：ResNet-50 對防禦圖的指數邊界損失，標籤取原圖 top-1）。
`--sampler ddpm --prompt-scope full --select last` 為 SPA 論文設定（附錄 B、整段 77 token、取最後一步）。
身分以 FaceNet 餘弦下限 τ = max(id_floor, cos(x_ref, x) − id_margin) 約束；`--struct-cap` 以灰階
LPIPS 限制防禦圖相對 x_ref 的結構改動。`--freeze-warm` 讓暖身段固定用原風格指令，梯度不再被截斷。
`--updates 0` 只產參考圖，即風格指令的無最佳化預覽。
"""


from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from immunization_core.artifacts.images import save_image
from immunization_core.editors.instruct_pix2pix import IP2PWrapper
from immunization_core.io import load_dataset_images, load_image_tensor, write_sorted_csv
from immunization_core.metrics.identity import embed_box, embed_box_differentiable, face_boxes
from immunization_core.optimization.attention import encode_text
from immunization_core.optimization.carrier import quantize
from immunization_style import layout
from immunization_style.method import (
    RESOLUTION, STYLES, StyleEditor, optimize, psnr, select_result,
)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output-dir", dest="out", type=Path, required=True)
    ap.add_argument("--data-root", dest="data", type=Path, default=layout.PORTRAITS)
    ap.add_argument("--images", nargs="+", default=None)
    ap.add_argument("--styles", nargs="+", default=list(STYLES), choices=list(STYLES))
    ap.add_argument("--s-i", type=float, nargs="+", default=[2.0])
    ap.add_argument("--s-t", type=float, default=7.5)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--warm", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--updates", type=int, default=0)
    ap.add_argument("--objective", default="free", choices=("free", "enc_gray", "attn", "xattn", "chaos",
                                                                    "classifier"))
    ap.add_argument("--cls-arch", default="resnet50", help="classifier：torchvision 模型名（論文 §4.1）")
    ap.add_argument("--cls-weights", default="IMAGENET1K_V1", help="classifier：權重名或 default")
    ap.add_argument("--cls-tau", type=float, default=1.0, help="classifier：邊界縮放 τ（論文 1.0）")
    ap.add_argument("--cls-kappa", type=float, default=9.0, help="classifier：飽和常數 κ（論文 9.0）")
    ap.add_argument("--class-word", default="", help="xattn／chaos：攻擊端的文字條件（類別詞，如 man、woman）")
    ap.add_argument("--chaos-steps", type=int, default=10, help="chaos：攻擊端短程編輯的 DDIM 步數")
    ap.add_argument("--chaos-val-k", type=int, default=2, help="chaos：固定驗證用幾組起始噪聲")
    ap.add_argument("--attack-s-i", type=float, default=1.8, help="chaos：攻擊端的影像引導強度")
    ap.add_argument("--freeze-warm", action="store_true", help="暖身段固定用原風格指令跑一次並快取")
    ap.add_argument("--sampler", default="ddim", choices=("ddim", "ddpm"), help="ddpm：SPA 附錄 B 的 (B.5)／(B.6)")
    ap.add_argument("--prompt-scope", default="instr", choices=("instr", "full"))
    ap.add_argument("--select", default="best", choices=("best", "last"),
                    help="best：可行步中目標最佳者；last：最後一步（論文）")
    ap.add_argument("--struct-cap", type=float, default=0.0, help="> 0 時防禦圖對 x_ref 的灰階 LPIPS 上限")
    ap.add_argument("--a-struct", type=float, default=20.0)
    ap.add_argument("--carrier", default="prompt", choices=("prompt", "latent", "prompt_latent"))
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--latent-lr", type=float, default=0.02)
    ap.add_argument("--latent-rms-cap", type=float, default=0.3)
    ap.add_argument("--init-rms", type=float, default=0.01)
    ap.add_argument("--delta-rms-cap", type=float, default=0.10)
    ap.add_argument("--a-perc", type=float, default=25.0)
    ap.add_argument("--a-adv", type=float, default=1.0)
    ap.add_argument("--a-prompt", type=float, default=1.0)
    ap.add_argument("--a-id", type=float, default=20.0)
    ap.add_argument("--w-enc", type=float, default=0.1)
    ap.add_argument("--id-floor", type=float, default=0.80)
    ap.add_argument("--id-margin", type=float, default=0.03)
    ap.add_argument("--bg-cap", type=float, default=0.0, help="> 0 時背景對 x_ref 的 LPIPS 上限")
    ap.add_argument("--a-bg", type=float, default=20.0)
    ap.add_argument("--col-cap", type=float, default=0.0, help="> 0 時限制相對 x_ref 往 a*＋／b*＋ 的位移")
    ap.add_argument("--col-face-only", action="store_true", help="色偏上限只在臉框內量")
    ap.add_argument("--a-col", type=float, default=0.05)
    ap.add_argument("--attn-k", type=int, default=2, help="attn：每步抽幾組 (t, ε)")
    ap.add_argument("--attn-val-k", type=int, default=4, help="attn：固定驗證用幾組 (t, ε)")
    ap.add_argument("--attn-tmin", type=int, default=100)
    ap.add_argument("--attn-tmax", type=int, default=900)
    ap.add_argument("--val-seed", type=int, default=777)
    ap.add_argument("--val-every", type=int, default=10)
    ap.add_argument("--patience", type=int, default=15, help="停滯判定：連續幾次評估沒有改善 1%%")
    ap.add_argument("--max-decays", type=int, default=2, help="停滯時最多降 lr 幾次（每次 /4）後停止")
    ap.add_argument("--snapshot-every", type=int, default=10)
    ap.add_argument("--noise-seed", type=int, default=0)
    ap.add_argument("--tf32", action="store_true")
    return ap


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    if args.objective in ("xattn", "chaos") and not args.class_word:
        raise SystemExit(f"--objective {args.objective} 需要 --class-word")

    args.out.mkdir(parents=True, exist_ok=True)
    if args.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    ip2p = IP2PWrapper(dtype=torch.float32)
    dev = ip2p.device
    import piq
    lp = piq.LPIPS().to(dev).eval()
    lp.requires_grad_(False)
    items = load_dataset_images(args.data, set(args.images) if args.images else None)

    rows, trace = [], []
    for item in items:
        name = item["name"]
        x = load_image_tensor(item["path"], dev, size=RESOLUTION)
        boxes = face_boxes(x, dev)
        if not boxes:
            raise SystemExit(f"{name} 偵測不到臉，身分下限無從定義")
        box = max(boxes, key=lambda q: (q[2] - q[0]) * (q[3] - q[1]))
        e0 = embed_box(x, box, dev).float()
        mask_png = args.data / "masks" / f"{name}.png"
        if not mask_png.is_file():
            raise SystemExit(f"找不到背景遮罩 {mask_png}")
        bg = (load_image_tensor(mask_png, dev, size=RESOLUTION)[:, :1] >= 0.5).float()
        face = torch.zeros_like(bg)
        fx0, fy0, fx1, fy1 = (int(round(v)) for v in box)
        face[..., max(0, fy0):fy1, max(0, fx0):fx1] = 1.0

        def id_cos(y):
            return float(F.cosine_similarity(embed_box(y, box, dev).float()[None], e0[None]))

        def id_cos_diff(y):
            return F.cosine_similarity(embed_box_differentiable(y, box, dev).float()[None],
                                       e0[None]).squeeze()

        save_image(x, args.out / f"{name}__orig.png")
        with torch.no_grad():
            v = quantize(ip2p.decode_latent(ip2p.encode_image(x)).float())
        save_image(v, args.out / f"{name}__vae.png")
        rows.append({"image": name, "style": "vae_roundtrip", "prompt": "", "s_i": "",
                     "id_ref": round(id_cos(v), 5),
                     "lpips_ref_x": round(float(lp(v, x).mean()), 5),
                     "psnr_ref_x": round(psnr(v, x), 3)})

        for style in args.styles:
            ids, e_txt = encode_text(ip2p, STYLES[style])
            for s_i in args.s_i:
                started = time.time()
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                tag = f"{name}__{style}_si{s_i:g}".replace(".", "p")
                ed = StyleEditor(ip2p, x, steps=args.steps, warm=args.warm,
                                 s_t=args.s_t, s_i=s_i, seed=args.seed, sampler=args.sampler)
                if args.freeze_warm:
                    ed.freeze_warm(e_txt)
                with torch.no_grad():
                    y_ref, x0_ref = ed.run(e_txt, grad=False)
                    x_ref = quantize(y_ref)
                save_image(x_ref, args.out / f"{tag}__ref.png")
                id_ref = id_cos(x_ref)
                row = {"image": name, "style": style, "prompt": STYLES[style], "s_i": s_i,
                       "s_t": args.s_t, "steps": args.steps, "warm": args.warm, "seed": args.seed,
                       "id_ref": round(id_ref, 5),
                       "lpips_ref_x": round(float(lp(x_ref, x).mean()), 5),
                       "psnr_ref_x": round(psnr(x_ref, x), 3)}
                if args.updates > 0:
                    tau = max(args.id_floor, id_ref - args.id_margin)
                    best, last, extra = optimize(
                        args, ip2p, ed, x_ref, e_txt, ids, x0_ref, id_cos, id_cos_diff, tau, lp,
                        trace, tag, bg=bg, face=face if args.col_face_only else None, x_orig=x)
                    (u, yd, rec), selection = select_result(best, last, args.select)
                    # last 政策仍輸出所選的最後一步，限制判定由 selection 記錄。
                    suffix = "def" if args.select == "last" or best is not None else "def_infeasible"
                    save_image(yd, args.out / f"{tag}__{suffix}.png")
                    row.update(extra)
                    row.update({
                        "updates": args.updates, "lr": args.lr, "delta_rms_cap": args.delta_rms_cap,
                        "a_perc": args.a_perc, "a_adv": args.a_adv, "a_prompt": args.a_prompt,
                        "a_id": args.a_id, "w_enc": args.w_enc, "tau": round(tau, 5),
                        **selection, "chosen_update": u,
                        "last_update": last[0], "stop": last[2].get("stop", ""),
                        "id_def": rec["id_cos"],
                        "lpips_def_x": round(float(lp(yd, x).mean()), 5),
                        "lpips_def_ref": round(rec["lpips_def_ref"], 5),
                        "psnr_def_x": round(psnr(yd, x), 3),
                        "psnr_def_ref": round(psnr(yd, x_ref), 3),
                        "l_adv": round(rec["l_adv"], 5), "l_perc": round(rec["l_perc"], 6),
                        **{k: round(rec[k], 5) for k in ("val_l_adv", "enc", "cond") if k in rec},
                        **{k: rec[k] for k in ("cls_margin", "cls_top1", "cls_p_true") if k in rec},
                        "tok_cos": round(rec["tok_cos"], 5), "delta_rms": round(rec["delta_rms"], 5)})
                row["seconds"] = round(time.time() - started, 1)
                if torch.cuda.is_available():
                    row["peak_mem_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
                rows.append(row)
                write_sorted_csv(args.out / "results.csv", rows)
                print(f"[{tag}] id_ref {row['id_ref']} lpips_ref_x {row['lpips_ref_x']}"
                      + (f"  id_def {row['id_def']} lpips_def_ref {row['lpips_def_ref']} "
                         f"feasible {row['feasible']}" if args.updates > 0 else "")
                      + f"  {row['seconds']}s", flush=True)
    print(f"完成：{len(rows)} 列 → {args.out}", flush=True)


if __name__ == "__main__":
    main()

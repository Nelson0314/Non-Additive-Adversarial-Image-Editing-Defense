"""以現成生成編輯模型的風格／色彩轉換作為免疫載體，產生防禦圖。

載體
────────────────────────────────────────────────────────────────────
防禦圖不是「原圖 ＋ 擾動」，而是**現成 Stable Diffusion v1.5 對原圖做一次
SDEdit（img2img）之後的輸出**：

    x_def = SDEdit(x ; emb, noise(seed), num_steps, strength, w)

`emb` 是文字條件的 CLIP embedding（77×768）。`src.models.sd.SDWrapper.sdedit`
整條取樣鏈可微分（`use_ckpt` / `vae_ckpt` 控制 UNet 與 VAE 的 gradient
checkpointing），所以 `emb` 可以當作求解變數，梯度穿過整條鏈回到它身上。

兩個臂
────────────────────────────────────────────────────────────────────
| 臂 | `emb` 怎麼來 | 這一臂回答什麼 |
|---|---|---|
| `style_random` | `encode_text(style_prompt)`，不最佳化 | 現成模型的風格轉換本身推得動多少 |
| `style_opt` | `encode_text(style_prompt) + δ`，最佳化 `δ` | 最佳化相對於未最佳化買到了什麼 |

兩臂共用同一顆 `--seed` 的噪聲、同一個 `strength`、同一個 `num_steps`，
差別只有 `δ`。缺 `style_random` 時整組讀數無法解讀——本專案已分別量到過
「最佳化贏隨機」與「臂間打平」兩種情形。

求解目標
────────────────────────────────────────────────────────────────────
威脅模型裡防禦方看不到攻擊指令，故目標不得含編輯指令。可選三種代理
（`--objective`），全部對受害模型族共用的 SD v1.x VAE／UNet：

| 值 | `L_def` | 出處形式 |
|---|---|---|
| `encoder_away` | `‖E(x_def) − E(x)‖² / n` | PhotoGuard 的 encoder attack，無目標版 |
| `encoder_target` | `−‖E(x_def) − z_tgt‖² / n` | PhotoGuard 的 encoder attack，有目標版 |
| `eps_error` | 空 prompt 下 `‖ε_θ(z_t, t) − ε‖² / n` 的時間平均 | 擴散重建誤差 |

**代理推得動不等於編輯端會動**（本專案已量到四個代理各推 0.003–3.8 倍而
編輯端全部沒垮）。因此本檔只負責產圖與記錄，成立與否由編輯端讀數與逐格
影像判定，不在這裡下結論。

約束
────────────────────────────────────────────────────────────────────
約束進損失，不做事後投影：

    L = −w_def·L_def
        + w_lpips·relu(LPIPS(x_def, x) − lpips_cap)²
        + w_id   ·relu(id_floor − cos_id(x_def, x))²

`lpips_cap` 管結構偏離，`id_floor` 管「防禦圖本身有沒有把人換掉」。後者是
本設計最主要的靜默失效點：防禦圖若換了人，之後的位移讀數很高但沒有意義，
因為分母（未防禦編輯）保護的是另一個人。人臉框由**原圖**偵測一次後固定，
不逐步重偵測。偵不到臉時該項停用，CSV 的 `id_term` 欄記為 `no_face`。

產出（版面與 `defence_run.py` 相同，`edit_preflight.py --defended` 直接吃）
────────────────────────────────────────────────────────────────────
    {out}/{image}__orig.png              原圖（512² 版本）
    {out}/{image}__{arm}__def.png        防禦圖
    {out}/results.csv                    逐列寫入，不是跑完才寫

用法
    python code/style_defence.py --arm style_random --out <目錄>
    python code/style_defence.py --arm style_opt --out <目錄> --images man_00
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import paths  # noqa: E402

paths.add_source_to_syspath()

import torch  # noqa: E402
import yaml  # noqa: E402

from src.metrics import identity as idmod  # noqa: E402
from src.models.sd import SDWrapper  # noqa: E402
from src.utils.artifacts import save_image  # noqa: E402
from src.utils.io import load_image_tensor  # noqa: E402

RESOLUTION = 512
CARRIER_MODEL = "runwayml/stable-diffusion-v1-5"

#: 風格目錄。鍵是臂名的後綴，值是文字條件。
#: 這是**防禦方自己選的**，與攻擊指令屬於不同的人，故可以進求解端。
STYLE_CATALOGUE = {
    "film": ("a vintage cross-processed film photograph, heavy grain, "
             "teal and orange colour grading, faded highlights"),
    "cyan": ("a photograph with a strong cyan-magenta colour filter, "
             "high contrast, cold cinematic grade"),
    "paint": ("an oil painting with thick visible brush strokes, "
              "saturated pigments, canvas texture"),
}


def load_images(root: Path, only):
    spec = yaml.safe_load((root / "prompts.yaml").read_text(encoding="utf-8"))
    out = []
    for cls in sorted(k for k in spec if k != "edits"):
        for img in sorted((root / cls).glob("*.png")):
            if only and img.stem not in only:
                continue
            out.append({"name": img.stem, "class": cls, "path": img,
                        "content": spec[cls]["content"]})
    if not out:
        raise SystemExit(f"{root} 找不到影像（--images 過濾後為空）")
    return out


def face_box(x01: torch.Tensor):
    """原圖的人臉框，偵測一次後固定。偵不到回傳 None。"""
    boxes = idmod.face_boxes(x01)
    if boxes is None or len(boxes) == 0:
        return None
    return boxes[0]


def identity_cos(sd_x: torch.Tensor, x: torch.Tensor, box) -> torch.Tensor:
    a = idmod.embed_box_differentiable(sd_x, box).reshape(1, -1)
    b = idmod.embed_box_differentiable(x, box).reshape(1, -1)
    return torch.nn.functional.cosine_similarity(a, b).mean()


def render(sd: SDWrapper, x01, emb, emb_uncond, noise, args) -> torch.Tensor:
    return sd.sdedit(
        x01, emb, noise,
        num_steps=args.num_steps,
        strength=args.strength,
        guidance_scale=args.guidance,
        emb_uncond=emb_uncond,
        use_ckpt=args.ckpt,
        vae_ckpt=args.ckpt,
    ).clamp(0, 1)


def defence_loss(sd: SDWrapper, x_def, x01, z_src, args):
    """回傳 (L_def, 逐項數值)。`L_def` 越大代表代理上推得越遠。"""
    if args.objective in ("encoder_away", "encoder_target"):
        z = sd.encode_image(x_def, use_ckpt=args.ckpt)
        if args.objective == "encoder_away":
            val = (z - z_src).pow(2).mean()
        else:
            val = -(z - args.target_latent).pow(2).mean()
        return val, {"proxy": float(val.detach())}
    # eps_error：空 prompt 下的重建誤差，對固定的一組 t 取平均
    z = sd.encode_image(x_def, use_ckpt=args.ckpt)
    uncond = sd.uncond_prompt()
    abar = sd.alphas_cumprod(z.device)
    total = 0.0
    g = torch.Generator(device="cpu").manual_seed(args.seed)
    for t_int in args.eps_timesteps:
        t = torch.tensor(int(t_int), device=z.device)
        eps = torch.randn(z.shape, generator=g).to(z.device, z.dtype)
        z_t = abar[t].sqrt() * z + (1 - abar[t]).sqrt() * eps
        pred = sd.unet_forward(z_t, t, uncond)
        total = total + (pred - eps).pow(2).mean()
    val = total / len(args.eps_timesteps)
    return val, {"proxy": float(val.detach())}


def solve_one(sd, lpips_fn, item, args, arm):
    x01 = load_image_tensor(item["path"], sd.device, size=RESOLUTION)
    z_src = sd.encode_image(x01).detach()
    noise = sd.sample_edit_noise(z_src, args.seed)
    base = sd.encode_text(args.style_prompt).detach()
    uncond = sd.uncond_prompt().detach()
    box = face_box(x01)

    row = {"image": item["name"], "class": item["class"], "arm": arm,
           "style": args.style, "style_prompt": args.style_prompt,
           "carrier_model": CARRIER_MODEL, "strength": args.strength,
           "num_steps": args.num_steps, "guidance": args.guidance,
           "seed": args.seed, "objective": args.objective,
           "lpips_cap": args.lpips_cap, "id_floor": args.id_floor,
           "id_term": "active" if box is not None else "no_face"}

    started = time.time()
    if arm.endswith("_opt"):
        delta = torch.zeros_like(base, requires_grad=True)
        opt = torch.optim.Adam([delta], lr=args.lr)
        history = []
        for step in range(args.steps):
            opt.zero_grad(set_to_none=True)
            x_def = render(sd, x01, base + delta, uncond, noise, args)
            proxy, parts = defence_loss(sd, x_def, x01, z_src, args)
            lp = lpips_fn(x_def, x01).mean()
            loss = -args.w_def * proxy
            loss = loss + args.w_lpips * torch.relu(lp - args.lpips_cap).pow(2)
            idv = None
            if box is not None:
                idv = identity_cos(x_def, x01, box)
                loss = loss + args.w_id * torch.relu(args.id_floor - idv).pow(2)
            loss.backward()
            opt.step()
            if step % args.log_every == 0 or step == args.steps - 1:
                history.append({"step": step, "loss": float(loss.detach()),
                                "proxy": parts["proxy"], "lpips": float(lp.detach()),
                                "id": None if idv is None else float(idv.detach())})
                print(f"  [{item['name']}] step {step:4d}  loss {float(loss):.5f}  "
                      f"proxy {parts['proxy']:.5f}  lpips {float(lp):.4f}", flush=True)
        emb = (base + delta).detach()
        row["opt_steps"] = args.steps
        row["opt_lr"] = args.lr
        row["history"] = json.dumps(history)
    else:
        emb = base
        row["opt_steps"] = 0
        row["opt_lr"] = 0.0
        row["history"] = "[]"

    with torch.no_grad():
        x_def = render(sd, x01, emb, uncond, noise, args)
        proxy, _ = defence_loss(sd, x_def, x01, z_src, args)
        row["proxy_final"] = float(proxy)
        row["lpips"] = float(lpips_fn(x_def, x01).mean())
        row["psnr"] = float(10 * torch.log10(1.0 / (x_def - x01).pow(2).mean()))
        row["linf"] = float((x_def - x01).abs().max())
        row["id_cos"] = (float(identity_cos(x_def, x01, box))
                         if box is not None else "")
        # 未防禦的基準 proxy：原圖對自己，供逐列對照
        base_proxy, _ = defence_loss(sd, x01, x01, z_src, args)
        row["proxy_identity"] = float(base_proxy)
    row["seconds"] = round(time.time() - started, 1)
    return x01, x_def, row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", required=True,
                    help="style_random / style_opt（後綴 _opt 才最佳化）")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--data", type=Path, default=paths.PORTRAITS)
    ap.add_argument("--images", nargs="+", default=None)
    ap.add_argument("--style", default="film", choices=sorted(STYLE_CATALOGUE))
    ap.add_argument("--strength", type=float, default=0.35)
    ap.add_argument("--num-steps", type=int, default=10)
    ap.add_argument("--guidance", type=float, default=7.5)
    ap.add_argument("--seed", type=int, default=20260812)
    ap.add_argument("--objective", default="encoder_away",
                    choices=("encoder_away", "encoder_target", "eps_error"))
    ap.add_argument("--steps", type=int, default=200, help="最佳化步數")
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--w-def", type=float, default=1.0)
    ap.add_argument("--w-lpips", type=float, default=50.0)
    ap.add_argument("--w-id", type=float, default=50.0)
    ap.add_argument("--lpips-cap", type=float, default=0.35)
    ap.add_argument("--id-floor", type=float, default=0.60)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--ckpt", action="store_true", default=True)
    ap.add_argument("--no-ckpt", dest="ckpt", action="store_false")
    ap.add_argument("--dtype", default="float16", choices=("float32", "float16"))
    args = ap.parse_args()

    args.style_prompt = STYLE_CATALOGUE[args.style]
    args.eps_timesteps = (200, 400, 600)
    args.target_latent = None

    args.out.mkdir(parents=True, exist_ok=True)
    sd = SDWrapper(CARRIER_MODEL, dtype=getattr(torch, args.dtype))

    import piq
    lpips_fn = piq.LPIPS()

    if args.objective == "encoder_target":
        tgt = paths.TARGETS / "MIST.png"
        t01 = load_image_tensor(tgt, sd.device, size=RESOLUTION)
        args.target_latent = sd.encode_image(t01).detach()

    items = load_images(args.data, set(args.images) if args.images else None)
    csv_path = args.out / "results.csv"
    rows = []
    for item in items:
        print(f"[{args.arm}] {item['name']}", flush=True)
        x01, x_def, row = solve_one(sd, lpips_fn, item, args, args.arm)
        save_image(x01, args.out / f"{item['name']}__orig.png")
        save_image(x_def, args.out / f"{item['name']}__{args.arm}__def.png")
        rows.append(row)
        _write_csv(csv_path, rows)
        print(f"  → lpips {row['lpips']:.4f}  psnr {row['psnr']:.2f}  "
              f"id {row['id_cos']}  {row['seconds']}s", flush=True)
    print(f"完成：{len(rows)} 張 → {args.out}", flush=True)


def _write_csv(path: Path, rows) -> None:
    """逐列重寫，不是跑完才寫——中途被砍掉時已完成的部分要留得下來。"""
    import csv
    keys = sorted({k for r in rows for k in r})
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()

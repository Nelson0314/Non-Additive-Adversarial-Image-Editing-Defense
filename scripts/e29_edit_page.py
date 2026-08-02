"""把 E29 的防禦圖真的拿去被編輯一次，做成人眼比對頁。

為什麼需要這一支。E29 帶 `--no_eval`，只留下防禦圖，沒有編輯結果；而綁定者
診斷表給的是 LPIPS、色度偏壓與 `edit_shift` 這些數字。本專案的既定作法是
指標之間出現矛盾時把影像做成比對頁由人眼定調（E20 的 τ_acut、E28 的
τ_chroma 兩個門檻都是這樣定出來的），現在要判斷的是同一類問題：

- site C 的解被色度 hinge 綁住，色度偏壓 0.76–0.86。**那個色偏看不看得出來。**
- site C 的 LPIPS 只有 site P 的一半，`edit_shift` 卻相當。**防禦是否真的有效。**

`edit_shift` 不是防禦成功的判準（E25 §1：726 格語意失敗 0 格），所以這裡
不報任何新數字，只把圖擺出來。要看的是編輯的內容有沒有照 prompt 發生，
不是兩張圖長不長得一樣——兩次擴散輸出即使都達成 prompt 也本來就不同。

攻擊設定與 E29／E30 一致：`guidance_scale=7.5`、`strength=0.5`、`n_edit=10`。
同一張影像的所有分支共用同一個 ε（`sdedit` 要求呼叫端提供 noise，正是為了
讓這件事成為介面上的硬性要求），否則看到的差異可能只是抽樣不同。

執行：
    python scripts/e29_edit_page.py runs/e29_C_lr0.1 runs/e29_C_lr0.3 runs/e29_P_lr0.03
"""

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from PIL import Image, ImageDraw

from src.models.sd import SDWrapper
from src.utils.device import get_device, tf32_enabled

ROOT = Path(__file__).resolve().parent.parent


def load_png(p: Path, device) -> torch.Tensor:
    import torchvision.transforms as T
    return T.ToTensor()(Image.open(p).convert("RGB")).unsqueeze(0).to(device)


def to_pil(x: torch.Tensor) -> Image.Image:
    import torchvision.transforms as T
    return T.ToPILImage()(x.detach().float().clamp(0, 1).squeeze(0).cpu())


def cells_of(run_dir: Path):
    """回傳 {影像名: (cell 目錄, prompt)}。prompt 取自 summary.csv 而非重新
    查 prompts.yaml：要重現的是這個 run 當初實際用的那一個。"""
    prompts = {}
    sm = run_dir / "summary.csv"
    if sm.exists():
        for r in csv.DictReader(sm.open(encoding="utf-8")):
            prompts[r["image"]] = r["prompt"]
    out = {}
    for d in sorted(run_dir.glob("*__*__r*")):
        if (d / "defended.png").exists():
            name = d.name.split("__")[0]
            out[name] = (d, prompts.get(name, ""))
    return out


def montage(rows, labels, cell=320, pad=6, header=22):
    """rows[i] 是一列 PIL 影像，labels 是欄標題。"""
    ncol = max(len(r) for r in rows)
    W = ncol * cell + (ncol + 1) * pad
    H = len(rows) * cell + (len(rows) + 1) * pad + header
    canvas = Image.new("RGB", (W, H), (128, 128, 128))
    d = ImageDraw.Draw(canvas)
    for j, lab in enumerate(labels):
        d.text((pad + j * (cell + pad) + 4, 5), lab, fill=(255, 255, 255))
    for i, row in enumerate(rows):
        for j, im in enumerate(row):
            canvas.paste(im.resize((cell, cell), Image.LANCZOS),
                         (pad + j * (cell + pad), header + pad + i * (cell + pad)))
    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", default="runs/e29_edit_page")
    ap.add_argument("--model", default="CompVis/stable-diffusion-v1-4")
    ap.add_argument("--n_edit", type=int, default=10)
    ap.add_argument("--strength", type=float, default=0.5)
    ap.add_argument("--guidance_scale", type=float, default=7.5)
    ap.add_argument("--seed", type=int, default=20260728)
    args = ap.parse_args()

    device = get_device()
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    print(f"[edit] device={device} tf32={tf32_enabled()} w={args.guidance_scale}")

    runs = []
    for r in args.runs:
        d = Path(r) if Path(r).is_absolute() else ROOT / r
        runs.append((d.name, cells_of(d)))

    sd = SDWrapper(args.model)
    emb_uncond = sd.encode_text("").detach()

    images = sorted(set().union(*[set(c) for _, c in runs]))
    labels = ["原圖", "原圖被編輯"] + sum(
        [[f"{n} 防禦圖", f"{n} 被編輯"] for n, _ in runs], [])
    rows = []

    for name in images:
        # 同一張影像的所有分支共用同一個 ε，差異才歸因於防禦而非抽樣。
        g = torch.Generator(device="cpu").manual_seed(args.seed)
        prompt = next(p for _, c in runs for k, (_, p) in c.items() if k == name)
        emb = sd.encode_text(prompt).detach()

        first_cell = next(c[name][0] for _, c in runs if name in c)
        x_orig = load_png(first_cell / "orig.png", device)
        lat = sd.latent_shape(x_orig.shape[-1], x_orig.shape[-1])
        noise = torch.randn(lat, generator=g).to(device)

        with torch.no_grad():
            y_orig = sd.sdedit(x_orig, emb, noise, args.n_edit,
                               strength=args.strength,
                               guidance_scale=args.guidance_scale,
                               emb_uncond=emb_uncond)
        row = [to_pil(x_orig), to_pil(y_orig)]
        to_pil(x_orig).save(out / f"{name}_00_orig.png")
        to_pil(y_orig).save(out / f"{name}_01_orig_edited.png")

        for rn, cells in runs:
            if name not in cells:
                continue
            x_def = load_png(cells[name][0] / "defended.png", device)
            with torch.no_grad():
                y_def = sd.sdedit(x_def, emb, noise, args.n_edit,
                                  strength=args.strength,
                                  guidance_scale=args.guidance_scale,
                                  emb_uncond=emb_uncond)
            row += [to_pil(x_def), to_pil(y_def)]
            to_pil(x_def).save(out / f"{name}_{rn}_def.png")
            to_pil(y_def).save(out / f"{name}_{rn}_def_edited.png")
            print(f"  {name}  {rn}  完成", flush=True)

        rows.append(row)
        montage([row], labels).save(out / f"{name}_montage.png")

    montage(rows, labels).save(out / "all_montage.png")
    (out / "prompt.txt").write_text(
        "\n".join(f"{n}: {next(p for _, c in runs for k, (_, p) in c.items() if k == n)}"
                  for n in images) +
        f"\n\nguidance_scale={args.guidance_scale} strength={args.strength} "
        f"n_edit={args.n_edit} seed={args.seed}\n", encoding="utf-8")
    print(f"[edit] 輸出於 {out}")


if __name__ == "__main__":
    sys.exit(main())

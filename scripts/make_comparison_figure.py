"""產生定性對照圖：加性 vs 非加性防禦在「淨化前後」的編輯結果。

不計算任何指標，只生成可視化 grid，供 meeting 定性展示。重用專案既有模組
（build_protection / edit_image / purify），與 stage1/stage2 同一套設定。

於 repo 根目錄執行（需先 source env.sh 啟用 GPU 環境）：
    python scripts/make_comparison_figure.py [args]

每張輸入影像輸出一張 grid（figures/comparison_<ts>/<image>_grid.png），每列一個
防禦方法，六欄依序為：
    Original | Protected | Purified | Edited(Original) | Edited(Protected) | Edited(Purified)
其中 Edited(Original) 為「無防禦時攻擊成功的樣子」（各列相同，作參考）；
Edited(Protected) 檢視未淨化下防禦是否生效；Edited(Purified) 檢視對手淨化後
防禦是否仍存活。個別 PNG 亦另存於 <image>/ 子目錄，方便單看或放入報告。

時間預估（V100，單張、sdedit 100 步）：pg_enc protect≈15s、advdiff≈9s、
apa≈170s、hybrid≈155s；每方法額外 2 次編輯≈20–30s。4 方法×2 張約 20 分鐘。
pg_diff（≈45min/張）預設不納入；如需請自行加入 --methods 並預留時間。
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from common import load_configs  # noqa: E402（common 會把 repo 根加入 sys.path）

from src.data.dataset import load_dataset
from src.edit import edit_image
from src.models.sd_wrapper import SDWrapper
from src.protect import build_protection
from src.purify import purify
from src.utils.io import save_image
from src.utils.seed import set_seed

# 各淨化手段的預設強度（可由 --purify-strength 覆寫）
PURIFY_DEFAULT = {"jpeg": 65, "blur": 1.5, "crop_resize": 0.2, "advclean_bfgf": None,
                  "advclean_bf": None}


def to_pil(img: torch.Tensor) -> Image.Image:
    """(1,3,H,W) [0,1] → PIL RGB。"""
    a = img.detach().float().clamp(0, 1).squeeze(0).permute(1, 2, 0).cpu().numpy()
    return Image.fromarray((a * 255 + 0.5).astype(np.uint8))


def purify_label(method: str, strength) -> str:
    if method == "jpeg":
        return f"jpeg_q{strength}"
    if method == "blur":
        return f"blur_s{strength}"
    if method == "crop_resize":
        return f"crop_r{strength}"
    return method


def build_grid(path: Path, rows, col_titles, thumb=256):
    """rows: list of (row_name, [PIL × ncol])；輸出拼接大圖。"""
    pad, left, title_h = 6, 104, 18
    ncol = len(col_titles)
    W = left + ncol * (thumb + pad) + pad
    H = title_h + pad + len(rows) * (thumb + pad) + pad
    canvas = Image.new("RGB", (W, H), (245, 245, 245))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for ci, t in enumerate(col_titles):
        draw.text((left + ci * (thumb + pad) + 3, 4), t, fill=(0, 0, 0), font=font)
    y = title_h + pad
    for name, pils in rows:
        draw.text((4, y + 4), name, fill=(0, 0, 0), font=font)
        for ci, p in enumerate(pils):
            canvas.paste(p.resize((thumb, thumb)), (left + ci * (thumb + pad), y))
        y += thumb + pad
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def main():
    ap = argparse.ArgumentParser(description="定性對照圖：淨化前後的防禦效果")
    ap.add_argument("--methods", default="pg_enc,advdiff,apa,hybrid",
                    help="逗號分隔；pg_diff 很貴（≈45min/張）預設不含")
    ap.add_argument("--max-images", type=int, default=2)
    ap.add_argument("--prompt-idx", type=int, default=0, help="用每張圖的第幾個惡意 prompt")
    ap.add_argument("--seed", type=int, default=None, help="固定 seed（預設取 base.runtime.seed）")
    ap.add_argument("--purify", default="jpeg",
                    help="jpeg|blur|crop_resize|advclean_bfgf|advclean_bf")
    ap.add_argument("--purify-strength", default=None)
    ap.add_argument("--sdedit-strength", type=float, default=None,
                    help="覆寫 edit.sdedit_strength（viability 用 0.8）")
    ap.add_argument("--protect-model", default=None)
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()

    cfgs = load_configs()
    base = cfgs["base"]
    model = args.protect_model or base["model"]["protect_model"]
    if args.sdedit_strength is not None:
        base["edit"]["sdedit_strength"] = args.sdedit_strength
    seed = args.seed if args.seed is not None else base["runtime"]["seed"]
    set_seed(seed)

    pm = args.purify
    if args.purify_strength is not None:
        pstr = int(args.purify_strength) if pm == "jpeg" else float(args.purify_strength)
    else:
        pstr = PURIFY_DEFAULT.get(pm)
    plabel = purify_label(pm, pstr)

    calibrated = (Path(__file__).resolve().parents[1] / "configs"
                  / "nonadditive_calibrated.yaml").exists()
    ts = time.strftime("%Y%m%d_%H%M%S")
    outdir = Path(args.outdir or f"figures/comparison_{ts}")
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"model={model} | seed={seed} | purify={plabel} | "
          f"sdedit_strength={base['edit']['sdedit_strength']} | "
          f"calibrated_overlay={'yes' if calibrated else 'NO（用 nonadditive.yaml 預設 eps，未 stage0 校準）'}",
          flush=True)

    sd = SDWrapper(model)
    data = load_dataset(base, max_images=args.max_images)
    methods = args.methods.split(",")
    col_titles = ["Original", "Protected", f"Purified({plabel})",
                  "Edited(Original)", "Edited(Protected)", "Edited(Purified)"]

    for sample in data:
        sid = sample["image_id"].replace("/", "__")
        x, concept = sample["image"], sample["concept"]
        prompt = sample["edit_prompts"][args.prompt_idx]
        idir = outdir / sid
        idir.mkdir(parents=True, exist_ok=True)
        print(f"\n=== {sid} | concept={concept} | prompt='{prompt}' ===", flush=True)

        save_image(idir / "original.png", x)
        t0 = time.time()
        edited_orig = edit_image(x, prompt, seed, "sdedit", config=base, sd=sd)
        save_image(idir / "edited_original.png", edited_orig)
        print(f"  edited(original): {time.time() - t0:.0f}s", flush=True)
        eo_pil, x_pil = to_pil(edited_orig), to_pil(x)

        rows = []
        for mkey in methods:
            print(f"  [{mkey}] protecting...", flush=True)
            t0 = time.time()
            protector = build_protection(mkey, sd, cfgs["additive"], cfgs["nonadditive"])
            protected = protector.protect(x, concept)
            tp = time.time() - t0
            purified = purify(protected, pm, pstr, config=cfgs["purify"])
            t1 = time.time()
            edited_prot = edit_image(protected, prompt, seed, "sdedit", config=base, sd=sd)
            edited_pur = edit_image(purified, prompt, seed, "sdedit", config=base, sd=sd)
            print(f"  [{mkey}] protect={tp:.0f}s edit×2={time.time() - t1:.0f}s "
                  f"peak={protector.peak_memory_mb():.0f}MB", flush=True)

            save_image(idir / f"{mkey}_protected.png", protected)
            save_image(idir / f"{mkey}_purified_{plabel}.png", purified)
            save_image(idir / f"{mkey}_edited_protected.png", edited_prot)
            save_image(idir / f"{mkey}_edited_purified_{plabel}.png", edited_pur)

            rows.append((mkey, [x_pil, to_pil(protected), to_pil(purified),
                                eo_pil, to_pil(edited_prot), to_pil(edited_pur)]))
            del protector
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        build_grid(outdir / f"{sid}_grid.png", rows, col_titles)
        print(f"  grid → {outdir / (sid + '_grid.png')}", flush=True)

    print(f"\n完成，輸出於 {outdir}", flush=True)


if __name__ == "__main__":
    main()

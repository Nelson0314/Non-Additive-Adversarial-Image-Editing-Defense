"""把 retention 量過的每一格「淨化後的影像」與「淨化後的編輯」存成 PNG。

`phase_retention.py` 只回報數字，中間影像用完即棄——於是報告裡的抗淨化那一
段沒有圖可看，違反本專案「每一格都必須有影像」的判準。本腳本不重算任何數
字，只是把同一條路徑上的兩張圖存下來：

    x_pur  = purifier.evaluate(x_def)
    x_edit = sdedit(x_pur, prompt, EDIT_SEED)

編輯只跑 `EDIT_SEED` 這一個種子（retention 的三個種子是為了估分母的標準差，
看圖不需要三份）。故本腳本的輸出**不參與任何統計**，純供人眼。

用法：
    python scripts/hb5_purify_gallery.py --out runs/hb5/purified \
        --runs runs/hb5 runs/hb5_pgc --data data/lo_aligned
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402

from apa_baseline import (  # noqa: E402
    EDIT_GUIDANCE, EDIT_SEED, EDIT_STEPS, EDIT_STRENGTH, MODEL_NAME,
    RESOLUTION, load_dataset,
)
from phase_retention import cell_of, label, purifier_set  # noqa: E402
from src.models.sd import SDWrapper  # noqa: E402
from src.utils.artifacts import save_image  # noqa: E402
from src.utils.io import load_image_tensor  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--runs", type=Path, nargs="+", required=True)
    ap.add_argument("--data", type=Path, default=Path("data/lo_aligned"))
    ap.add_argument("--images", nargs="+", default=None)
    ap.add_argument("--conditions", nargs="+", default=None)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    cells = []
    for run in args.runs:
        with (run / "results.csv").open(encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                c = cell_of(r)
                c["run"] = run
                cells.append(c)
    if args.images:
        keep = set(args.images)
        cells = [c for c in cells if c["image"] in keep]
    if args.conditions:
        keep = set(args.conditions)
        cells = [c for c in cells if c["condition"] in keep]
    if not cells:
        raise SystemExit("沒有符合條件的格")

    sd = SDWrapper(MODEL_NAME, dtype=torch.float32)
    purifiers = purifier_set(sd, seed=0)
    dataset = {d["name"]: d for d in load_dataset(args.data)}

    def edit(x01, item):
        noise = sd.sample_edit_noise(sd.encode_image(x01), seed=EDIT_SEED)
        emb, emb_u = sd.encode_text(item["prompt"]), sd.uncond_prompt()
        with torch.no_grad():
            return sd.sdedit(x01.clamp(0, 1), emb, noise, EDIT_STEPS,
                             strength=EDIT_STRENGTH, guidance_scale=EDIT_GUIDANCE,
                             emb_uncond=emb_u)

    n = 0
    for cell in cells:
        item = dataset[cell["image"]]
        def_png = cell["run"] / f"{cell['image']}__{cell['tag']}__def.png"
        if not def_png.exists():
            raise FileNotFoundError(f"缺少防禦圖 {def_png}")
        x_def = load_image_tensor(def_png, sd.device, size=RESOLUTION)
        print(f"=== {cell['image']} / {cell['condition']} ===", flush=True)
        t0 = time.time()
        for p in purifiers:
            name = label(p)
            x_pur = p.evaluate(x_def)
            stem = f"{cell['image']}__{cell['condition']}__{name}"
            save_image(x_pur.clamp(0, 1), args.out / f"{stem}__pur.png")
            save_image(edit(x_pur, item), args.out / f"{stem}__edit.png")
            n += 2
        print(f"    {len(purifiers)} 個算子，{time.time() - t0:.1f}s", flush=True)

    print(f"\n{n} 張圖 → {args.out}")


if __name__ == "__main__":
    main()

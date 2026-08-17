"""用新的頭部遮罩重算編輯，不重跑攻擊。

2026-08-17：五張人物影像的頭部遮罩原本全部偏高，下巴與下顎露在遮罩外
（見 `scripts/make_headmasks.py` 的紀錄）。遮罩只作用在編輯階段的 latent
混合，攻擊本身不讀它，所以 `*__def.png` 完全不受影響——重跑攻擊只會把
photoguard_c 的 6429 s/張再付一次，共 8.9 小時，換到逐位相同的防禦圖。

本腳本只做編輯這一段：讀既有的 `*__def.png`，用新遮罩重算 `edit_orig`
與 `edit_def`，覆寫 `results.csv` 的 `edit_*` 六欄與兩張編輯圖。`fid_*`
不動——它們量的是防禦圖對原圖，與遮罩無關。

一個必須講明的差異：這裡的 `x_def` 是從 8 位元 PNG 讀回來的，原批次用的是
攻擊當下的浮點張量。兩者相差一次量化。淨化階段（`phase_retention.py`）本來
就是讀 PNG，所以重算後的 `edit_*` 與淨化階段同源，比原本更可比，不是更不可比。

`edit_orig` 只與原圖和遮罩有關，與條件無關，故每張影像只算一次，再依各條件
的檔名分別存出。

    python scripts/reeval_edits.py --run runs/s0817/merged --data data/set0817
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
    RESOLUTION, head_keep, load_dataset,
)
from phase_retention import cell_of  # noqa: E402
from src.metrics.suite import MetricSuite  # noqa: E402
from src.models.sd import SDWrapper  # noqa: E402
from src.utils.artifacts import save_image  # noqa: E402
from src.utils.io import load_image_tensor, write_csv  # noqa: E402

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--images", nargs="+", default=None,
                    help="預設是所有帶頭部遮罩的影像")
    ap.add_argument("--prompt-index", type=int, default=0)
    args = ap.parse_args()

    csv_path = args.run / "results.csv"
    with csv_path.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise FileNotFoundError(f"{csv_path} 讀不到任何列")

    sd = SDWrapper(MODEL_NAME, dtype=torch.float32)
    suite = MetricSuite(device=sd.device)
    dataset = {d["name"]: d for d in load_dataset(args.data, prompt_index=args.prompt_index)}

    targets = args.images or [n for n, d in dataset.items() if d.get("head_mask")]
    missing = [n for n in targets if n not in dataset]
    if missing:
        raise KeyError(f"資料集裡沒有這些影像：{missing}")
    no_mask = [n for n in targets if not dataset[n].get("head_mask")]
    if no_mask:
        raise ValueError(f"這些影像沒有頭部遮罩，重算沒有意義：{no_mask}")
    print(f"重算 {len(targets)} 張影像的編輯：{' '.join(targets)}", flush=True)

    n_done = 0
    for name in targets:
        item = dataset[name]
        x01 = item["path01"]
        keep = head_keep(item, x01)
        if keep is None:
            raise ValueError(f"{name}：head_keep 回傳 None，遮罩沒有被讀到")

        emb, emb_u = sd.encode_text(item["prompt"]), sd.uncond_prompt()
        noise = sd.sample_edit_noise(sd.encode_image(x01), seed=EDIT_SEED)
        with torch.no_grad():
            edit_orig = sd.sdedit(x01, emb, noise, EDIT_STEPS,
                                  strength=EDIT_STRENGTH, guidance_scale=EDIT_GUIDANCE,
                                  emb_uncond=emb_u, keep01=keep)
        so = suite.semantic(edit_orig, item["prompt"])

        for row in rows:
            if row["image"] != name:
                continue
            tag = cell_of(row)["tag"]
            def_png = args.run / f"{name}__{tag}__def.png"
            if not def_png.exists():
                raise FileNotFoundError(def_png)

            t0 = time.time()
            x_def = load_image_tensor(def_png, sd.device, size=RESOLUTION)
            with torch.no_grad():
                edit_def = sd.sdedit(x_def.clamp(0, 1), emb, noise, EDIT_STEPS,
                                     strength=EDIT_STRENGTH,
                                     guidance_scale=EDIT_GUIDANCE,
                                     emb_uncond=emb_u, keep01=keep)
            sdf = suite.semantic(edit_def, item["prompt"])

            before = row.get("edit_lpips")
            row["edit_lpips"] = round(
                float(suite.pairwise(edit_orig, edit_def)["lpips"]), 4)
            row["edit_clip_orig"] = round(so["clip"], 4)
            row["edit_clip_def"] = round(sdf["clip"], 4)
            row["edit_clip_drop"] = round(so["clip"] - sdf["clip"], 4)
            row["edit_siglip_orig"] = round(so["siglip"], 4)
            row["edit_siglip_def"] = round(sdf["siglip"], 4)
            row["edit_siglip_drop"] = round(so["siglip"] - sdf["siglip"], 4)

            save_image(edit_orig, args.run / f"{name}__{tag}__edit_orig.png")
            save_image(edit_def, args.run / f"{name}__{tag}__edit_def.png")
            write_csv(csv_path, rows)
            n_done += 1
            print(f"  {name:12s} {tag:18s} edit_lpips {before} -> "
                  f"{row['edit_lpips']}  ({time.time() - t0:.1f} s)", flush=True)

    print(f"\n重算完成：{n_done} 列 -> {csv_path}")


if __name__ == "__main__":
    main()

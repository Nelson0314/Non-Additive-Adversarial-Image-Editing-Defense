"""各族的防禦圖實際把 `‖E(x_def)‖₂` 壓到哪裡。**不訓練，只讀已存的圖。**

問的是一個歸因問題：色彩族的位移只有相位族的一半上下，**是因為它壓不動
損失，還是因為同樣的損失值換不到同樣的位移**？

`latent_norm` 的定義逐字取自 `scripts/ip2p_run.py`：

    loss(x) = ip2p.encode_image(x).flatten().norm(p=2)      （要最小化）

所以把各族已存的防禦圖丟進同一個編碼器算一次，就得到它們各自實際到達的
損失值，不需要重跑任何訓練。`image_guidance` 的防禦圖也一起量——它訓練時
optimise 的不是這個量，量出來只是為了看它落在哪裡。

用法：

    python scripts/latent_norm_probe.py --out runs/latent_norm_probe/results.csv \\
        --entry phase_gain=runs/ip2p_mainline/ours_pg_n \\
        --entry color_grid_1000=runs/ip2p_color/grid_r010
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from src.utils.io import load_image_tensor, write_csv  # noqa: E402

RESOLUTION = 512


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--entry", action="append", required=True,
                    metavar="LABEL=DIR",
                    help="一族的防禦圖目錄。條件名由該目錄的 results.csv 讀出")
    ap.add_argument("--images", nargs="+", required=True)
    ap.add_argument("--data", type=Path, default=Path("data/omniedit150"))
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    from src.models.ip2p import IP2PWrapper

    ip2p = IP2PWrapper(dtype=torch.float32)
    dev = ip2p.device

    rows = []
    for name in args.images:
        orig = args.data / name / f"{name}.png"
        x = load_image_tensor(orig, dev, size=RESOLUTION)
        with torch.no_grad():
            base = float(ip2p.encode_image(x).flatten().norm(p=2))
        rows.append({"image": name, "label": "original", "condition": "",
                     "latent_norm": round(base, 4), "ratio_to_original": 1.0,
                     "fid_dists": "", "edit_lpips": ""})
        print(f"{name[15:]:26s}{'original':26s}{base:10.3f}", flush=True)

        for spec in args.entry:
            label, _, d = spec.partition("=")
            d = Path(d)
            recs = [r for r in csv.DictReader(
                (d / "results.csv").open(encoding="utf-8"))
                if r["image"] == name]
            if not recs:
                continue
            r = recs[0]
            png = d / f"{name}__{r['condition']}__def.png"
            if not png.exists():
                print(f"[skip] 缺 {png}", flush=True)
                continue
            xd = load_image_tensor(png, dev, size=RESOLUTION)
            with torch.no_grad():
                val = float(ip2p.encode_image(xd).flatten().norm(p=2))
            rows.append({
                "image": name, "label": label, "condition": r["condition"],
                "latent_norm": round(val, 4),
                "ratio_to_original": round(val / base, 4),
                "fid_dists": r.get("fid_dists", ""),
                "edit_lpips": r.get("edit_lpips", ""),
            })
            print(f"{name[15:]:26s}{label:26s}{val:10.3f}  "
                  f"({val / base:.3f}× 原圖)  DISTS={r.get('fid_dists','')} "
                  f"位移={r.get('edit_lpips','')}", flush=True)

    write_csv(args.out, rows)
    print(f"\n寫出 {args.out}")


if __name__ == "__main__":
    main()

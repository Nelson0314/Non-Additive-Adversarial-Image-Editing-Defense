"""對防禦圖或未防禦原圖套用 VAE-Trans 淨化（`immunization_baseline.vae_trans`），輸出編輯器讀取的版面。

輸出 `<--output-root>/<條件>/vae_trans/ip2p_<條件>_vae_trans/<影像>__orig.png`，與主表 ip2p 存下的淨化後輸入同一版面，
`run_flux_edits --purifier vae_trans`、`run_ultraedit_edits --purifiers vae_trans` 以 `--purified-edits-root` 指向同一根目錄讀取。
`--fidelity-csv` 記淨化後 PNG（8 位元）對淨化前輸入的 PSNR 與 LPIPS（`piq.LPIPS`）。

用法（遠端 baseline 專案根，需 GPU）：
    python -m immunization_baseline.cli.apply_vae_trans --encoder <encoder.pt> \\
        --defenses-dir artifacts/cross_model/flux/defenses/dct_shield --output-root <淨化根> --fidelity-csv <CSV>
    python -m immunization_baseline.cli.apply_vae_trans --encoder <encoder.pt> \\
        --data-root data/portraits --output-root <淨化根> --fidelity-csv <CSV>
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch

from immunization_baseline.vae_trans import LABEL, RESOLUTION, VAETrans
from immunization_core.artifacts.images import save_image
from immunization_core.io import load_image_tensor, write_csv
from immunization_core.pipelines.purification import dataset_sources, defended_sources


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument("--defenses-dir", dest="defended", type=Path, help="`<影像>__<條件>__def.png` 所在目錄")
    source.add_argument("--data-root", dest="data", type=Path, help="資料集根目錄，淨化未防禦的原圖")
    ap.add_argument("--encoder", type=Path, required=True)
    ap.add_argument("--output-root", dest="out", type=Path, required=True)
    ap.add_argument("--fidelity-csv", type=Path, required=True)
    ap.add_argument("--condition", default=None,
                    help="輸出目錄與 CSV 的條件名；預設取 --defenses-dir 的目錄名，或未防禦時的 `undefended`")
    args = ap.parse_args()

    if args.defended is not None:
        sources = defended_sources(args.defended)
        condition = args.condition or args.defended.name
    else:
        sources = dataset_sources(args.data)
        condition = args.condition or "undefended"

    import piq

    device = torch.device("cuda")
    purify = VAETrans(args.encoder, device)
    lpips = piq.LPIPS().to(device)
    target = args.out / condition / LABEL / f"ip2p_{condition}_{LABEL}"
    target.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, path in sources.items():
        x = load_image_tensor(path, device, size=RESOLUTION)
        out = target / f"{name}__orig.png"
        save_image(purify(x), out)
        with torch.no_grad():
            y = load_image_tensor(out, device)
            mse = float(((y - x) ** 2).mean())
            rows.append({"condition": condition, "purifier": LABEL, "image": name,
                         "psnr": round(10 * math.log10(1.0 / mse), 4),
                         "lpips": round(float(lpips(y, x).mean()), 5),
                         "source_png": path.as_posix(), "purified_png": out.as_posix(),
                         "encoder": args.encoder.as_posix(), "decoder_model": purify.model_name})
        print(f"[PURIFIED] {condition} {name} lpips={rows[-1]['lpips']}", flush=True)
    write_csv(args.fidelity_csv, rows)
    print(f"[ALLDONE] {args.fidelity_csv}（{len(rows)} 列）", flush=True)


if __name__ == "__main__":
    main()

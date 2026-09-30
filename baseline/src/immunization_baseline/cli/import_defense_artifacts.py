"""把顏色方法的產物整理成與其他防禦條件同一個版面，好進同一張主表。

為什麼要這一支
────────────────────────────────────────────────────────────────────
`../scripts/immunise.py` 的輸出是逐結構的 `<名稱>__<結構>__def.png` 加上選中的
`<名稱>__immunised.png`；`../scripts/paper_baseline.py` 的輸出是
`<名稱>__<臂>__defended.png`；`generate_defenses` 的輸出則是
`<名稱>__<條件>__def.png` 與一份 `results*.csv`。下游（`--defended`、淨化、
版面、主表）都照最後那個版面讀檔，所以前兩者要先換成那個版面才能並列。

`--variant` 指定 `paper_baseline.py` 的臂名；不給就找 `immunise.py` 選中的
`<名稱>__immunised.png`。

**保真那一欄在這裡重算，不從 `immunise.csv` 搬。** 那份 CSV 的欄位是求解端
自己的診斷量（score、違反量、λ），與其他條件的 `fid_*` 不是同一個定義；
搬過來會讓主表同一欄底下混進兩種量法。這裡走 `MetricSuite.pairwise` 與
`standard_row`，與 `generate_defenses` 同一段程式、同一份權重。

用法
    python -m immunization_baseline.cli.import_defense_artifacts --run <求解輸出目錄> \\
        --out artifacts/defenses/colour_curve_ours
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from immunization_baseline import layout  # noqa: E402

import torch  # noqa: E402

from immunization_core.metrics.standard import standard_row  # noqa: E402
from immunization_core.metrics.suite import MetricSuite  # noqa: E402
from immunization_core.artifacts.images import save_image  # noqa: E402
from immunization_core.io import load_image_tensor, write_csv  # noqa: E402

RESOLUTION = 512

#: 求解端看到的文字條件。顏色線**整條求解路徑不含任何文字**：
#: `immunise.py` 與 `paper_baseline.py` 都會遞迴檢查設定檔，讀到 instruction
#: 或 prompt 的鍵就拒絕啟動。
SOLVER_PROMPT = ("", "無文字條件：三個項都不經過 text encoder"
                     "（設定檔的 assert_no_instructions 擋下含指令的設定）")


def dataset_images(root: Path) -> dict:
    out = {}
    for directory in sorted(p for p in root.iterdir() if p.is_dir()):
        if directory.name in ("masks", "headmasks", "overview"):
            continue
        for image in sorted(directory.glob("*.png")):
            out[image.stem] = (image, directory.name)
    if not out:
        raise SystemExit(f"{root} 底下找不到任何影像")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, required=True,
                        help="immunise.py 的輸出目錄")
    parser.add_argument("--data", type=Path, default=layout.PORTRAITS)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--condition", default=None,
                        help="條件名。預設取 --variant，沒有 --variant 時是 colour")
    parser.add_argument("--variant", default=None,
                        help="paper_baseline.py 的臂名，對應 "
                             "`<名稱>__<臂>__defended.png`")
    parser.add_argument("--norm", default="delta_e00_cap",
                        help="束縛的種類，逐列寫進 CSV。**不同臂不一樣**："
                             "有色差上限的填 delta_e00_cap，只有半徑的填 "
                             "advcf_radius，填錯等於在表上宣告一個不存在的預算")
    parser.add_argument("--budget", default="",
                        help="該束縛的數值（例如 16.0 或 radius=1.0），寫進 eps 欄")
    args = parser.parse_args()
    if args.condition is None:
        args.condition = args.variant or "colour"

    sources = dataset_images(args.data)
    stem = (f"__{args.variant}__defended.png" if args.variant
            else "__immunised.png")
    picked = {}
    for name in sources:
        path = args.run / f"{name}{stem}"
        if not path.is_file():
            raise SystemExit(
                f"{path} 不存在：還沒為 {name} 產出這個臂的防禦圖。"
                "逐張齊全才整理，缺一張會讓這個條件的分母與其他條件不同")
        picked[name] = path

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    suite = MetricSuite(device=device)
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, defended in picked.items():
        source, cls = sources[name]
        started = time.time()
        x01 = load_image_tensor(source, device, size=RESOLUTION)
        y01 = load_image_tensor(defended, device, size=RESOLUTION)
        with torch.no_grad():
            pair = suite.pairwise(x01, y01)
        save_image(x01, args.out / f"{name}__orig.png")
        save_image(y01, args.out / f"{name}__{args.condition}__def.png")
        rows.append({
            "image": name, "class": cls, "condition": args.condition,
            "solver_prompt": SOLVER_PROMPT[0],
            "solver_prompt_source": SOLVER_PROMPT[1],
            "seed": "",
            # 顏色線沒有 L∞／L2 預算，束縛的種類逐臂不同，由 --norm 指定：
            # `colour_curve_ours` 是 CIEDE2000 上限（投影施加），
            # `advcf_paper` 只有 AdvCF 自己的半徑，**沒有色差上限**。
            "eps": args.budget, "eps_pixel01": "", "norm": args.norm,
            "steps": "", "grad_reps": "", "q_alg": "", "pg_strength": "",
            "modified_from_paper": "", "modification_note": "",
            "spec_source": (f"../scripts/paper_baseline.py 的 {args.variant} 臂"
                            if args.variant else
                            "本專案的顏色載體（../scripts/immunise.py）"),
            "total_seconds": "",
            "fid_psnr": round(float(pair["psnr"]), 4),
            "fid_lpips": round(float(pair["lpips"]), 4),
            "fid_rms": round(float(pair["rms"]), 6),
            "fid_linf": round(float(pair["linf"]), 6),
            **standard_row("fid_", pair),
            "data_root": str(args.data).replace("\\", "/"),
            "source_png": defended.as_posix(),
            "measured_seconds": round(time.time() - started, 1),
        })
        print(f"[DONE] {name:12s} psnr={pair['psnr']:.3f} "
              f"lpips={pair['lpips']:.4f} rms={pair['rms']:.5f}", flush=True)
    write_csv(args.out / "results_all.csv", rows)
    print(f"[ALLDONE] {args.out / 'results_all.csv'}（{len(rows)} 列）", flush=True)


if __name__ == "__main__":
    main()

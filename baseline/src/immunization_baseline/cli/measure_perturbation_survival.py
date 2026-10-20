"""淨化後的擾動存續比：LPIPS(P(x̃), P(x)) ÷ LPIPS(x̃, x)。

x 為原圖、x̃ 為防禦圖、P 為淨化。分子與分母皆讀交付的 8-bit PNG，經 `load_image_tensor(size=512)`，
以 `MetricSuite` 的 `piq.LPIPS` 在 CPU 上計算（防禦圖失真類讀數的裝置約定）。

淨化後的影像取自評測鏈實際送進編輯器的輸入：
    P(x̃)  <淨化根>/<條件>/<算子>/<場景>_<條件>_<算子>/<影像>__orig.png
    P(x)   <淨化根>/undefended/<算子>/<場景>_undefended_<算子>/<影像>__orig.png
另可以 `--extra-purifier 名稱 防禦側樣板 原圖側樣板` 加入其他淨化（例如 VAE-Trans）的輸出，
樣板中的 `{condition}`、`{image}` 會被代換。

輸出每列一個（條件, 影像, 淨化）：`lpips_defense`（分母）、`lpips_purified`（分子）、`survival`。

用法（遠端 baseline 專案根）：
    python -m immunization_baseline.cli.measure_perturbation_survival --defenses-root artifacts/aligned/defenses \\
        --purified-edits-root artifacts/aligned/purified_edits --output-csv <CSV>
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from immunization_core.purifiers.protocol import purifier_labels

RESOLUTION = 512
FIELDS = ("condition", "image", "purifier", "lpips_defense", "lpips_purified", "survival",
          "defended_png", "purified_png", "purified_reference_png", "device")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--defenses-root", type=Path, required=True,
                        help="`<條件>/<影像>__orig.png` 與 `<條件>/<影像>__<條件>__def.png`")
    parser.add_argument("--purified-edits-root", type=Path, required=True)
    parser.add_argument("--scenario", default="ip2p")
    parser.add_argument("--conditions", nargs="+", default=None)
    parser.add_argument("--purifiers", nargs="*", default=purifier_labels(include_identity=False))
    parser.add_argument("--extra-purifier", nargs=3, action="append", default=[],
                        metavar=("NAME", "DEFENSE_TEMPLATE", "REFERENCE_TEMPLATE"))
    parser.add_argument("--output-csv", type=Path, required=True)
    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    conditions = args.conditions or sorted(d.name for d in args.defenses_root.iterdir()
                                           if d.is_dir() and not d.name.startswith("_"))
    plan = []
    for c in conditions:
        for defended in sorted((args.defenses_root / c).glob(f"*__{c}__def.png")):
            image = defended.name.split("__")[0]
            original = defended.with_name(f"{image}__orig.png")
            for p in args.purifiers:
                arm = args.purified_edits_root / c / p / f"{args.scenario}_{c}_{p}"
                ref = (args.purified_edits_root / "undefended" / p
                       / f"{args.scenario}_undefended_{p}")
                plan.append((c, image, p, defended, original,
                             arm / f"{image}__orig.png", ref / f"{image}__orig.png"))
            for name, dt, rt in args.extra_purifier:
                plan.append((c, image, name, defended, original,
                             Path(dt.format(condition=c, image=image)),
                             Path(rt.format(condition=c, image=image))))
    missing = sorted({str(p) for row in plan for p in row[3:] if not p.is_file()})
    if missing:
        raise SystemExit(f"{len(missing)} 個輸入缺檔，首個：{missing[0]}")

    import torch

    from immunization_core.io import load_image_tensor
    from immunization_core.metrics.suite import MetricSuite

    device = torch.device("cpu")
    suite = MetricSuite(device=device)
    load = lambda p: load_image_tensor(p, device, size=RESOLUTION)  # noqa: E731
    base = {}
    rows = []
    for c, image, p, defended, original, pd, pr in plan:
        with torch.no_grad():
            if (c, image) not in base:
                base[(c, image)] = float(suite.lpips_module(load(original), load(defended)))
            num = float(suite.lpips_module(load(pr), load(pd)))
        den = base[(c, image)]
        rows.append({"condition": c, "image": image, "purifier": p,
                     "lpips_defense": round(den, 5), "lpips_purified": round(num, 5),
                     "survival": round(num / den, 5) if den > 0 else "",
                     "defended_png": defended.as_posix(), "purified_png": pd.as_posix(),
                     "purified_reference_png": pr.as_posix(), "device": str(device)})
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    print(f"[ALLDONE] {args.output_csv}（{len(rows)} 列）", flush=True)


if __name__ == "__main__":
    main()

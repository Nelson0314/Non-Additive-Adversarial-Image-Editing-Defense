"""淨增益：淨化洗掉了多少防禦。

定義（`docs/EVALUATION.md`）
────────────────────────────────────────────────────────────────────
    淨增益 = metrics( 編輯(原圖), 編輯(防禦圖) )
           − metrics( 編輯(淨化(原圖)), 編輯(淨化(防禦圖)) )

兩項都是**位移**，量法與 `scripts/edit_displacement.py` 完全相同（同一份 LPIPS
權重、同一個遮罩極性），差別只在輸入有沒有先過一道淨化。

**兩側都要淨化。** 只淨化防禦圖那一側會把「淨化本身改變了畫面」算進位移裡，
那不是防禦被洗掉，是分母動了。所以淨化後那一項的分母是
`runs/edit_purified/undefended/<算子>/`。

怎麼讀
────────────────────────────────────────────────────────────────────
`retained` = 淨化後位移 ÷ 未淨化位移。1.0 代表淨化完全沒洗掉防禦，0 代表洗光。
`net_gain` 是兩者相減的絕對量，與位移同單位。兩個都報：比值在位移本來就小的
條件上會放大雜訊，絕對量在位移大的條件上才有意義。

**幾何類（`crop_resize0.1`、`rotate15`）要分開看**：取景本身被改掉，兩側的
編輯結果會一起偏移，比值同時含「防禦被洗掉」與「畫面被移動」。

用法
    python scripts/edit_retention.py --purified-root runs/edit_purified \\
        --displacement runs/edit_defended/displacement.csv \\
        --data data/portraits --out runs/edit_purified/retention.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402

from src.metrics.regional import RegionalLPIPS, split_displacement  # noqa: E402
from src.metrics.standard import SIGLIP_BLOCKED_THRESHOLD  # noqa: E402
from src.metrics.suite import MetricSuite  # noqa: E402
from src.utils.io import load_image_tensor, write_csv  # noqa: E402

from edit_displacement import subject_mask  # noqa: E402

RESOLUTION = 512
UNDEFENDED = "undefended"
GEOMETRIC = ("crop_resize0.1", "rotate15")


def read_csv(path: Path) -> list:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def cell(root: Path, condition: str, purifier: str, scenario: str,
         name: str, index: str) -> Path:
    arm = f"{scenario}_{condition}_{purifier}"
    return root / condition / purifier / arm / f"{name}__p{index}.png"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--purified-root", type=Path, required=True)
    parser.add_argument("--displacement", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("data/portraits"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    base = {}
    for row in read_csv(args.displacement):
        key = (row["condition"], row["scenario"], row["image"], row["prompt_index"])
        base[key] = row

    conditions = sorted(d.name for d in args.purified_root.iterdir()
                        if d.is_dir() and not d.name.startswith("_")
                        and d.name != UNDEFENDED)
    if not conditions:
        raise SystemExit(f"{args.purified_root} 底下沒有任何條件目錄")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    suite = MetricSuite(device=device)
    regional = RegionalLPIPS(suite.lpips_module)
    masks = {}

    rows = []
    for condition in conditions:
        for purifier_dir in sorted((args.purified_root / condition).iterdir()):
            if not purifier_dir.is_dir():
                continue
            purifier = purifier_dir.name
            for key, row in sorted(base.items()):
                if key[0] != condition:
                    continue
                _, scenario, name, index = key
                a = cell(args.purified_root, UNDEFENDED, purifier, scenario, name, index)
                b = cell(args.purified_root, condition, purifier, scenario, name, index)
                if not a.is_file() or not b.is_file():
                    raise SystemExit(
                        f"缺檔：{a if not a.is_file() else b}。"
                        "淨化後的兩側必須齊全，少一邊就不是同一個分母")
                if name not in masks:
                    repaint = load_image_tensor(args.data / "masks" / f"{name}.png",
                                                device, size=RESOLUTION)[:, :1]
                    masks[name] = subject_mask(repaint)
                xa = load_image_tensor(a, device, size=RESOLUTION)
                xb = load_image_tensor(b, device, size=RESOLUTION)
                with torch.no_grad():
                    split = split_displacement(regional, xa, xb, masks[name])
                    similar = suite.image_similarity(xa, xb)
                plain = float(base[key]["disp_lpips_full"])
                purified = float(split["lpips_full"])
                siglip = float(similar["siglip"])
                rows.append({
                    "condition": condition, "purifier": purifier,
                    "geometric": purifier in GEOMETRIC,
                    "scenario": scenario, "image": name, "prompt_index": index,
                    "disp_plain": round(plain, 5),
                    "disp_purified": round(purified, 5),
                    "net_gain": round(plain - purified, 5),
                    "retained": round(purified / plain, 5) if plain else "",
                    "disp_purified_subject": round(float(split["lpips_subject"]), 5),
                    "disp_purified_background": round(float(split["lpips_background"]), 5),
                    "siglip_pair": round(siglip, 5),
                    "blocked": siglip < SIGLIP_BLOCKED_THRESHOLD,
                    "siglip_blocked_threshold": SIGLIP_BLOCKED_THRESHOLD,
                })
            write_csv(args.out, rows)
            print(f"[DONE] {condition:18s} {purifier:16s} 累計 {len(rows)} 列", flush=True)
    print(f"[ALLDONE] {args.out}（{len(rows)} 列）", flush=True)


if __name__ == "__main__":
    main()

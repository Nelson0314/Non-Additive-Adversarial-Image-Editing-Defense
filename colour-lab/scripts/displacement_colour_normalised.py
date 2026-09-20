"""在既有的 displacement.csv 上，多算一欄「色彩校正還不回去的位移」。

`scripts/edit_displacement.py` 的 `disp_lpips` 是兩張編輯結果的全圖 LPIPS。
這條線的載體是純顏色全域映射，所以那個數裡有一部分是**防禦的色偏原封不動
穿過編輯**，攻擊方對輸出做一次全域色彩校正就還得回去。本腳本把那一部分扣掉：
先對兩張編輯結果做逐通道 CDF 直方圖匹配，再量一次 LPIPS。

輸入的 CSV 必須帶 `defended_png` 與 `undefended_png` 兩欄（`edit_displacement.py`
會寫）。輸出是原欄位加上四個新欄位，原欄位一個都不動。

**兩個數都照報，本腳本不下判準。**

用法
    python scripts/displacement_colour_normalised.py \
        --displacement runs/amplitude_ladder_edits/displacement.csv \
        --out runs/amplitude_ladder_edits/displacement_colour_normalised.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from src.metrics.colour_normalised import colour_normalised_pair  # noqa: E402
from src.metrics.suite import MetricSuite  # noqa: E402
from src.utils.io import load_image_tensor, write_csv  # noqa: E402

RESOLUTION = 512


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--displacement", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    with args.displacement.open(encoding="utf-8", newline="") as stream:
        source = list(csv.DictReader(stream))
    if not source:
        raise SystemExit(f"{args.displacement} 沒有資料列")
    for column in ("defended_png", "undefended_png"):
        if column not in source[0]:
            raise SystemExit(f"輸入的 CSV 缺 {column} 欄，補不出影像路徑")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    suite = MetricSuite(device=device)
    lpips = suite.lpips_module

    rows = []
    for k, row in enumerate(source, start=1):
        a = load_image_tensor(Path(row["undefended_png"]), device, size=RESOLUTION)
        b = load_image_tensor(Path(row["defended_png"]), device, size=RESOLUTION)
        scores = colour_normalised_pair(lpips, a, b)
        rows.append({**row,
                     **{key: round(value, 6) for key, value in scores.items()}})
        if k % 16 == 0:
            print(f"  {k}/{len(source)} 列", flush=True)
        write_csv(args.out, rows)
    print(f"[ALLDONE] {args.out}（{len(rows)} 列）", flush=True)


if __name__ == "__main__":
    main()
